//! Per-worker buffers for increments of saturated objects (`lxr_defer_saturated_incs`).
//!
//! An increment whose target's RC field reads `MAX_REF_COUNT` must add 1 to the target's excess in
//! `OVERFLOW_RC_TABLE`. On workloads with a few very popular objects (`biojava`: ~43) every GC
//! worker does that `fetch_add` on the same few words, and the cache lines bounce between cores.
//! Here each worker instead counts its saturated increments per object in a private table, and the
//! totals are applied with one `fetch_add(n)` per (worker, object):
//!
//! * when a worker's table is half full mid-pause (`SatIncBuf::add` returns `false`), and
//! * once per GC for every worker, at the end of the pause. With the 1024-slot table (RC field of
//!   8 bits or more) `LXR::release` drains every table itself, on one thread. With a larger table
//!   ([`PARALLEL_FLUSH`]) it instead adds one [`SatIncFlush`] packet per non-empty table to the
//!   `Release` bucket, so the GC threads drain them in parallel, still inside the pause.
//!
//! The parallel flush keeps the same order as the old serial drain in `LXR::release`: `Release`
//! opens only after every `ProcessIncs` has finished (all workers parked with every open bucket
//! empty), and every reader of the true count comes later -- a `RefCount` pause's decrements are
//! released only when the pause ends (`on_gc_finished`), which cannot happen while `Release` holds
//! packets; `FullRC` (or STW) decrements run in `STWRCDecsAndSweep`, and `sanity` in `Final`, both
//! stages that open only after `Release` is empty. Each packet owns one worker's table, so no two
//! packets touch the same table.
//!
//! # Why applying late is safe
//!
//! Increments only commute with other increments. `overflow_rc_cache.rs` invariant 2 already says
//! increments (`ProcessIncs`, STW) and decrements (`ProcessDecs`, after the pause, or in
//! `STWRCDecsAndSweep` for `FullRC`) never overlap; the decrements of a pause are released into a
//! runnable bucket only by `release_rc` (FullRC) or `on_gc_finished` (RefCount), both after the
//! drain. So every buffered increment is in the table before any decrement can read it.
//!
//! While an increment sits in a buffer the object's RC field reads `MAX_REF_COUNT`, which is all
//! any in-pause reader looks at, except the `sanity`-only `rc.rs` check for a true count of exactly
//! `2 * MAX_REF_COUNT`, which merely chooses which objects the sanity GC re-checks.
//!
//! See `~/mmtk/SATURATED_INCS_DEFERRAL_PLAN.md` (stage S1 only; S2 is not implemented).
//!
//! # Second use: cycle-collector mark decrements (`lxr_defer_saturated_cc_decs`)
//!
//! A separate set of buffers (`LXR::sat_cc_decs`) counts the decrements `mark` applies to children
//! whose RC field reads `MAX_REF_COUNT`. On `biojava` these are ~22 M per timed iteration, almost all
//! on a few dozen very popular objects, and every mark worker CASes the same few excess words. The
//! counts here are *decrements*; `LXR::apply_saturated_cc_decs` applies them.
//!
//! Applying them late is safe for the same reason as the increments: nothing that depends on an
//! exact count runs while they are buffered. During mark nothing increments (`ProcessIncs` is in
//! the pause, `scan_black` runs after mark) and no other phase decrements (`ProcessDecs` has
//! drained before cycle collection starts; `collect_*` run after scan). Mark itself reads only the
//! RC *field*, which stays `MAX_REF_COUNT` while a decrement is buffered. `scan` is the first
//! reader of the true count, and the buffers are drained before it can start: in
//! `LXR::on_lazy_mark_finished`, which runs only after the last mark token is dropped, and after
//! the mark loop of `all_buff_gc` for `FullRC`. A mark decrement never kills an object (death is in
//! collect), so the apply has no death or candidate side effects; the only boundary is the excess
//! running out, handled in `RefCountWithOverflow::dec_saturated_n`.

use std::cell::UnsafeCell;

use crossbeam::utils::CachePadded;

use super::LXR;
use crate::scheduler::{GCWork, GCWorker};
use crate::util::{Address, ObjectReference};
use crate::vm::VMBinding;
use crate::MMTK;

/// Table size. Power of two; 12 bytes of key and count per slot, per worker. Depends on the RC
/// field width: below 8 bits far more objects saturate, so the table is 4x larger (48 KiB) to keep
/// it from filling and flushing as often; at 8 bits or more it is 1024 (12 KiB).
const CAPACITY: usize = if crate::util::rc::REF_COUNT_BITS >= 8 { 1024 } else { 4096 };
const LOG_CAPACITY: u32 = CAPACITY.trailing_zeros();
/// The table takes no new object once it holds this many: kept at half occupancy so linear-probe
/// runs stay short on every lookup.  Counting an object already present is always allowed.
const MAX_ENTRIES: usize = CAPACITY / 2;
/// Whether the end-of-pause flush is split into one `SatIncFlush` packet per table. At 1024 slots
/// or fewer (at most 512 entries per table) one thread drains everything in `LXR::release`; with
/// the 4096-slot table (RC field below 8 bits) the tables are large enough to split.
pub const PARALLEL_FLUSH: bool = CAPACITY > 1024;
/// Linear probing gives up after this many slots and reports the table full.  At most half full
/// this is a backstop for an unlucky cluster, not the normal trigger.
const MAX_PROBE: usize = 16;

pub struct SatIncBuf {
    /// `ObjectReference` as an address; 0 is empty.
    keys: Box<[usize; CAPACITY]>,
    counts: Box<[u32; CAPACITY]>,
    /// Occupied slots, so draining and clearing cost O(entries) rather than O(CAPACITY).
    used: Vec<u16>,
}

impl SatIncBuf {
    fn new() -> Self {
        Self {
            keys: Box::new([0; CAPACITY]),
            counts: Box::new([0; CAPACITY]),
            used: Vec::with_capacity(CAPACITY),
        }
    }

    #[inline(always)]
    fn hash(key: usize) -> usize {
        // Fibonacci hashing of the 8-byte-aligned address.
        ((key >> 3).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (usize::BITS - LOG_CAPACITY)) as usize
    }

    /// Count one increment of `o`. Returns `false`, changing nothing, if `o` is not in the table
    /// and the table already holds `MAX_ENTRIES` objects (or the probe run is too long); the caller
    /// then drains the table and retries.
    #[inline(always)]
    pub fn add(&mut self, o: ObjectReference) -> bool {
        let key = o.to_raw_address().as_usize();
        let mut i = Self::hash(key);
        for _ in 0..MAX_PROBE {
            let k = self.keys[i];
            if k == key {
                self.counts[i] += 1;
                return true;
            }
            if k == 0 {
                if self.used.len() >= MAX_ENTRIES {
                    return false;
                }
                self.keys[i] = key;
                self.counts[i] = 1;
                self.used.push(i as u16);
                return true;
            }
            i = (i + 1) & (CAPACITY - 1);
        }
        false
    }

    pub fn is_empty(&self) -> bool {
        self.used.is_empty()
    }

    /// Hand every (object, count) to `f` and leave the table empty.
    pub fn drain(&mut self, mut f: impl FnMut(ObjectReference, u32)) {
        for &i in &self.used {
            let i = i as usize;
            let o = unsafe { ObjectReference::from_raw_address_unchecked(Address::from_usize(self.keys[i])) };
            f(o, self.counts[i]);
            self.keys[i] = 0;
            self.counts[i] = 0;
        }
        self.used.clear();
    }
}

/// One `SatIncBuf` per GC worker, indexed by `GCWorker::ordinal`.
pub struct SatIncBuffers {
    bufs: Box<[CachePadded<UnsafeCell<SatIncBuf>>]>,
}

// Each buffer is written only by its own worker, and read by another thread only in `drain_all`,
// when no packet that writes it can be running. For the increments the scheduler's park/open-bucket
// handshake orders the two; for the mark decrements it is the SeqCst `fetch_sub` chain on the mark
// counter, whose last decrement runs `on_lazy_mark_finished`.
unsafe impl Sync for SatIncBuffers {}
unsafe impl Send for SatIncBuffers {}

impl SatIncBuffers {
    pub fn new(num_workers: usize) -> Self {
        Self {
            bufs: (0..num_workers)
                .map(|_| CachePadded::new(UnsafeCell::new(SatIncBuf::new())))
                .collect(),
        }
    }

    /// # Safety
    ///
    /// Only the worker with this `ordinal` may call it, and the returned reference must not outlive
    /// the current increment.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get(&self, ordinal: usize) -> &mut SatIncBuf {
        &mut *self.bufs[ordinal].get()
    }

    /// Drain every worker's buffer into `f`.
    ///
    /// # Safety
    ///
    /// No packet that writes these buffers may be running or runnable: for `LXR::sat_incs`, no
    /// `ProcessIncs` (call from the `Release` stage only); for `LXR::sat_cc_decs`, no mark packet
    /// (call from `on_lazy_mark_finished`, or from the single `CycleFullRC` packet after its mark).
    pub unsafe fn drain_all(&self, mut f: impl FnMut(ObjectReference, u32)) {
        for b in self.bufs.iter() {
            (*b.get()).drain(&mut f);
        }
    }

    /// Ordinals of the buffers that hold anything, under the same rule as `drain_all`.
    pub unsafe fn non_empty(&self) -> Vec<usize> {
        (0..self.bufs.len())
            .filter(|&i| !(*self.bufs[i].get()).is_empty())
            .collect()
    }

    /// Drain one worker's buffer into `f`.
    ///
    /// # Safety
    ///
    /// As `drain_all`, for this buffer, and no other thread may drain it at the same time
    /// ([`SatIncFlush`] packets each own one ordinal).
    pub unsafe fn drain_one(&self, ordinal: usize, f: impl FnMut(ObjectReference, u32)) {
        (*self.bufs[ordinal].get()).drain(f);
    }

    /// True if every buffer is empty. For debug assertions, under the same rule as `drain_all`.
    #[cfg(debug_assertions)]
    pub unsafe fn all_empty(&self) -> bool {
        self.bufs.iter().all(|b| (*b.get()).is_empty())
    }
}

/// Apply one worker's buffered saturated increments at the end of the pause, in the `Release`
/// bucket. One packet per non-empty table (see the module doc for why the order is safe).
pub struct SatIncFlush {
    ordinal: usize,
}

impl SatIncFlush {
    pub fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }
}

impl<VM: VMBinding> GCWork<VM> for SatIncFlush {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        // SAFETY: `Release` -- no `ProcessIncs` is running or runnable, and this ordinal belongs
        // to this packet alone.
        unsafe {
            lxr.sat_incs
                .drain_one(self.ordinal, |o, n| lxr.apply_saturated_incs(o, n))
        };
    }
}
