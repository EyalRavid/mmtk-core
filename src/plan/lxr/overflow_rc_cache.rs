//! Exact reference counts for objects whose RC side-metadata field has saturated.
//!
//! When `RefCountHelper::{inc,dec}` reports `Err(MAX_REF_COUNT)` the real count no longer fits in
//! the RC table, and the remainder lives here.
//!
//! # Why this is a side table and not a map
//!
//! Until 2026-09-14 this was a `DashMap<ObjectReference, AtomicUsize>`, and every saturated
//! operation paid a hash, a shard `RwLock`, a probe and an atomic. `FINDINGS.md` 20 measured what
//! that costs: **99.1% of `biojava`'s increments reach this path**, and 99.4% of them come from the
//! cycle collector's trial deletion. Only **43 objects** are ever saturated, so a general-purpose
//! concurrent hash map was managing a population that fits in a cache line.
//!
//! `OVERFLOW_RC_TABLE` replaces it: a 32-bit side-metadata spec at object granularity, so the slot
//! for an object is *computed from its address*. No hashing, no locking, no probing, no allocation,
//! no insertion or removal. The cost is address space, not memory — side metadata is demand-zero
//! mmapped (`memory.rs::dzmmap`, no `MAP_POPULATE`; the explicit zeroing is
//! `cfg(not(target_os = "linux"))`) at 4 KiB pages (`MmapStrategy::SIDE_METADATA` is
//! `HugePageSupport::No`), so pages that are never written are never committed. 43 scattered
//! objects commit at most 43 pages.
//!
//! # Encoding: the table holds the EXCESS, not the count
//!
//! `OVERFLOW_RC_TABLE[o] == true_count - MAX_REF_COUNT`, consulted only when the RC table reads
//! `MAX_REF_COUNT`.
//!
//!     0      not overflowed — the true count is exactly `MAX_REF_COUNT`
//!     n > 0  the true count is `MAX_REF_COUNT + n`
//!
//! Storing the excess rather than the count is what makes **`inc` a single unconditional
//! `fetch_add`**. Under the old encoding the first overflow was a distinct case ("insert an entry
//! seeded at `MAX`"), so `inc` needed to branch on whether an entry existed — and doing that
//! race-free without a lock would have needed a CAS loop. With the excess, "no entry" and "excess
//! zero" are the same state, and `0 -> 1` is just another increment. `inc` is the hot path here, so
//! it gets the cheapest possible form.
//!
//! Zero is therefore the correct initial state and needs no initialisation: it is what demand-zero
//! mmap already provides.
//!
//! # What this encoding deletes
//!
//! The old module documented a transient `MAX_REF_COUNT` slot — the window between the CAS that
//! stored `MAX` and the `remove_if` that dropped the slot, plus slots deliberately left behind when
//! an `inc` raced the removal — and every path had to treat that value as "absent". **None of that
//! exists now.** There is no insertion, no removal, and therefore no window between them; excess 0
//! *is* "absent". The old invariant 3 (every value mutation happens under at least a shard read
//! lock, so `remove_if`'s predicate can decide atomically) is likewise gone with the shard locks.
//!
//! # Invariants
//!
//! 1. **A non-zero excess implies the object's RC field is `MAX_REF_COUNT`.** Every path to death
//!    walks the count back down through `MAX + 1 -> MAX`, which zeroes the excess first;
//!    `Block::rc_dead` needs an all-zero RC table, so an overflowed object can never be reclaimed
//!    while its excess is non-zero. This is what keeps a stale excess safe against address reuse: a
//!    recycled address finds excess 0, i.e. "not overflowed". Note the argument depends on
//!    `lxr_no_mature_evac`: `Block::clear_rc_table` bulk zeroes an RC table without consulting this
//!    table, and its only caller is mature-evacuation sweeping.
//!
//!    ⚠ **This invariant is now load-bearing in a way it was not before.** A map entry was dropped
//!    when it drained; a side-metadata slot is not, so a stale non-zero excess would persist at
//!    that address indefinitely. The invariant says that cannot happen, and `dec` zeroing the
//!    excess before the RC field leaves `MAX` is what enforces it.
//! 2. **`inc` and `dec` never run concurrently with each other.** `ProcessIncs` is STW,
//!    `ProcessDecs` is concurrent, and since the `decs -> sweep -> cc` reorder `CycleCollector` is
//!    a single packet ordered after sweeping, which is itself ordered after decs. What keeps the
//!    *next* GC's increments behind the current GC's concurrent window is the scheduler, not any
//!    LXR-level lock: a GC request becomes a `ScheduleCollection` packet only when the **last GC
//!    worker parks** (`scheduler/scheduler.rs`, `on_last_parked` -> `respond_to_requests`), and a
//!    worker parks only after `poll_schedulable_work` finds every bucket and every other worker's
//!    queue empty. Concurrent work therefore holds the next GC off by occupying a worker.
//!
//!    Nothing enforces this; it is relied upon by the gap between `RefCountHelper::{inc,dec}`
//!    returning `Err(MAX_REF_COUNT)` and this table being consulted, and by the `*_exclusive`
//!    methods below, which additionally require that **no** other thread is doing RC work at all.

#[cfg(feature = "s_rc_stats")]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use crate::util::rc::{RefCountHelper, MAX_REF_COUNT, OVERFLOW_RC_TABLE, RC_DEATH_TRANSIENT};
use crate::util::ObjectReference;
use crate::vm::VMBinding;

const MAX_RC_USIZE: usize = MAX_REF_COUNT as usize;

/// The excess is a `u32` (`log_num_of_bits: 5`). The largest true count observed is ~14.7M on
/// `biojava` (`FINDINGS.md` 20), four orders of magnitude below the wrap point, but a wrap would be
/// silent and catastrophic, so it is asserted rather than assumed.
type Excess = u32;

pub struct RefCountWithOverflow<VM: VMBinding> {
    /// Objects whose excess is currently non-zero, maintained on the 0 <-> non-zero transitions
    /// because a side table cannot be counted the way `DashMap::len` could.
    ///
    /// **Gated on `s_rc_stats`, and the field itself is absent without it.** This is
    /// instrumentation, and a publication build must not carry it at all: the transitions sit on
    /// the saturated RC path, which is 99% of `biojava`'s increment traffic (`FINDINGS.md` 20), so
    /// "only two atomics per overflow episode" is an argument about the workloads measured so far
    /// rather than a guarantee. `EVALUATION_PLAN.md` 6 rule 4, `ARTIFACT.md` 7.
    #[cfg(feature = "s_rc_stats")]
    overflowed: AtomicUsize,
    rc: RefCountHelper<VM>,
}

impl<VM: VMBinding> RefCountWithOverflow<VM> {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "s_rc_stats")]
            overflowed: AtomicUsize::new(0),
            rc: RefCountHelper::NEW,
        }
    }

    #[inline]
    fn excess(&self, o: ObjectReference) -> Excess {
        OVERFLOW_RC_TABLE.load_atomic::<Excess>(o.to_raw_address(), Ordering::Relaxed)
    }

    pub fn inc(&self, o: ObjectReference) -> usize {
        // `self.rc.inc` is a CAS retry loop internally, but it returns exactly one
        // `Result` per logical inc, so matching on that result counts each operation
        // once however many times the CAS was retried.
        match self.rc.inc(o) {
            Ok(prev) => {
                // Fast: the count was below MAX_REF_COUNT and the side-metadata CAS
                // completed the operation. The overflow table was never consulted.
                #[cfg(feature = "lxr_rc_path_stats")]
                super::rc_path_stats::inc_fast();
                prev as usize
            }

            Err(MAX_REF_COUNT) => {
                #[cfg(feature = "lxr_rc_path_stats")]
                super::rc_path_stats::inc_slow();

                // ONE atomic on an address computed from `o`. The old implementation probed the
                // map with `get`, then fell through to `entry` on a miss, paying a second hash and
                // probe under the shard write lock. Both cases are this single instruction now,
                // because excess 0 and "no entry" are the same state.
                let prev: Excess = OVERFLOW_RC_TABLE.fetch_add_atomic::<Excess>(
                    o.to_raw_address(),
                    1,
                    Ordering::Relaxed,
                );
                debug_assert!(prev != Excess::MAX, "overflow excess wrapped for {:?}", o);
                #[cfg(feature = "s_rc_stats")]
                if prev == 0 {
                    self.overflowed.fetch_add(1, Ordering::Relaxed);
                }
                MAX_RC_USIZE + prev as usize
            }

            Err(other) => {
                panic!("unexpected RC increment error: {:?}", other);
            }
        }
    }

    pub fn dec(&self, o: ObjectReference) -> usize {
        // As in `inc`: one `Result` per logical dec regardless of internal CAS retries.
        match self.rc.dec(o) {
            Ok(prev_rc) => {
                #[cfg(feature = "lxr_rc_path_stats")]
                super::rc_path_stats::dec_fast();
                prev_rc as usize
            }

            Err(MAX_REF_COUNT) => {
                // Counted once here, covering both outcomes below -- the excess was decremented,
                // or it was already zero and the RC table itself is stepped down. That fallback
                // re-enters `RefCountHelper`, not this method, so it cannot count a second time.
                #[cfg(feature = "lxr_rc_path_stats")]
                super::rc_path_stats::dec_slow();

                let addr = o.to_raw_address();

                // A plain `fetch_sub` would underflow when the excess is already 0, which is the
                // legitimate "true count is exactly MAX_REF_COUNT" case, so the decrement keeps the
                // CAS retry loop the map version had. `inc` needs no equivalent because increasing
                // has no boundary to respect.
                let mut cur: Excess = self.excess(o);
                while cur > 0 {
                    match OVERFLOW_RC_TABLE.compare_exchange_atomic::<Excess>(
                        addr,
                        cur,
                        cur - 1,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            #[cfg(feature = "s_rc_stats")]
                            if cur == 1 {
                                self.overflowed.fetch_sub(1, Ordering::Relaxed);
                            }
                            return MAX_RC_USIZE + cur as usize;
                        }
                        Err(observed) => cur = observed,
                    }
                }

                // Excess zero: the real RC was exactly MAX_REF_COUNT, so step the table down from
                // MAX to MAX - 1. `dec` guards MAX as sticky, hence the unconditional variant.
                self.rc.dec_unconditionally(o) as usize
            }

            Err(other) => {
                panic!("unexpected RC decrement error: {:?}", other);
            }
        }
    }

    /// Single-writer [`Self::inc`].
    ///
    /// Identical in behaviour and return value, but reaches the RC table through
    /// [`RefCountHelper::inc_exclusive`] and updates the excess with a plain load/store instead of
    /// a `fetch_add`. Both replacements drop an atomic read-modify-write.
    ///
    /// # Safety
    ///
    /// The caller must be the only thread performing RC operations for the duration of the call,
    /// which is stronger than invariant 2 above: that invariant only excludes `inc` against `dec`.
    /// See [`RefCountHelper::inc_exclusive`] for what a violation costs.
    pub unsafe fn inc_exclusive(&self, o: ObjectReference) -> usize {
        let prev = unsafe { self.rc.inc_exclusive(o) };

        if prev != MAX_REF_COUNT {
            #[cfg(feature = "lxr_rc_path_stats")]
            super::rc_path_stats::inc_fast();
            return prev as usize;
        }

        #[cfg(feature = "lxr_rc_path_stats")]
        super::rc_path_stats::inc_slow();

        let addr = o.to_raw_address();
        let cur: Excess = self.excess(o);
        debug_assert!(cur != Excess::MAX, "overflow excess wrapped for {:?}", o);
        OVERFLOW_RC_TABLE.store_atomic::<Excess>(addr, cur + 1, Ordering::Relaxed);
        #[cfg(feature = "s_rc_stats")]
        if cur == 0 {
            self.overflowed.fetch_add(1, Ordering::Relaxed);
        }
        MAX_RC_USIZE + cur as usize
    }

    /// Single-writer [`Self::dec`].
    ///
    /// Identical in behaviour and return value, but reaches the RC table through
    /// [`RefCountHelper::dec_exclusive`] / [`RefCountHelper::dec_unconditionally_exclusive`], and
    /// walks the excess down with a load/compare/store instead of a `compare_exchange` retry loop.
    /// Under exclusive access the CAS can never fail, so the loop is dead weight.
    ///
    /// The old map version carried a `debug_assert!(cur > MAX_RC_USIZE)` here, guarding against a
    /// transient `MAX_REF_COUNT` slot left behind by the concurrent `dec` between its CAS and its
    /// `remove`. **That state cannot exist under the excess encoding** -- there is no insertion or
    /// removal to be caught between -- so the assertion is not merely unnecessary, it is
    /// unexpressible. Excess 0 is the ordinary "not overflowed" case, handled below.
    ///
    /// # Safety
    ///
    /// As [`Self::inc_exclusive`].
    pub unsafe fn dec_exclusive(&self, o: ObjectReference) -> usize {
        let prev = unsafe { self.rc.dec_exclusive(o) };

        if prev != MAX_REF_COUNT {
            // `dec` panics on `Err(0)`; `dec_exclusive` reports the same condition as a returned
            // 0, so reproduce the panic rather than silently returning it as a count.
            debug_assert!(prev != 0, "unexpected RC decrement error: {:?}", prev);

            #[cfg(feature = "lxr_rc_path_stats")]
            super::rc_path_stats::dec_fast();
            return prev as usize;
        }

        #[cfg(feature = "lxr_rc_path_stats")]
        super::rc_path_stats::dec_slow();

        let addr = o.to_raw_address();
        let cur: Excess = self.excess(o);

        if cur > 0 {
            OVERFLOW_RC_TABLE.store_atomic::<Excess>(addr, cur - 1, Ordering::Relaxed);
            #[cfg(feature = "s_rc_stats")]
            if cur == 1 {
                self.overflowed.fetch_sub(1, Ordering::Relaxed);
            }
            MAX_RC_USIZE + cur as usize
        } else {
            // Excess zero: the real RC was exactly MAX_REF_COUNT, so step the table down from MAX
            // to MAX - 1. `dec_exclusive` will not do this -- it guards MAX as sticky -- so this
            // needs the unconditional variant, exactly as `dec` needs `dec_unconditionally`.
            unsafe { self.rc.dec_unconditionally_exclusive(o) as usize }
        }
    }

    pub fn get(&self, o: ObjectReference) -> usize {
        let table_rc = self.rc.count(o) as usize;

        if table_rc != MAX_RC_USIZE {
            // Fast: a single side-metadata load answered the query. Note `get`'s boundary
            // is a plain load-and-compare, not a CAS result like `inc`/`dec`.
            #[cfg(feature = "lxr_rc_path_stats")]
            super::rc_path_stats::get_fast();
            return table_rc;
        }

        // Slow: saturated, so the excess must be added to tell MAX_REF_COUNT from anything above
        // it. An excess of 0 returns `table_rc` unchanged, which is what a map miss used to do.
        #[cfg(feature = "lxr_rc_path_stats")]
        super::rc_path_stats::get_slow();

        MAX_RC_USIZE + self.excess(o) as usize
    }

    pub fn is_alive(&self, o: ObjectReference) -> bool {
        self.rc.count(o) as usize > RC_DEATH_TRANSIENT
    }

    /// Objects whose excess is currently non-zero, i.e. currently overflowed.
    ///
    /// **Only exists under `s_rc_stats`**, along with the counter behind it; its one caller
    /// (`ProcessIncs`'s `inc.overflow_peak` flush) is gated the same way. The old `capacity()` had
    /// no meaning for a side table and is gone -- its only caller was a commented-out `println` in
    /// `gc_work.rs`.
    #[cfg(feature = "s_rc_stats")]
    pub fn num_entries(&self) -> usize {
        self.overflowed.load(Ordering::Relaxed)
    }
}
