//! Exact reference counts for objects whose RC side-metadata field has saturated.
//!
//! When `RefCountHelper::{inc,dec}` reports `Err(MAX_REF_COUNT)` the real count no longer fits in
//! the RC table, and the remainder lives here.
//!
//! # Encoding
//!
//! A map value of `MAX_REF_COUNT` means *no overflow* — it is indistinguishable from having no
//! entry at all. `dec` walks an overflowed count down through `MAX + 1 -> MAX` and then drops the
//! slot, so in the steady state the map holds only objects that are *currently* overflowed.
//!
//! The `MAX_REF_COUNT` value is nevertheless a legal, transient state that every path must handle,
//! because the removal cannot be fused with the decrement: the CAS runs under a shard **read** lock
//! and `remove_if` needs the shard **write** lock, and DashMap's shard lock is not reentrant. Two
//! things can therefore be observed by another thread:
//!
//! * the window between the CAS that stores `MAX` and the `remove_if` that drops the slot, and
//! * a slot deliberately left behind because an `inc` raced in and pushed the value back above
//!   `MAX`, which makes the removal predicate fail (see `dec`).
//!
//! Treating `MAX_REF_COUNT` as "absent" everywhere is what makes both harmless: `inc` on such a
//! slot `fetch_add`s it to `MAX + 1` exactly as a fresh insert would, `dec` falls through to
//! `dec_unconditionally`, and `get` returns the table value it already read.
//!
//! # Why the value is an `AtomicUsize`
//!
//! With a plain `usize` value every increment would need DashMap's shard **write** lock. Roughly
//! one in seven to one in ten RC operations reaches this path (`rc_path.*` counters), and that
//! traffic concentrates on a small number of heavily-referenced objects — so the hot objects would
//! re-serialise on their shard. Holding an atomic instead means the common case takes only a shard
//! **read** lock, which readers do not contend on, and the mutation is a `fetch_add`/CAS on the
//! value itself. The shard write lock is then taken only when an address first overflows and when
//! it stops being overflowed.
//!
//! # Invariants
//!
//! 1. **An entry with value `> MAX_REF_COUNT` implies the object's RC field is `MAX_REF_COUNT`.**
//!    Every path to death walks the count back down through `MAX + 1 -> MAX`, which empties the
//!    entry first; `Block::rc_dead` needs an all-zero RC table, so an overflowed object can never
//!    be reclaimed while its entry is live. This is what keeps a stranded entry safe against
//!    address reuse: a recycled address either finds no entry or finds `MAX_REF_COUNT`, i.e.
//!    "absent". Note the argument depends on `lxr_no_mature_evac`: `Block::clear_rc_table` bulk
//!    zeroes an RC table without consulting this map, and its only caller is mature-evacuation
//!    sweeping.
//! 2. **`inc` and `dec` never run concurrently with each other.** `ProcessIncs` is STW,
//!    `ProcessDecs` is concurrent, and since the `decs -> sweep -> cc` reorder `CycleCollector` is
//!    a single packet ordered after sweeping, which is itself ordered after decs. What keeps the
//!    *next* GC's increments behind the current GC's concurrent window is the scheduler, not any
//!    LXR-level lock: a GC request becomes a `ScheduleCollection` packet only when the **last GC
//!    worker parks** (`scheduler/scheduler.rs`, `on_last_parked` -> `respond_to_requests`), and a
//!    worker parks only after `poll_schedulable_work` finds every bucket and every other worker's
//!    queue empty. Concurrent work therefore holds the next GC off by occupying a worker.
//!
//!    *This justification changed on 2026-08-24.* It previously rested on the
//!    `decide_cycle_collection` condvar, which has since been removed. The condvar was never the
//!    binding constraint — it was released from `on_lazy_sweeping_finished`, i.e. inside the last
//!    packet's `drop`, and so was always already granted by the time `ScheduleCollection` ran.
//!    Nothing enforces this invariant; it is relied upon by the gap between
//!    `RefCountHelper::{inc,dec}` returning `Err(MAX_REF_COUNT)` and this map being consulted, and
//!    by the `*_exclusive` methods below, which additionally require that **no** other thread is
//!    doing RC work at all.
//! 3. **Every mutation of an entry's value happens while at least a shard read lock is held.**
//!    This is what lets `remove_if` decide atomically whether a slot is still empty: its predicate
//!    runs under the shard write lock, which excludes every `fetch_add`/CAS below.

use std::sync::atomic::{AtomicUsize, Ordering};

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;

use crate::util::rc::{RefCountHelper, MAX_REF_COUNT, RC_DEATH_TRANSIENT};
use crate::util::ObjectReference;
use crate::vm::VMBinding;

const MAX_RC_USIZE: usize = MAX_REF_COUNT as usize;

/// Initial slot count, split across DashMap's shards (`available_parallelism() * 4`, rounded up to
/// a power of two), so this is a per-shard hint rather than a global bound. The map grows if it is
/// wrong.
const DEFAULT_CAPACITY: usize = 1 << 12;

pub struct RefCountWithOverflow<VM: VMBinding> {
    entries: DashMap<ObjectReference, AtomicUsize>,
    rc: RefCountHelper<VM>,
}

impl<VM: VMBinding> RefCountWithOverflow<VM> {
    pub fn new() -> Self {
        Self {
            entries: DashMap::with_capacity(DEFAULT_CAPACITY),
            rc: RefCountHelper::NEW,
        }
    }

    pub fn inc(&self, o: ObjectReference) -> usize {
        // `self.rc.inc` is a CAS retry loop internally, but it returns exactly one
        // `Result` per logical inc, so matching on that result counts each operation
        // once however many times the CAS was retried.
        match self.rc.inc(o) {
            Ok(prev) => {
                // Fast: the count was below MAX_REF_COUNT and the side-metadata CAS
                // completed the operation. The overflow map was never consulted.
                #[cfg(feature = "lxr_rc_path_stats")]
                super::rc_path_stats::inc_fast();
                prev as usize
            }

            Err(MAX_REF_COUNT) => {
                // Slow: the count is saturated. Counted here, before the map is touched, so that
                // the hit and insert cases below share this single increment.
                #[cfg(feature = "lxr_rc_path_stats")]
                super::rc_path_stats::inc_slow();

                if let Some(e) = self.entries.get(&o) {
                    return e.fetch_add(1, Ordering::Relaxed);
                }

                self.entries
                    .entry(o)
                    .or_insert_with(|| AtomicUsize::new(MAX_RC_USIZE))
                    .fetch_add(1, Ordering::Relaxed)
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
                // Fast: the count was neither 0 nor MAX_REF_COUNT, so the side-metadata
                // CAS completed the operation without touching the overflow map.
                #[cfg(feature = "lxr_rc_path_stats")]
                super::rc_path_stats::dec_fast();
                prev_rc as usize
            }

            Err(MAX_REF_COUNT) => {
                // Slow: the count is saturated. Counted once here, which covers both outcomes
                // below -- the entry was decremented, or there was none and the RC table itself
                // is stepped down. That fallback re-enters `RefCountHelper`, not this method,
                // so it cannot count a second time.
                #[cfg(feature = "lxr_rc_path_stats")]
                super::rc_path_stats::dec_slow();

                let mut prev: Option<usize> = None;

                if let Some(e) = self.entries.get(&o) {
                    let mut cur = e.load(Ordering::Relaxed);

                    while cur > MAX_RC_USIZE {
                        match e.compare_exchange_weak(
                            cur,
                            cur - 1,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => {
                                prev = Some(cur);
                                break;
                            }
                            Err(observed) => cur = observed,
                        }
                    }
                }

                match prev {
                    Some(cur) => {
                        if cur == MAX_RC_USIZE + 1 {

                            let _ = self
                                .entries
                                .remove(&o);
                        }
                        cur
                    }

                    // No live entry: the real RC was exactly MAX_REF_COUNT, so step the table down
                    // from MAX to MAX - 1.
                    None => self.rc.dec_unconditionally(o) as usize,
                }
            }

            Err(other) => {
                panic!("unexpected RC decrement error: {:?}", other);
            }
        }
    }

    /// Single-writer [`Self::inc`].
    ///
    /// Identical in behaviour and return value, but reaches the RC table through
    /// [`RefCountHelper::inc_exclusive`] and mutates the map slot with a plain load/store instead
    /// of a `fetch_add`.  Both replacements drop an atomic read-modify-write: the table access
    /// stops being a CAS retry loop, and the slot update stops being a `lock xadd`.
    ///
    /// The DashMap shard locking is unchanged — the map's *structure* is still shared, so insert
    /// and remove still take the shard write lock.  Only the value RMW is relaxed.
    ///
    /// # Safety
    ///
    /// The caller must be the only thread performing RC operations for the duration of the call,
    /// which is stronger than invariant 2 above: that invariant only excludes `inc` against `dec`.
    /// See [`RefCountHelper::inc_exclusive`] for what a violation costs.
    pub unsafe fn inc_exclusive(&self, o: ObjectReference) -> usize {
        let prev = unsafe { self.rc.inc_exclusive(o) };

        if prev != MAX_REF_COUNT {
            // Fast: the count was below MAX_REF_COUNT and the table store completed the
            // operation. The overflow map was never consulted. `prev != MAX` is the exclusive
            // spelling of `inc`'s `Ok(prev)`.
            #[cfg(feature = "lxr_rc_path_stats")]
            super::rc_path_stats::inc_fast();
            return prev as usize;
        }

        // Slow: saturated. Counted before the map is touched so the hit and insert cases share it.
        #[cfg(feature = "lxr_rc_path_stats")]
        super::rc_path_stats::inc_slow();

        // ONE lookup, not two. `inc` probes with `get` first so the common hit takes only a
        // shard *read* lock, paying a second hash+probe when it has to fall through to `entry`.
        // With no contention a read lock and a write lock cost the same single atomic on the lock
        // word, so that trade stops paying: take the write lock once and handle both cases under
        // it.
        match self.entries.entry(o) {
            Entry::Occupied(mut e) => {
                let cur = e.get().load(Ordering::Relaxed);
                e.get_mut().store(cur + 1, Ordering::Relaxed);
                cur
            }
            Entry::Vacant(e) => {
                // First overflow for this address: the table is saturated, so the real count is
                // MAX_REF_COUNT and this inc takes it to MAX + 1.
                e.insert(AtomicUsize::new(MAX_RC_USIZE + 1));
                MAX_RC_USIZE
            }
        }
    }

    /// Single-writer [`Self::dec`].
    ///
    /// Identical in behaviour and return value, but reaches the RC table through
    /// [`RefCountHelper::dec_exclusive`] / [`RefCountHelper::dec_unconditionally_exclusive`], and
    /// walks the map slot down with a load/compare/store instead of a `compare_exchange_weak`
    /// retry loop.  Under exclusive access the CAS can never fail, so the loop is dead weight.
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

            // Fast: the count was neither 0 nor MAX_REF_COUNT, so the table store completed the
            // operation without touching the overflow map.
            #[cfg(feature = "lxr_rc_path_stats")]
            super::rc_path_stats::dec_fast();
            return prev as usize;
        }

        // Slow: saturated. Counted once here, covering both outcomes below.
        #[cfg(feature = "lxr_rc_path_stats")]
        super::rc_path_stats::dec_slow();

        // ONE lookup and one lock acquisition, where `dec` needs two: it decrements under a shard
        // read lock, drops that guard, then re-hashes and re-probes to `remove` under the write
        // lock, because DashMap's shard lock is not reentrant. Holding the write lock across both
        // steps is only sound because nothing else can be touching the map.
        //
        // That fusion also removes the transient `MAX_REF_COUNT` slot the module header describes:
        // on this path the decrement to MAX and the removal are a single step, so no other thread
        // can observe the in-between state. The concurrent `dec` above can still produce one, but
        // it cannot survive into this method -- see the assertion below, which enforces that
        // rather than tolerating it.
        match self.entries.entry(o) {
            Entry::Occupied(mut e) => {
                let cur = e.get().load(Ordering::Relaxed);

                // An occupied slot must be strictly above MAX_REF_COUNT here, so unlike `dec`
                // there is no "treat MAX as absent" fallback -- reaching that state means an
                // assumption this method rests on has already broken.
                //
                // A slot at exactly MAX_REF_COUNT is the transient state the concurrent `dec`
                // passes through between its CAS (`MAX + 1 -> MAX`) and its `remove`, both in one
                // thread's straight-line code with no early return between them.  It therefore
                // cannot outlive the decrement phase, and that phase fully drains before
                // `CycleCollector` is scheduled.  `inc` never publishes one either: its
                // `or_insert_with` holds the shard write guard across both the insert and the
                // `fetch_add`.
                //
                // If this fires, that ordering no longer holds -- most likely decrement work has
                // been allowed to overlap cycle collection.  Do **not** repair it by restoring a
                // fallback: the exclusivity this whole method assumes would already have been
                // violated, so the RC table is suspect too, not just this slot.
                debug_assert!(
                    cur > MAX_RC_USIZE,
                    "overflow slot for {:?} holds {}, expected > MAX_REF_COUNT ({}): a transient \
                     slot outlived the decrement phase, so RC work is no longer exclusive to the \
                     cycle collector",
                    o,
                    cur,
                    MAX_RC_USIZE
                );

                if cur == MAX_RC_USIZE + 1 {
                    // Would land on MAX_REF_COUNT, i.e. "no longer overflowed". Drop the slot
                    // instead of storing the sentinel.
                    e.remove();
                } else {
                    e.get_mut().store(cur - 1, Ordering::Relaxed);
                }
                cur
            }

            // No entry: the real RC was exactly MAX_REF_COUNT, so step the table down from MAX to
            // MAX - 1. `dec_exclusive` will not do this -- it guards MAX as sticky -- so this
            // needs the unconditional variant, exactly as `dec` needs `dec_unconditionally`.
            Entry::Vacant(_) => unsafe { self.rc.dec_unconditionally_exclusive(o) as usize },
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

        // Slow: the count is saturated, so the map must be consulted to tell MAX_REF_COUNT from
        // anything above it. Counted before the lookup, so the hit and the miss share it. A slot
        // still holding MAX_REF_COUNT returns the same answer as a miss.
        #[cfg(feature = "lxr_rc_path_stats")]
        super::rc_path_stats::get_slow();

        self.entries
            .get(&o)
            .map(|e| e.load(Ordering::Relaxed))
            .unwrap_or(table_rc)
    }

    pub fn is_alive(&self, o: ObjectReference) -> bool {
        self.rc.count(o) as usize > RC_DEATH_TRANSIENT
    }

    /// Slots currently held: objects that are currently overflowed, plus any slot transiently left
    /// at `MAX_REF_COUNT` by `dec` (see the encoding note above).
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Slots the map has room for before it grows, summed across shards.
    pub fn capacity(&self) -> usize {
        self.entries.capacity()
    }
}
