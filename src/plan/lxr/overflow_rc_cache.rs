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
//!    `ProcessDecs` is concurrent, `CycleCollector` is a single packet ordered after decs, and the
//!    `decide_cycle_collection` condvar keeps the next GC's increments behind the current GC's
//!    concurrent window. Nothing enforces this; it is relied upon by the gap between
//!    `RefCountHelper::{inc,dec}` returning `Err(MAX_REF_COUNT)` and this map being consulted.
//! 3. **Every mutation of an entry's value happens while at least a shard read lock is held.**
//!    This is what lets `remove_if` decide atomically whether a slot is still empty: its predicate
//!    runs under the shard write lock, which excludes every `fetch_add`/CAS below.

use std::sync::atomic::{AtomicUsize, Ordering};

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
