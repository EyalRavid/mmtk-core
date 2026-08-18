//! Instrumentation for the LXR overflow reference-count cache
//! ([`super::overflow_rc_cache::RefCountWithOverflow`]).
//!
//! It answers two questions, per RC width:
//!
//! 1. How many `inc`/`dec`/`get` operations are served by the RC **side table** alone,
//!    and how many fall through to the **overflow struct**? The side-table path is a
//!    single side-metadata access; the overflow path takes a global mutex and scans a
//!    `Vec`, so the split is what determines the cost of a width.
//! 2. How much **scanning** does the overflow struct do? `*.scanned` counts entries
//!    examined, so `scanned / overflow` is the mean scan length and `*.miss` counts the
//!    scans that ran to the end of the vector without a hit.
//!
//! # Cost when disabled
//!
//! Everything here is gated on the `lxr_overflow_stats` cargo feature. With the feature
//! off the counters do not exist and every entry point is an empty `#[inline(always)]`
//! function, so the arguments are dead and the calls vanish; `RefCountWithOverflow` is
//! then identical to the uninstrumented version.
//!
//! # Cost when enabled
//!
//! Counters are **per thread** and are aggregated only at `harness_end`. A single global
//! atomic would put every GC worker onto one cache line inside `ProcessIncs` — which is
//! precisely the pathology these counters exist to measure, so the instrument would
//! change the reading. Per-thread counters are still `AtomicUsize` because the
//! aggregating thread reads them, but each is written only by its owning thread, so the
//! line stays exclusive in that core's cache.
//!
//! The one exception is `rc_overflow.entries.max`, which is a property of the shared
//! table rather than of a thread. It is updated only while the table's mutex is already
//! held, so a global atomic there adds no contention that the lock does not already
//! impose.

#[cfg(feature = "lxr_overflow_stats")]
mod imp {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// Declares the counter set once, and derives the per-thread struct, the aggregation,
    /// the reset and the stats-block columns from it. The literal is the column name that
    /// reaches `parsed.csv` — `running-parser` is schema-agnostic, so adding a line here
    /// is all it takes to get a new column (EVALUATION_PLAN.md §3.4).
    macro_rules! counters {
        ($($name:ident : $key:literal,)*) => {
            #[derive(Default)]
            pub struct ThreadStats {
                $(pub(super) $name: AtomicUsize,)*
            }

            #[derive(Default, Clone, Copy)]
            struct Totals {
                $($name: usize,)*
            }

            fn aggregate() -> Totals {
                let mut t = Totals::default();
                for s in REGISTRY.lock().unwrap().iter() {
                    $(t.$name += s.$name.load(Ordering::Relaxed);)*
                }
                t
            }

            fn clear_all() {
                for s in REGISTRY.lock().unwrap().iter() {
                    $(s.$name.store(0, Ordering::Relaxed);)*
                }
            }

            pub fn print_keys() {
                $(print!("{}\t", $key);)*
                print!("rc_overflow.entries.max\trc_overflow.entries.live\t");
            }

            pub fn print_values() {
                let t = RETIRED.lock().unwrap().unwrap_or_default();
                $(print!("{}\t", t.$name);)*
                print!(
                    "{}\t{}\t",
                    ENTRIES_MAX.load(Ordering::Relaxed),
                    t.inc_inserts.saturating_sub(t.dec_removes),
                );
            }
        };
    }

    counters! {
        // inc: `RefCountHelper::inc` succeeded, i.e. the count was below MAX_REF_COUNT.
        inc_side:      "rc_overflow.inc.side",
        // inc: the count was saturated, so the overflow table was locked and scanned.
        inc_overflow:  "rc_overflow.inc.overflow",
        inc_scanned:   "rc_overflow.inc.scanned",
        // inc: the scan found no entry, so a new one was pushed (a full-length scan).
        inc_inserts:   "rc_overflow.inc.inserts",

        dec_side:      "rc_overflow.dec.side",
        dec_overflow:  "rc_overflow.dec.overflow",
        dec_scanned:   "rc_overflow.dec.scanned",
        // dec: full-length scan with no hit; the real count was exactly MAX_REF_COUNT
        // and the side table is decremented unconditionally instead.
        dec_miss:      "rc_overflow.dec.miss",
        // dec: the entry dropped back to MAX_REF_COUNT and was removed from the table.
        dec_removes:   "rc_overflow.dec.removes",

        get_side:      "rc_overflow.get.side",
        get_overflow:  "rc_overflow.get.overflow",
        get_scanned:   "rc_overflow.get.scanned",
        // get: full-length scan with no hit; the count is exactly MAX_REF_COUNT.
        get_miss:      "rc_overflow.get.miss",
    }

    static REGISTRY: Mutex<Vec<Arc<ThreadStats>>> = Mutex::new(Vec::new());
    /// High-water mark of the overflow vector's length. Written under the table's mutex.
    static ENTRIES_MAX: AtomicUsize = AtomicUsize::new(0);
    /// Snapshot taken at `harness_end`, mirroring `crate::RETIRED_COUNTERS`: the values
    /// printed describe the harness window, not whatever happened after it closed.
    /// `None` until the first `stop()`, which is only reachable if the stats block is
    /// printed without a harness.
    static RETIRED: Mutex<Option<Totals>> = Mutex::new(None);

    thread_local! {
        static LOCAL: Arc<ThreadStats> = {
            let stats = Arc::new(ThreadStats::default());
            REGISTRY.lock().unwrap().push(stats.clone());
            stats
        };
    }

    /// Records against the calling thread's counters. A thread already in TLS teardown
    /// cannot reach `LOCAL`; dropping such a sample is correct — it would otherwise
    /// resurrect a destroyed thread-local.
    #[inline(always)]
    fn bump(f: impl FnOnce(&ThreadStats)) {
        let _ = LOCAL.try_with(|s| f(s));
    }

    #[inline(always)]
    pub fn inc_side() {
        bump(|s| {
            s.inc_side.fetch_add(1, Ordering::Relaxed);
        })
    }

    #[inline(always)]
    pub fn inc_overflow(scanned: usize) {
        bump(|s| {
            s.inc_overflow.fetch_add(1, Ordering::Relaxed);
            s.inc_scanned.fetch_add(scanned, Ordering::Relaxed);
        })
    }

    #[inline(always)]
    pub fn inc_insert(entries_len: usize) {
        bump(|s| {
            s.inc_inserts.fetch_add(1, Ordering::Relaxed);
        });
        ENTRIES_MAX.fetch_max(entries_len, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dec_side() {
        bump(|s| {
            s.dec_side.fetch_add(1, Ordering::Relaxed);
        })
    }

    #[inline(always)]
    pub fn dec_overflow(scanned: usize) {
        bump(|s| {
            s.dec_overflow.fetch_add(1, Ordering::Relaxed);
            s.dec_scanned.fetch_add(scanned, Ordering::Relaxed);
        })
    }

    #[inline(always)]
    pub fn dec_remove() {
        bump(|s| {
            s.dec_removes.fetch_add(1, Ordering::Relaxed);
        })
    }

    #[inline(always)]
    pub fn dec_miss() {
        bump(|s| {
            s.dec_miss.fetch_add(1, Ordering::Relaxed);
        })
    }

    #[inline(always)]
    pub fn get_side() {
        bump(|s| {
            s.get_side.fetch_add(1, Ordering::Relaxed);
        })
    }

    #[inline(always)]
    pub fn get_overflow(scanned: usize) {
        bump(|s| {
            s.get_overflow.fetch_add(1, Ordering::Relaxed);
            s.get_scanned.fetch_add(scanned, Ordering::Relaxed);
        })
    }

    #[inline(always)]
    pub fn get_miss() {
        bump(|s| {
            s.get_miss.fetch_add(1, Ordering::Relaxed);
        })
    }

    /// Called from `harness_begin`, so the counters cover the timed window only.
    pub fn reset() {
        clear_all();
        ENTRIES_MAX.store(0, Ordering::Relaxed);
        *RETIRED.lock().unwrap() = None;
    }

    /// Called from `harness_end`, before the stats block is printed.
    pub fn stop() {
        *RETIRED.lock().unwrap() = Some(aggregate());
    }
}

#[cfg(not(feature = "lxr_overflow_stats"))]
#[allow(unused_variables)]
mod imp {
    #[inline(always)]
    pub fn inc_side() {}
    #[inline(always)]
    pub fn inc_overflow(scanned: usize) {}
    #[inline(always)]
    pub fn inc_insert(entries_len: usize) {}
    #[inline(always)]
    pub fn dec_side() {}
    #[inline(always)]
    pub fn dec_overflow(scanned: usize) {}
    #[inline(always)]
    pub fn dec_remove() {}
    #[inline(always)]
    pub fn dec_miss() {}
    #[inline(always)]
    pub fn get_side() {}
    #[inline(always)]
    pub fn get_overflow(scanned: usize) {}
    #[inline(always)]
    pub fn get_miss() {}
    /// No columns are added to the stats block when the feature is off, so a build
    /// without it parses exactly as it did before.
    pub fn print_keys() {}
    pub fn print_values() {}
    pub fn reset() {}
    pub fn stop() {}
}

pub(crate) use imp::*;
