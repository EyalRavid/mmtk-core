//! Fast-path / slow-path counters for the LXR overflow reference count
//! ([`super::overflow_rc_cache::RefCountWithOverflow`]).
//!
//! For each of `inc`, `dec` and `get` this counts how many logical operations were served
//! by the RC side table alone (**fast**) and how many had to take the overflow table's
//! global mutex and scan its `Vec` (**slow**).
//!
//! The counts are emitted as six columns of MMTk's statistics block, so `running-parser`
//! picks them up with no parser change and `extract.py` aggregates them like any other
//! metric (EVALUATION_PLAN.md §3.3). A human-readable block is printed alongside.
//!
//! # Zero cost when disabled
//!
//! This module is compiled **only** when the `lxr_rc_path_stats` cargo feature is enabled:
//! its `mod` declaration in [`super`] carries `#[cfg(feature = "lxr_rc_path_stats")]`, and
//! every call site — in `overflow_rc_cache.rs`, `crate::mmtk` and
//! `crate::util::statistics::stats` — is a `#[cfg(feature = "lxr_rc_path_stats")]`-gated
//! *statement*.
//!
//! So with the feature off there is nothing for the optimiser to remove: the statics, the
//! functions and the calls are all stripped by `cfg` expansion, before type checking and
//! long before code generation. No no-op shim is inlined away, no `if` is folded, no
//! counter occupies a byte of `.bss`, and no column is added to the stats block. A call
//! site that loses its `cfg` is a compile error, not silent runtime cost.
//!
//! # Cost when enabled
//!
//! Six plain global `AtomicU64`s incremented with `Relaxed`. Every GC worker shares those
//! cache lines, which is bad for the *instrumented* build's speed and irrelevant to its
//! purpose — these are counts, not timings.

use std::sync::atomic::{AtomicU64, Ordering};

/// `inc`: `RefCountHelper::inc` succeeded, so the count was below `MAX_REF_COUNT` and the
/// side-metadata CAS was the whole operation.
static INC_FAST: AtomicU64 = AtomicU64::new(0);
/// `inc`: the count was saturated, so the overflow table's mutex was taken and its `Vec`
/// scanned.
static INC_SLOW: AtomicU64 = AtomicU64::new(0);
/// `dec`: `RefCountHelper::dec` succeeded, so the count was neither 0 nor `MAX_REF_COUNT`.
static DEC_FAST: AtomicU64 = AtomicU64::new(0);
/// `dec`: the count was saturated, so the overflow table's mutex was taken and its `Vec`
/// scanned. Includes the miss case that then falls back to `dec_unconditionally`.
static DEC_SLOW: AtomicU64 = AtomicU64::new(0);
/// `get`: the side-table count was not saturated, so it is the answer.
static GET_FAST: AtomicU64 = AtomicU64::new(0);
/// `get`: the side-table count was saturated, so the overflow table's mutex was taken.
static GET_SLOW: AtomicU64 = AtomicU64::new(0);

/// The counter set, declared once. `print_keys` and `print_values` both walk this array,
/// which is what keeps them positionally aligned — the stats block matches keys to values
/// by order, not by name, so two independent lists would eventually drift.
static ALL: [(&str, &AtomicU64, &AtomicU64); 3] = [
    ("inc", &INC_FAST, &INC_SLOW),
    ("dec", &DEC_FAST, &DEC_SLOW),
    ("get", &GET_FAST, &GET_SLOW),
];

#[inline(always)]
pub(super) fn inc_fast() {
    INC_FAST.fetch_add(1, Ordering::Relaxed);
}

#[inline(always)]
pub(super) fn inc_slow() {
    INC_SLOW.fetch_add(1, Ordering::Relaxed);
}

#[inline(always)]
pub(super) fn dec_fast() {
    DEC_FAST.fetch_add(1, Ordering::Relaxed);
}

#[inline(always)]
pub(super) fn dec_slow() {
    DEC_SLOW.fetch_add(1, Ordering::Relaxed);
}

#[inline(always)]
pub(super) fn get_fast() {
    GET_FAST.fetch_add(1, Ordering::Relaxed);
}

#[inline(always)]
pub(super) fn get_slow() {
    GET_SLOW.fetch_add(1, Ordering::Relaxed);
}

/// Called from `harness_begin` next to `crate::reset_counters()`, so the counts cover the
/// same timed window as every other counter in the stats block and exclude warmup.
pub(crate) fn reset() {
    for (_, fast, slow) in ALL {
        fast.store(0, Ordering::Relaxed);
        slow.store(0, Ordering::Relaxed);
    }
}

/// Column names for MMTk's statistics block. Called from `Stats::print_column_names`.
pub(crate) fn print_keys() {
    for (op, _, _) in ALL {
        print!("rc_path.{}.fast\trc_path.{}.slow\t", op, op);
    }
}

/// Column values for MMTk's statistics block. Called from `Stats::print_stats`, in the
/// same position within the row as [`print_keys`] uses within the header.
pub(crate) fn print_values() {
    for (_, fast, slow) in ALL {
        print!(
            "{}\t{}\t",
            fast.load(Ordering::Relaxed),
            slow.load(Ordering::Relaxed)
        );
    }
}

/// Human-readable summary, printed after the statistics block at `harness_end`.
///
/// The marker line deliberately does not begin with `=`, so `running-parser`'s state
/// machine treats this block as ordinary log output and ignores it
/// (`running-parser/src/main.rs:97`).
pub(crate) fn print_report() {
    println!("--- LXR RC fast/slow path counts ---");
    for (op, fast, slow) in ALL {
        let fast = fast.load(Ordering::Relaxed);
        let slow = slow.load(Ordering::Relaxed);
        let total = fast + slow;
        // An operation a benchmark never performs reports 0%, not NaN.
        let pct = |x: u64| {
            if total == 0 {
                0.0
            } else {
                (x as f64) * 100.0 / (total as f64)
            }
        };
        println!("{}:", op);
        println!("  fast: {}", fast);
        println!("  slow: {}", slow);
        println!("  total: {}", total);
        println!("  fast %: {:.4}", pct(fast));
        println!("  slow %: {:.4}", pct(slow));
    }
}
