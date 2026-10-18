//! The set of live `java.lang.ref.Reference` objects. Always on.
//!
//! `~/mmtk/REFERENCE_PROCESSING_PLAN.md` §3. A Reference is inserted when it is promoted (its first
//! increment, `ProcessIncs::promote`) and removed when it dies (either `process_dead_object`: RC
//! death in `ProcessDecs`, or the cycle collector's). A Reference that dies before its first GC is
//! never promoted, so it is never inserted and never removed.
//!
//! Address reuse needs no handling: a Reference's memory is reallocated only after it has died
//! (removed) and a mutator has allocated a new object there; that object's promotion (insert) is in
//! a later pause. So the set always describes what lives at an address now.
//!
//! Stage R0: the set is maintained and checked, nothing reads it yet. Debug builds assert that an
//! insert is new, that a remove finds its entry, and (at every FullRC) that every entry is a live
//! Reference. Under `lxr_rc_path_stats` only, counters report sizes (`inserts`, `removes`, `peak`;
//! compare with the census, `FINDINGS.md` §34) and contention (`ops`, `op_ns` — time inside insert / remove, shard lock included, §3.3).

use std::hash::{BuildHasherDefault, Hasher};

use dashmap::DashSet;

use crate::util::ObjectReference;

/// Fibonacci hashing of a heap address. The keys are addresses, so `std`'s DoS-resistant default
/// (SipHash) buys nothing and costs several times more. Objects are 16-byte aligned; the low 4 bits
/// are shifted out before the multiply.
#[derive(Default, Clone, Copy)]
pub struct AddrHasher(u64);

impl Hasher for AddrHasher {
    fn write(&mut self, _bytes: &[u8]) {
        // `ObjectReference` derives `Hash` from a `NonZeroUsize`, which hashes via `write_usize`.
        unreachable!("AddrHasher only hashes addresses (write_usize)")
    }
    #[inline(always)]
    fn write_usize(&mut self, a: usize) {
        self.0 = ((a >> 4) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// A sharded concurrent set: each shard has its own lock, and an address picks its shard from the
/// high bits of its hash, so different References rarely contend (`FINDINGS.md` §35: one `Mutex`
/// cost jython 108 ms of lock wait per iteration). Fibonacci hashing spreads well into the high
/// bits, so the shards stay balanced.
pub type RefSet = DashSet<ObjectReference, BuildHasherDefault<AddrHasher>>;

/// Statistics (`lxr_rc_path_stats` only): R0 counters and lock-wait timing. None of this is
/// compiled into a normal build; there the registry costs one `reference_kind` test, one lock and
/// one set operation per Reference promotion or death. The FullRC `check` walk is a correctness
/// check, not a statistic, and is compiled into debug builds (`debug_assertions`) instead.
#[cfg(feature = "lxr_rc_path_stats")]
mod diag {
    pub(super) use std::sync::atomic::{AtomicU64, Ordering};

    /// Counters for the timed iteration (reset at `harness_begin`) and for the whole run.
    pub(super) struct Counter {
        pub(super) iter: AtomicU64,
        pub(super) run: AtomicU64,
    }

    impl Counter {
        const fn new() -> Self {
            Self { iter: AtomicU64::new(0), run: AtomicU64::new(0) }
        }
        #[inline(always)]
        pub(super) fn add(&self, n: u64) {
            self.iter.fetch_add(n, Ordering::Relaxed);
            self.run.fetch_add(n, Ordering::Relaxed);
        }
        #[inline(always)]
        pub(super) fn max(&self, n: u64) {
            self.iter.fetch_max(n, Ordering::Relaxed);
            self.run.fetch_max(n, Ordering::Relaxed);
        }
    }

    pub(super) static INSERTS: Counter = Counter::new();
    pub(super) static REMOVES: Counter = Counter::new();
    /// Time spent inside `insert` / `remove`, shard lock included (DashMap's locks cannot be timed
    /// separately).
    pub(super) static OP_NS: Counter = Counter::new();
    pub(super) static OPS: Counter = Counter::new();
    pub(super) static PEAK: Counter = Counter::new();
    /// Entries now (insert +1, remove -1): `DashSet::len` locks every shard, so it is not used on
    /// the hot path.
    pub(super) static LIVE: AtomicU64 = AtomicU64::new(0);
}
#[cfg(feature = "lxr_rc_path_stats")]
use diag::*;

pub struct RefRegistry {
    set: RefSet,
}

impl RefRegistry {
    pub fn new() -> Self {
        Self { set: RefSet::with_hasher(BuildHasherDefault::default()) }
    }

    /// Called from `ProcessIncs::promote` for a Reference object, on any GC worker.
    #[inline]
    pub fn insert(&self, o: ObjectReference) {
        #[cfg(feature = "lxr_rc_path_stats")]
        let t = std::time::Instant::now();
        let fresh = self.set.insert(o);
        // Impossible by §3.2: the same Reference promoted twice, or a missed removal followed by a
        // new Reference at the same address.
        debug_assert!(fresh, "reference {:?} registered twice", o);
        #[cfg(feature = "lxr_rc_path_stats")]
        {
            OP_NS.add(t.elapsed().as_nanos() as u64);
            OPS.add(1);
            INSERTS.add(1);
            if fresh {
                PEAK.max(LIVE.fetch_add(1, Ordering::Relaxed) + 1);
            }
        }
    }

    /// Called from both `process_dead_object`s for a dying Reference object, on any GC worker.
    #[inline]
    pub fn remove(&self, o: ObjectReference) {
        #[cfg(feature = "lxr_rc_path_stats")]
        let t = std::time::Instant::now();
        let present = self.set.remove(&o).is_some();
        // Impossible: only promoted (registered) objects reach `process_dead_object`.
        debug_assert!(present, "dying reference {:?} was never registered", o);
        #[cfg(feature = "lxr_rc_path_stats")]
        {
            OP_NS.add(t.elapsed().as_nanos() as u64);
            OPS.add(1);
            REMOVES.add(1);
            if present {
                LIVE.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    /// FullRC check (§3.4), debug builds only: assert that every entry is a mapped, live object that
    /// is still a Reference. Must run where no insert / remove can race: `CycleFullRC`.
    ///
    /// A net, not a proof (plan §3.4): an entry whose memory was reused by a larger object covering
    /// its address would read the middle of that object as a header.
    #[cfg(debug_assertions)]
    pub fn assert_valid<VM: crate::vm::VMBinding>(&self, rc: &crate::util::rc::RefCountHelper<VM>) {
        // `iter` read-locks one shard at a time; nothing inserts or removes during CycleFullRC, so
        // it cannot deadlock against a writer.
        for entry in self.set.iter() {
            let r = *entry;
            assert!(r.to_raw_address().is_mapped(), "reference registry: {:?} is not mapped", r);
            assert!(rc.count(r) > 0, "reference registry: {:?} is dead (rc 0) but registered", r);
            assert!(
                <VM::VMScanning as crate::vm::Scanning<VM>>::reference_kind(r) != 0,
                "reference registry: {:?} is no longer a Reference",
                r
            );
        }
    }
}

/// Reset the timed-iteration counters (from `harness_begin`).
#[cfg(feature = "lxr_rc_path_stats")]
pub(crate) fn reset() {
    for c in [&INSERTS, &REMOVES, &OP_NS, &OPS, &PEAK] {
        c.iter.store(0, Ordering::Relaxed);
    }
}

/// Print the R0 report (from the stats dump at `harness_end`). Only at `MMTK_VERBOSE` >= 2: the
/// stats block of publication runs is parsed, so it must not grow lines there.
#[cfg(feature = "lxr_rc_path_stats")]
pub(crate) fn print_report() {
    if !crate::verbose(2) {
        return;
    }
    let r = Ordering::Relaxed;
    println!("--- Reference registry (R0): timed iteration | run ---");
    for (name, c) in [
        ("inserts", &INSERTS), ("removes", &REMOVES), ("peak", &PEAK),
        ("ops", &OPS), ("op_ns", &OP_NS),
    ] {
        println!("  refs.{}: {} | {}", name, c.iter.load(r), c.run.load(r));
    }
}
