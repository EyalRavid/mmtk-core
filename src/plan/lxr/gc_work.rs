use super::cm::LXRWeakRefProcessEdges;
use super::{barrier, LXR};
use crate::scheduler::{gc_work::*, GCWork, GCWorker, WorkBucketStage};
use crate::util::ObjectReference;
use crate::{vm::*, Plan, MMTK};
use crate::util::rc::{cc, MAX_STRONG_REF_COUNT, BLACK_OUT_OF_STACK, BLACK_IN_STACK, GREY, WHITE, STRONG_RC_LAST_BEFORE_ZERO};
use atomic::Ordering;
use crate::util::address::CLDScanPolicy;
use crate::util::address::RefScanPolicy;
use std::marker::PhantomData;
use crate::vm;
use crate::vm::slot::Slot;
use crate::util::VMWorkerThread;
use crate::util::VMThread;
use crate::plan::SlotIterator;
use crate::vm::{Scanning, VMBinding};
use crate::policy::space::Space;
use crate::policy::immix::block::Block;
use crate::util::rc::RefCountHelper;
use crate::vm::slot::MemorySlice;
use crate::plan::lxr::global::NUM_OF_CANDIDATES_VECTORS;
use crate::plan::lxr::buffer::LocalBuffer;
use crate::plan::lxr::buffer::FinalBuffers;
use crate::util::rc::RcBits;
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::LazySweepingJobsCounter;
#[cfg(feature = "graph_project")]
use crate::plan::lxr:: graphs_project::{*};
#[cfg(feature = "graph_project")]
use crate::plan::lxr::buffer::FinalIterMut;
use crate::Pause;

pub(super) struct LXRGCWorkContext<E: ProcessEdgesWork>(std::marker::PhantomData<E>);

impl<E: ProcessEdgesWork> crate::scheduler::GCWorkContext for LXRGCWorkContext<E> {
    type VM = E::VM;
    type PlanType = LXR<E::VM>;
    type DefaultProcessEdges = E;
    type PinningProcessEdges = UnsupportedProcessEdges<Self::VM>;
}

pub(super) struct LXRWeakRefWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);

impl<VM: VMBinding> crate::scheduler::GCWorkContext for LXRWeakRefWorkContext<VM> {
    type VM = VM;
    type PlanType = LXR<VM>;
    type DefaultProcessEdges = LXRWeakRefProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<Self::VM>;
}

pub struct FastRCPrepare;

impl<VM: VMBinding> GCWork<VM> for FastRCPrepare {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        #[allow(invalid_reference_casting)]
        let lxr = unsafe { &mut *(lxr as *const LXR<VM> as *mut LXR<VM>) };
        lxr.prepare(worker.tls)
    }
}


pub struct ReleaseLOSNursery;

impl<VM: VMBinding> GCWork<VM> for ReleaseLOSNursery {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        lxr.los().release_rc_nursery_objects();
    }
}

fn in_stack(o: ObjectReference) -> bool {
    cc::colour(o, Ordering::Relaxed) == BLACK_IN_STACK
}

fn is_black(o: ObjectReference) -> bool {
    cc::colour(o, Ordering::Relaxed) <= BLACK_IN_STACK
}

#[cfg(feature = "s_rc_stats")]
#[derive(Default)]
struct CycleCollectorStats {
    raw_cycle_candidates: usize,
    candidates_after_filter: usize,
    objects_in_mark: std::cell::Cell<usize>,
    objects_in_scan: std::cell::Cell<usize>,
    objects_in_scan_black: std::cell::Cell<usize>,
    objects_in_collect: std::cell::Cell<usize>,
    satb_map_size: usize,
    satb_reads: std::cell::Cell<usize>,
    // Parallel-mark baseline; see `Counters` for what each one is for.
    mark_packets: std::cell::Cell<usize>,
    claim_attempts: std::cell::Cell<usize>,
    claim_wins: std::cell::Cell<usize>,
    pushes: std::cell::Cell<usize>,
    mark_logged: std::cell::Cell<usize>,
    peak_stack: usize,
}

#[cfg(feature = "s_rc_stats")]
impl CycleCollectorStats {
    /// Publish this cycle collection's counts into the global `Counters`.
    ///
    /// Replaces the old CSV sink, which appended `cycle_collector_stats.csv` to the JVM's working
    /// directory. Under running-ng that file landed unattributed -- not per benchmark, not per
    /// invocation -- and `running-parser` never saw it. `Counters` is reset at `harness_begin`
    /// and printed at `harness_end` into the MMTk statistics block, so these flow through the
    /// schema-agnostic parser, `load.py`, CI computation and `compare` with no change anywhere
    /// downstream. `EVALUATION_PLAN.md` E2(a).
    ///
    /// Called once per `do_work`, so the atomics run a handful of times per GC rather than once
    /// per object: the per-object accumulation is the non-atomic `Cell`s above.
    fn flush_to_counters(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let c = crate::counters();
        c.cc_raw_candidates.fetch_add(self.raw_cycle_candidates, Relaxed);
        c.cc_candidates_after_filter
            .fetch_add(self.candidates_after_filter, Relaxed);
        c.cc_objects_in_mark.fetch_add(self.objects_in_mark.get(), Relaxed);
        c.cc_objects_in_scan.fetch_add(self.objects_in_scan.get(), Relaxed);
        c.cc_objects_in_scan_black
            .fetch_add(self.objects_in_scan_black.get(), Relaxed);
        c.cc_objects_in_collect
            .fetch_add(self.objects_in_collect.get(), Relaxed);
        c.cc_satb_reads.fetch_add(self.satb_reads.get(), Relaxed);
        // A level, not a sum: the peak SATB map size across the run.
        c.cc_satb_map_peak.fetch_max(self.satb_map_size, Relaxed);
        c.cc_mark_packets.fetch_add(self.mark_packets.get(), Relaxed);
        c.cc_claim_attempts.fetch_add(self.claim_attempts.get(), Relaxed);
        c.cc_claim_wins.fetch_add(self.claim_wins.get(), Relaxed);
        c.cc_pushes.fetch_add(self.pushes.get(), Relaxed);
        c.cc_mark_logged.fetch_add(self.mark_logged.get(), Relaxed);
        // Also a level.
        c.cc_peak_stack.fetch_max(self.peak_stack, Relaxed);
    }
}

/// The trial-deletion traversal itself, with no notion of scheduling.
///
/// Three packets drive it: `CycleFullRC` (all phases, `Pause::FullRC`), and the `CycleMark` /
/// `CycleScanCollect` pair that splits the concurrent path so mark can run on several workers.
/// Separating it from the packets is what lets those three own different tokens while sharing one
/// implementation of every phase.
pub struct CycleTraversal<VM: VMBinding> {
    rc: RefCountHelper<VM>,
    #[cfg(feature = "s_rc_stats")]
    stats: CycleCollectorStats,
}

impl<VM: VMBinding> CycleTraversal<VM> {
    fn new() -> Self {
        Self {
            rc: RefCountHelper::NEW,
            #[cfg(feature = "s_rc_stats")]
            stats: CycleCollectorStats::default(),
        }
    }
}

/// Cycle collection for `Pause::FullRC`: every phase, over all three candidate pools, inside the
/// pause.  The concurrent path uses `CycleMark` + `CycleScanCollect` instead.
pub struct CycleFullRC<VM: VMBinding> {
    t: CycleTraversal<VM>,
    #[cfg(not(feature = "lxr_stw"))]
    _c: LazySweepingJobsCounter,
}

impl<VM: VMBinding> GCWork<VM> for CycleFullRC<VM> {

    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {

        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        lxr.satb_map.clear();
        lxr.in_cycle_collection.store(true, Ordering::SeqCst);
        #[cfg(feature = "s_rc_stats")]
        { self.t.stats.mark_packets.set(self.t.stats.mark_packets.get() + 1); }
        //println!("size of rc cache: {}, num of entreies: {}", lxr.rc_with_overflow.capacity(), lxr.rc_with_overflow.num_entries());

        // The two scratch DFS stacks, reused by every traversal below.
        //
        // Two are needed, and exactly two: the phases run in sequence, but `scan` nests into
        // `scan_black` and `collect_whites` nests into `collect_blacks`, so at most two are live
        // at once.  Neither nested traversal is recursive or reachable from anywhere else, so the
        // depth is 2 and never grows.  Running the phases over three candidate buffers instead of
        // one does not change that: the buffers are walked in sequence within a phase, so the
        // nesting depth is a property of the phases, not of how many candidates they see.
        //
        // They are locals rather than `CycleFullRC` fields on purpose: this packet is
        // constructed fresh per GC and `do_work` runs once, so a field would retain nothing across
        // GCs, and a local cannot be aliased by construction -- which keeps them out of the
        // exclusivity argument in the impl header below.
        //
        // `Vec`, not `ChunkedStack`: a `Vec` keeps its capacity as it drains, so it grows once per
        // GC and every later candidate reuses resident pages.  `ChunkedStack::pop` frees the top
        // chunk once it empties, so reusing one would still churn 16 KB malloc/free pairs at every
        // chunk boundary -- which is the cost this change exists to remove.
        let mut dfs_stack = Vec::<ObjectReference>::with_capacity(4096);
        let mut nested_stack = Vec::<ObjectReference>::with_capacity(4096);

        // How many candidate pools this collection drains.
        //
        // The test is against `FullRC`, and swapping it for `== Some(Pause::RefCount)` is NOT
        // equivalent: in an ordinary reference-counting GC this packet runs in the *concurrent
        // tail*, scheduled by `on_lazy_cc_finished` after `gc_pause_end` has already stored `None`
        // into `current_pause` (`global.rs:553`).  So for the ordinary case `current_pause()` is
        // `None`, not `Some(Pause::RefCount)`, and a `== RefCount` test would send every ordinary
        // pause down `all_buff_gc`.  `FullRC` is the only pause that runs this packet inside the
        // pause, which is exactly what makes it the one that can afford to drain everything.
        match lxr.current_pause() {
            Some(Pause::FullRC) => self.t.all_buff_gc(lxr, &mut dfs_stack, &mut nested_stack),
            // The concurrent path is `CycleMark` + `CycleScanCollect`, scheduled by
            // `LXR::schedule_cycle_mark`; nothing constructs this packet for it.  The one caller
            // that would is the `lxr_stw` site in `global.rs`, which does not currently compile.
            _ => unreachable!("CycleFullRC is the Pause::FullRC packet; see schedule_cycle_mark"),
        }

        // Timestamp the end of cycle collection. Together with "lazy decs finished",
        // "lazy sweep finished" and "lazy jobs finished" this splits the concurrent window
        // into its phases for `scripts/lxr/window.py`.
        //
        // Since the 2026-08-23 reorder the chain is decs -> sweep -> cc, so the interval that
        // ends here starts at "lazy sweep finished", not at "lazy decs finished". This packet
        // is still single-threaded, so every millisecond it runs is time the other GC workers
        // spend idle -- the reorder moved reclamation off the far side of that wait, it did
        // not shorten the wait.
        gc_log!([2]
            " - lazy cc finished since-gc-start={:.3}ms",
            crate::gc_start_time_ms(),
        );
    }
}

/// # A note on the `*_exclusive` RC calls below
///
/// The RC traversal in `scan`, `scan_black`, `collect_whites` and `collect_blacks` uses the
/// `*_exclusive` variants of the reference-count helpers, which skip the atomic read-modify-write
/// (a CAS retry loop) that the ordinary `inc`/`dec` need.  Every one of them carries the same
/// safety obligation: **this thread must be the only one performing RC work**.
///
/// ⚠ **`mark` and `mark_buffer` are the exception: they use the ATOMIC variants.**  They are being
/// prepared for parallel execution (`~/mmtk/CYCLE_COLLECTOR_PARALLEL_PLAN.md`), where the obligation
/// below cannot hold for them, and an object's grey claim goes through `cc::try_claim_grey` so that
/// exactly one thread walks its fields.  The conversion is behaviour-preserving while the packet is
/// still single-threaded.  The four phases above remain exclusive because they run after mark has
/// joined, and the split must not change that.
///
/// That holds because `CycleFullRC` is a single `GCWork` packet, so it runs on exactly one GC
/// worker, and the concurrent chain is ordered `decs -> sweep -> cc`, so `ProcessDecs` has fully
/// drained before it starts.  The next GC cannot begin either: a GC request becomes a
/// `ScheduleCollection` packet only when the *last* GC worker parks, and this packet is occupying
/// one.  The mutator write barrier stays clear of the RC tables during this window -- it touches
/// only `OBJ_COLOR_TABLE` and `satb_map`.
///
/// The same applies to the `OBJ_COLOR_TABLE` and `CANDIDATES_STATUS` stores, which use
/// `store_atomic_exclusive`.  Both specs are **2 bits per entry**, so four objects share a byte and
/// the store is a read-merge-write of that byte with no CAS -- meaning the obligation there is the
/// wider one: no other thread may write *any* entry in the same byte.  Concurrent *readers* are
/// fine, because the byte store is still atomic and a reader's own bits are untouched by a write to
/// a neighbour; that is what lets the mutator barrier keep loading `OBJ_COLOR_TABLE`
/// (`plan/lxr/barrier.rs`) while this runs.
///
/// The two tables differ in how much they depend on the phase order:
///
/// * `OBJ_COLOR_TABLE` is written **only** by this impl, anywhere in the codebase, so it needs no
///   assumption beyond "one worker runs this packet".
/// * `CANDIDATES_STATUS` is also written by `ProcessDecs` (`plan/lxr/rc.rs`), so it is safe only
///   because decrements have fully drained first.  It is the one that breaks first if the
///   `decs -> sweep -> cc` order is relaxed.
///
/// **If any of that changes -- the remaining phases are split across workers, cycle collection is
/// moved off the GC worker pool, or decrement work is allowed to overlap it -- every `unsafe` block
/// in this impl becomes unsound and must go back to the atomic variants**, as `mark` already has.
/// What `mark` needs in order to hand part of its DFS stack to another worker.
///
/// `Copy`, so it threads down `mark_buffer` -> `mark` -> `mark_from_stack` without reborrowing.
/// `None` means "do not spill", which is how the `Pause::FullRC` path opts out: `CycleFullRC`
/// runs mark, scan and collect in one packet using the `*_exclusive` metadata stores, so it must
/// not create packets that would still be running during its own scan -- see the impl header.
#[derive(Clone, Copy)]
struct Spill<'a> {
    /// Borrowed from the spilling packet, so every packet it mints is counted in the same
    /// generation's mark counter and `end_of_mark` cannot fire while any of them is outstanding.
    c: &'a LazySweepingJobsCounter,
    vec_index: u8,
    curr_vec: u8,
}

impl<VM: VMBinding> CycleTraversal<VM>{
    
    pub const UNLOGGED_VALUE: u8 = 0b1;
    pub const LOGGED_VALUE: u8 = 0b0;

    /// Hand work off once the DFS stack passes twice this, in chunks of this size.
    ///
    /// Mirrors `ProcessDecs::CAPACITY` (`crate::args::BUFFER_SIZE`, 1024), which is the same
    /// mechanism: a bounded buffer that becomes a new packet instead of growing.  The high-water
    /// mark is `2 *` so that crossing the line exports one chunk and leaves a full chunk to work
    /// on, rather than exporting on every push from then on.
    const SPLIT_THRESHOLD: usize = 2048;

    const UNLOG_BITS: SideMetadataSpec = *VM::VMObjectModel::GLOBAL_FIELD_UNLOG_BIT_SPEC
        .as_spec()
        .extract_side_spec();


    /// Run mark over one candidate buffer.
    ///
    /// Shared by both collection modes so they cannot drift: `single_buff_gc` calls it once and
    /// `all_buff_gc` calls it once per pool, and the per-candidate decision is identical in both.
    ///
    /// `vec_index` must be the tag of the pool `candidates` was drained from. The pool/tag
    /// relation is `tag(pool i) == i + 1`, with 0 meaning "not a candidate" -- three pools and a
    /// null value is exactly the four states `CANDIDATES_STATUS` can hold (`spec_defs.rs`,
    /// `log_num_of_bits: 1`). That is the hard ceiling on `NUM_OF_CANDIDATES_VECTORS`.
    ///
    /// `cand_buffer` must be a handle on pool `lxr.curr_vec`, because `mark` tags what it pushes
    /// with `curr_vec + 1` while this pushes it into whichever pool the handle came from. If the
    /// two disagree, the object lands in one pool carrying another pool's tag, `should_mark` can
    /// never match it again, and it is leaked as a candidate for the life of the process.
    fn mark_buffer(
        &self,
        lxr: &LXR<VM>,
        candidates: &mut FinalBuffers<ObjectReference>,
        vec_index: u8,
        curr_vec: u8,
        work: &mut Vec<ObjectReference>,
        cand_buffer: &mut LocalBuffer<'_, ObjectReference>,
        spill: Option<Spill<'_>>,
    ) {
        // Filter first, walk second.
        //
        // `should_mark` + `set_tag(cand, 0)` is the *candidate* claim, and it applies only to
        // objects that came out of a pool.  Doing it for the whole buffer up front leaves a plain
        // list of objects to walk, which is exactly the shape a spilled chunk has -- so both enter
        // the traversal the same way and there is one loop rather than one per candidate.
        //
        // The pruning is not just bookkeeping: what survives here is what `CycleScanCollect` will
        // scan, so rejects must leave the buffer.
        work.reserve(candidates.len());
        let mut it = candidates.iter_mut();
        while let Some(cand) = it.next() {
            if self.should_mark(*cand, vec_index) {
                cc::set_tag(*cand, 0, Ordering::Relaxed);
                debug_assert!(self.rc.count(*cand) > 0);
                work.push(*cand);
            } else {
                if cc::strong_rc(*cand, Ordering::Relaxed) > 0 {
                    cc::set_tag(*cand, 0, Ordering::Relaxed);
                }
                it.swap_remove_current();
            }
        }

        // One object's children, collected before any of them is decremented -- see
        // `mark_from_stack`.  Owned here rather than per object: `pmd` marks ~123 k candidates per
        // GC, and this keeps its capacity across all of them.
        let mut children = Vec::<ObjectReference>::with_capacity(64);
        self.mark_from_stack(lxr, curr_vec, work, &mut children, cand_buffer, spill);
    }

    /// Cycle collection over **all three** candidate pools, phase-major:
    /// mark(all) -> scan(all) -> collect_whites(all).
    ///
    /// Only `Pause::FullRC` takes this path. It removes the N-2 lag: candidates registered by this
    /// GC's own decrements sit in pool `curr_vec` and are collected here rather than two GCs from
    /// now, which is the point of a stop-the-world collection at an iteration boundary.
    ///
    /// **Phase-major is load-bearing, not stylistic.** A cycle whose members were registered in
    /// different GCs spans two pools, and trial deletion only classifies it correctly if every
    /// candidate has been marked before anything is scanned. Calling `single_buff_gc` three times
    /// would run a full mark/scan/collect per pool and mis-classify exactly those cycles.
    ///
    /// One consequence to be aware of: `should_mark` filters on `STRONG_RC_TABLE == 0`, and `mark`
    /// decrements the strong RC of children -- so marking one buffer can make a later buffer's
    /// candidate qualify that did not before, or `mark`'s restore path can disqualify one. Which
    /// candidates get marked is therefore order-dependent here, which it never was with one pool.
    /// It errs towards collecting more, and every candidate that misses out keeps its tag and is
    /// drained by a later GC.
    fn all_buff_gc(
        &mut self,
        lxr: &LXR<VM>,
        dfs_stack: &mut Vec<ObjectReference>,
        nested_stack: &mut Vec<ObjectReference>,
    ) {

        // `curr_vec` on entry. Rotated below to walk the pools, then restored before the scan
        // phase so that a FullRC pause is invisible to the global rotation: the next GC's
        // `schedule_collection` advances from this same value, exactly as it would have.
        //
        // Mutating it here is sound for the same reason every `*_exclusive` call in this impl is
        // (see the header): this is a single packet on one worker, `ProcessDecs` has fully
        // drained, and no next GC can start while this packet occupies a worker. Nothing else
        // reads `curr_vec` in that window.
        let entry_vec = lxr.curr_vec.get();

        // FINALIZER_RC_PLAN step 3A -- cut the ONE root edge into the unfinalized chain.
        //
        // `Finalizer.unfinalized` is a static field holding the head of a chain that is otherwise
        // linked only by its own `Finalizer.next`/`prev`. That static is therefore the single
        // reference into the chain from outside it. Take it away, seed the head as a candidate,
        // and ordinary trial deletion walks the whole chain by itself -- `mark` decrements both
        // `rc` and `strong_rc` of every child, so each node's own walk is what makes its successor
        // eligible. Every `Finalizer.referent` edge is then subtracted exactly once, by `mark`.
        //
        // THIS REPLACES A PER-REFERENT SUBTRACTION, and that is the bug it fixes. The previous
        // version reached into each referent and decremented it by hand to cancel the finalizer's
        // edge -- but `mark` then walked the `Finalizer` and subtracted the *same* edge again. One
        // edge, two subtractions: the referent's count fell one below the truth every time both
        // happened, and on `batik` it reached the +1-bias floor and tripped
        // `debug_assert!(prev != 1)` in `mark`. Confirmed by instrumenting the assertion, which
        // reported `is_a_registered_Finalizer=true` on the parent and `was_dec_by_step3A=true` on
        // the child. `lusearch` has no finalizers, so none of it ran and it passed.
        //
        // The raw `*_exclusive` decrements are deliberate, as before: a `ProcessDecs`-style
        // decrement crossing `RC_DEATH_THRESHOLD` would run death processing and free the object
        // outright, before the CC ever looks at it.
        //
        // Must run BEFORE `into_final_buffers()` drains the pools below, so the head seeded here
        // is picked up by the mark phase of THIS collection.
        let pending_finalizers: Vec<(ObjectReference, ObjectReference)> =
            <VM::VMCollection as Collection<VM>>::finalizer_candidates();
        let mut cut_head: Option<ObjectReference> = None;
        if !pending_finalizers.is_empty() {
            // The pool drained first by the loop below (i = 0, `curr_vec` still `entry_vec`),
            // and its tag. `tag(pool i) == i + 1`; `CANDIDATES_STATUS` is 2 bits and all four
            // values are taken, so there is no spare tag for a buffer of our own.
            let tag = ((entry_vec + 1) % NUM_OF_CANDIDATES_VECTORS + 1) as u8;
            let pool = unsafe { lxr.s_cycle_candidates() };
            let mut buf = pool.local_buffer();
            // The true head, NOT `pending_finalizers[0].0`: the VM walker skips finalizers
            // already handed to Java, so its first report need not be the head. Cutting the edge
            // into a non-head node would leave the real head rooted and the nodes in front of the
            // cut unreachable from the seed.
            if let Some((head, mirror)) =
                <VM::VMCollection as Collection<VM>>::finalizer_take_list_head()
            {
                // The static is ALREADY cleared by the upcall -- the pointer is gone from the
                // heap, not just from the count below. That is the fix: `Finalizer.unfinalized`
                // lives in the `Finalizer` class mirror, an ordinary object whose static fields
                // `InstanceMirrorKlass::oop_iterate` walks unconditionally, so while the pointer
                // was still there `mark` would find it whenever it greyed the mirror and subtract
                // the same edge the hand-decrement below had already taken off. Step 3B restores
                // both the count and the pointer.
                //
                // DIAGNOSTIC (log only): `mirror` owns the static, and this state is what decides
                // whether trial deletion walks it.
                gc_log!([2]
                    "    - finalizers: mirror={:?} rc={} s_rc={} tag={} colour={}",
                    mirror,
                    lxr.rc.count(mirror),
                    cc::strong_rc(mirror, Ordering::SeqCst),
                    cc::tag(mirror, Ordering::SeqCst),
                    cc::colour(mirror, Ordering::SeqCst),
                );
                // Already at real rc 0: nothing to subtract, and decrementing would underflow.
                if lxr.rc.count(head) > 1 {
                    // SAFETY: single-threaded CycleFullRC -- see the impl header.
                    unsafe { lxr.rc_with_overflow.dec_exclusive(head) };
                    // `s_rc == 0` is what makes an object a candidate: `should_mark` requires it.
                    // Nothing is lost -- `scan_black` reconstructs `s_rc` from `rc` in step 3B.
                    // SAFETY: as above.
                    unsafe { cc::strong_rc_dec_exclusive(head) };
                    cut_head = Some(head);
                    // Only seed if the head is not already filed in some pool. If it is, it will
                    // be marked through that pool and a second entry would mark it twice.
                    if cc::tag(head, Ordering::Relaxed) == 0 {
                        // SAFETY: as above.
                        unsafe {
                            cc::set_tag_exclusive(head, tag, Ordering::Relaxed)
                        };
                        buf.push(head);
                    }
                } else {
                    // Not cutting after all, so the pointer goes back now: nothing later will do
                    // it, and mutators must not resume to a nulled `unfinalized`.
                    <VM::VMCollection as Collection<VM>>::finalizer_restore_list_head(head);
                }
            }
            gc_log!([2]
                "    - finalizers: {} pending, head {}",
                pending_finalizers.len(),
                if cut_head.is_some() { "cut" } else { "not cut" },
            );
        }

        // Drain all three pools BEFORE any producer handle exists.
        //
        // Up front, and not per round, for two reasons. `into_final_buffers` takes `&mut self` and
        // is only valid with no live `LocalBuffer`, so it cannot be interleaved with the marking
        // it feeds. And draining everything first means nothing this collection *produces* can be
        // consumed by it -- without that, the pool filled by the first round's `mark` is the pool
        // the third round would drain, and `mark`'s retry path would re-process objects it had
        // just deferred. That path fires on `is_logged`, which is a static property of the
        // object's field unlog bits, so re-processing them could only defer them again.
        //
        // Rotating `curr_vec` here rather than indexing the pools directly keeps the two existing
        // accessors honest: `s_cycle_candidates_mut()` is defined as pool `curr_vec + 1` and the
        // `vec_index` expression is its tag, so advancing `curr_vec` re-points both together and
        // they cannot disagree. The pools visited are `curr+1`, `curr+2`, `curr` -- all three,
        // each exactly once.
        let mut buffers: Vec<(FinalBuffers<ObjectReference>, u8)> =
            Vec::with_capacity(NUM_OF_CANDIDATES_VECTORS as usize);
        for i in 0..NUM_OF_CANDIDATES_VECTORS {
            lxr.curr_vec.set((entry_vec + i) % NUM_OF_CANDIDATES_VECTORS);
            let vec_index = ((lxr.curr_vec.get() + 1) % NUM_OF_CANDIDATES_VECTORS + 1) as u8;
            let candidates = unsafe { lxr.s_cycle_candidates_mut() }.into_final_buffers();
            buffers.push((candidates, vec_index));
        }

        #[cfg(feature = "graph_project")]
        {
            let mut reporter = lxr.graph_reporter.lock().unwrap();
            for (candidates, vec_index) in buffers.iter_mut() {
                let it = candidates.iter_mut();
                self.report_candidates_sub_graph(it, &mut reporter, *vec_index);
            }
        }

        #[cfg(feature = "s_rc_stats")]
        {
            self.stats.raw_cycle_candidates = buffers.iter().map(|(c, _)| c.len()).sum();
        }

        // Mark phase, over every buffer, rotating `curr_vec` between them.
        //
        // The handle is re-acquired every round and dropped at the end of it. It cannot be hoisted
        // the way `single_buff_gc` hoists it: the handle is bound to a pool at acquisition while
        // `mark` reads the tag from `curr_vec` at push time, so a handle that outlived a rotation
        // would write one pool's objects under another pool's tag. Nothing would catch that -- the
        // accessors mint references out of an `UnsafeCell`, so it is not a borrow error, and the
        // damage is silent: `should_mark` never matches those objects again.
        for (i, (candidates, vec_index)) in buffers.iter_mut().enumerate() {
            lxr.curr_vec
                .set((entry_vec + i as u8) % NUM_OF_CANDIDATES_VECTORS);
            let mut cand_buffer = unsafe { lxr.curr_s_cycle_candidates() }.local_buffer();
            // Read after the `set` above and passed by value, so the handle and the tag `mark`
            // writes provably come from the same rotation step.
            // `None`: a FullRC collection must not spill.  Its scan and collect run in this same
            // packet with the `*_exclusive` stores, so a spilled packet still marking during them
            // would break the one-thread contract the impl header sets out.
            self.mark_buffer(lxr, candidates, *vec_index, lxr.curr_vec.get(), dfs_stack, &mut cand_buffer, None);
        }

        lxr.curr_vec.set(entry_vec);

        #[cfg(feature = "s_rc_stats")]
        {
            self.stats.candidates_after_filter = buffers.iter().map(|(c, _)| c.len()).sum();
        }

        // Scan phase, over every buffer. `scan` and `scan_black` produce no candidates, so this
        // phase needs no handle.
        for (candidates, _) in buffers.iter_mut() {
            let mut it = candidates.iter_mut();
            while let Some(cand) = it.next() {
                self.scan(*cand, lxr, dfs_stack, nested_stack);
            }
        }

        lxr.in_cycle_collection.store(false, Ordering::Relaxed);

        // FINALIZER_RC_PLAN step 3B -- read the verdict, restore the chain, enqueue.
        //
        // THIS SEAM IS LOAD-BEARING. `all_buff_gc` runs mark-over-all-buffers, then
        // scan-over-all-buffers, then collect-over-all-buffers. That global phase structure is
        // the only reason it is safe to act on a WHITE object here: no `collect_whites` has run
        // yet, so nothing has been freed. If the phases are ever restructured per-candidate, this
        // breaks SILENTLY -- a neighbouring candidate would free a referent out from under us.
        if !pending_finalizers.is_empty() {
            // 1. READ THE VERDICT FIRST. `scan_black` below re-increments every referent and
            //    repaints it black, which erases exactly the colouring being read here. The two
            //    steps cannot be merged into one loop.
            //
            //    WHITE == the collector actually decided this is garbage, i.e. nothing outside the
            //    collected subgraph wants it -- which, with the finalizer edge subtracted by
            //    `mark`, is precisely "wanted by nothing but its own Finalizer".
            //
            //    NOT `rc.count(t) == 1`: `mark` aborts and reverts whenever it meets an
            //    SATB-logged slot, recolouring the object BLACK_IN_STACK and deferring it to a
            //    later GC. `scan` only acts on GREY, so an aborted candidate is never scanned and
            //    its count can read 1 without the collector ever having reasoned about its graph.
            //    Treating that as "finished" resurrected such an object and `collect_whites` then
            //    walked into it. Crashed `pmd`.
            //
            //    Not-WHITE simply means "not decided this time"; the finalizer is retried at the
            //    next FullRC.
            let mut to_enqueue: Vec<ObjectReference> = vec![];
            // Gated on the cut actually having happened. Without it the chain is still rooted, so
            // a WHITE referent could only have come from some unrelated candidate -- and the
            // restore below, which is rooted at the head, would not cover it. Enqueueing an object
            // we have not restored hands the VM something `collect_whites` is about to free.
            if cut_head.is_some() {
                for (f, t) in pending_finalizers.iter() {
                    if cc::colour(*t, Ordering::Relaxed) == WHITE {
                        to_enqueue.push(*f);
                    }
                }
            }

            // 2. RESTORE. Put the root edge back and re-blacken the chain.
            //
            // One `scan_black` from the head covers the whole chain AND every referent, because
            // each `Finalizer` reaches its referent as an ordinary field. Where it stops early at
            // an already-black node, that node was restored by `scan`'s own `scan_black`, which
            // re-incremented its children -- so the chain past it is restored too. Every node is
            // therefore either already black, or reached from here.
            //
            // The finalizers have to survive regardless of the verdict: `finalize()` has not run
            // yet, so nothing here is garbage until the VM says so.
            if let Some(head) = cut_head {
                // `scan_black` asserts `count > 1` on entry, so the edge goes back first.
                // SAFETY: single-threaded CycleFullRC -- see the impl header.
                unsafe { lxr.rc_with_overflow.inc_exclusive(head) };
                // And the POINTER goes back with it, undoing step 3A's clear. It must be back
                // before `collect_whites` runs and before the pause ends -- mutators resuming to
                // a nulled `unfinalized` would lose every pending finalizer.
                <VM::VMCollection as Collection<VM>>::finalizer_restore_list_head(head);
                if !is_black(head) {
                    self.scan_black(head, lxr, nested_stack);
                }
            }

            // 3. ENQUEUE. Anything still not black was not restored by step 2, which means the
            //    reasoning above is wrong for this object -- drop it rather than hand the VM an
            //    object `collect_whites` is about to free. It is retried at the next FullRC.
            to_enqueue.retain(|f| {
                let ok = is_black(*f);
                debug_assert!(ok, "finalizer {:?} not restored by scan_black from the list head", f);
                ok
            });
            if !to_enqueue.is_empty() {
                gc_log!([2] "    - finalizers: {} enqueued for Java", to_enqueue.len());
                let old_head =
                    <VM::VMCollection as Collection<VM>>::enqueue_finalizers(&to_enqueue);

                // Account for the edges the splice just created.
                //
                // `enqueue_finalizers` chains the batch through the `discovered` fields and
                // stores them RAW -- `set_next_reference` writes the slot directly, with no write
                // barrier, so no increment is applied. But the store that later DESTROYS each of
                // those edges is an ordinary Java field write: `ReferenceHandler` does
                // `ref.discovered = null`, which goes through the barrier and emits a real
                // DECREMENT. Without the increments below, every enqueued finalizer takes one
                // unmatched decrement, its count falls one too far, and it is freed while still
                // live -- after which a later `collect_whites` walks into freed memory. That is
                // the `pmd` crash: SIGSEGV at offset 12 of a null klass. Confirmed by bisection
                // (enqueue disabled: 3/3 pmd runs pass; enabled: crashes).
                //
                // The baseline gets away with the identical raw store in `ProcessDiscoveredList`
                // because it only runs in `Pause::Full`, where a full trace rebuilds RC
                // afterwards. `FullRC` has no trace, so an undercount here is permanent. Same
                // shape as the original bug: baseline machinery quietly relying on a trace this
                // fork does not run.
                //
                // The edge targets are every element after the first (each is pointed at by its
                // predecessor's `discovered`) plus the previous pending-list head, which the tail
                // now points at. `to_enqueue[0]` is the target of no edge we created -- the VM's
                // pending-list root is not a Java slot and carries no count.
                //
                // `rc` ONLY -- deliberately not `strong_rc`. The two error directions are not
                // symmetric. An `s_rc` that is too low is handled: `strong_rc_dec` clamps at zero
                // and returns `Err(STRONG_RC_ALREADY_ZERO)`, which `process_decs` does not act on
                // (rc.rs:1143), and `scan_black` reconstructs the value from `rc` the next time
                // the collector looks at the object. An `s_rc` that is too high is a silent,
                // permanent cycle leak: the object never reaches `s_rc == 0`, so it never becomes
                // a candidate and the collector never examines it again.
                let edge_targets = to_enqueue[1..].iter().copied().chain(old_head);
                for x in edge_targets {
                    // SAFETY: single-threaded CycleFullRC -- see the impl header.
                    unsafe { lxr.rc_with_overflow.inc_exclusive(x) };
                }
            }
        }

        // Collect phase, over every buffer. `curr_vec` is back to its entry value, so the
        // candidates `collect_blacks` discovers are tagged and filed exactly as they would be by
        // an ordinary collection, and are drained two GCs from now.
        let mut cand_buffer = unsafe { lxr.curr_s_cycle_candidates() }.local_buffer();
        for (candidates, _) in buffers.iter_mut() {
            let mut it = candidates.iter_mut();
            while let Some(cand) = it.next() {
                debug_assert!(cc::colour(*cand, Ordering::SeqCst) != GREY);
                self.collect_whites(*cand, lxr, dfs_stack, nested_stack, &mut cand_buffer);
            }
        }
        drop(cand_buffer);

        #[cfg(feature = "s_rc_stats")]
        {
            self.stats.satb_map_size = lxr.satb_map.len();
            // Capacity, not length: the stacks are reused for the whole collection, so this is the
            // high-water mark to within a doubling, read once here at no per-object cost.
            self.stats.peak_stack = dfs_stack.capacity().max(nested_stack.capacity());
            self.stats.flush_to_counters();
        }
    }

    fn get_slot_logging_state(&self, slot: VM::VMSlot) -> u8 {
        Self::UNLOG_BITS.load_atomic(slot.to_address(), Ordering::SeqCst)
    }

    fn get_child(&self, slot: <VM as vm::VMBinding>::VMSlot, lxr: &LXR<VM>) -> Option<ObjectReference>{
        let child = slot.load();
        if self.get_slot_logging_state(slot) == Self::UNLOGGED_VALUE{
            return child;
        }
        else{
            debug_assert!(self.get_slot_logging_state(slot) != Self::UNLOGGED_VALUE);
            // The increment is INSIDE the loop, at the return.
            //
            // It used to sit after the loop, where it was unreachable: the loop `return`s out of
            // the function on success and has no `break`, so control never reached the line below
            // it. `cc.satb_reads` therefore read 0 no matter how much this path ran -- confirmed on
            // `pmd` and `biojava` (`~/prod-ae/bundle/FINDINGS.md` 19), where it was 0 beside a
            // non-zero `barrier.satb_inserts`, i.e. entries were being written and never counted
            // as read.
            //
            // NOTE this counts SUCCESSFUL READS, one per call, which is what the name says. It does
            // NOT count spin iterations, so it still cannot answer
            // `~/mmtk/CYCLE_COLLECTOR_PERF_REVIEW.md` 4(a)'s question -- whether this spin ever
            // actually spins. That needs a separate counter incremented in the failure arm.
            loop {
                if let Some(m) = lxr.satb_map.get(&slot).map(|v| *v) {
                    #[cfg(feature = "s_rc_stats")]
                    { self.stats.satb_reads.set(self.stats.satb_reads.get() + 1); }
                    return m;
                }
                std::hint::spin_loop();
            }
        }
    }

    /// Trial deletion: decrements RC of children for each GREY candidate via DFS.
    ///
    /// **Decrements are applied on commit, not while walking.**  The field walk only collects the
    /// object's children; the writes happen afterwards, once it is known whether a logged slot
    /// turned up.  On the logged path nothing has been decremented, so there is nothing to revert.
    ///
    /// That replaces an unwind loop that popped `num_of_childs` entries off `dfs_stack` and assumed
    /// they were exactly the children it had just pushed -- true only while one thread owns the
    /// stack, and false as soon as it can be split.  It also makes this phase **dec-only**, which
    /// is what keeps `OVERFLOW_RC_TABLE` invariant 2 ("inc and dec never run concurrently") true
    /// with several workers marking.
    ///
    /// `dfs_stack` is scratch owned by `do_work` and reused across every candidate.  It is drained
    /// to empty before returning, so it needs no clearing on entry; the debug assertions hold that
    /// invariant in place.
    /// `children` is scratch owned by `mark_buffer`, cleared once per object.
    /// `cand_buffer` is the shared producer handle owned by `do_work`; see its comment there for
    /// why one handle covers every phase.
    /// The marking loop, over everything in `work` and everything it discovers.
    ///
    /// Two buffers, as in `ProcessDecs`: `work` is drained, `new_work` is filled with the children
    /// found along the way, and a `new_work` that reaches [`Self::SPLIT_THRESHOLD`] becomes another
    /// packet instead of growing.  When `work` runs dry the two swap, so whatever was not exported
    /// is processed here.  Filled chunks leave, the partial remainder stays -- exactly
    /// `ProcessDecs`, which exports `new_decs` on overflow and drains the leftover locally.
    ///
    /// Two buffers rather than one stack with a `drain(..n)`: taking a chunk out of the middle of a
    /// single vector copies it out *and* shifts everything after it down, thousands of times per
    /// GC.  Handing over a whole buffer is a pointer swap.
    ///
    /// Order is breadth-first by chunk rather than depth-first.  Trial deletion is indifferent --
    /// it needs each object visited once and each out-edge subtracted once, neither of which is
    /// order-dependent.
    ///
    /// `work` is drained to empty before returning, so `all_buff_gc` can reuse one buffer across
    /// its three rounds and hand the same one to `scan`.
    ///
    /// `curr_vec` is the pool `cand_buffer` was acquired from, passed rather than read from
    /// `lxr.curr_vec`: the tag written below and the pool pushed into must come from the same
    /// value, and in the concurrent path several workers share that `Cell`.
    fn mark_from_stack(
        &self,
        lxr: &LXR<VM>,
        curr_vec: u8,
        work: &mut Vec<ObjectReference>,
        children: &mut Vec<ObjectReference>,
        cand_buffer: &mut LocalBuffer<'_, ObjectReference>,
        spill: Option<Spill<'_>>,
    ) {
        #[cfg(feature = "lxr_stw")]
        let _ = spill;
        let mut new_work = Vec::<ObjectReference>::with_capacity(Self::SPLIT_THRESHOLD);
        loop {
            let curr = match work.pop() {
                Some(o) => o,
                None => {
                    if new_work.is_empty() {
                        break;
                    }
                    std::mem::swap(work, &mut new_work);
                    continue;
                }
            };
            debug_assert!(self.rc.count(curr) > 0);
            let mut is_logged = false;
            children.clear();

            if is_black(curr)
                && cc::strong_rc(curr, Ordering::Relaxed) == 0
                && cc::tag(curr, Ordering::Relaxed) == 0
            {
                #[cfg(feature = "s_rc_stats")]
                { self.stats.claim_attempts.set(self.stats.claim_attempts.get() + 1); }
                // Claim the object before walking it.  Losing means another worker greyed it and is
                // walking it, so this thread must not: that is what keeps each object's out-edges
                // subtracted exactly once.  Cannot fail while mark is single-threaded.
                if !cc::try_claim_grey(curr) {
                    continue;
                }
                #[cfg(feature = "s_rc_stats")]
                { self.stats.claim_wins.set(self.stats.claim_wins.get() + 1); }
                if VM::VMScanning::is_obj_array(curr) {
                    // An object array's references are one contiguous, indexable run of slots, so
                    // this walk can be written directly -- and, unlike `iterate_fields`, *stopped*
                    // at the first logged field.
                    //
                    // Through `iterate_fields` there is no way out: `SlotVisitor` has no abort
                    // hook and the binding's `oop_iterate` loops unconditionally, so once
                    // `is_logged` is set every remaining element is still visited, each one paying
                    // a heap load and a metadata load to do nothing.  On a large array whose first
                    // element is logged that is a full array walk for no result.
                    //
                    // These are exactly the slots `iterate_fields` would visit, in the same order:
                    // under `CLDScanPolicy::Ignore`, `ObjArrayKlass::oop_iterate` skips `do_klass`
                    // and walks `array.data(T_OBJECT)`, which is the same base and length that
                    // `obj_array_data` describes.  So the collected set is identical to the generic
                    // path's -- this changes only *when the loop stops*, never what it decides.
                    let data = VM::VMScanning::obj_array_data(curr);
                    for i in 0..data.len() {
                        let slot = data.get(i);
                        // Load the value *before* testing this slot's unlog bit, exactly as the
                        // generic visitor does.  A mutator that writes and logs between the two is
                        // caught by the test; one that did so between a test and a *later* load
                        // would not be, and the collector would then collect and decrement the NEW
                        // referent while `scan` reads the old one back out of `satb_map` -- leaving
                        // the new one permanently short.  These two lines must not be reordered.
                        let child = slot.load();
                        if self.get_slot_logging_state(slot) == Self::LOGGED_VALUE {
                            is_logged = true;
                            break;
                        }
                        if let Some(x) = child {
                            children.push(x);
                        }
                    }
                } else {
                    let visitor = |slot: <VM as vm::VMBinding>::VMSlot, _| {
                        let child = slot.load();
                        if self.get_slot_logging_state(slot) == Self::LOGGED_VALUE{
                                is_logged = true;
                        }
                        else if let Some(x) = child{
                            if !is_logged{
                                children.push(x);
                            }

                        }
                    };
                    curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
                }
                #[cfg(feature = "s_rc_stats")]
                { self.stats.objects_in_mark.set(self.stats.objects_in_mark.get() + 1); }
                if !is_logged {
                    // Commit: subtract each collected edge exactly once, then hand the children to
                    // the walk.
                    for &x in children.iter() {
                        let prev = lxr.rc_with_overflow.dec(x);
                        debug_assert!(prev != 1);
                        debug_assert!(prev != 0);
                        let _ = cc::strong_rc_dec(x);
                        new_work.push(x);
                        #[cfg(feature = "s_rc_stats")]
                        { self.stats.pushes.set(self.stats.pushes.get() + 1); }

                        // A full buffer becomes a packet; the swap above never sees it.
                        //
                        // `x` was decremented on the line above before it was pushed, so a chunk is
                        // complete the moment it leaves: the receiving packet pops it and applies
                        // the same guard and `try_claim_grey` this one would have.
                        //
                        // `None` on the `Pause::FullRC` path, where `new_work` simply grows -- the
                        // same single unbounded stack the loop had before spilling existed.
                        #[cfg(not(feature = "lxr_stw"))]
                        if let Some(sp) = spill {
                            if new_work.len() >= Self::SPLIT_THRESHOLD {
                                let chunk = std::mem::replace(
                                    &mut new_work,
                                    Vec::with_capacity(Self::SPLIT_THRESHOLD),
                                );
                                GCWorker::<VM>::current().add_work_prioritized(
                                    WorkBucketStage::Unconstrained,
                                    CycleMark::<VM>::spilled(
                                        chunk,
                                        sp.vec_index,
                                        sp.curr_vec,
                                        sp.c.clone_with_mark(),
                                    ),
                                );
                            }
                        }
                    }
                } else {
                    #[cfg(feature = "s_rc_stats")]
                    { self.stats.mark_logged.set(self.stats.mark_logged.get() + 1); }
                    // Nothing was decremented, so there is nothing to revert -- but the NET effect
                    // of the old walk-then-revert has to be reproduced exactly, and it is
                    // asymmetric on purpose:
                    //
                    //   rc         unchanged   (walk decremented, revert incremented)
                    //   strong_rc  -1          (walk decremented; the revert's `strong_rc_inc` is
                    //                           commented out, so it was never restored)
                    //   strong_rc  -2 in total when the child is not black (the second decrement
                    //                           below, which the revert applied as an extra)
                    //
                    // Do NOT "correct" this while reading it: `strong_rc` is a sinking heuristic,
                    // not a reference count, and making it accurate stops cyclic garbage being
                    // collected at all (`sanity_checker.rs` fires; `FINDINGS.md` 26.5).
                    for &x in children.iter() {
                        let _ = cc::strong_rc_dec(x);
                        cand_buffer.push(x);
                        cc::set_tag(x, (curr_vec + 1) as u8, Ordering::Relaxed);
                        if !is_black(x) { // this condition is unnecessary. it is only to satisfy assertion (should be remove after assertion removal)
                            let _ = cc::strong_rc_dec(x);
                        }
                    }

                    cand_buffer.push(curr);

                    cc::set_tag(curr, (curr_vec + 1) as u8, Ordering::Relaxed);

                    // Restore the colour LAST, and only here.  Deferring `curr` means its
                    // out-edges are never subtracted, and BLACK_IN_STACK is what tells `scan`
                    // (GREY only) and `scan_black` (`!is_black`) to leave it alone.  Written any
                    // earlier, `curr` is BLACK with `tag == 0` and `strong_rc == 0` for the whole
                    // loop above -- claimable again -- and a second mark packet re-greys it.  It
                    // then re-enters `scan` with its out-edges un-subtracted: RC > 1 has
                    // `scan_black` increment edges nothing decremented, and RC == 1 paints it
                    // WHITE, so `collect_whites` frees it and death-processes children whose RC
                    // then falls below the `real_refs + 1` floor.
                    //
                    // SeqCst, not Relaxed: the store must not become visible before the tag above,
                    // or the window reopens between the two.  While the colour is GREY,
                    // `try_claim_grey`'s CAS cannot succeed whatever a racing packet read for the
                    // tag, which is what actually closes it.
                    cc::set_colour(curr, BLACK_IN_STACK, Ordering::SeqCst);
                }
            } else if is_black(curr) {
                // Could not be claimed -- tagged, still strongly referenced, or claimed by
                // someone else -- so just record it as in-stack.  CAS, not a store: `is_black`
                // was read at the top of the guard and another packet can win `try_claim_grey`
                // before we get here, which a store would clobber back to black.  See
                // `cc::try_mark_in_stack`.
                let _ = cc::try_mark_in_stack(curr);
            }
        }
    }

    /// Scan phase: GREY objects with RC > 1 are externally referenced — restore them (scan_black).
    /// GREY objects with RC == 1 are garbage — mark WHITE.
    /// `dfs_stack` is scratch owned by `do_work` and reused across every candidate.  It is drained
    /// to empty before returning, so it needs no clearing on entry; the debug assertions hold that
    /// invariant in place.
    /// `black_stack` is handed straight through to `scan_black`, which nests inside this
    /// traversal and therefore needs a second stack of its own.
    fn scan(
        &self,
        o: ObjectReference,
        lxr: &LXR<VM>,
        dfs_stack: &mut Vec<ObjectReference>,
        black_stack: &mut Vec<ObjectReference>,
    ) {
        debug_assert!(dfs_stack.is_empty());
        dfs_stack.push(o);
        while let Some(curr) = dfs_stack.pop() {
            #[cfg(feature = "s_rc_stats")]
            { self.stats.objects_in_scan.set(self.stats.objects_in_scan.get() + 1); }
            debug_assert!(self.rc.count(curr) > 0);
            let visitor = |slot: <VM as vm::VMBinding>::VMSlot, _| {
                if let Some(x) = self.get_child(slot, lxr) {
                    debug_assert!(self.rc.count(x) > 0);
                    dfs_stack.push(x);
                }
            };

            if cc::colour(curr, Ordering::Relaxed) == GREY {
                debug_assert!(cc::strong_rc(curr, Ordering::SeqCst) == 0);
                if RefCountHelper::<VM>::NEW.count(curr) > 1 {
                    self.scan_black(curr, lxr, black_stack);
                } else {
                    // SAFETY: single-threaded CycleFullRC -- see the impl header.
                    unsafe { cc::set_colour_exclusive(curr, WHITE, Ordering::Relaxed) };
                    curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
                }
            }
        }
    }


    /// Restores RCs for objects found to be externally reachable.
    /// Reconstructs strong_rc and marks objects BLACK.
    /// `dfs_stack` is scratch owned by `do_work` and reused across every candidate.  It is drained
    /// to empty before returning, so it needs no clearing on entry; the debug assertions hold that
    /// invariant in place.
    fn scan_black(&self, o: ObjectReference, lxr: &LXR<VM>, dfs_stack: &mut Vec<ObjectReference>) {
        debug_assert!(dfs_stack.is_empty());
        dfs_stack.push(o);
        // `Vec::last` borrows immutably and copies out here, so the visitor below is free to push.
        // `ChunkedStack::last` took `&mut self` (it dropped an emptied chunk as a side effect),
        // which is what the old `curr_copy` dance was working around.
        while let Some(&curr) = dfs_stack.last() {
            debug_assert!(self.rc.count(curr) > 1);
            if !is_black(curr) {
                // SAFETY: single-threaded CycleFullRC -- see the impl header.
                unsafe { cc::set_colour_exclusive(curr, BLACK_IN_STACK, Ordering::SeqCst) };
                let s_rc = self.rc.count(curr) - 1;
                // SAFETY: single-threaded CycleFullRC -- see the impl header.  Note the
                // obligation here is the *wider* one: `STRONG_RC_TABLE` is 4 bits per entry, so
                // two adjacent objects share a byte and the exclusive store is a read-merge-write
                // of that byte with no CAS.  No other thread may write *any* entry in the same
                // byte -- in practice, nothing else may write this table at all.  That holds only
                // because `ProcessDecs` has fully drained under the `decs -> sweep -> cc` order;
                // see `RefCountHelper::strong_rc_inc_exclusive` for the full argument.
                if s_rc > MAX_STRONG_REF_COUNT as RcBits{
                    unsafe { cc::set_strong_rc_exclusive(curr, MAX_STRONG_REF_COUNT, Ordering::Relaxed) };
                }
                else{
                    unsafe { cc::set_strong_rc_exclusive(curr, s_rc as u8, Ordering::Relaxed) };
                }
                curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, |slot: <VM as vm::VMBinding>::VMSlot, b| {
                    if let Some(x) = self.get_child(slot, lxr) {
                        // SAFETY: single-threaded CycleFullRC -- see the impl header.
                        let _prev = unsafe { lxr.rc_with_overflow.inc_exclusive(x) };
                        if !in_stack(x) {
                            // SAFETY: as above.
                            unsafe { cc::strong_rc_inc_exclusive(x) };
                            dfs_stack.push(x);
                        }
                        debug_assert!(self.rc.count(x) > 1);
                    }
                });
                #[cfg(feature = "s_rc_stats")]
                { self.stats.objects_in_scan_black.set(self.stats.objects_in_scan_black.get() + 1); }
              
            } else {
                // SAFETY: single-threaded CycleFullRC -- see the impl header.
                unsafe { cc::set_colour_exclusive(curr, BLACK_OUT_OF_STACK, Ordering::Relaxed) };
                debug_assert!(cc::strong_rc(curr, Ordering::SeqCst) != 0);
                dfs_stack.pop();
            }
        }
    }
    

    /// Frees WHITE (garbage) objects. Also handles BLACK_IN_STACK objects with RC == 1
    /// by delegating to collect_blacks.
    /// `dfs_stack` is scratch owned by `do_work` and reused across every candidate.  It is drained
    /// to empty before returning, so it needs no clearing on entry; the debug assertions hold that
    /// invariant in place.
    /// `black_stack` is handed straight through to `collect_blacks`, which nests inside this
    /// traversal and therefore needs a second stack of its own.
    fn collect_whites(
        &self,
        o: ObjectReference,
        lxr: &LXR<VM>,
        dfs_stack: &mut Vec<ObjectReference>,
        black_stack: &mut Vec<ObjectReference>,
        cand_buffer: &mut LocalBuffer<'_, ObjectReference>,
    ) {
        debug_assert!(dfs_stack.is_empty());
        dfs_stack.push(o);
        while let Some(curr) = dfs_stack.pop() {
            #[cfg(feature = "s_rc_stats")]
            { self.stats.objects_in_collect.set(self.stats.objects_in_collect.get() + 1); }
            let visitor = |slot: <VM as vm::VMBinding>::VMSlot, _| {
                debug_assert!(self.get_slot_logging_state(slot) == Self::UNLOGGED_VALUE);
                if let Some(x) = slot.load() {
                    dfs_stack.push(x);
                }
            };

            debug_assert!(cc::strong_rc(curr, Ordering::SeqCst) as RcBits <= lxr.rc.count(curr));

            if cc::colour(curr, Ordering::Relaxed) == WHITE {
                #[cfg(feature = "graph_project")]
                {
                    let mut reporter = lxr.graph_reporter.lock().unwrap();
                    reporter.add_cycle_collector_freed(curr.to_raw_address().as_usize());
                }
                debug_assert!(self.rc.count(curr) == 1);
                debug_assert!(cc::strong_rc(curr, Ordering::SeqCst) == 0);
                // SAFETY: single-threaded CycleFullRC -- see the impl header.
                unsafe { cc::set_tag_exclusive(curr, 0, Ordering::Relaxed) };
                // SAFETY: single-threaded CycleFullRC -- see the impl header.
                unsafe { cc::set_colour_exclusive(curr, BLACK_OUT_OF_STACK, Ordering::Relaxed) };
                debug_assert!(lxr.rc.count(curr) == 1);
                curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
                self.process_dead_object(curr, lxr);
                // SAFETY: single-threaded CycleFullRC -- see the impl header.
                unsafe { self.rc.dec_exclusive(curr) };
            } else if cc::colour(curr, Ordering::Relaxed) == BLACK_IN_STACK
                && self.rc.count(curr) == 1
            {
                debug_assert!(cc::tag(curr, Ordering::Relaxed) != 0);
                self.collect_blacks(curr, lxr, black_stack, cand_buffer);
            }
        }
    }

    
    /// Frees BLACK objects with RC == 1 by decrementing children and reclaiming.
    /// Children that reach RC == 2 (death threshold) are also freed recursively.
    /// `dfs_stack` is scratch owned by `do_work` and reused across every candidate.  It is drained
    /// to empty before returning, so it needs no clearing on entry; the debug assertions hold that
    /// invariant in place.
    /// `cand_buffer` is the shared producer handle owned by `do_work`; see its comment there for
    /// why one handle covers every phase.
    fn collect_blacks(
        &self,
        o: ObjectReference,
        lxr: &LXR<VM>,
        dfs_stack: &mut Vec<ObjectReference>,
        cand_buffer: &mut LocalBuffer<'_, ObjectReference>,
    ) {
        debug_assert!(dfs_stack.is_empty());
        dfs_stack.push(o);
        while let Some(curr) = dfs_stack.pop() {
            #[cfg(feature = "s_rc_stats")]
            { self.stats.objects_in_collect.set(self.stats.objects_in_collect.get() + 1); }
            debug_assert!(self.rc.count(curr) == 1);
            let visitor = |slot: <VM as vm::VMBinding>::VMSlot, _| {
                debug_assert!(self.get_slot_logging_state(slot) == Self::UNLOGGED_VALUE);
                if let Some(x) = slot.load() {
                    debug_assert!(self.rc.count(x) > 1);
                    // SAFETY: single-threaded CycleFullRC -- see the impl header.
                    let prev_rc = unsafe { lxr.rc_with_overflow.dec_exclusive(x) };
                    // SAFETY: as above.  Returns the previous value directly rather than a
                    // `Result`, so the `Ok(1)` test below becomes a plain comparison against
                    // STRONG_RC_LAST_BEFORE_ZERO -- the same value, unwrapped.
                    let prev_s_rc = unsafe { cc::strong_rc_dec_exclusive(x) };
                    debug_assert!(prev_rc != 1);
                    if prev_rc == 2 {
                        dfs_stack.push(x);
                    } else if prev_s_rc == STRONG_RC_LAST_BEFORE_ZERO {
                        // SAFETY: single-threaded CycleFullRC -- see the impl header.
                        unsafe { cc::set_tag_exclusive(x, (lxr.curr_vec.get() + 1) as u8, Ordering::Relaxed) };
                        cand_buffer.push(x);
                    }
                }
            };
            curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
            debug_assert!(cc::strong_rc(curr, Ordering::SeqCst) as RcBits <= lxr.rc.count(curr));
            debug_assert!(is_black(curr));
            // SAFETY: single-threaded CycleFullRC -- see the impl header.
            unsafe { cc::set_tag_exclusive(curr, 0, Ordering::Relaxed) };
            // SAFETY: single-threaded CycleFullRC -- see the impl header.
            unsafe { cc::set_colour_exclusive(curr, BLACK_OUT_OF_STACK, Ordering::Relaxed) };
            #[cfg(feature = "graph_project")]
            {
                let mut reporter = lxr.graph_reporter.lock().unwrap();
                reporter.add_cycle_collector_freed(curr.to_raw_address().as_usize());
            }
            self.process_dead_object(curr, lxr);
            debug_assert!(lxr.rc.count(curr) == 1);
            // SAFETY: single-threaded CycleFullRC -- see the impl header.
            unsafe { self.rc.dec_exclusive(curr) };
        }
    }

    fn process_dead_object(&self, o: ObjectReference, lxr: &LXR<VM>) -> bool {
        crate::stat(|s| {
            s.dead_mature_objects += 1;
            s.dead_mature_volume += o.get_size::<VM>();

            s.dead_mature_rc_objects += 1;
            s.dead_mature_rc_volume += o.get_size::<VM>();

            if !lxr.immix_space.in_space(o) {
                s.dead_mature_los_objects += 1;
                s.dead_mature_los_volume += o.get_size::<VM>();

                s.dead_mature_rc_los_objects += 1;
                s.dead_mature_rc_los_volume += o.get_size::<VM>();
            }
        });
        let in_ix_space = lxr.immix_space.in_space(o);
        if !crate::args::BLOCK_ONLY && in_ix_space {
            self.rc.unmark_straddle_object(o);
        }
        #[cfg(feature = "sanity")]
        crate::util::sanity::sanity_checker::SANITY_DEAD_CYCLE_COUNT
            .store_atomic::<u8>(o.to_raw_address(), 0, Ordering::SeqCst);
        if cfg!(feature = "sanity") || ObjectReference::STRICT_VERIFICATION {
            unsafe { o.to_raw_address().store(0xdeadusize) };
        }
        if in_ix_space {
            if cfg!(feature = "lxr_log_reclaim") {
                lxr.immix_space
                    .rc_killed_bytes
                    .fetch_add(o.get_size::<VM>(), Ordering::Relaxed);
            }
            let block = Block::containing(o);
            lxr.immix_space
                .add_to_possibly_dead_mature_blocks(block, false);
            false
        } else {
            if cfg!(feature = "lxr_log_reclaim") {
                lxr.los()
                    .rc_killed_bytes
                    .fetch_add(o.get_size::<VM>(), Ordering::Relaxed);
            }
            lxr.los().rc_free(o);
            true
        }
    }

    fn should_mark(&self, o: ObjectReference, vec_index: u8) -> bool {
        cc::tag(o, Ordering::Relaxed) == vec_index
            && cc::strong_rc(o, Ordering::Relaxed) == 0
        // cc::strong_rc(o, Ordering::Relaxed) == 0
        //     && cc::tag(o, Ordering::Relaxed) != 0
    }

}



impl<VM: VMBinding> CycleFullRC<VM> {
    pub fn new(#[cfg(not(feature = "lxr_stw"))] c: LazySweepingJobsCounter) -> CycleFullRC<VM> {
        CycleFullRC::<VM> {
            t: CycleTraversal::new(),
            // Take ownership of the token handed over by `end_of_decs`.  Cloning it here and
            // dropping the original would be a redundant +1/-1 on the same cc counter.
            #[cfg(not(feature = "lxr_stw"))]
            _c: c,
        }
    }
}

/// The mark phase of a concurrent cycle collection, over one or more candidate buffers.
///
/// Scheduled by `LXR::schedule_cycle_mark`.  Its token is a `clone_with_mark`, so the generation's
/// mark counter stays non-zero while any mark packet is outstanding; when the last one drops,
/// `end_of_mark` fires and `LXR::on_lazy_mark_finished` schedules `CycleScanCollect`.
///
/// Several run at once, one per candidate buffer.  That is what the `try_claim_grey` claim and the
/// commit-ordered decrements of `mark` were put in place for: the claim is what keeps an object's
/// out-edges subtracted exactly once when two packets reach it from different candidates.
pub struct CycleMark<VM: VMBinding> {
    t: CycleTraversal<VM>,
    /// Candidate buffers this packet marks -- one, as scheduled.  Taken in `do_work`, pruned, and
    /// deposited in `lxr.mark_survivors`, where every packet's survivors are concatenated for the
    /// scan phase.
    buffers: Vec<Vec<ObjectReference>>,
    /// DFS work handed over by another packet's spill; empty in a seed packet.
    ///
    /// Resumed through `mark_from_stack`, NOT `mark_buffer`: these are objects found mid-traversal,
    /// already decremented by their parent, and must skip the candidate filter.
    stack: Vec<ObjectReference>,
    /// Tag of the pool the buffers were drained from; `should_mark` matches on it.
    vec_index: u8,
    /// `curr_vec` as it stood at fan-out -- the pool this packet's own candidates go INTO.
    curr_vec: u8,
    #[cfg(not(feature = "lxr_stw"))]
    _c: LazySweepingJobsCounter,
}

impl<VM: VMBinding> CycleMark<VM> {
    pub fn new(
        buffers: Vec<Vec<ObjectReference>>,
        vec_index: u8,
        curr_vec: u8,
        #[cfg(not(feature = "lxr_stw"))] c: LazySweepingJobsCounter,
    ) -> Self {
        Self {
            t: CycleTraversal::new(),
            buffers,
            stack: Vec::new(),
            vec_index,
            curr_vec,
            #[cfg(not(feature = "lxr_stw"))]
            _c: c,
        }
    }

    /// A packet for one chunk spilled out of another packet's DFS stack.
    ///
    /// No candidate buffers, so it deposits nothing in `mark_survivors` and never runs the
    /// candidate filter.  `vec_index` is carried anyway rather than defaulted: 0 is the tag for
    /// "not a candidate", so a defaulted one would make `should_mark` match non-candidates if this
    /// packet ever gained buffers.
    #[cfg(not(feature = "lxr_stw"))]
    pub fn spilled(
        stack: Vec<ObjectReference>,
        vec_index: u8,
        curr_vec: u8,
        c: LazySweepingJobsCounter,
    ) -> Self {
        Self {
            t: CycleTraversal::new(),
            buffers: Vec::new(),
            stack,
            vec_index,
            curr_vec,
            _c: c,
        }
    }
}

impl<VM: VMBinding> GCWork<VM> for CycleMark<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        #[cfg(feature = "s_rc_stats")]
        { self.t.stats.mark_packets.set(self.t.stats.mark_packets.get() + 1); }

        // Whatever was spilled to us, if anything.  `mark_buffer` appends this packet's own
        // candidates to it and runs one traversal over the lot, so a seed packet and a spill packet
        // take the same path -- a seed packet just starts from an empty `work`.
        let mut work = std::mem::take(&mut self.stack);
        let mut candidates = FinalBuffers::from_vecs(std::mem::take(&mut self.buffers));

        #[cfg(feature = "graph_project")]
        {
            let it = candidates.iter_mut();
            let mut reporter = lxr.graph_reporter.lock().unwrap();
            self.t.report_candidates_sub_graph(it, &mut reporter, self.vec_index);
        }

        #[cfg(feature = "s_rc_stats")]
        { self.t.stats.raw_cycle_candidates = candidates.len(); }

        // The pool named by the SNAPSHOT, so the handle and the tag `mark` writes cannot disagree.
        let mut cand_buffer = unsafe { lxr.s_cycle_candidates_at(self.curr_vec) }.local_buffer();

        // Spilling is on for this packet, and only this packet type -- `CycleFullRC` passes
        // `None`.  Borrowing our own token means a chunk we hand off is counted in the same
        // generation, and our token stays alive until `do_work` returns, so the mark counter cannot
        // reach zero while we are still splitting.
        #[cfg(not(feature = "lxr_stw"))]
        let spill = Some(Spill {
            c: &self._c,
            vec_index: self.vec_index,
            curr_vec: self.curr_vec,
        });
        #[cfg(feature = "lxr_stw")]
        let spill: Option<Spill<'_>> = None;

        self.t.mark_buffer(
            lxr,
            &mut candidates,
            self.vec_index,
            self.curr_vec,
            &mut work,
            &mut cand_buffer,
            spill,
        );

        #[cfg(feature = "s_rc_stats")]
        {
            self.t.stats.candidates_after_filter = candidates.len();
            self.t.stats.peak_stack = work.capacity();
            self.t.stats.flush_to_counters();
        }

        // Hand the survivors on, BEFORE `_c` drops: that drop is what schedules the packet which
        // reads them.
        lxr.mark_survivors
            .lock()
            .unwrap()
            .extend(candidates.into_vecs());
    }
}

/// Scan, after every mark packet has finished.
///
/// **Single-threaded, and out of scope for parallelisation** --
/// `CYCLE_COLLECTOR_PARALLEL_COLLECT_PLAN.md` §10.  `scan` and `scan_black` keep every
/// `*_exclusive` store, which is sound because the mark counter guarantees this starts only after
/// mark is complete and the scan counter guarantees collect starts only after this.
///
/// It nonetheless has a counter and a callback of its own, like the other three phases, so that the
/// day scan is parallelised the join already exists.
pub struct CycleScan<VM: VMBinding> {
    t: CycleTraversal<VM>,
    /// The surviving candidates from mark.  Scan neither adds nor removes any, so these are handed
    /// on to collect unchanged.
    buffers: Vec<Vec<ObjectReference>>,
    #[cfg(not(feature = "lxr_stw"))]
    _c: LazySweepingJobsCounter,
}

impl<VM: VMBinding> CycleScan<VM> {
    pub fn new(
        buffers: Vec<Vec<ObjectReference>>,
        #[cfg(not(feature = "lxr_stw"))] c: LazySweepingJobsCounter,
    ) -> Self {
        Self {
            t: CycleTraversal::new(),
            buffers,
            #[cfg(not(feature = "lxr_stw"))]
            _c: c,
        }
    }
}

/// Collect, after scan has finished.
///
/// One packet today; `collect_whites` and `collect_blacks` still use the `*_exclusive` stores and
/// still read `lxr.curr_vec` directly, both of which are sound only while that is true.  Stage C3
/// of the plan converts them; **do not schedule more than one of these before it has.**
pub struct CycleCollect<VM: VMBinding> {
    t: CycleTraversal<VM>,
    buffers: Vec<Vec<ObjectReference>>,
    /// The pool this packet's `LocalBuffer` is taken from -- `collect_blacks` produces candidates
    /// for the next GC.
    curr_vec: u8,
    #[cfg(not(feature = "lxr_stw"))]
    _c: LazySweepingJobsCounter,
}

impl<VM: VMBinding> CycleCollect<VM> {
    pub fn new(
        buffers: Vec<Vec<ObjectReference>>,
        curr_vec: u8,
        #[cfg(not(feature = "lxr_stw"))] c: LazySweepingJobsCounter,
    ) -> Self {
        Self {
            t: CycleTraversal::new(),
            buffers,
            curr_vec,
            #[cfg(not(feature = "lxr_stw"))]
            _c: c,
        }
    }
}

impl<VM: VMBinding> GCWork<VM> for CycleScan<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        let mut candidates = FinalBuffers::from_vecs(std::mem::take(&mut self.buffers));

        // Two stacks: `scan` nests into `scan_black`.
        //
        // No `LocalBuffer`: `scan` and `scan_black` produce no candidates, so this phase needs no
        // producer handle.  Only collect does.
        let mut dfs_stack = Vec::<ObjectReference>::with_capacity(4096);
        let mut nested_stack = Vec::<ObjectReference>::with_capacity(4096);

        // Classify GREY objects as WHITE (garbage) or restore them to BLACK.
        let mut it = candidates.iter_mut();
        while let Some(cand) = it.next() {
            self.t.scan(*cand, lxr, &mut dfs_stack, &mut nested_stack);
        }

        // Stays here, as the last thing scanning does: collect runs with the barrier off.
        lxr.in_cycle_collection.store(false, Ordering::Relaxed);

        #[cfg(feature = "s_rc_stats")]
        {
            self.t.stats.peak_stack = dfs_stack.capacity().max(nested_stack.capacity());
            self.t.stats.flush_to_counters();
        }

        // Hand the candidates on, BEFORE `_c` drops: that drop is what fires `end_of_scan`, which
        // reads them.  Scan removed none of them.
        lxr.scan_survivors
            .lock()
            .unwrap()
            .extend(candidates.into_vecs());
    }
}

impl<VM: VMBinding> GCWork<VM> for CycleCollect<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        let mut candidates = FinalBuffers::from_vecs(std::mem::take(&mut self.buffers));

        // Two stacks: `collect_whites` nests into `collect_blacks`.
        let mut dfs_stack = Vec::<ObjectReference>::with_capacity(4096);
        let mut nested_stack = Vec::<ObjectReference>::with_capacity(4096);
        let mut cand_buffer = unsafe { lxr.s_cycle_candidates_at(self.curr_vec) }.local_buffer();

        // Free WHITE objects and handle the remaining BLACK_IN_STACK ones.
        let mut it = candidates.iter_mut();
        while let Some(cand) = it.next() {
            debug_assert!(cc::colour(*cand, Ordering::SeqCst) != GREY);
            self.t.collect_whites(*cand, lxr, &mut dfs_stack, &mut nested_stack, &mut cand_buffer);
        }

        #[cfg(feature = "s_rc_stats")]
        {
            // A peak reading, so it belongs after collecting.  `fetch_max` in `flush_to_counters`,
            // so several packets taking it is correct.
            self.t.stats.satb_map_size = lxr.satb_map.len();
            self.t.stats.peak_stack = dfs_stack.capacity().max(nested_stack.capacity());
            self.t.stats.flush_to_counters();
        }
    }
}

#[cfg(feature = "graph_project")]
impl<VM: VMBinding> CycleTraversal<VM> {

    fn report_candidates_sub_graph(&self, mut candidates: FinalIterMut<'_, ObjectReference>, reporter: &mut GcCycleReport, vec_index: u8) {
        while let Some(cand) = candidates.next(){
            if self.should_mark(*cand, vec_index) {
                reporter.mark_candidate_sub_graph::<VM>(*cand);
                
            }
        }

        while let Some(cand) = candidates.next(){
            if self.should_mark(*cand, vec_index) {
                reporter.sweep_candidate_sub_graph::<VM>(*cand);
                
            }
        }
        reporter.append_to_default_file();
        reporter.clear();
    }
}
