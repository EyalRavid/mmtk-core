use super::cm::LXRWeakRefProcessEdges;
use super::{barrier, LXR};
use crate::scheduler::{gc_work::*, GCWork, GCWorker};
use crate::util::ObjectReference;
use crate::{vm::*, Plan, MMTK};
use crate::util::rc::{CANDIDATES_STATUS, MAX_STRONG_REF_COUNT, OBJ_COLOR_TABLE, STRONG_RC_TABLE, BLACK_OUT_OF_STACK, BLACK_IN_STACK, GREY, WHITE, STRONG_RC_LAST_BEFORE_ZERO};
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
    OBJ_COLOR_TABLE.load_atomic::<u8>(o.to_raw_address(), Ordering::Relaxed) == BLACK_IN_STACK
}

fn is_black(o: ObjectReference) -> bool {
    OBJ_COLOR_TABLE.load_atomic::<u8>(o.to_raw_address(), Ordering::Relaxed) <= BLACK_IN_STACK
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
}

#[cfg(feature = "s_rc_stats")]
impl CycleCollectorStats {
    const LOG_FILE: &'static str = "cycle_collector_stats.csv";

    fn log_to_file(&self) {
        use std::fs::OpenOptions;
        use std::io::Write;

        let needs_header = !std::path::Path::new(Self::LOG_FILE).exists()
            || std::fs::metadata(Self::LOG_FILE).map(|m| m.len() == 0).unwrap_or(true);

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(Self::LOG_FILE)
            .expect("failed to open cycle collector stats file");

        if needs_header {
            writeln!(file,
                "raw_cycle_candidates,candidates_after_filter,\
                 objects_in_mark,objects_in_scan,objects_in_scan_black,\
                 objects_in_collect,satb_map_size,satb_reads"
            ).unwrap();
        }

        writeln!(file, "{},{},{},{},{},{},{},{}",
            self.raw_cycle_candidates,
            self.candidates_after_filter,
            self.objects_in_mark.get(),
            self.objects_in_scan.get(),
            self.objects_in_scan_black.get(),
            self.objects_in_collect.get(),
            self.satb_map_size,
            self.satb_reads.get(),
        ).unwrap();
    }
}

pub struct CycleCollector<VM: VMBinding> {
    rc: RefCountHelper<VM>,
    #[cfg(not(feature = "lxr_stw"))]
    _c: LazySweepingJobsCounter,
    #[cfg(feature = "s_rc_stats")]
    stats: CycleCollectorStats,
}

impl<VM: VMBinding> GCWork<VM> for CycleCollector<VM> {

    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {

        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        lxr.satb_map.clear();
        lxr.in_cycle_collection.store(true, Ordering::SeqCst);
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
        // They are locals rather than `CycleCollector` fields on purpose: this packet is
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
            Some(Pause::FullRC) => self.all_buff_gc(lxr, &mut dfs_stack, &mut nested_stack),
            _ => self.single_buff_gc(lxr, &mut dfs_stack, &mut nested_stack),
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
/// The RC traversal in `mark`, `scan`, `scan_black`, `collect_whites` and `collect_blacks` uses the
/// `*_exclusive` variants of the reference-count helpers, which skip the atomic read-modify-write
/// (a CAS retry loop) that the ordinary `inc`/`dec` need.  Every one of them carries the same
/// safety obligation: **this thread must be the only one performing RC work**.
///
/// That holds because `CycleCollector` is a single `GCWork` packet, so it runs on exactly one GC
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
/// **If any of that changes -- the packet is split across workers, cycle collection is moved off
/// the GC worker pool, or decrement work is allowed to overlap it -- every `unsafe` block in this
/// impl becomes unsound and must go back to the atomic variants.**
impl<VM: VMBinding> CycleCollector<VM>{
    
    pub const UNLOGGED_VALUE: u8 = 0b1;
    pub const LOGGED_VALUE: u8 = 0b0;

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
        dfs_stack: &mut Vec<ObjectReference>,
        cand_buffer: &mut LocalBuffer<'_, ObjectReference>,
    ) {
        let mut it = candidates.iter_mut();
        while let Some(cand) = it.next() {
            if self.should_mark(*cand, vec_index) {
                // SAFETY: single-threaded CycleCollector -- see the impl header.
                unsafe { CANDIDATES_STATUS.store_atomic_exclusive::<u8>(cand.to_raw_address(), 0, Ordering::Relaxed) };
                debug_assert!(self.rc.count(*cand) > 0);
                self.mark(*cand, lxr, dfs_stack, cand_buffer);
            } else {
                if STRONG_RC_TABLE.load_atomic::<u8>(cand.to_raw_address(), Ordering::Relaxed) > 0 {
                    // SAFETY: single-threaded CycleCollector -- see the impl header.
                    unsafe { CANDIDATES_STATUS.store_atomic_exclusive::<u8>(cand.to_raw_address(), 0, Ordering::Relaxed) };
                }
                it.swap_remove_current();
            }
        }
    }

    /// Cycle collection over **one** candidate pool: the one filled two GCs ago.
    ///
    /// This is the ordinary path, taken by every collection except `Pause::FullRC`, and it is the
    /// behaviour this collector has always had. `curr_vec` is not touched, so the pool rotation is
    /// driven solely by `schedule_collection`.
    ///
    /// Candidates registered by *this* GC's decrements land in pool `curr_vec` and are not looked
    /// at here; they are drained two GCs later. That lag is what `all_buff_gc` removes.
    fn single_buff_gc(
        &mut self,
        lxr: &LXR<VM>,
        dfs_stack: &mut Vec<ObjectReference>,
        nested_stack: &mut Vec<ObjectReference>,
    ) {
        let vec_index = ((lxr.curr_vec.get() + 1) % NUM_OF_CANDIDATES_VECTORS + 1) as u8;
        let mut candidates = unsafe { lxr.s_cycle_candidates_mut() }.into_final_buffers();

        #[cfg(feature = "graph_project")]
        {
            let it = candidates.iter_mut();
            let mut reporter = lxr.graph_reporter.lock().unwrap();
            self.report_candidates_sub_graph(it, &mut reporter, vec_index);
        }

        #[cfg(feature = "s_rc_stats")]
        {
            self.stats.raw_cycle_candidates = candidates.len();
        }

        // One producer handle into the pool being filled, held for both producing phases.
        //
        // `mark` and `collect_blacks` are the only producers, they both target `curr_vec`, and
        // nothing drains that pool between them -- it is read by `into_final_buffers` two GCs
        // from now, by which time this handle has long been dropped. So one handle covers both.
        //
        // Acquiring per candidate (`mark`) and per qualifying edge (`collect_blacks`) costs a
        // `BufferPool` mutex on both `acquire_buffer` and `LocalBuffer::drop`; hoisting it pays
        // that pair once per GC instead.
        //
        // Taken through the shared accessor rather than `curr_s_cycle_candidates_mut`: pushing
        // only needs `&BufferPool`, so this avoids minting a `&mut` out of the `UnsafeCell` and
        // keeps the two call sites from ever holding overlapping `&mut`s to the same pool.
        let mut cand_buffer = unsafe { lxr.curr_s_cycle_candidates() }.local_buffer();

        // Mark phase: trial-delete candidates with strong_rc == 0
        self.mark_buffer(lxr, &mut candidates, vec_index, dfs_stack, &mut cand_buffer);

        #[cfg(feature = "s_rc_stats")]
        {
            self.stats.candidates_after_filter = candidates.len();
        }

        // Scan phase: classify GREY objects as WHITE (garbage) or restore to BLACK
        let mut it = candidates.iter_mut();
        while let Some(cand) = it.next() {
            self.scan(*cand, lxr, dfs_stack, nested_stack);
        }

        lxr.in_cycle_collection.store(false, Ordering::Relaxed);

        // Collect phase: free WHITE objects and handle remaining BLACK_IN_STACK
        let mut it = candidates.iter_mut();
        while let Some(cand) = it.next() {
            debug_assert!(OBJ_COLOR_TABLE.load_atomic::<u8>(cand.to_raw_address(), Ordering::SeqCst) != GREY);
            self.collect_whites(*cand, lxr, dfs_stack, nested_stack, &mut cand_buffer);
        }

        #[cfg(feature = "s_rc_stats")]
        {
            self.stats.satb_map_size = lxr.satb_map.len();
            self.stats.log_to_file();
        }
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
        println!("in full gc");

        // `curr_vec` on entry. Rotated below to walk the pools, then restored before the scan
        // phase so that a FullRC pause is invisible to the global rotation: the next GC's
        // `schedule_collection` advances from this same value, exactly as it would have.
        //
        // Mutating it here is sound for the same reason every `*_exclusive` call in this impl is
        // (see the header): this is a single packet on one worker, `ProcessDecs` has fully
        // drained, and no next GC can start while this packet occupies a worker. Nothing else
        // reads `curr_vec` in that window.
        let entry_vec = lxr.curr_vec.get();

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
            self.mark_buffer(lxr, candidates, *vec_index, dfs_stack, &mut cand_buffer);
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

        // Collect phase, over every buffer. `curr_vec` is back to its entry value, so the
        // candidates `collect_blacks` discovers are tagged and filed exactly as they would be by
        // an ordinary collection, and are drained two GCs from now.
        let mut cand_buffer = unsafe { lxr.curr_s_cycle_candidates() }.local_buffer();
        for (candidates, _) in buffers.iter_mut() {
            let mut it = candidates.iter_mut();
            while let Some(cand) = it.next() {
                debug_assert!(OBJ_COLOR_TABLE.load_atomic::<u8>(cand.to_raw_address(), Ordering::SeqCst) != GREY);
                self.collect_whites(*cand, lxr, dfs_stack, nested_stack, &mut cand_buffer);
            }
        }
        drop(cand_buffer);

        #[cfg(feature = "s_rc_stats")]
        {
            self.stats.satb_map_size = lxr.satb_map.len();
            self.stats.log_to_file();
        }
    }

    fn get_slot_logging_state(&self, slot: VM::VMSlot) -> u8 {
        Self::UNLOG_BITS.load_atomic(slot.to_address(), Ordering::SeqCst)
    }

    pub fn new(#[cfg(not(feature = "lxr_stw"))] c: LazySweepingJobsCounter) -> CycleCollector<VM> {
        CycleCollector::<VM> {
            rc: RefCountHelper::NEW,
            // Take ownership of the token handed over by `end_of_decs`.  Cloning it here and
            // dropping the original would be a redundant +1/-1 on the same cc counter.
            #[cfg(not(feature = "lxr_stw"))]
            _c: c,
            #[cfg(feature = "s_rc_stats")]
            stats: CycleCollectorStats::default(),
        }
    }

    fn get_child(&self, slot: <VM as vm::VMBinding>::VMSlot, lxr: &LXR<VM>) -> Option<ObjectReference>{
        let child = slot.load();
        if self.get_slot_logging_state(slot) == Self::UNLOGGED_VALUE{
            return child;
        }
        else{
            debug_assert!(self.get_slot_logging_state(slot) != Self::UNLOGGED_VALUE);
            let satb_child = loop {
                if let Some(m) = lxr.satb_map.get(&slot).map(|v| *v) {
                    return m;
                }
                std::hint::spin_loop();
            };
            #[cfg(feature = "s_rc_stats")]
            { self.stats.satb_reads.set(self.stats.satb_reads.get() + 1); }
            return satb_child;
        }
    }

    /// Trial deletion: decrements RC of children for each GREY candidate via DFS.
    /// If an SATB-logged slot is found, reverts all decrements.
    /// `dfs_stack` is scratch owned by `do_work` and reused across every candidate.  It is drained
    /// to empty before returning, so it needs no clearing on entry; the debug assertions hold that
    /// invariant in place.
    /// `cand_buffer` is the shared producer handle owned by `do_work`; see its comment there for
    /// why one handle covers every phase.
    fn mark(
        &self,
        o: ObjectReference,
        lxr: &LXR<VM>,
        dfs_stack: &mut Vec<ObjectReference>,
        cand_buffer: &mut LocalBuffer<'_, ObjectReference>,
    ) {
        debug_assert!(dfs_stack.is_empty());
        dfs_stack.push(o);
        debug_assert!(self.rc.count(o) > 0);
        while let Some(curr) = dfs_stack.pop() {
            debug_assert!(self.rc.count(curr) > 0);
            let mut is_logged = false;
            let mut num_of_childs = 0;

            if is_black(curr)
                && STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) == 0
                && CANDIDATES_STATUS.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) == 0
            {
                // SAFETY: single-threaded CycleCollector -- see the impl header.
                unsafe { OBJ_COLOR_TABLE.store_atomic_exclusive::<u8>(curr.to_raw_address(), GREY, Ordering::SeqCst) };
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
                    // `obj_array_data` describes.  The per-field work below is unchanged, so the
                    // children pushed, `num_of_childs`, and therefore the revert set the caller
                    // acts on are all identical to the generic path -- this changes only *when the
                    // loop stops*, never what it decides.
                    let data = VM::VMScanning::obj_array_data(curr);
                    for i in 0..data.len() {
                        let slot = data.get(i);
                        // Load the value *before* testing this slot's unlog bit, exactly as the
                        // generic visitor does.  A mutator that writes and logs between the two is
                        // caught by the test; one that did so between a test and a *later* load
                        // would not be, and the collector would then decrement the new referent
                        // while the restore path increments the old one -- leaving the new one
                        // permanently short.  These two lines must not be reordered.
                        let child = slot.load();
                        if self.get_slot_logging_state(slot) == Self::LOGGED_VALUE {
                            is_logged = true;
                            break;
                        }
                        if let Some(x) = child {
                            // SAFETY: single-threaded CycleCollector -- see the impl header.
                            let prev = unsafe { lxr.rc_with_overflow.dec_exclusive(x) };
                            debug_assert!(prev != 1);
                            debug_assert!(prev != 0);
                            // SAFETY: as above.
                            unsafe { self.rc.strong_rc_dec_exclusive(x) };
                            dfs_stack.push(x);
                            num_of_childs += 1;
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
                                // SAFETY: single-threaded CycleCollector -- see the impl header.
                                let prev = unsafe { lxr.rc_with_overflow.dec_exclusive(x) };
                                debug_assert!(prev != 1);
                                debug_assert!(prev != 0);
                                // SAFETY: as above.
                                unsafe { self.rc.strong_rc_dec_exclusive(x) };
                                dfs_stack.push(x);
                                num_of_childs += 1;
                            }

                        }
                    };
                    curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
                }
                #[cfg(feature = "s_rc_stats")]
                { self.stats.objects_in_mark.set(self.stats.objects_in_mark.get() + 1); }
                if is_logged{
                    // SAFETY: single-threaded CycleCollector -- see the impl header.
                    unsafe { OBJ_COLOR_TABLE.store_atomic_exclusive::<u8>(curr.to_raw_address(),BLACK_IN_STACK, Ordering::Relaxed) };
                    for _ in 0..num_of_childs{
                        if let Some(curr_child) = dfs_stack.pop(){
                            // SAFETY: single-threaded CycleCollector -- see the impl header.
                            let _ = unsafe { lxr.rc_with_overflow.inc_exclusive(curr_child) };
                            //let _ = self.rc.strong_rc_inc(curr_child); 
                           

                            cand_buffer.push(curr_child);
                        
                            // SAFETY: single-threaded CycleCollector -- see the impl header.
                            unsafe { CANDIDATES_STATUS.store_atomic_exclusive::<u8>(curr_child.to_raw_address(),(lxr.curr_vec.get() + 1) as u8, Ordering::Relaxed) };
                                if !is_black(curr_child){ // this condition is unnecessary. it is only to satisfy assertion (should be remove after assertion removal)
                                // SAFETY: as above.
                                let _ = unsafe { self.rc.strong_rc_dec_exclusive(curr_child) };
                                }
                            }
                            else{
                                panic!("num_of_childs is greater than the actual number of childs");
                            }

                    }

                    cand_buffer.push(curr);
                
                    // SAFETY: single-threaded CycleCollector -- see the impl header.
                    unsafe { CANDIDATES_STATUS.store_atomic_exclusive::<u8>(curr.to_raw_address(),(lxr.curr_vec.get() + 1) as u8, Ordering::Relaxed) };
                }
            } else if is_black(curr) {
                // SAFETY: single-threaded CycleCollector -- see the impl header.
                unsafe { OBJ_COLOR_TABLE.store_atomic_exclusive::<u8>(curr.to_raw_address(), BLACK_IN_STACK, Ordering::Relaxed) };
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

            if OBJ_COLOR_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) == GREY {
                debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) == 0);
                if RefCountHelper::<VM>::NEW.count(curr) > 1 {
                    self.scan_black(curr, lxr, black_stack);
                } else {
                    // SAFETY: single-threaded CycleCollector -- see the impl header.
                    unsafe { OBJ_COLOR_TABLE.store_atomic_exclusive::<u8>(curr.to_raw_address(), WHITE, Ordering::Relaxed) };
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
                // SAFETY: single-threaded CycleCollector -- see the impl header.
                unsafe { OBJ_COLOR_TABLE.store_atomic_exclusive::<u8>(curr.to_raw_address(), BLACK_IN_STACK, Ordering::SeqCst) };
                let s_rc = self.rc.count(curr) - 1;
                // SAFETY: single-threaded CycleCollector -- see the impl header.  Note the
                // obligation here is the *wider* one: `STRONG_RC_TABLE` is 4 bits per entry, so
                // two adjacent objects share a byte and the exclusive store is a read-merge-write
                // of that byte with no CAS.  No other thread may write *any* entry in the same
                // byte -- in practice, nothing else may write this table at all.  That holds only
                // because `ProcessDecs` has fully drained under the `decs -> sweep -> cc` order;
                // see `RefCountHelper::strong_rc_inc_exclusive` for the full argument.
                if s_rc > MAX_STRONG_REF_COUNT as RcBits{
                    unsafe { STRONG_RC_TABLE.store_atomic_exclusive::<u8>(curr.to_raw_address(),MAX_STRONG_REF_COUNT, Ordering::Relaxed) };
                }
                else{
                    unsafe { STRONG_RC_TABLE.store_atomic_exclusive::<u8>(curr.to_raw_address(),s_rc as u8, Ordering::Relaxed) };
                }
                curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, |slot: <VM as vm::VMBinding>::VMSlot, b| {
                    if let Some(x) = self.get_child(slot, lxr) {
                        // SAFETY: single-threaded CycleCollector -- see the impl header.
                        let _prev = unsafe { lxr.rc_with_overflow.inc_exclusive(x) };
                        if !in_stack(x) {
                            // SAFETY: as above.
                            unsafe { self.rc.strong_rc_inc_exclusive(x) };
                            dfs_stack.push(x);
                        }
                        debug_assert!(self.rc.count(x) > 1);
                    }
                });
                #[cfg(feature = "s_rc_stats")]
                { self.stats.objects_in_scan_black.set(self.stats.objects_in_scan_black.get() + 1); }
              
            } else {
                // SAFETY: single-threaded CycleCollector -- see the impl header.
                unsafe { OBJ_COLOR_TABLE.store_atomic_exclusive::<u8>(curr.to_raw_address(), BLACK_OUT_OF_STACK, Ordering::Relaxed) };
                debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) != 0);
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

            debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) as RcBits <= lxr.rc.count(curr));

            if OBJ_COLOR_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) == WHITE {
                #[cfg(feature = "graph_project")]
                {
                    let mut reporter = lxr.graph_reporter.lock().unwrap();
                    reporter.add_cycle_collector_freed(curr.to_raw_address().as_usize());
                }
                debug_assert!(self.rc.count(curr) == 1);
                debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) == 0);
                // SAFETY: single-threaded CycleCollector -- see the impl header.
                unsafe { CANDIDATES_STATUS.store_atomic_exclusive::<u8>(curr.to_raw_address(), 0, Ordering::Relaxed) };
                // SAFETY: single-threaded CycleCollector -- see the impl header.
                unsafe { OBJ_COLOR_TABLE.store_atomic_exclusive::<u8>(curr.to_raw_address(), BLACK_OUT_OF_STACK, Ordering::Relaxed) };
                debug_assert!(lxr.rc.count(curr) == 1);
                curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
                self.process_dead_object(curr, lxr);
                // SAFETY: single-threaded CycleCollector -- see the impl header.
                unsafe { self.rc.dec_exclusive(curr) };
            } else if OBJ_COLOR_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) == BLACK_IN_STACK
                && self.rc.count(curr) == 1
            {
                debug_assert!(CANDIDATES_STATUS.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) != 0);
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
                    // SAFETY: single-threaded CycleCollector -- see the impl header.
                    let prev_rc = unsafe { lxr.rc_with_overflow.dec_exclusive(x) };
                    // SAFETY: as above.  Returns the previous value directly rather than a
                    // `Result`, so the `Ok(1)` test below becomes a plain comparison against
                    // STRONG_RC_LAST_BEFORE_ZERO -- the same value, unwrapped.
                    let prev_s_rc = unsafe { self.rc.strong_rc_dec_exclusive(x) };
                    debug_assert!(prev_rc != 1);
                    if prev_rc == 2 {
                        dfs_stack.push(x);
                    } else if prev_s_rc == STRONG_RC_LAST_BEFORE_ZERO {
                        // SAFETY: single-threaded CycleCollector -- see the impl header.
                        unsafe { CANDIDATES_STATUS.store_atomic_exclusive::<u8>(x.to_raw_address(), (lxr.curr_vec.get() + 1) as u8, Ordering::Relaxed) };
                        cand_buffer.push(x);
                    }
                }
            };
            curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
            debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) as RcBits <= lxr.rc.count(curr));
            debug_assert!(is_black(curr));
            // SAFETY: single-threaded CycleCollector -- see the impl header.
            unsafe { CANDIDATES_STATUS.store_atomic_exclusive::<u8>(curr.to_raw_address(), 0, Ordering::Relaxed) };
            // SAFETY: single-threaded CycleCollector -- see the impl header.
            unsafe { OBJ_COLOR_TABLE.store_atomic_exclusive::<u8>(curr.to_raw_address(), BLACK_OUT_OF_STACK, Ordering::Relaxed) };
            #[cfg(feature = "graph_project")]
            {
                let mut reporter = lxr.graph_reporter.lock().unwrap();
                reporter.add_cycle_collector_freed(curr.to_raw_address().as_usize());
            }
            self.process_dead_object(curr, lxr);
            debug_assert!(lxr.rc.count(curr) == 1);
            // SAFETY: single-threaded CycleCollector -- see the impl header.
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
        CANDIDATES_STATUS.load_atomic::<u8>(o.to_raw_address(), Ordering::Relaxed) == vec_index
            && STRONG_RC_TABLE.load_atomic::<u8>(o.to_raw_address(), Ordering::Relaxed) == 0
        // STRONG_RC_TABLE.load_atomic::<u8>(o.to_raw_address(), Ordering::Relaxed) == 0
        //     && CANDIDATES_STATUS.load_atomic::<u8>(o.to_raw_address(), Ordering::Relaxed) != 0
    }

}



#[cfg(feature = "graph_project")]
impl<VM: VMBinding> CycleCollector<VM> {

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
