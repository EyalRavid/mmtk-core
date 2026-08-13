use super::cm::LXRWeakRefProcessEdges;
use super::{barrier, LXR};
use crate::scheduler::{gc_work::*, GCWork, GCWorker};
use crate::util::ObjectReference;
use crate::{vm::*, Plan, MMTK};
use crate::util::rc::{CANDIDATES_STATUS, MAX_STRONG_REF_COUNT, OBJ_COLOR_TABLE, STRONG_RC_TABLE, BLACK_OUT_OF_STACK, BLACK_IN_STACK, GREY, WHITE};
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
use crate::plan::lxr::stack::ChunkedStack;
use crate::util::rc::RcBits;
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::LazySweepingJobsCounter;
#[cfg(feature = "graph_project")]
use crate::plan::lxr:: graphs_project::{*};
#[cfg(feature = "graph_project")]
use crate::plan::lxr::buffer::FinalIterMut;

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

        let vec_index = ((lxr.curr_vec.get() + 1) % NUM_OF_CANDIDATES_VECTORS + 1) as u8;
        let mut candidates = unsafe {lxr.s_cycle_candidates_mut()}.into_final_buffers();


        #[cfg(feature = "graph_project")]
        {
            let mut it = candidates.iter_mut();
            let mut reporter = lxr.graph_reporter.lock().unwrap();
            self.report_candidates_sub_graph(it, &mut reporter, vec_index);

        }


        #[cfg(feature = "s_rc_stats")]
        {
            self.stats.raw_cycle_candidates = candidates.len();
        }

        // Mark phase: trial-delete candidates with strong_rc == 0
        {
            let mut it = candidates.iter_mut();
            while let Some(cand) = it.next(){
                if self.should_mark(*cand, vec_index) {
                    CANDIDATES_STATUS.store_atomic::<u8>(cand.to_raw_address(), 0, Ordering::Relaxed);
                    debug_assert!(self.rc.count(*cand) > 0);
                    self.mark(*cand, lxr);
                } else {
                    if STRONG_RC_TABLE.load_atomic::<u8>(cand.to_raw_address(), Ordering::Relaxed) > 0 {
                        CANDIDATES_STATUS.store_atomic::<u8>(cand.to_raw_address(), 0, Ordering::Relaxed);
                    }
                    it.swap_remove_current();
                }
            }
        }

        #[cfg(feature = "s_rc_stats")]
        {
            self.stats.candidates_after_filter = candidates.len();
        }

        // Scan phase: classify GREY objects as WHITE (garbage) or restore to BLACK
        let mut it = candidates.iter_mut();
        while let Some(cand) = it.next() {
            self.scan(*cand, lxr);
        }



        lxr.in_cycle_collection.store(false, Ordering::Relaxed);

        // Collect phase: free WHITE objects and handle remaining BLACK_IN_STACK
        let mut it = candidates.iter_mut();
        while let Some(cand) = it.next() {
            debug_assert!(OBJ_COLOR_TABLE.load_atomic::<u8>(cand.to_raw_address(), Ordering::SeqCst) != GREY);
            self.collect_whites(*cand, lxr);
        }
        #[cfg(feature = "s_rc_stats")]
        {
            self.stats.satb_map_size = lxr.satb_map.len();
            self.stats.log_to_file();
        }
    }
}

impl<VM: VMBinding> CycleCollector<VM>{
    
    pub const UNLOGGED_VALUE: u8 = 0b1;
    pub const LOGGED_VALUE: u8 = 0b0;

    const UNLOG_BITS: SideMetadataSpec = *VM::VMObjectModel::GLOBAL_FIELD_UNLOG_BIT_SPEC
        .as_spec()
        .extract_side_spec();

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
    fn mark(&self, o: ObjectReference, lxr: &LXR<VM>) {
        let mut local_buffer = unsafe {lxr.curr_s_cycle_candidates_mut()}.local_buffer();
        let mut dfs_stack = ChunkedStack::<ObjectReference>::new();
        dfs_stack.push(o);
        debug_assert!(self.rc.count(o) > 0);
        while let Some(curr) = dfs_stack.pop() {
            debug_assert!(self.rc.count(curr) > 0);
            let mut is_logged = false;
            let mut num_of_childs = 0;
            let visitor = |slot: <VM as vm::VMBinding>::VMSlot, b| {
                let child = slot.load();
                if self.get_slot_logging_state(slot) == Self::LOGGED_VALUE{
                        is_logged = true;      
                }
                else if let Some(x) = child{
                    if !is_logged{
                        let prev = lxr.rc_with_overflow.dec(x);
                        debug_assert!(prev != 1);
                        debug_assert!(prev != 0);
                        self.rc.strong_rc_dec(x);
                        dfs_stack.push(x);
                        num_of_childs += 1;
                    }

                }
            };

            if is_black(curr)
                && STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) == 0
                && CANDIDATES_STATUS.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) == 0
            {
                OBJ_COLOR_TABLE.store_atomic::<u8>(curr.to_raw_address(), GREY, Ordering::SeqCst);
                curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
                #[cfg(feature = "s_rc_stats")]
                { self.stats.objects_in_mark.set(self.stats.objects_in_mark.get() + 1); }
                if is_logged{
                    OBJ_COLOR_TABLE.store_atomic::<u8>(curr.to_raw_address(),BLACK_IN_STACK, Ordering::Relaxed);
                    for _ in 0..num_of_childs{
                        if let Some(curr_child) = dfs_stack.pop(){
                            let _ = lxr.rc_with_overflow.inc(curr_child);
                            //let _ = self.rc.strong_rc_inc(curr_child); 
                           

                            local_buffer.push(curr_child);
                        
                            CANDIDATES_STATUS.store_atomic::<u8>(curr_child.to_raw_address(),(lxr.curr_vec.get() + 1) as u8, Ordering::Relaxed);
                                if !is_black(curr_child){ // this condition is unnecessary. it is only to satisfy assertion (should be remove after assertion removal)
                                let _ = self.rc.strong_rc_dec(curr_child);
                                }
                            }
                            else{
                                panic!("num_of_childs is greater than the actual number of childs");
                            }

                    }

                    local_buffer.push(curr);
                
                    CANDIDATES_STATUS.store_atomic::<u8>(curr.to_raw_address(),(lxr.curr_vec.get() + 1) as u8, Ordering::Relaxed);
                }
            } else if is_black(curr) {
                OBJ_COLOR_TABLE.store_atomic::<u8>(curr.to_raw_address(), BLACK_IN_STACK, Ordering::Relaxed);
            }
        }
    }

    /// Scan phase: GREY objects with RC > 1 are externally referenced — restore them (scan_black).
    /// GREY objects with RC == 1 are garbage — mark WHITE.
    fn scan(&self, o: ObjectReference, lxr: &LXR<VM>) {
        let mut dfs_stack = ChunkedStack::<ObjectReference>::new();
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
                    self.scan_black(curr, lxr);
                } else {
                    OBJ_COLOR_TABLE.store_atomic::<u8>(curr.to_raw_address(), WHITE, Ordering::Relaxed);
                    curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
                }
            }
        }
    }


    /// Restores RCs for objects found to be externally reachable.
    /// Reconstructs strong_rc and marks objects BLACK.
    fn scan_black(&self, o: ObjectReference, lxr: &LXR<VM>) {
        let mut dfs_stack = ChunkedStack::<ObjectReference>::new();
        dfs_stack.push(o);
        while let Some(curr) = dfs_stack.last() {
            assert!(self.rc.count(*curr) > 1);
            let curr_copy = *curr; // needed for the borrow checker
            if !is_black(*curr) {
                OBJ_COLOR_TABLE.store_atomic::<u8>(curr_copy.to_raw_address(), BLACK_IN_STACK, Ordering::SeqCst);
                let s_rc = self.rc.count(*curr) - 1;
                if s_rc > MAX_STRONG_REF_COUNT as RcBits{
                    STRONG_RC_TABLE.store_atomic::<u8>(curr.to_raw_address(),MAX_STRONG_REF_COUNT, Ordering::Relaxed);
                }
                else{
                    STRONG_RC_TABLE.store_atomic::<u8>(curr.to_raw_address(),s_rc as u8, Ordering::Relaxed);
                }
                curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, |slot: <VM as vm::VMBinding>::VMSlot, b| {
                    if let Some(x) = self.get_child(slot, lxr) {
                        let _prev = lxr.rc_with_overflow.inc(x);
                        if !in_stack(x) {
                            self.rc.strong_rc_inc(x);
                            dfs_stack.push(x);
                        }
                        debug_assert!(self.rc.count(x) > 1);
                    }
                });
                #[cfg(feature = "s_rc_stats")]
                { self.stats.objects_in_scan_black.set(self.stats.objects_in_scan_black.get() + 1); }
              
            } else {
                OBJ_COLOR_TABLE.store_atomic::<u8>(curr.to_raw_address(), BLACK_OUT_OF_STACK, Ordering::Relaxed);
                debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) != 0);
                dfs_stack.pop();
            }
        }
    }
    

    /// Frees WHITE (garbage) objects. Also handles BLACK_IN_STACK objects with RC == 1
    /// by delegating to collect_blacks.
    fn collect_whites(&self, o: ObjectReference, lxr: &LXR<VM>) {
        let mut dfs_stack = ChunkedStack::<ObjectReference>::new();
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
                CANDIDATES_STATUS.store_atomic::<u8>(curr.to_raw_address(), 0, Ordering::Relaxed);
                OBJ_COLOR_TABLE.store_atomic::<u8>(curr.to_raw_address(), BLACK_OUT_OF_STACK, Ordering::Relaxed);
                debug_assert!(lxr.rc.count(curr) == 1);
                curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
                self.process_dead_object(curr, lxr);
                self.rc.dec(curr);
            } else if OBJ_COLOR_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) == BLACK_IN_STACK
                && self.rc.count(curr) == 1
            {
                debug_assert!(CANDIDATES_STATUS.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) != 0);
                self.collect_blacks(curr, lxr);
            }
        }
    }

    
    /// Frees BLACK objects with RC == 1 by decrementing children and reclaiming.
    /// Children that reach RC == 2 (death threshold) are also freed recursively.
    fn collect_blacks(&self, o: ObjectReference, lxr: &LXR<VM>) {
        let mut dfs_stack = ChunkedStack::<ObjectReference>::new();
        dfs_stack.push(o);
        while let Some(curr) = dfs_stack.pop() {
            #[cfg(feature = "s_rc_stats")]
            { self.stats.objects_in_collect.set(self.stats.objects_in_collect.get() + 1); }
            debug_assert!(self.rc.count(curr) == 1);
            let visitor = |slot: <VM as vm::VMBinding>::VMSlot, _| {
                debug_assert!(self.get_slot_logging_state(slot) == Self::UNLOGGED_VALUE);
                if let Some(x) = slot.load() {
                    assert!(self.rc.count(x) > 1);
                    let prev_rc = lxr.rc_with_overflow.dec(x);;
                    let prev_s_rc = self.rc.strong_rc_dec(x);
                    debug_assert!(prev_rc != 1);
                    if prev_rc == 2 {
                        dfs_stack.push(x);
                    } else if prev_s_rc == Ok(1) {
                        let s_candidates = unsafe { lxr.curr_s_cycle_candidates_mut() };
                        CANDIDATES_STATUS.store_atomic::<u8>(x.to_raw_address(), (lxr.curr_vec.get() + 1) as u8, Ordering::Relaxed);
                        s_candidates.local_buffer().push(x);
                    }
                }
            };
            curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
            debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) as RcBits <= lxr.rc.count(curr));
            debug_assert!(is_black(curr));
            CANDIDATES_STATUS.store_atomic::<u8>(curr.to_raw_address(), 0, Ordering::Relaxed);
            OBJ_COLOR_TABLE.store_atomic::<u8>(curr.to_raw_address(), BLACK_OUT_OF_STACK, Ordering::Relaxed);
            #[cfg(feature = "graph_project")]
            {
                let mut reporter = lxr.graph_reporter.lock().unwrap();
                reporter.add_cycle_collector_freed(curr.to_raw_address().as_usize());
            }
            self.process_dead_object(curr, lxr);
            debug_assert!(lxr.rc.count(curr) == 1);
            self.rc.dec(curr);
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
