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

pub struct CycleCollector<VM: VMBinding> {
    rc: RefCountHelper<VM>,
    _c: LazySweepingJobsCounter,
}

impl<VM: VMBinding> GCWork<VM> for CycleCollector<VM> {
    
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        println!("hash_map size before cycle collection = {}", lxr.satb_map.len());
        lxr.satb_map.clear();

        let vec_index = ((lxr.curr_vec.get() + 1) % NUM_OF_CANDIDATES_VECTORS + 1) as u8;

        #[cfg(feature = "s_rc_stats")]
        self.print_stats(lxr, vec_index);

        // Mark phase: trial-delete candidates with strong_rc == 0
        let mut i = 0;
        loop {
            let cand = unsafe {
                let v = lxr.s_cycle_candidates_mut();
                if i >= v.len() { break }
                v[i] // ObjectReference is Copy
            };

            if self.should_mark(cand, vec_index) {
                CANDIDATES_STATUS.store_atomic::<u8>(cand.to_raw_address(), 0, Ordering::Relaxed);
                debug_assert!(self.rc.count(cand) > 0);
                self.mark(cand, lxr);
                i += 1;
            } else {
                if STRONG_RC_TABLE.load_atomic::<u8>(cand.to_raw_address(), Ordering::Relaxed) > 0 {
                    CANDIDATES_STATUS.store_atomic::<u8>(cand.to_raw_address(), 0, Ordering::Relaxed);
                }
                unsafe { lxr.s_cycle_candidates_mut().swap_remove(i); }
            }
        }

        #[cfg(feature = "s_rc_stats")]
        println!("num of real candidates = {}", s_candidates.len());

        // Scan phase: classify GREY objects as WHITE (garbage) or restore to BLACK
        let len = unsafe { lxr.s_cycle_candidates_mut().len() };
        for i in 0..len {
            let cand = unsafe { lxr.s_cycle_candidates_mut()[i] };
            self.scan(cand, lxr);
        }

        #[cfg(feature = "s_rc_stats")]
        {
            let mut num_of_garbage_candidates = 0;
            for obj in s_candidates.iter() {
                if OBJ_COLOR_TABLE.load_atomic::<u8>((*obj).to_raw_address(), Ordering::SeqCst) == WHITE {
                    num_of_garbage_candidates += 1;
                }
            }
            println!("num of s_rc dead scanned candidates = {}", num_of_garbage_candidates);

            let mut num_of_scanned = lxr.num_of_scanned_s_rc_candidates.lock().unwrap();
            println!("num of scanned objects in s_rc scan = {}", num_of_scanned);
            *num_of_scanned = 0;

            println!("===ENDED CYCLE COLLECTION PHASE===");
        }

        // Collect phase: free WHITE objects and handle remaining BLACK_IN_STACK
        let len = unsafe { lxr.s_cycle_candidates_mut().len() };
        for i in 0..len {
            let cand = unsafe { lxr.s_cycle_candidates_mut()[i] };
            debug_assert!(OBJ_COLOR_TABLE.load_atomic::<u8>(cand.to_raw_address(), Ordering::SeqCst) != GREY);
            self.collect_whites(cand, lxr);
        }
        println!("hash_map size after cycle collection = {}", lxr.satb_map.len());
        unsafe { lxr.s_cycle_candidates_mut() }.clear();
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

    pub fn new(c: LazySweepingJobsCounter) -> CycleCollector<VM> {
        CycleCollector::<VM> {
            rc: RefCountHelper::NEW,
            _c: c.clone_with_cc(),
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
                    println!("got child from satb map");
                    break m;
                }
                println!("loop in get child");
                std::hint::spin_loop();
            };
            return satb_child;
        }
    }

    /// Trial deletion: decrements RC of children for each GREY candidate via DFS.
    /// If an SATB-logged slot is found, reverts all decrements.
    fn mark(&self, o: ObjectReference, lxr: &LXR<VM>) {
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
                    let prev = self.rc.dec(x);
                    debug_assert!(prev != Ok(1));
                    debug_assert!(prev != Err(0));
                    self.rc.strong_rc_dec(x);
                    dfs_stack.push(x);
                    num_of_childs += 1;
                }
            };

            if is_black(curr)
                && STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) == 0
                && CANDIDATES_STATUS.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) == 0
            {
                OBJ_COLOR_TABLE.store_atomic::<u8>(curr.to_raw_address(), GREY, Ordering::SeqCst);
                curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
                #[cfg(feature = "s_rc_stats")]
                {
                    let mut num_of_scanned = lxr.num_of_scanned_s_rc_candidates.lock().unwrap();
                    *num_of_scanned += 1;
                }
                if is_logged{
                    OBJ_COLOR_TABLE.store_atomic::<u8>(curr.to_raw_address(),BLACK_IN_STACK, Ordering::Relaxed);
                    for _ in 0..num_of_childs{
                        if let Some(curr_child) = dfs_stack.pop(){
                            let _ = self.rc.inc(curr_child);
                            let _ = self.rc.strong_rc_inc(curr_child); 
                            if !is_black(curr_child){ // this condition is unnecessary. it is only to satisfy assertion (should be remove after assertion removal)
                                let _ = self.rc.strong_rc_dec(curr_child);
                            }
                        }
                        else{
                            panic!("num_of_childs is greater than the actual number of childs");
                        }

                    }
                    unsafe {
                        lxr.curr_s_cycle_candidates_mut().push(curr);
                    }
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
            debug_assert!(self.rc.count(curr) > 0);
            let visitor = |slot: <VM as vm::VMBinding>::VMSlot, _| {
                if let Some(x) = self.get_child(slot, lxr) {
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
                let s_rc = self.rc.count(*curr) - 1;
                if s_rc > MAX_STRONG_REF_COUNT as RcBits{
                    STRONG_RC_TABLE.store_atomic::<u8>(curr.to_raw_address(),MAX_STRONG_REF_COUNT, Ordering::Relaxed);
                }
                else{
                    STRONG_RC_TABLE.store_atomic::<u8>(curr.to_raw_address(),s_rc as u8, Ordering::Relaxed);
                }
                curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, |slot: <VM as vm::VMBinding>::VMSlot, b| {
                    if let Some(x) = self.get_child(slot, lxr) {
                        let _prev = self.rc.inc(x);
                        if !in_stack(x) {
                            self.rc.strong_rc_inc(x);
                            dfs_stack.push(x);
                        }
                        debug_assert!(self.rc.count(x) > 1);
                    }
                });
                OBJ_COLOR_TABLE.store_atomic::<u8>(curr_copy.to_raw_address(), BLACK_IN_STACK, Ordering::SeqCst);
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
            let visitor = |slot: <VM as vm::VMBinding>::VMSlot, _| {
                if let Some(x) = self.get_child(slot, lxr) {
                    dfs_stack.push(x);
                }
            };

            debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) as RcBits <= lxr.rc.count(curr));

            if OBJ_COLOR_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) == WHITE {
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
                && CANDIDATES_STATUS.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) != 0
            {
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
            debug_assert!(self.rc.count(curr) == 1);
            let visitor = |slot: <VM as vm::VMBinding>::VMSlot, _| {
                if let Some(x) = self.get_child(slot, lxr) {
                    assert!(self.rc.count(x) > 1);
                    let prev_rc = self.rc.dec(x);
                    let prev_s_rc = self.rc.strong_rc_dec(x);
                    debug_assert!(prev_rc != Ok(1));
                    if prev_rc == Ok(2) {
                        dfs_stack.push(x);
                    } else if prev_s_rc == Ok(1) {
                        let s_candidates = unsafe { lxr.curr_s_cycle_candidates_mut() };
                        CANDIDATES_STATUS.store_atomic::<u8>(x.to_raw_address(), (lxr.curr_vec.get() + 1) as u8, Ordering::Relaxed);
                        s_candidates.push(x);
                    }
                }
            };
            curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
            debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) as RcBits <= lxr.rc.count(curr));
            debug_assert!(is_black(curr));
            CANDIDATES_STATUS.store_atomic::<u8>(curr.to_raw_address(), 0, Ordering::Relaxed);
            OBJ_COLOR_TABLE.store_atomic::<u8>(curr.to_raw_address(), BLACK_OUT_OF_STACK, Ordering::Relaxed);
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
    }

    
    #[cfg(feature = "s_rc_stats")]
    fn print_stats(&self, lxr: &LXR<VM>, vec_index: u8) {
        println!("===GOT TO CYCLE COLLECTION PHASE===");
        let s_candidates = unsafe { lxr.s_cycle_candidates_mut() };
        let mut candidates = lxr.cycle_candidates.lock().unwrap();
        let mut real_candidates = Vec::<ObjectReference>::new();
        println!("num of trial deletion candidates with dead objects and duplicates = {}", candidates.len());
        let mut num_of_dupcs = 0;
        let mut num_of_dead_candidates = 0;
        // Remove candidates with RC == 0 (already freed)
        for obj in candidates.iter() {
            if lxr.rc.count(*obj) > 0 {
                if !real_candidates.contains(obj) {
                    real_candidates.push(*obj);
                } else {
                    num_of_dupcs += 1;
                }
            } else {
                num_of_dead_candidates += 1;
            }
        }
        println!("num of duplicates candidates in trial deletion not including dead duplicates = {}", num_of_dupcs);
        println!("num of dead candidates in trial deletion = {}", num_of_dead_candidates);
        println!("num of trial deletion candidates after duplicates and dead objects removal = {}", real_candidates.len());
        candidates.clear();
        let mut real_strong_candidates = Vec::<ObjectReference>::new();
        num_of_dupcs = 0;
        num_of_dead_candidates = 0;
        for obj in s_candidates.iter() {
            if self.should_mark(*obj, vec_index) {
                if !real_strong_candidates.contains(obj) {
                    real_strong_candidates.push(*obj);
                } else {
                    num_of_dupcs += 1;
                }
            } else {
                num_of_dead_candidates += 1;
            }
        }
        println!("num of strong candidates before dead object removal = {}", s_candidates.len());
        println!("num of duplicates candidates in s_rc scan not including dead duplicates = {}", num_of_dupcs);
        println!("num of dead candidates in s_rc scan = {}", num_of_dead_candidates);
        println!("num of s_rc candidates = {}", real_strong_candidates.len());
    }

}
