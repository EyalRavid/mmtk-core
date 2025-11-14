use super::cm::LXRWeakRefProcessEdges;
use super::{barrier, LXR};
use crate::scheduler::{gc_work::*, GCWork, GCWorker};
use crate::util::ObjectReference;
use crate::{vm::*, Plan, MMTK};
use crate::util::rc::{IN_STACK_TABLE, MAX_REF_COUNT, MAX_STRONG_REF_COUNT, OBJ_COLOR_TABLE, RC_TABLE, STRONG_RC_TABLE};
use atomic::Ordering;
use chunked_vec::ChunkedVec;
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
//Eyal added this:
use crate::plan::lxr::stack::ChunkedStack;
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



pub const BLACK_OUT_OF_STACK: u8 = 0;
pub const BLACK_IN_STACK: u8 = 1;
pub const GREY: u8 = 2;
pub const WHITE: u8 = 3;

unsafe fn in_stack(o: ObjectReference) -> bool{
    OBJ_COLOR_TABLE.load::<u8>(o.to_raw_address()) == BLACK_IN_STACK  
}

unsafe fn is_black(o: ObjectReference) -> bool{
    OBJ_COLOR_TABLE.load::<u8>(o.to_raw_address()) <= BLACK_IN_STACK  
}

pub struct CycleCollector<VM: VMBinding>{
    rc : RefCountHelper<VM>,
}

impl<VM: VMBinding> GCWork<VM> for CycleCollector<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        
        #[cfg(feature = "s_rc_stats")]
        self.print_stats(lxr);
        
        let mut s_candidates = unsafe {
            lxr.s_cycle_candidates_mut()
        };
        // for obj in s_candidates.iter(){
        //     debug_assert!(OBJ_COLOR_TABLE.load_atomic::<u8>((*obj).to_raw_address(), Ordering::SeqCst) != WHITE);
        //     if lxr.rc.count(*obj) > 0{
        //         self.mark(*obj, #[cfg(feature = "s_rc_stats")] lxr);
        //     }    
        // }
        let mut i = 0;
        while i < s_candidates.len() {
            if self.should_mark(s_candidates[i]) {
                self.mark(s_candidates[i], #[cfg(feature = "s_rc_stats")] lxr);
                IN_STACK_TABLE.store_atomic::<u8>(s_candidates[i].to_raw_address(),0 , Ordering::Relaxed);
                i+=1;
            }
            else {
                s_candidates.swap_remove(i);
            }
        }
        for obj in s_candidates.iter(){
            self.scan(*obj);
        }
        #[cfg(feature = "s_rc_stats")]
        {
            let mut num_of_garbage_candidates = 0;
            for obj in s_candidates.iter(){
                if OBJ_COLOR_TABLE.load_atomic::<u8>((*obj).to_raw_address(), Ordering::SeqCst) == WHITE{
                    num_of_garbage_candidates+=1;     
                }
            }
            
            println!("num of s_rc dead scanned candidates = {}", num_of_garbage_candidates);

            let mut  num_of_scanned = lxr.num_of_scanned_s_rc_candidates.lock().unwrap();
            println!("num of scanned objects in s_rc scan = {}", num_of_scanned);
            *num_of_scanned = 0;

            let num_of_root_candidates: u64 = 0;
            
            println!("===ENDED CYCLE COLLECTION PHAZE===");
            println!("\n\n");

        }

        
        for obj in s_candidates.iter(){
            debug_assert!(OBJ_COLOR_TABLE.load_atomic::<u8>((*obj).to_raw_address(), Ordering::SeqCst) != GREY);
            self.collect_whites(*obj, lxr);    

        }
        

        *s_candidates = ChunkedVec::with_capacity(1024);
    }
}

impl<VM: VMBinding> CycleCollector<VM>{

    pub fn new()->CycleCollector<VM>{
        CycleCollector::<VM>{
            rc: RefCountHelper::NEW
        }
    }



    fn mark(&self, o: ObjectReference, #[cfg(feature = "s_rc_stats")] lxr: &LXR<VM>){
        debug_assert!(RefCountHelper::<VM>::NEW.count(o) > 0 
        || OBJ_COLOR_TABLE.load_atomic::<u8>(o.to_raw_address(), Ordering::SeqCst) == GREY);
        let mut dfs_stack = ChunkedStack::<ObjectReference>::new();
        dfs_stack.push(o);

        while let Some(curr) = dfs_stack.pop() {
            let visitor = |slot: <VM as vm::VMBinding>::VMSlot, b| {
                if let Some(x) = slot.load(){
                    let prev = self.rc.dec(x);
                    debug_assert!(prev != Err(0));
                    debug_assert!(self.rc.count(x) < MAX_REF_COUNT);
                    dfs_stack.push(x);
                }
            };
            unsafe{
                if OBJ_COLOR_TABLE.load::<u8>(curr.to_raw_address()) == BLACK_OUT_OF_STACK{
                    //STRONG_RC_TABLE.store_atomic::<u8>(curr.to_raw_address(),0, Ordering::Relaxed);
                    OBJ_COLOR_TABLE.store::<u8>(curr.to_raw_address(),GREY);
                    curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);

                    #[cfg(feature = "s_rc_stats")]
                    {
                        let mut  num_of_scanned = lxr.num_of_scanned_s_rc_candidates.lock().unwrap();
                        *num_of_scanned+=1
                    }
;
                }
            }  
        }
    }

    fn scan(&self, o: ObjectReference){
        let mut dfs_stack = ChunkedStack::<ObjectReference>::new();
        dfs_stack.push(o);

        while let Some(curr) = dfs_stack.pop(){
            let visitor = |slot: <VM as vm::VMBinding>::VMSlot, b| {
                if let Some(x) = slot.load(){
                    dfs_stack.push(x);
                }
            };
            unsafe {
                if OBJ_COLOR_TABLE.load::<u8>(curr.to_raw_address()) == GREY{
                    debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) == 0);
                    if RefCountHelper::<VM>::NEW.count(curr) > 0{
                        self.scan_black(curr);
                    }
                    else{
                        OBJ_COLOR_TABLE.store::<u8>(curr.to_raw_address(),WHITE);
                        curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
                    }
               }    
            }

        }
    }


    fn scan_black(&self, o: ObjectReference){
        let mut dfs_stack = ChunkedStack::<ObjectReference>::new();
        dfs_stack.push(o);
        while let Some(curr) = dfs_stack.last(){
            debug_assert!(self.rc.count(*curr) > 0);
            unsafe {
                if !is_black(*curr){
                    debug_assert!(IN_STACK_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) == 0);
                    OBJ_COLOR_TABLE.store::<u8>(curr.to_raw_address(),BLACK_IN_STACK);
                    //IN_STACK_TABLE.store::<u8>(curr.to_raw_address(), 1 as u8);
                    let s_rc = self.rc.count(*curr);
                    if s_rc > MAX_STRONG_REF_COUNT{
                        STRONG_RC_TABLE.store::<u8>(curr.to_raw_address(),MAX_STRONG_REF_COUNT);
                    }
                    else{
                        STRONG_RC_TABLE.store::<u8>(curr.to_raw_address(),s_rc as u8);
                    }
                    curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, |slot: <VM as vm::VMBinding>::VMSlot, b| {
                        if let Some(x) = slot.load() {
                            debug_assert!(self.rc.count(x) < MAX_REF_COUNT);
                            let _prev = self.rc.inc(x);
                            if !in_stack(x){
                                self.rc.strong_rc_inc(x);
                                dfs_stack.push(x);
                            }
                        }
                    });
                    
                }
                else{
                    OBJ_COLOR_TABLE.store::<u8>(curr.to_raw_address(),BLACK_OUT_OF_STACK);
                    debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) != 0);
                    dfs_stack.pop();
                }
            }

        }
  
    }
    

    fn collect_whites(&self, o: ObjectReference, lxr: &LXR<VM>){

        let mut dfs_stack = ChunkedStack::<ObjectReference>::new();
        dfs_stack.push(o);
        while let Some(curr) = dfs_stack.pop(){
            let mut visitor = |slot: <VM as vm::VMBinding>::VMSlot, b| {
                if let Some(x) = slot.load() {
                    dfs_stack.push(x);
                }
            };
            unsafe {
                if OBJ_COLOR_TABLE.load::<u8>(curr.to_raw_address()) == WHITE{
                    STRONG_RC_TABLE.store::<u8>(curr.to_raw_address(),0);
                    OBJ_COLOR_TABLE.store::<u8>(curr.to_raw_address(),BLACK_OUT_OF_STACK);
                    debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) == 0);
                    debug_assert!(lxr.rc.count(curr) == 0);
                    curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
                    self.process_dead_object(curr, lxr);
                }
            }

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
        // if self.cm_in_progress {
        //     let marked = lxr.mark(o);
        //     if cfg!(feature = "lxr_satb_live_bytes_counter") && marked {
        //         crate::record_live_bytes(o.get_size::<VM>());
        //     }
        // }
        // println!(" - dead {:?}", o);
        // debug_assert_eq!(self::count(o), 0);
        // Recursively decrease field ref counts
        if false
            && VM::VMScanning::is_obj_array(o)
            && VM::VMScanning::obj_array_data(o).bytes() > 1024
        {
            // Buggy. Dead array can be recycled at any time.
            unimplemented!()
        }
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

    fn should_mark(&self, o: ObjectReference) -> bool{
        IN_STACK_TABLE.load_atomic::<u8>(o.to_raw_address(), Ordering::Relaxed) == 1 as u8 && unsafe {
            OBJ_COLOR_TABLE.load::<u8>(o.to_raw_address()) != GREY
        } 
    }
    #[cfg(feature = "s_rc_stats")]
    fn print_stats(&self, lxr: &LXR<VM>) {
        println!("===GOT TO CYCLE COLLECTION PHAZE===");
        let mut  s_candidates = lxr.s_cycle_candidates.lock().unwrap();
        let mut  candidates = lxr.cycle_candidates.lock().unwrap();
        let mut real_candidates = Vec::<ObjectReference>::new();
        println!("num of trial deletion candidates with dead objects and duplicates = {}", candidates.len()); 
        let mut num_of_dupcs = 0;
        let mut num_of_dead_candidates = 0;
        // removing from candidates objects with 0 rc (because this objects allready freed)
        for obj in candidates.iter(){
            if lxr.rc.count(*obj) > 0{
                if !real_candidates.contains(obj){
                    real_candidates.push(*obj);
                }
                else{
                    num_of_dupcs +=1 ;
                }
            }
            else{
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
        for obj in s_candidates.iter(){
            if lxr.rc.count(*obj) > 0{
                if !real_strong_candidates.contains(obj){
                    real_strong_candidates.push(*obj);
                }
                else{
                    num_of_dupcs +=1 ;
                }
            }
            else{
                num_of_dead_candidates += 1;
            }
        }  
        println!("num of strong candidates before dead object removal = {}", s_candidates.len());
        println!("num of duplicates candidates in s_rc scan not including dead duplicates = {}", num_of_dupcs);
        println!("num of dead candidates in s_rc scan = {}", num_of_dead_candidates);
        println!("num of s_rc candidates = {}", real_strong_candidates.len());
        
    
    }

}
