use super::cm::LXRWeakRefProcessEdges;
use super::{barrier, LXR};
use crate::scheduler::{gc_work::*, GCWork, GCWorker};
use crate::util::ObjectReference;
use crate::{vm::*, Plan, MMTK};
use crate::util::rc::{IN_STACK_TABLE, MAX_REF_COUNT, MAX_STRONG_REF_COUNT, OBJ_COLOR_TABLE, RC_TABLE, STRONG_RC_TABLE};
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




pub const BLACK: u8 = 0;
pub const GREY: u8 = 1;
pub const WHITE: u8 = 2;
pub struct CycleCollector<VM: VMBinding>{
    rc : RefCountHelper<VM>,
}

impl<VM: VMBinding> GCWork<VM> for CycleCollector<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        println!("GOT TO CYCLE COLLECTION PHAZE");
        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        let mut  candidates = lxr.cycle_candidates.lock().unwrap();
        let mut real_candidate = Vec::<ObjectReference>::new();
       
        println!("##################################");

        println!("num of regular candidates before dead object removal = {}", candidates.len()); 
        let mut num_of_dupcs = 0;  
        // removing from candidates objects with 0 rc (because this objects allready freed)
        for obj in candidates.iter(){
            if RC_TABLE.load_atomic::<u16>(obj.to_raw_address(), Ordering::SeqCst) > 0{
                if !real_candidate.contains(obj){
                    real_candidate.push(*obj);
                }
                else{
                    num_of_dupcs +=1 ;
                }
                
            }
        }
        println!("num_of_dupcs in regular candidates = {}", num_of_dupcs); 
        
        candidates.clear();
        let mut  s_candidates = lxr.s_cycle_candidates.lock().unwrap();
        let mut real_s_candidate = Vec::<ObjectReference>::new();

        for obj in s_candidates.iter(){
            if RC_TABLE.load_atomic::<u16>(obj.to_raw_address(), Ordering::SeqCst) > 0{
                real_s_candidate.push(*obj);
            }
        }
        println!("num of s_rc candidates before dead object removal = {}", s_candidates.len());


        s_candidates.clear();




        println!("num of regular candidates = {}", real_candidate.len());
        println!("num of s_rc candidates = {}", real_s_candidate.len());

        println!("##################################");

        for obj in real_s_candidate.iter(){
            assert!(OBJ_COLOR_TABLE.load_atomic::<u8>((*obj).to_raw_address(), Ordering::SeqCst) != WHITE);
            self.mark(*obj);
        }

        for obj in real_s_candidate.iter(){
            self.scan(*obj);
        }

         for obj in real_s_candidate.iter(){
            assert!(OBJ_COLOR_TABLE.load_atomic::<u8>((*obj).to_raw_address(), Ordering::SeqCst) != GREY);
            self.collect_whites(*obj, lxr);
        }
    }
}

impl<VM: VMBinding> CycleCollector<VM>{

    pub fn new()->CycleCollector<VM>{
        CycleCollector::<VM>{
            rc: RefCountHelper::NEW
        }
    }

    fn mark(&self, o: ObjectReference){
        debug_assert!(RC_TABLE.load_atomic::<u16>(o.to_raw_address(), Ordering::SeqCst) > 0 
        || OBJ_COLOR_TABLE.load_atomic::<u8>(o.to_raw_address(), Ordering::SeqCst) == GREY);
        let mut dfs_stack = Vec::<ObjectReference>::new();
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
            if OBJ_COLOR_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) == BLACK{
                STRONG_RC_TABLE.store_atomic::<u8>(curr.to_raw_address(),0, Ordering::SeqCst);
                OBJ_COLOR_TABLE.store_atomic::<u8>(curr.to_raw_address(),GREY, Ordering::SeqCst);
                curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
            }
            
        }
    }

    fn scan(&self, o: ObjectReference){
        let mut dfs_stack = Vec::<ObjectReference>::new();
        dfs_stack.push(o);

        while let Some(curr) = dfs_stack.pop(){
            let visitor = |slot: <VM as vm::VMBinding>::VMSlot, b| {
                if let Some(x) = slot.load(){
                    dfs_stack.push(x);
                }
            };
            if OBJ_COLOR_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) == GREY{
                debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) == 0);
                if RC_TABLE.load_atomic::<u16>(curr.to_raw_address(), Ordering::SeqCst) > 0{
                    self.scan_black(curr);
                }
                else{
                    OBJ_COLOR_TABLE.store_atomic::<u8>(curr.to_raw_address(),WHITE, Ordering::SeqCst);
                    curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
                }
           }
        }
    }


    fn scan_black(&self, o: ObjectReference){
        let mut dfs_stack = Vec::<ObjectReference>::new();
        dfs_stack.push(o);
        while let Some(curr) = dfs_stack.pop(){
            debug_assert!(self.rc.count(curr) > 0);
            if OBJ_COLOR_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) != BLACK{
                dfs_stack.push(curr);
                OBJ_COLOR_TABLE.store_atomic::<u8>(curr.to_raw_address(),BLACK, Ordering::SeqCst);
                IN_STACK_TABLE.store_atomic::<u8>(curr.to_raw_address(), 1 as u8, Ordering::SeqCst);
                let s_rc = self.rc.count(curr);
                if s_rc > MAX_STRONG_REF_COUNT as u16{
                    STRONG_RC_TABLE.store_atomic::<u8>(curr.to_raw_address(),MAX_STRONG_REF_COUNT, Ordering::SeqCst);
                }
                else{
                    STRONG_RC_TABLE.store_atomic::<u8>(curr.to_raw_address(),s_rc as u8, Ordering::SeqCst);
                }
                curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, |slot: <VM as vm::VMBinding>::VMSlot, b| {
                    if let Some(x) = slot.load() {
                        debug_assert!(self.rc.count(x) < MAX_REF_COUNT);
                        let prev = self.rc.inc(x);
                        if IN_STACK_TABLE.load_atomic::<u8>(x.to_raw_address(), Ordering::SeqCst) == 0{
                            self.rc.strong_rc_inc(x);
                            dfs_stack.push(x);
                        }
                    }
                });
                
            }
            else{
                IN_STACK_TABLE.store_atomic::<u8>(curr.to_raw_address(), 0 as u8, Ordering::SeqCst);
                debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) != 0);
            }
        }
  
    }
    

    fn collect_whites(&self, o: ObjectReference, lxr: &LXR<VM>){

        let mut dfs_stack = Vec::<ObjectReference>::new();
        dfs_stack.push(o);
        while let Some(curr) = dfs_stack.pop(){
            let mut visitor = |slot: <VM as vm::VMBinding>::VMSlot, b| {
                if let Some(x) = slot.load() {
                    dfs_stack.push(x);
                }
            };
            if OBJ_COLOR_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) == WHITE{
                OBJ_COLOR_TABLE.store_atomic::<u8>(curr.to_raw_address(),BLACK, Ordering::SeqCst);
                debug_assert!(STRONG_RC_TABLE.load_atomic::<u8>(curr.to_raw_address(), Ordering::SeqCst) == 0);
                debug_assert!(RC_TABLE.load_atomic::<u16>(curr.to_raw_address(), Ordering::SeqCst) == 0);
                curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
                self.process_dead_object(o, lxr);
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
            true
        }
    }
}
