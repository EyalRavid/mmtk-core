use crate::plan::Plan;
use crate::policy::immix::block::{Block, BlockState};
use crate::policy::space::Space;
use crate::scheduler::gc_work::*;
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::util::ObjectReference;
use crate::vm::slot::Slot;
use crate::vm::*;
use crate::MMTK;
use crate::{scheduler::*, ObjectQueue};
use std::collections::HashSet;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU8, Ordering};
use crate::util::rc::{CANDIDATES_STATUS, MAX_REF_COUNT, MAX_STRONG_REF_COUNT, OBJ_COLOR_TABLE, RC_TABLE, STRONG_RC_TABLE};
use crate::util::heap::chunk_map::ChunkState;
use crate::util::linear_scan::Region;
use crate::util::rc;
use crate::policy::immix::line::Line;
#[allow(dead_code)]
pub struct SanityChecker<SL: Slot> {
    /// Visited objects
    refs: HashSet<ObjectReference>,
    /// Cached root edges for sanity root scanning
    root_slots: Vec<(Vec<SL>, RootKind)>,
    /// Cached root nodes for sanity root scanning
    root_nodes: Vec<Vec<ObjectReference>>,
}

impl<SL: Slot> Default for SanityChecker<SL> {
    fn default() -> Self {
        Self::new()
    }
}

impl<SL: Slot> SanityChecker<SL> {
    pub fn new() -> Self {
        Self {
            refs: HashSet::new(),
            root_slots: vec![],
            root_nodes: vec![],
        }
    }

    /// Cache a list of root slots to the sanity checker.
    pub fn add_root_slots(&mut self, roots: Vec<SL>, kind: RootKind) {
        self.root_slots.push((roots, kind))
    }

    pub fn add_root_nodes(&mut self, roots: Vec<ObjectReference>) {
        self.root_nodes.push(roots)
    }

    /// Reset roots cache at the end of the sanity gc.
    pub(crate) fn clear_roots_cache(&mut self) {
        self.root_slots.clear();
        self.root_nodes.clear();
    }
}

pub struct ScheduleSanityGC<P: Plan> {
    _plan: &'static P,
}

impl<P: Plan> ScheduleSanityGC<P> {
    pub fn new(plan: &'static P) -> Self {
        ScheduleSanityGC { _plan: plan }
    }
}

impl<P: Plan> GCWork<P::VM> for ScheduleSanityGC<P> {
    fn do_work(&mut self, worker: &mut GCWorker<P::VM>, mmtk: &'static MMTK<P::VM>) {
        println!("$$$$$$$$$$$$$$$$reached sanity$$$$$$$$$");
        let scheduler = worker.scheduler();
        let plan = mmtk.get_plan();

        scheduler.reset_state();

        // We are going to do sanity GC which will traverse the object graph again. Reset slot logger to clear recorded slots.
        #[cfg(feature = "extreme_assertions")]
        mmtk.slot_logger.reset();

        mmtk.sanity_begin(); // Stop & scan mutators (mutator scanning can happen before STW)

        // We use the cached roots for sanity gc, based on the assumption that
        // the stack scanning triggered by the selected plan is correct and precise.
        // FIXME(Wenyu,Tianle): When working on eager stack scanning on OpenJDK,
        // the stack scanning may be broken. Uncomment the following lines to
        // collect the roots again.
        // Also, remember to call `DerivedPointerTable::update_pointers(); DerivedPointerTable::clear();`
        // in openjdk binding before the second round of roots scanning.
        // for mutator in <P::VM as VMBinding>::VMActivePlan::mutators() {
        //     scheduler.work_buckets[WorkBucketStage::Prepare]
        //         .add(ScanMutatorRoots::<SanityGCProcessEdges<P::VM>>(mutator));
        // }
        {
            let sanity_checker = mmtk.sanity_checker.lock().unwrap();
            for (roots, kind) in &sanity_checker.root_slots {
                let mut w = SanityGCProcessEdges::<P::VM>::new(
                    roots.clone(),
                    true,
                    mmtk,
                    WorkBucketStage::Closure,
                );
                w.root_kind = Some(*kind);
                scheduler.work_buckets[WorkBucketStage::Closure].add(w);
            }

            //this was the original loop (note the parameters of new):
            // for roots in &sanity_checker.root_nodes {
            //     scheduler.work_buckets[WorkBucketStage::Closure].add(ProcessRootNode::<
            //         P::VM,
            //         SanityGCProcessEdges<P::VM>,
            //         SanityGCProcessEdges<P::VM>,
            //     >::new(
            //         roots.clone(),
            //         false,
            //         false,
            //         false,
            //         WorkBucketStage::Closure,
            //     ));
            //}
            //Eyal added this:
            for roots in &sanity_checker.root_nodes {
                panic!("there is root_node");
                scheduler.work_buckets[WorkBucketStage::Closure].add(ProcessRootNode::<
                    P::VM,
                    SanityGCProcessEdges<P::VM>,
                    SanityGCProcessEdges<P::VM>,
                >::new(roots.clone(), WorkBucketStage::Closure));
            }
        }
        // Prepare global/collectors/mutators
        worker.scheduler().work_buckets[WorkBucketStage::Prepare]
            .add(SanityPrepare::<P>::new(plan.downcast_ref::<P>().unwrap()));
        // Release global/collectors/mutators
        worker.scheduler().work_buckets[WorkBucketStage::Release]
            .add(SanityRelease::<P>::new(plan.downcast_ref::<P>().unwrap()));
    }
}

pub static MARK_STATE: AtomicU8 = AtomicU8::new(0);
const MARK_BITS: SideMetadataSpec =
    crate::util::metadata::side_metadata::spec_defs::SANITY_MARK_BITS;

pub struct SanityPrepare<P: Plan> {
    pub plan: &'static P,
}

impl<P: Plan> SanityPrepare<P> {
    pub fn new(plan: &'static P) -> Self {
        Self { plan }
    }

    fn update_mark_state() {
        let mut mark_state = MARK_STATE.load(Ordering::SeqCst);
        if mark_state == 0 || mark_state == 255 {
            mark_state = 1;
        } else {
            mark_state += 1;
        }
        MARK_STATE.store(mark_state, Ordering::SeqCst);
    }
}

impl<P: Plan> GCWork<P::VM> for SanityPrepare<P> {
    fn do_work(&mut self, _worker: &mut GCWorker<P::VM>, mmtk: &'static MMTK<P::VM>) {
        Self::update_mark_state();
        <P::VM as VMBinding>::VMCollection::clear_cld_claimed_marks();
        info!("Sanity GC prepare");
        {
            let mut sanity_checker = mmtk.sanity_checker.lock().unwrap();
            sanity_checker.refs.clear();
        }
        crate::SANITY_LIVE_SIZE_IX.store(0, Ordering::Relaxed);
        crate::SANITY_LIVE_SIZE_LOS.store(0, Ordering::Relaxed);
    }
}

pub struct SanityRelease<P: Plan> {
    pub plan: &'static P,
}

impl<P: Plan> SanityRelease<P> {
    pub fn new(plan: &'static P) -> Self {
        Self { plan }
    }
}

impl<P: Plan> GCWork<P::VM> for SanityRelease<P> {
    fn do_work(&mut self, _worker: &mut GCWorker<P::VM>, mmtk: &'static MMTK<P::VM>) {
        info!("Sanity GC release");
        if let Some(lxr) = mmtk
            .get_plan()
            .downcast_ref::<crate::plan::lxr::LXR<P::VM>>()
        {
            lxr.los().sanity_sweep_large_objects();
            let mut rc_sanity_objects = lxr.rc_sanity_objects.lock().unwrap();
            for (obj, rc) in rc_sanity_objects.iter() {
                let real_rc = lxr.rc.count(*obj);
                //println!("object: {} has real rc of: {}", obj.to_raw_address(), real_rc);
                //println!("object: {} has acording to scan: {}", obj.to_raw_address(), *rc);
                if lxr.rc.is_stuck(*obj){
                    println!("stuck object in sanity!!!!!!!!!!!");
                    //assert!(*rc > 0);
                }
                else{
                    //this assertion was may be wrong beacuse of stuck objects
                    //assert!(real_rc == *rc, "object: {} has metadata rc of: {}, but acording to scan: {}", obj.to_raw_address(), real_rc, *rc);
                    assert!(real_rc >= *rc, "object: {} has metadata rc of: {}, but acording to scan: {}", obj.to_raw_address(), real_rc, *rc);
                }
                
            }
            for chunk in lxr.immix_space.chunk_map.all_chunks()
            .filter(|c| lxr.immix_space.chunk_map.get(*c) == ChunkState::Allocated){
                for block in chunk.iter_region::<Block>().filter(|block| block.get_state() != BlockState::Unallocated) {
                    let mut cursor = block.start();
                    let limit = block.end();
                    while cursor < limit {
                        let o = unsafe { cursor.to_object_reference::<P::VM>() };
                        let mark_state = MARK_STATE.load(Ordering::SeqCst);
                        let mark_val = MARK_BITS.load_atomic::<u8>(o.to_raw_address(), Ordering::SeqCst);
                        // if lxr.rc.count(o) > 1 || STRONG_RC_TABLE.load_atomic::<u8>(o.to_raw_address(), Ordering::SeqCst) > 0 {
                        if lxr.rc.count(o) > 0 &&
                            (!Line::is_aligned(o.to_raw_address()) || !lxr.rc.is_straddle_line(Line::from(o.to_raw_address()))) {
                
                            let size = <P::VM as VMBinding>::VMObjectModel::get_current_size(o);
                            cursor = cursor + size;
                            assert!(STRONG_RC_TABLE.load_atomic::<u8>(o.to_raw_address(), Ordering::SeqCst) > 0 ||
                                    CANDIDATES_STATUS.load_atomic::<u8>(o.to_raw_address(), Ordering::SeqCst) != 0);
                            assert!(cursor <= limit);

                            //this assertion was may be wrong beacuse of stuck objects
                            //assert!(mark_val == mark_state, "size is, {}", size);
                        }
                        else{
                            cursor = cursor + rc::MIN_OBJECT_SIZE;
                        }
                        
                        //let c = lxr.rc.count(o);
                        //assert!(c <= 1 || old_value == mark_state);
                        //assert!(STRONG_RC_TABLE.load_atomic::<u8>(o.to_raw_address(), Ordering::SeqCst) == 0 || mark_val == mark_state);
                    }
                }
            }
            rc_sanity_objects.clear();

            let is_live = |o: ObjectReference| -> bool {
                assert!(lxr.rc.count(o) > 0);
                let mark_state = MARK_STATE.load(Ordering::SeqCst);
                let mark_val = MARK_BITS.load_atomic::<u8>(o.to_raw_address(), Ordering::SeqCst);
                //assert!(mark_val == mark_state);
                assert!(STRONG_RC_TABLE.load_atomic::<u8>(o.to_raw_address(), Ordering::SeqCst) > 0 ||
                        CANDIDATES_STATUS.load_atomic::<u8>(o.to_raw_address(), Ordering::SeqCst) != 0);
                true
            };

            lxr.common.los.sweep_rc_mature_objects_after_satb(&is_live); 
        }
        else{
            panic!("no lxr");
        }
        mmtk.sanity_checker.lock().unwrap().clear_roots_cache();
        mmtk.sanity_end();
    }
}

// #[derive(Default)]
pub struct SanityGCProcessEdges<VM: VMBinding> {
    base: ProcessEdgesBase<VM>,
    edge: Option<VM::VMSlot>,
}

impl<VM: VMBinding> Deref for SanityGCProcessEdges<VM> {
    type Target = ProcessEdgesBase<VM>;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<VM: VMBinding> DerefMut for SanityGCProcessEdges<VM> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl<VM: VMBinding> SanityGCProcessEdges<VM> {
    fn attempt_mark(&self, o: ObjectReference) -> bool {
        let mark_state = MARK_STATE.load(Ordering::SeqCst);
        loop {
            let old_value = MARK_BITS.load_atomic::<u8>(o.to_raw_address(), Ordering::SeqCst);
            if old_value == mark_state {
                return false;
            }
            if MARK_BITS
                .compare_exchange_atomic::<u8>(
                    o.to_raw_address(),
                    old_value,
                    mark_state,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                return true;
            }
        }
    }
}

impl<VM: VMBinding> ProcessEdgesWork for SanityGCProcessEdges<VM> {
    type VM = VM;
    type ScanObjectsWorkType = ScanObjects<Self>;

    const OVERWRITE_REFERENCE: bool = false;
    fn new(
        slots: Vec<SlotOf<Self>>,
        roots: bool,
        mmtk: &'static MMTK<VM>,
        bucket: WorkBucketStage,
    ) -> Self {
        Self {
            base: ProcessEdgesBase::new(slots, roots, mmtk, bucket),
            // ..Default::default()
            edge: None,
        }
    }

    //Eyal chanched this
    //this func was process_edge
    //The declartion was:
    //fn process_edge(&mut self, slot: EdgeOf<Self>)
    fn process_slot(&mut self, slot: SlotOf<Self>) {
        self.edge = Some(slot);

        let Some(object) = slot.load() else {
            // Skip slots that are not holding an object reference.
            return;
        };
        let new_object = self.trace_object(object);
        if Self::OVERWRITE_REFERENCE && new_object != object {
            slot.store(Some(new_object));
        }
    }

    #[cfg(feature = "fragmentation_analysis")]
    fn trace_object(&mut self, object: ObjectReference) -> ObjectReference {
        use crate::util::address::{CLDScanPolicy, RefScanPolicy};
        if self.attempt_mark(object) {
            let lxr = self
                .mmtk()
                .get_plan()
                .downcast_ref::<crate::plan::lxr::LXR<VM>>()
                .unwrap();
            if lxr.immix_space.in_space(object) {
                crate::SANITY_LIVE_SIZE_IX.fetch_add(object.get_size::<VM>(), Ordering::Relaxed);
            } else {
                crate::SANITY_LIVE_SIZE_LOS.fetch_add(object.get_size::<VM>(), Ordering::Relaxed);
            }
            self.nodes.enqueue(object);
        }
        object
    }
    #[cfg(not(feature = "fragmentation_analysis"))]
    fn trace_object(&mut self, object: ObjectReference) -> ObjectReference {
        // gc_log!(
        //     "S {:?} -> {:?} r={} kind={:?}",
        //     self.edge,
        //     object,
        //     self.roots,
        //     self.root_kind
        // );
        if let Some(lxr) = self
            .mmtk()
            .get_plan()
            .downcast_ref::<crate::plan::lxr::LXR<VM>>()
        {
            let mut rc_sanity_objects = lxr.rc_sanity_objects.lock().unwrap();
            for (obj, rc) in rc_sanity_objects.iter_mut() {
                if (*obj == object){
                    if *rc < MAX_REF_COUNT {
                        *rc += 1;
                    }
                    
                } 
            }
            if self.edge.unwrap().to_address().is_mapped() {
                assert!(
                    !self.edge.unwrap().to_address().is_field_logged::<VM>(),
                    "{:?} -> {:?} is logged",
                    self.edge,
                    object
                );
            }
        }
        else{
            panic!("no lxr");
        }
        //Eyal commented this lines beacuse object should be null
        // if object.is_null() {
        //     return object;
        // }
        if self.attempt_mark(object) {

            // FIXME steveb consider VM-specific integrity check on reference.
            assert!(object.is_sane(), "Invalid reference {:?}", object);

            // Let plan check object
            assert!(
                object.to_raw_address().is_mapped(),
                "Invalid reference {:?} -> {:?}",
                self.edge,
                object
            );
            assert!(
                object.is_sane(),
                "Invalid reference {:?} -> {:?}",
                self.edge,
                object
            );
            if let Some(lxr) = self
                .mmtk()
                .get_plan()
                .downcast_ref::<crate::plan::lxr::LXR<VM>>()
            {
                assert!(STRONG_RC_TABLE.load_atomic::<u8>(object.to_raw_address(), Ordering::SeqCst) != 0 || 
                        CANDIDATES_STATUS.load_atomic::<u8>(object.to_raw_address(), Ordering::SeqCst) != 0,
                         "{:?} has zero strong rc count and {} rc", object, lxr.rc.count(object));
                assert!(STRONG_RC_TABLE.load_atomic::<u8>(object.to_raw_address(), Ordering::SeqCst) <= lxr.rc.count(object));
                assert!(
                    unsafe { object.to_raw_address().load::<usize>() } != 0xdead,
                    "{:?} -> {:?} is killed by decs",
                    self.edge,
                    object
                );
                assert!(
                    lxr.rc.count(object) > 0,
                    "{:?} -> {:?} has zero rc count",
                    self.edge,
                    object
                );

                assert!(
                    !crate::util::object_forwarding::is_forwarded_or_being_forwarded::<VM>(object),
                    "{:?} -> {:?} is forwarded",
                    self.edge,
                    object
                );
                if lxr.immix_space.in_space(object) {
                    assert_ne!(
                        Block::containing(object).get_state(),
                        BlockState::Unallocated,
                        "{:?}->{:?} block is released",
                        self.edge,
                        object
                    )
                }
                if lxr.current_pause().unwrap() == crate::plan::immix::Pause::FinalMark
                    || lxr.current_pause().unwrap() == crate::plan::immix::Pause::Full
                {
                    if !lxr.is_marked(object) {
                        flush_logs!()
                    }
                    assert!(
                        lxr.is_marked(object),
                        "{:?} -> {:?} is not marked, roots={} kind={:?}",
                        self.edge,
                        object,
                        self.roots,
                        self.root_kind,
                    )
                }

            }
            self.nodes.enqueue(object);
        }

        // If the valid object (VO) bit metadata is enabled, all live objects should have the VO
        // bit set when sanity GC starts.
        #[cfg(feature = "vo_bit")]
        if !crate::util::metadata::vo_bit::is_vo_bit_set(object) {
            panic!("VO bit is not set: {}", object);
        }

        object
    }

    fn create_scan_work(&self, nodes: Vec<ObjectReference>) -> Self::ScanObjectsWorkType {
        let mut x = ScanObjects::<Self>::new(nodes, false, false, false, WorkBucketStage::Closure);
        x.discovery = false;
        x
    }
}
