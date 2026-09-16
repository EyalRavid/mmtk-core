//! Read/Write barrier implementations.

use std::sync::atomic::AtomicUsize;

use atomic::Ordering;
use atomic_traits::fetch::Or;

use super::LXR;
use crate::plan::barriers::BarrierSemantics;
use crate::plan::barriers::LOGGED_VALUE;
use crate::plan::barriers::UNLOGGED_VALUE;
use crate::plan::lxr::rc::ProcessDecs;
use crate::plan::lxr::rc::ProcessIncs;
use crate::plan::lxr::rc::EDGE_KIND_MATURE;
use crate::plan::VectorQueue;
#[cfg(feature = "lxr_precise_incs_counter")]
use crate::policy::space::Space;
use crate::scheduler::WorkBucketStage;
use crate::util::address::CLDScanPolicy;
use crate::util::address::RefScanPolicy;
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::util::rc::{cc, BLACK_IN_STACK};
use crate::util::*;
use crate::vm::slot::MemorySlice;
use crate::vm::slot::Slot;
use crate::vm::*;
use crate::LazySweepingJobsCounter;
use crate::MMTK;

pub const TAKERATE_MEASUREMENT: bool = crate::args::TAKERATE_MEASUREMENT;
pub static FAST_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static SLOW_COUNT: AtomicUsize = AtomicUsize::new(0);

/// The LXR field write barrier.
///
/// # The concurrent-marking / SATB half was removed on 2026-09-15, because it cannot run
///
/// This barrier used to carry a second job beside reference counting: capturing the pre-write
/// value for a concurrent mark closure (SATB). **None of that machinery was reachable in this
/// fork**, so it was deleted. The proof, which is short and worth keeping because nothing else
/// in the tree states it:
///
/// * `LXR::select_collection_kind` is the function that can return `Pause::InitialMark`. It has
///   exactly one call site, `global.rs:272`, and that line is **commented out**. The pause is
///   chosen instead by the override just below it: `Pause::FullRC` for a user-triggered GC
///   outside the harness, `Pause::RefCount` otherwise.
/// * So `InitialMark`, `FinalMark` and `Full` are never selected. Nothing reaches
///   `set_concurrent_marking_state(true)`, so `LXR::cm_in_progress()` is permanently false, and
///   the binding's `CONCURRENT_MARKING_ACTIVE` stays 0 -- which is what gates
///   `MMTkFieldBarrierSetRuntime::load_reference` in `mmtkFieldBarrier.cpp`, so the VM never even
///   calls into the read barrier.
/// * Therefore `cm_in_progress() || current_pause() == Some(Pause::FinalMark)` -- the old
///   `should_create_satb_packets()` -- was permanently false.
///
/// Note `LXR::cm_enabled()` is nevertheless **true** (`global.rs:1520`,
/// `!cfg!(feature = "lxr_no_cm")`), so none of this was compiled out. It was live code that
/// evaluated to "do nothing" on every flush.
///
/// ## What was removed
///
/// * `refs: VectorQueue<ObjectReference>` -- one per mutator, fed only by `load_reference`, so
///   always empty.
/// * `load_reference` -- the override is gone; `BarrierSemantics`' default no-op (`barriers.rs`)
///   is now used, which is what it already did at run time.
/// * `flush_weak_refs` -- called on every `flush()` to discover an empty queue.
/// * `should_create_satb_packets`, and with it the `Arc` + `ProcessModBufSATB` arm of what is now
///   `flush_decs`.
///
/// `ProcessModBufSATB` (`cm.rs`) and `ProcessDecs::new_arc` (`rc.rs`) are left in place: they are
/// `pub`, this was their only caller, and they belong to the concurrent-marking module rather
/// than here.
///
/// ## ⚠ If concurrent marking is ever turned back on, this must come back FIRST
///
/// Uncommenting `global.rs:272` alone is **not** enough, and the failure would be silent. The
/// deleted `load_reference` called `LXR::is_marked`, which reads `LOCAL_MARK_BIT_SPEC` -- and
/// `5286e80d` stopped maintaining that table entirely (`OPTIMIZATION_AUDIT.md` B.1: the
/// clean-block and line-reuse paths no longer initialise it, on the grounds that nothing reads
/// it). A revived mark closure would read uninitialised metadata. Restoring the SATB barrier
/// therefore means restoring mark-table maintenance in the same commit.
///
/// Full account: `~/prod-ae/bundle/FINDINGS.md` §22.
pub struct LXRFieldBarrierSemantics<VM: VMBinding> {
    mmtk: &'static MMTK<VM>,
    incs: VectorQueue<VM::VMSlot>,
    decs: VectorQueue<ObjectReference>,
    lxr: &'static LXR<VM>,
    #[cfg(feature = "lxr_precise_incs_counter")]
    stat: crate::LocalRCStat,
    /// Barrier slow paths taken while the cycle collector is running, and SATB inserts performed.
    ///
    /// PLAIN `usize`, deliberately: these live in the per-mutator barrier struct and are
    /// accumulated with no atomic at all, then folded into `Counters` at the barrier's existing
    /// flush points. An atomic here would sit on the mutator's hot path and contaminate
    /// `time.other`, which is the very quantity these counters exist to explain
    /// (`EVALUATION_PLAN.md` §6 rule 5). `~/mmtk/OPTIMIZATION_AUDIT.md` B.2.
    #[cfg(feature = "s_rc_stats")]
    slow_in_cc: usize,
    #[cfg(feature = "s_rc_stats")]
    satb_inserts: usize,
}

impl<VM: VMBinding> LXRFieldBarrierSemantics<VM> {
    const UNLOG_BITS: SideMetadataSpec = *VM::VMObjectModel::GLOBAL_FIELD_UNLOG_BIT_SPEC
        .as_spec()
        .extract_side_spec();

    #[allow(unused)]
    pub fn new(mmtk: &'static MMTK<VM>) -> Self {
        Self {
            mmtk,
            incs: VectorQueue::default(),
            decs: VectorQueue::default(),
            lxr: mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap(),
            #[cfg(feature = "lxr_precise_incs_counter")]
            stat: crate::LocalRCStat::default(),
            #[cfg(feature = "s_rc_stats")]
            slow_in_cc: 0,
            #[cfg(feature = "s_rc_stats")]
            satb_inserts: 0,
        }
    }

    fn get_slot_logging_state(&self, slot: VM::VMSlot) -> u8 {
        unsafe { Self::UNLOG_BITS.load(slot.to_address()) }
    }

    fn attempt_to_log_field(&self, slot: VM::VMSlot) -> bool {
        loop {
            // Bailout if logged
            if self.get_slot_logging_state(slot) == LOGGED_VALUE {
                return false;
            }
            // Attempt to log the slots
            match Self::UNLOG_BITS.compare_exchange_atomic(
                slot.to_address(),
                UNLOGGED_VALUE,
                LOGGED_VALUE,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(current) => {
                    if current == LOGGED_VALUE {
                        return false;
                    }
                }
            }
            // Failed to log the slot. Spin.
            std::hint::spin_loop();
        }
    }

    fn log_slot_and_get_old_target(&self, slot: VM::VMSlot) -> Result<Option<ObjectReference>, ()> {

        if self.get_slot_logging_state(slot) == LOGGED_VALUE {
            return Err(());
        }
        let old = slot.load();
        if self.attempt_to_log_field(slot) {
            Ok(old)
        } else {
            Err(())
        }
    }

    #[allow(unused)]
    fn log_slot_and_get_old_target_sloppy(
        &self,
        slot: VM::VMSlot,
    ) -> Result<Option<ObjectReference>, ()> {
        if !slot.to_address().is_field_logged::<VM>() {
            let old = slot.load();
            slot.to_address().log_field::<VM>();
            Ok(old)
        } else {
            Err(())
        }
    }

    fn slow(
        &mut self,
        _src: Option<ObjectReference>,
        slot: VM::VMSlot,
        old: Option<ObjectReference>,
    ) {
        // FIXME: This assertion may fail!
        // #[cfg(any(
        //     feature = "sanity",
        //     feature = "field_barrier_validation",
        //     debug_assertions
        // ))]
        // debug_assert!(
        //     old.is_null() || self.lxr.rc.count(old) != 0,
        //     "zero rc count {:?} -> {:?}",
        //     slot,
        //     old
        // );
        if cfg!(feature = "field_barrier_validation") {
            let o = super::LAST_REFERENTS
                .lock()
                .unwrap()
                .get(&slot.to_address())
                .cloned()
                .expect(&format!("Unknown slot {:?} -> {:?}", slot, old));
            if old != o {
                println!("barrier {:?} old={:?}", slot, old);
                {
                    let _g = super::LAST_REFERENTS.lock();
                    // println!("{:?} {}", old, VM::VMObjectModel::dump_object_s(old));
                    // println!("{:?} {}", _src, VM::VMObjectModel::dump_object_s(_src));
                }
                assert!(
                    old == o,
                    "Untracked old referent {:?} -> {:?} should be {:?}  ",
                    slot,
                    old,
                    o,
                )
            }
        }
        // Reference counting
        if let Some(old) = old {
            if !cfg!(feature = "lxr_no_decs") || !self.lxr.is_marked(old) {
                self.decs.push(old);
                if self.decs.is_full() {
                    self.flush_decs();
                }
            }
        }
        self.incs.push(slot);
        #[cfg(feature = "lxr_precise_incs_counter")]
        {
            self.stat.total_incs += 1;
            if self.lxr.los().address_in_space(slot.to_address()) {
                self.stat.los_incs += 1;
            }
        }
        if self.incs.is_full() {
            self.flush_incs();
        }
        //self.lxr.satb_map.insert(slot, old);
        if (self.lxr.in_cycle_collection.load(Ordering::SeqCst) == true) {
            #[cfg(feature = "s_rc_stats")]
            { self.slow_in_cc += 1; }
            if let Some(obj) = _src {
                if cc::colour(obj, Ordering::SeqCst) >= BLACK_IN_STACK {
                    #[cfg(feature = "s_rc_stats")]
                    { self.satb_inserts += 1; }
                    self.lxr.satb_map.insert(slot, old);
                }
         }
            else {
                #[cfg(feature = "s_rc_stats")]
                { self.satb_inserts += 1; }
                self.lxr.satb_map.insert(slot, old);
            }
        }

    }

    fn enqueue_node(
        &mut self,
        src: Option<ObjectReference>,
        slot: VM::VMSlot,
        _new: Option<ObjectReference>,
    ) -> bool {
        if TAKERATE_MEASUREMENT && self.mmtk.inside_harness() {
            FAST_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        if let Ok(old) = self.log_slot_and_get_old_target(slot) {
            if TAKERATE_MEASUREMENT && self.mmtk.inside_harness() {
                SLOW_COUNT.fetch_add(1, Ordering::SeqCst);
            }
            self.slow(src, slot, old);
            true
        } else {
            false
        }
    }

    #[cold]
    fn flush_incs(&mut self) {
        if !self.incs.is_empty() {
            let incs = self.incs.take();
            self.lxr.rc.increase_inc_buffer_size(incs.len());
            self.mmtk.scheduler.work_buckets[WorkBucketStage::RCProcessIncs].add(ProcessIncs::<
                _,
                EDGE_KIND_MATURE,
            >::new(
                incs, self.lxr
            ));
        }
    }

    /// Formerly `flush_decs_and_satb`. The SATB half is gone -- see the note on
    /// `LXRFieldBarrierSemantics` -- so this only ever built the `ProcessDecs` packet.
    #[cold]
    fn flush_decs(&mut self) {
        if !self.decs.is_empty() {
            if cfg!(feature = "decs_counter") {
                self.lxr
                    .barrier_decs
                    .fetch_add(self.decs.len(), Ordering::SeqCst);
            }
            let decs = self.decs.take();
            let w = ProcessDecs::new(decs, LazySweepingJobsCounter::new_decs());
            if crate::args::LAZY_DECREMENTS {
                self.mmtk.scheduler.postpone_prioritized(w);
            } else {
                self.mmtk.scheduler.work_buckets[WorkBucketStage::STWRCDecsAndSweep].add(w);
            }
        }
    }

    /// Fold the per-mutator counters into `Counters`. Called from `flush`, i.e. at the same points
    /// the barrier already drains its inc/dec buffers -- a handful of atomics per flush rather
    /// than one per barrier hit.
    #[cfg(feature = "s_rc_stats")]
    #[cold]
    fn flush_barrier_stats(&mut self) {
        if self.slow_in_cc != 0 || self.satb_inserts != 0 {
            let c = crate::counters();
            c.barrier_slow_in_cc
                .fetch_add(self.slow_in_cc, Ordering::Relaxed);
            c.barrier_satb_inserts
                .fetch_add(self.satb_inserts, Ordering::Relaxed);
            self.slow_in_cc = 0;
            self.satb_inserts = 0;
        }
    }

}

impl<VM: VMBinding> BarrierSemantics for LXRFieldBarrierSemantics<VM> {
    type VM = VM;

    #[cold]
    fn flush(&mut self) {
        #[cfg(feature = "s_rc_stats")]
        self.flush_barrier_stats();
        self.flush_incs();
        self.flush_decs();
        #[cfg(feature = "lxr_precise_incs_counter")]
        {
            crate::RC_STAT.merge(&mut self.stat);
        }
    }

    fn object_reference_write_slow(
        &mut self,
        src: Option<ObjectReference>,
        slot: VM::VMSlot,
        target: Option<ObjectReference>,
    ) {
        self.enqueue_node(src, slot, target);
    }

    fn memory_region_copy_slow(&mut self, _src: VM::VMMemorySlice, dst: VM::VMMemorySlice) {
        #[cfg(feature = "lxr_precise_incs_counter")]
        let mut slots = 0;
        for s in dst.iter_slots() {
            let _succ = self.enqueue_node(ObjectReference::NULL, s, None);
            #[cfg(feature = "lxr_precise_incs_counter")]
            if _succ {
                slots += 1;
            }
        }
        #[cfg(feature = "lxr_precise_incs_counter")]
        {
            self.stat.ac_incs += slots;
            self.stat.ac_calls += 1;
            if self.lxr.los().address_in_space(dst.start()) {
                self.stat.los_ac_incs += slots;
                self.stat.los_ac_calls += 1;
            }
        }
    }

    fn object_probable_write_slow(&mut self, obj: ObjectReference) {
        // assert_eq!(self.lxr.rc.count(obj), 1);
        #[cfg(feature = "lxr_precise_incs_counter")]
        let mut slots = 0;
        obj.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, |s, _| {
            let _succ = self.enqueue_node(Some(obj), s, None);
            #[cfg(feature = "lxr_precise_incs_counter")]
            {
                assert!(_succ);
                slots += 1;
            }
        });
        #[cfg(feature = "lxr_precise_incs_counter")]
        {
            self.stat.opw_calls += 1;
            self.stat.opw_incs += slots;
            if self.lxr.los().in_space(obj) {
                self.stat.los_opw_calls += 1;
                self.stat.los_opw_incs += slots;
            }
        }
    }
}
