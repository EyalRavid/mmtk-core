use crate::plan::global::BasePlan;
use crate::plan::global::CreateGeneralPlanArgs;
use crate::plan::global::CreateSpecificPlanArgs;
use crate::plan::mygc::mutator::ALLOCATOR_MAPPING;
use crate::plan::AllocationSemantics;
use crate::plan::Plan;
use crate::plan::PlanConstraints;
use crate::policy::immortalspace::ImmortalSpace;
use crate::policy::space::Space;
use crate::scheduler::GCWorkScheduler;
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::heap::gc_trigger::SpaceStats;
#[allow(unused_imports)]
use crate::util::heap::VMRequest;
use crate::util::metadata::side_metadata::SideMetadataContext;
use crate::util::opaque_pointer::*;
use crate::vm::VMBinding;
use enum_map::EnumMap;
use mmtk_macros::HasSpaces;

use crate::policy::immortalspace::ImmortalSpace as MyGCImmortalSpace;

#[derive(HasSpaces)]
pub struct MyGC<VM: VMBinding> {
    #[parent]
    pub base: BasePlan<VM>,
    #[space]
    pub mygc_space: MyGCImmortalSpace<VM>,
    #[space]
    pub immortal: ImmortalSpace<VM>,
    #[space]
    pub los: ImmortalSpace<VM>,
}

/// The plan constraints for the no gc plan.
pub const NOGC_CONSTRAINTS: PlanConstraints = PlanConstraints {
    collects_garbage: false,
    ..PlanConstraints::default()
};

impl<VM: VMBinding> Plan for MyGC<VM> {
    fn constraints(&self) -> &'static PlanConstraints {
        &NOGC_CONSTRAINTS
    }

    fn collection_required(&self, space_full: bool, _space: Option<SpaceStats<Self::VM>>) -> bool {
        self.base().collection_required(self, space_full)
    }

    fn base(&self) -> &BasePlan<VM> {
        &self.base
    }

    fn base_mut(&mut self) -> &mut BasePlan<Self::VM> {
        &mut self.base
    }

    fn prepare(&mut self, _tls: VMWorkerThread) {
        unreachable!()
    }

    fn release(&mut self, _tls: VMWorkerThread) {
        unreachable!()
    }

    fn get_allocator_mapping(&self) -> &'static EnumMap<AllocationSemantics, AllocatorSelector> {
        &ALLOCATOR_MAPPING
    }

    fn schedule_collection(&'static self, _scheduler: &GCWorkScheduler<VM>) {
        unreachable!("GC triggered in mygc")
    }

    fn current_gc_may_move_object(&self) -> bool {
        false
    }

    fn get_used_pages(&self) -> usize {
        self.mygc_space.reserved_pages()
            + self.immortal.reserved_pages()
            + self.los.reserved_pages()
            + self.base.get_used_pages()
    }
}

impl<VM: VMBinding> MyGC<VM> {
    pub fn new(args: CreateGeneralPlanArgs<VM>) -> Self {
        let mut plan_args = CreateSpecificPlanArgs {
            global_args: args,
            constraints: &NOGC_CONSTRAINTS,
            global_side_metadata_specs: SideMetadataContext::new_global_specs(&[]),
        };

        let res = MyGC {
            mygc_space: MyGCImmortalSpace::new(plan_args.get_space_args(
                "mygc_space",
                cfg!(not(feature = "mygc_no_zeroing")),
                false,
                VMRequest::discontiguous(),
            )),
            immortal: ImmortalSpace::new(plan_args.get_space_args(
                "immortal",
                true,
                false,
                VMRequest::discontiguous(),
            )),
            los: ImmortalSpace::new(plan_args.get_space_args(
                "los",
                true,
                false,
                VMRequest::discontiguous(),
            )),
            base: BasePlan::new(plan_args),
        };

        res.verify_side_metadata_sanity();

        res
    }
}
