use crate::plan::barriers::NoBarrier;
use crate::plan::mutator_context::unreachable_prepare_func;
use crate::plan::mutator_context::unreachable_release_func;
use crate::plan::mutator_context::Mutator;
use crate::plan::mutator_context::MutatorConfig;
use crate::plan::mutator_context::{
    create_allocator_mapping, create_space_mapping, ReservedAllocators,
};
use crate::plan::mygc::MyGC;
use crate::plan::AllocationSemantics;
use crate::util::alloc::allocators::{AllocatorSelector, Allocators};
use crate::util::VMMutatorThread;
use crate::vm::VMBinding;
use crate::MMTK;
use enum_map::{enum_map, EnumMap};

/// We use three bump allocators when enabling mygc_multi_space.
const MULTI_SPACE_RESERVED_ALLOCATORS: ReservedAllocators = ReservedAllocators {
    n_bump_pointer: 3,
    ..ReservedAllocators::DEFAULT
};

lazy_static! {
    /// When mygc_multi_space is disabled, force all the allocation go to the default allocator and space.
    static ref ALLOCATOR_MAPPING_SINGLE_SPACE: EnumMap<AllocationSemantics, AllocatorSelector> = enum_map! {
        _ => AllocatorSelector::BumpPointer(0),
    };
    pub static ref ALLOCATOR_MAPPING: EnumMap<AllocationSemantics, AllocatorSelector> = {
        if cfg!(feature = "mygc_multi_space") {
            let mut map = create_allocator_mapping(MULTI_SPACE_RESERVED_ALLOCATORS, false);
            map[AllocationSemantics::Default] = AllocatorSelector::BumpPointer(0);
            map[AllocationSemantics::Immortal] = AllocatorSelector::BumpPointer(1);
            map[AllocationSemantics::Los] = AllocatorSelector::BumpPointer(2);
            map
        } else {
            *ALLOCATOR_MAPPING_SINGLE_SPACE
        }
    };
}

pub fn create_mygc_mutator<VM: VMBinding>(
    mutator_tls: VMMutatorThread,
    mmtk: &'static MMTK<VM>,
) -> Mutator<VM> {
    let plan = mmtk.get_plan().downcast_ref::<MyGC<VM>>().unwrap();
    let config = MutatorConfig {
        allocator_mapping: &ALLOCATOR_MAPPING,
        space_mapping: Box::new({
            let mut vec = create_space_mapping(MULTI_SPACE_RESERVED_ALLOCATORS, false, plan);
            vec.push((AllocatorSelector::BumpPointer(0), &plan.mygc_space));
            vec.push((AllocatorSelector::BumpPointer(1), &plan.immortal));
            vec.push((AllocatorSelector::BumpPointer(2), &plan.los));
            vec
        }),
        prepare_func: &unreachable_prepare_func,
        release_func: &unreachable_release_func,
    };

    Mutator {
        allocators: Allocators::<VM>::new(mutator_tls, mmtk, &config.space_mapping),
        barrier: Box::new(NoBarrier),
        mutator_tls,
        config,
        plan,
    }
}
