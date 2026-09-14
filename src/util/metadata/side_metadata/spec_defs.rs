use crate::policy::immix::block::Block;
use crate::util::constants::*;
use crate::util::heap::layout::vm_layout::*;
use crate::util::linear_scan::Region;
use crate::util::metadata::side_metadata::constants::{
    GLOBAL_SIDE_METADATA_BASE_OFFSET, LOCAL_SIDE_METADATA_BASE_OFFSET,
};
use crate::util::metadata::side_metadata::SideMetadataOffset;
use crate::util::metadata::side_metadata::SideMetadataSpec;

// This macro helps define side metadata specs, and layout their offsets one after another.
// The macro is implemented with the incremental TT muncher pattern (see https://danielkeep.github.io/tlborm/book/pat-incremental-tt-munchers.html).
// This should only be used twice within mmtk-core: one for global specs, and one for local specs.
// This should not be used to layout VM specs (we have provided side_first()/side_after() for the VM side metadata specs).
macro_rules! define_side_metadata_specs {
    // Internal patterns

    // Define the first spec with offset at either GLOBAL/LOCAL_SIDE_METADATA_BASE_OFFSET
    (@first_spec $name: ident = (global: $is_global: expr, log_num_of_bits: $log_num_of_bits: expr, log_bytes_in_region: $log_bytes_in_region: expr)) => {
        pub const $name: SideMetadataSpec = SideMetadataSpec {
            name: stringify!($name),
            is_global: $is_global,
            offset: if $is_global { GLOBAL_SIDE_METADATA_BASE_OFFSET } else { LOCAL_SIDE_METADATA_BASE_OFFSET },
            log_num_of_bits: $log_num_of_bits,
            log_bytes_in_region: $log_bytes_in_region,
        };
    };
    // Define any spec that follows a previous spec. The new spec will be created and laid out after the previous spec.
    (@prev_spec $last_spec: ident as $last_spec_ident: ident, $name: ident = (global: $is_global: expr, log_num_of_bits: $log_num_of_bits: expr, log_bytes_in_region: $log_bytes_in_region: expr), $($tail:tt)*) => {
        pub const $name: SideMetadataSpec = SideMetadataSpec {
            name: stringify!($name),
            is_global: $is_global,
            offset: SideMetadataOffset::layout_after(&$last_spec),
            log_num_of_bits: $log_num_of_bits,
            log_bytes_in_region: $log_bytes_in_region,
        };
        define_side_metadata_specs!(@prev_spec $name as $last_spec_ident, $($tail)*);
    };
    // Define the last spec with the given identifier.
    (@prev_spec $last_spec: ident as $last_spec_ident: ident,) => {
        pub const $last_spec_ident: SideMetadataSpec = $last_spec;
    };

    // The actual macro

    // This is the pattern that should be used outside this macro.
    (last_spec_as $last_spec_ident: ident, $name0: ident = (global: $is_global0: expr, log_num_of_bits: $log_num_of_bits0: expr, log_bytes_in_region: $log_bytes_in_region0: expr), $($tail:tt)*) => {
        // Defines the first spec
        define_side_metadata_specs!(@first_spec $name0 = (global: $is_global0, log_num_of_bits: $log_num_of_bits0, log_bytes_in_region: $log_bytes_in_region0));
        // The rest specs
        define_side_metadata_specs!(@prev_spec $name0 as $last_spec_ident, $($tail)*);
    };
}

// This defines all GLOBAL side metadata used by mmtk-core.
define_side_metadata_specs!(
    last_spec_as LAST_GLOBAL_SIDE_METADATA_SPEC,
    // Mark the start of an object
    VO_BIT       = (global: true, log_num_of_bits: 0, log_bytes_in_region: LOG_MIN_OBJECT_SIZE as usize),
    // Track chunks used by (malloc) marksweep
    MS_ACTIVE_CHUNK = (global: true, log_num_of_bits: 3, log_bytes_in_region: LOG_BYTES_IN_CHUNK),
    // Track the index in SFT map for a chunk (only used for SFT sparse chunk map)
    SFT_DENSE_CHUNK_MAP_INDEX   = (global: true, log_num_of_bits: 3, log_bytes_in_region: LOG_BYTES_IN_CHUNK),
    // Reference counts
    RC_TABLE = (global: true, log_num_of_bits: crate::util::rc::LOG_REF_COUNT_BITS, log_bytes_in_region: crate::util::rc::LOG_MIN_OBJECT_SIZE),
    // Exact reference counts for objects whose `RC_TABLE` entry has SATURATED at `MAX_REF_COUNT`.
    //
    // WHY THIS EXISTS.  The fork has no mark-and-sweep, so it cannot let a saturated count stick
    // the way the baseline does -- nothing would ever recover the object.  It must therefore track
    // the exact count of every saturated object, and `FINDINGS.md` 20 measures what that costs:
    // 99.1% of `biojava`'s increments reach the overflow structure, and today that structure is a
    // `DashMap` (hash + shard RwLock + probe + fetch_add per operation).
    //
    // WHY 32 BITS.  The true counts are large -- at least 2.58M observed on `biojava`, and only 43
    // objects are ever saturated.  16 bits (65_535) does not reach; 32 bits does.
    //
    // WHY A SEPARATE SPARSE TABLE RATHER THAN A WIDER `RC_TABLE`.  `RC_TABLE` is DENSE -- every
    // object has an entry -- so widening it to 32 bits costs heap/4 of committed memory.  This
    // table is written only for objects that actually saturate, and side metadata is demand-zero
    // mmapped (`memory.rs::dzmmap`, no MAP_POPULATE; the explicit zeroing is
    // `cfg(not(target_os = "linux"))`) at 4 KiB pages (`MmapStrategy::SIDE_METADATA` is
    // `HugePageSupport::No`).  So an untouched page costs address space and NO RSS: 43 scattered
    // objects commit at most 43 pages, ~172 KiB, against heap/4 for the dense alternative.
    //
    // ENCODING, chosen to mirror the `DashMap` it replaces exactly:
    //     consulted ONLY when `RC_TABLE[o] == MAX_REF_COUNT`
    //     0      the true count is EXACTLY `MAX_REF_COUNT`   (the map's "no live entry" case)
    //     n > 0  the true count is `n`, and `n > MAX_REF_COUNT`
    // Zero is therefore the correct initial state and needs no explicit initialisation -- it is
    // what demand-zero mmap already gives.  An increment from the 0 state stores
    // `MAX_REF_COUNT + 1`, mirroring `or_insert_with(MAX_RC_USIZE)` followed by `fetch_add(1)`;
    // a decrement that reaches `MAX_REF_COUNT` stores 0, mirroring `entries.remove`.
    //
    // ⚠ BUDGET.  At 1/4 of the address space this is the LARGEST global spec in the system, and
    // the global budget (< 1/2, `LOG_GLOBAL_SIDE_METADATA_WORST_CASE_RATIO`) is NOT enforced
    // anywhere -- `LOG_MAX_GLOBAL_SIDE_METADATA_SIZE` has exactly one use, computing where LOCAL
    // metadata starts.  With this spec the global total is 0.484 of 0.500.  Gating the
    // `SANITY_*`/`GRAPH_*` declarations behind their features frees 0.102 and takes it to 0.383.
    OVERFLOW_RC_TABLE = (global: true, log_num_of_bits: 5, log_bytes_in_region: crate::util::rc::LOG_MIN_OBJECT_SIZE),
    // Record defrag state for immix blocks
    IX_BLOCK_DEFRAG = (global: true, log_num_of_bits: 3, log_bytes_in_region: crate::policy::immix::block::Block::LOG_BYTES),
    // Mark table for sanity GC
    // 4 bits, NOT 8. This holds the sanity mark EPOCH (`sanity_checker::MARK_STATE`), so its
    // width sets how many sanity GCs pass before a stale mark aliases a current one. 8 bits cost
    // 1/16 of the address space and pushed the global budget to 0.5000305 -- over the hard 1/2
    // limit by 0.0000305, which is the entire size of `IX_BLOCK_DEFRAG`. At 4 bits it costs 1/32
    // and the total is 0.4687805.
    //
    // WHY 4 BITS IS ENOUGH, and what it is coupled to: the epoch now wraps every 15 sanity GCs,
    // so an object last marked exactly 15 epochs ago reads as current. That can never mask a
    // leak, because the only consumer of the comparison is the `SANITY_DEAD_CYCLE_COUNT` check,
    // which asserts at **5** consecutive unmarked cycles -- and an object must pass distances
    // 1..14 to reach 15, so the assert has already fired long before the wrap can alias.
    //
    // THIS COUPLES TWO OTHERWISE UNRELATED CONSTANTS. If that 5-cycle threshold is ever raised
    // above 15, or `MARK_STATE`'s wrap in `SanityPrepare::update_mark_state` stops matching this
    // width, sanity starts lying. Both are commented in kind.
    SANITY_MARK_BITS = (global: true, log_num_of_bits: 2, log_bytes_in_region: crate::util::rc::LOG_MIN_OBJECT_SIZE),
    //Cycle collection colors table
    OBJ_COLOR_TABLE = (global: true, log_num_of_bits: 1, log_bytes_in_region: crate::util::rc::LOG_MIN_OBJECT_SIZE),

    CANDIDATES_STATUS = (global: true, log_num_of_bits: 1, log_bytes_in_region: crate::util::rc::LOG_MIN_OBJECT_SIZE),
    // Strong Reference count
    STRONG_RC_TABLE = (global: true, log_num_of_bits: crate::util::rc::LOG_STRONG_REF_COUNT_BITS, log_bytes_in_region: crate::util::rc::LOG_MIN_OBJECT_SIZE),
    // Sanity: counts how many GC cycles pass before a dead object is collected (2 bits per object)
    SANITY_DEAD_CYCLE_COUNT = (global: true, log_num_of_bits: 2, log_bytes_in_region: crate::util::rc::LOG_MIN_OBJECT_SIZE),

    GRAPH_REPORT_MARK = (global: true, log_num_of_bits: 0, log_bytes_in_region: crate::util::rc::LOG_MIN_OBJECT_SIZE),
);

/// **The global side-metadata budget, enforced.**
///
/// Global specs are laid out sequentially from `GLOBAL_SIDE_METADATA_BASE_ADDRESS`, and
/// `LOCAL_SIDE_METADATA_BASE_ADDRESS` begins immediately after the region reserved for them
/// (`GLOBAL_BASE + 2^LOG_MAX_GLOBAL_SIDE_METADATA_SIZE`). If the global specs ever sum past that
/// point, the last of them silently overlaps LOCAL metadata: two unrelated specs writing the same
/// addresses, which is heap corruption with no error anywhere.
///
/// Until 2026-09-14 nothing checked this. `LOG_MAX_GLOBAL_SIDE_METADATA_SIZE` had exactly one use
/// in the tree -- computing where local metadata starts -- and the constants file still carries a
/// `TODO - we should check this limit somewhere` for the local equivalent.
///
/// This matters now because `OVERFLOW_RC_TABLE` is 1/4 of the address space, the largest global
/// spec in the system: the total goes from 0.234 to 0.484 of a 0.500 budget, leaving 3.1%.
/// Gating the `SANITY_*` / `GRAPH_*` declarations behind their features would free 0.102.
const _: () = assert!(
    LAST_GLOBAL_SIDE_METADATA_SPEC
        .upper_bound_address_for_contiguous()
        .as_usize()
        <= crate::util::metadata::side_metadata::constants::LOCAL_SIDE_METADATA_BASE_ADDRESS
            .as_usize(),
    "global side metadata specs exceed LOG_MAX_GLOBAL_SIDE_METADATA_SIZE and would overlap local \
     side metadata -- remove a spec, narrow one, or cfg-gate the ones whose features are off"
);

// This defines all LOCAL side metadata used by mmtk-core.
define_side_metadata_specs!(
    last_spec_as LAST_LOCAL_SIDE_METADATA_SPEC,
    // Mark pages by (malloc) marksweep
    MALLOC_MS_ACTIVE_PAGE  = (global: false, log_num_of_bits: 3, log_bytes_in_region: crate::util::malloc::library::LOG_BYTES_IN_MALLOC_PAGE as usize),
    // Record objects allocated with some offset
    MS_OFFSET_MALLOC = (global: false, log_num_of_bits: 0, log_bytes_in_region: LOG_MIN_OBJECT_SIZE as usize),
    // Mark lines by immix
    IX_LINE_MARK    = (global: false, log_num_of_bits: 3, log_bytes_in_region: crate::policy::immix::line::Line::LOG_BYTES),
    // Mark blocks by immix
    IX_BLOCK_MARK   = (global: false, log_num_of_bits: 3, log_bytes_in_region: crate::policy::immix::block::Block::LOG_BYTES),
    // Striddle line marks
    RC_STRADDLE_LINES = (global: false, log_num_of_bits: 3, log_bytes_in_region: crate::policy::immix::line::Line::LOG_BYTES),
    // LXR Block logging bits
    IX_BLOCK_LOG   = (global: false, log_num_of_bits: 0, log_bytes_in_region: crate::policy::immix::block::Block::LOG_BYTES),
    NURSERY_PROMOTION_STATE   = (global: false, log_num_of_bits: 3, log_bytes_in_region: crate::policy::immix::block::Block::LOG_BYTES),
    PHASE_EPOCH   = (global: false, log_num_of_bits: 3, log_bytes_in_region: crate::policy::immix::block::Block::LOG_BYTES),
    IX_BLOCK_DEAD_WORDS = (global: false, log_num_of_bits: 5 /* u32 */, log_bytes_in_region: Block::LOG_BYTES),
    // Mark chunks (any plan that uses the chunk map should include this spec in their local sidemetadata specs)
    CHUNK_MARK   = (global: false, log_num_of_bits: 3, log_bytes_in_region: crate::util::heap::chunk_map::Chunk::LOG_BYTES),
    CHUNK_BIN   = (global: false, log_num_of_bits: 3, log_bytes_in_region: crate::util::heap::chunk_map::Chunk::LOG_BYTES),
    CHUNK_LIVE_BLOCKS   = (global: false, log_num_of_bits: 4, log_bytes_in_region: crate::util::heap::chunk_map::Chunk::LOG_BYTES),
    CHUNK_PREV   = (global: false, log_num_of_bits: 6, log_bytes_in_region: crate::util::heap::chunk_map::Chunk::LOG_BYTES),
    CHUNK_NEXT   = (global: false, log_num_of_bits: 6, log_bytes_in_region: crate::util::heap::chunk_map::Chunk::LOG_BYTES),
    // The block is in a mutator allocator's local allocation buffer
    BLOCK_OWNER   = (global: false, log_num_of_bits: 6, log_bytes_in_region: crate::policy::immix::block::Block::LOG_BYTES),
    // The block is being used by the allocator
    BLOCK_IN_USE   = (global: false, log_num_of_bits: 3, log_bytes_in_region: crate::policy::immix::block::Block::LOG_BYTES),
    IX_BLOCK_ALLOC_BITS   = (global: false, log_num_of_bits: 3, log_bytes_in_region: crate::policy::immix::block::Block::LOG_BYTES),
    // Mark blocks by (native mimalloc) marksweep
    MS_BLOCK_MARK   = (global: false, log_num_of_bits: 3, log_bytes_in_region: crate::policy::marksweepspace::native_ms::Block::LOG_BYTES),
    // Next block in list for native mimalloc
    MS_BLOCK_NEXT   = (global: false, log_num_of_bits: LOG_BITS_IN_ADDRESS, log_bytes_in_region: crate::policy::marksweepspace::native_ms::Block::LOG_BYTES),
    // Previous block in list for native mimalloc
    MS_BLOCK_PREV   = (global: false, log_num_of_bits: LOG_BITS_IN_ADDRESS, log_bytes_in_region: crate::policy::marksweepspace::native_ms::Block::LOG_BYTES),
    // Pointer to owning list for blocks for native mimalloc
    MS_BLOCK_LIST   = (global: false, log_num_of_bits: LOG_BITS_IN_ADDRESS, log_bytes_in_region: crate::policy::marksweepspace::native_ms::Block::LOG_BYTES),
    // Size of cells in block for native mimalloc FIXME: do we actually need usize?
    MS_BLOCK_SIZE         = (global: false, log_num_of_bits: LOG_BITS_IN_ADDRESS, log_bytes_in_region: crate::policy::marksweepspace::native_ms::Block::LOG_BYTES),
    // TLS of owning mutator of block for native mimalloc
    MS_BLOCK_TLS    = (global: false, log_num_of_bits: LOG_BITS_IN_ADDRESS, log_bytes_in_region: crate::policy::marksweepspace::native_ms::Block::LOG_BYTES),
    // First cell of free list in block for native mimalloc
    MS_FREE         = (global: false, log_num_of_bits: LOG_BITS_IN_ADDRESS, log_bytes_in_region: crate::policy::marksweepspace::native_ms::Block::LOG_BYTES),
    // The following specs are only used for manual malloc/free
    // First cell of local free list in block for native mimalloc
    MS_LOCAL_FREE   = (global: false, log_num_of_bits: LOG_BITS_IN_ADDRESS, log_bytes_in_region: crate::policy::marksweepspace::native_ms::Block::LOG_BYTES),
    // First cell of thread free list in block for native mimalloc
    MS_THREAD_FREE  = (global: false, log_num_of_bits: LOG_BITS_IN_ADDRESS, log_bytes_in_region: crate::policy::marksweepspace::native_ms::Block::LOG_BYTES),
);

#[cfg(test)]
mod tests {
    // We assert on constants to test if the macro is working properly.
    #![allow(clippy::assertions_on_constants)]

    use super::*;
    #[test]
    fn first_global_spec() {
        define_side_metadata_specs!(last_spec_as LAST_GLOBAL_SPEC, TEST_SPEC = (global: true, log_num_of_bits: 0, log_bytes_in_region: 3),);
        assert!(TEST_SPEC.is_global);
        assert!(TEST_SPEC.offset == GLOBAL_SIDE_METADATA_BASE_OFFSET);
        assert_eq!(TEST_SPEC.log_num_of_bits, 0);
        assert_eq!(TEST_SPEC.log_bytes_in_region, 3);
        assert_eq!(TEST_SPEC, LAST_GLOBAL_SPEC);
    }

    #[test]
    fn first_local_spec() {
        define_side_metadata_specs!(last_spec_as LAST_LOCAL_SPEC, TEST_SPEC = (global: false, log_num_of_bits: 0, log_bytes_in_region: 3),);
        assert!(!TEST_SPEC.is_global);
        assert!(TEST_SPEC.offset == LOCAL_SIDE_METADATA_BASE_OFFSET);
        assert_eq!(TEST_SPEC.log_num_of_bits, 0);
        assert_eq!(TEST_SPEC.log_bytes_in_region, 3);
        assert_eq!(TEST_SPEC, LAST_LOCAL_SPEC);
    }

    #[test]
    fn two_global_specs() {
        define_side_metadata_specs!(
            last_spec_as LAST_GLOBAL_SPEC,
            TEST_SPEC1 = (global: true, log_num_of_bits: 0, log_bytes_in_region: 3),
            TEST_SPEC2 = (global: true, log_num_of_bits: 1, log_bytes_in_region: 4),
        );

        assert!(TEST_SPEC1.is_global);
        assert!(TEST_SPEC1.offset == GLOBAL_SIDE_METADATA_BASE_OFFSET);
        assert_eq!(TEST_SPEC1.log_num_of_bits, 0);
        assert_eq!(TEST_SPEC1.log_bytes_in_region, 3);

        assert!(TEST_SPEC2.is_global);
        assert!(TEST_SPEC2.offset == SideMetadataOffset::layout_after(&TEST_SPEC1));
        assert_eq!(TEST_SPEC2.log_num_of_bits, 1);
        assert_eq!(TEST_SPEC2.log_bytes_in_region, 4);

        assert_eq!(TEST_SPEC2, LAST_GLOBAL_SPEC);
    }

    #[test]
    fn three_global_specs() {
        define_side_metadata_specs!(
            last_spec_as LAST_GLOBAL_SPEC,
            TEST_SPEC1 = (global: true, log_num_of_bits: 0, log_bytes_in_region: 3),
            TEST_SPEC2 = (global: true, log_num_of_bits: 1, log_bytes_in_region: 4),
            TEST_SPEC3 = (global: true, log_num_of_bits: 2, log_bytes_in_region: 5),
        );

        assert!(TEST_SPEC1.is_global);
        assert!(TEST_SPEC1.offset == GLOBAL_SIDE_METADATA_BASE_OFFSET);
        assert_eq!(TEST_SPEC1.log_num_of_bits, 0);
        assert_eq!(TEST_SPEC1.log_bytes_in_region, 3);

        assert!(TEST_SPEC2.is_global);
        assert!(TEST_SPEC2.offset == SideMetadataOffset::layout_after(&TEST_SPEC1));
        assert_eq!(TEST_SPEC2.log_num_of_bits, 1);
        assert_eq!(TEST_SPEC2.log_bytes_in_region, 4);

        assert!(TEST_SPEC3.is_global);
        assert!(TEST_SPEC3.offset == SideMetadataOffset::layout_after(&TEST_SPEC2));
        assert_eq!(TEST_SPEC3.log_num_of_bits, 2);
        assert_eq!(TEST_SPEC3.log_bytes_in_region, 5);

        assert_eq!(TEST_SPEC3, LAST_GLOBAL_SPEC);
    }

    #[test]
    fn both_global_and_local() {
        define_side_metadata_specs!(
            last_spec_as LAST_GLOBAL_SPEC,
            TEST_GSPEC1 = (global: true, log_num_of_bits: 0, log_bytes_in_region: 3),
            TEST_GSPEC2 = (global: true, log_num_of_bits: 1, log_bytes_in_region: 4),
        );
        define_side_metadata_specs!(
            last_spec_as LAST_LOCAL_SPEC,
            TEST_LSPEC1 = (global: false, log_num_of_bits: 2, log_bytes_in_region: 5),
            TEST_LSPEC2 = (global: false, log_num_of_bits: 3, log_bytes_in_region: 6),
        );

        assert!(TEST_GSPEC1.is_global);
        assert!(TEST_GSPEC1.offset == GLOBAL_SIDE_METADATA_BASE_OFFSET);
        assert_eq!(TEST_GSPEC1.log_num_of_bits, 0);
        assert_eq!(TEST_GSPEC1.log_bytes_in_region, 3);

        assert!(TEST_GSPEC2.is_global);
        assert!(TEST_GSPEC2.offset == SideMetadataOffset::layout_after(&TEST_GSPEC1));
        assert_eq!(TEST_GSPEC2.log_num_of_bits, 1);
        assert_eq!(TEST_GSPEC2.log_bytes_in_region, 4);

        assert_eq!(TEST_GSPEC2, LAST_GLOBAL_SPEC);

        assert!(!TEST_LSPEC1.is_global);
        assert!(TEST_LSPEC1.offset == LOCAL_SIDE_METADATA_BASE_OFFSET);
        assert_eq!(TEST_LSPEC1.log_num_of_bits, 2);
        assert_eq!(TEST_LSPEC1.log_bytes_in_region, 5);

        assert!(!TEST_LSPEC2.is_global);
        assert!(TEST_LSPEC2.offset == SideMetadataOffset::layout_after(&TEST_LSPEC1));
        assert_eq!(TEST_LSPEC2.log_num_of_bits, 3);
        assert_eq!(TEST_LSPEC2.log_bytes_in_region, 6);

        assert_eq!(TEST_LSPEC2, LAST_LOCAL_SPEC);
    }
}
