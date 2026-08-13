use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, AtomicUsize};

use crate::util::linear_scan::Region;
use crate::util::{metadata::side_metadata::address_to_meta_address, Address};
use crate::{
    policy::immix::{block::Block, line::Line},
    util::{metadata::side_metadata::SideMetadataSpec, ObjectReference},
    vm::*,
};
use atomic::Ordering;

//Strong reference count constants
pub const LOG_STRONG_REF_COUNT_BITS: usize = 2; 
pub const STRONG_REF_COUNT_BITS: u8 = 1 << LOG_STRONG_REF_COUNT_BITS;
pub const STRONG_REF_COUNT_MASK: u8 = (((1u16 << STRONG_REF_COUNT_BITS) - 1) & 0xff) as u8;
pub const MAX_STRONG_REF_COUNT: u8 = STRONG_REF_COUNT_MASK;


//size of rc field defintion
#[cfg(feature = "lxr_rc_bits_2")]
pub type RcBits = u8;
#[cfg(feature = "lxr_rc_bits_4")]
pub type RcBits = u8;
#[cfg(feature = "lxr_rc_bits_8")]
pub type RcBits = u8;
#[cfg(feature = "lxr_rc_bits_16")]
pub type RcBits = u16;
#[cfg(feature = "lxr_rc_bits_32")]
pub type RcBits = u32;
#[cfg(feature = "lxr_rc_bits_64")]
pub type RcBits = u64;

#[cfg(not(any(
    feature="lxr_rc_bits_2",
    feature="lxr_rc_bits_4",
    feature="lxr_rc_bits_8",
    feature="lxr_rc_bits_16",
    feature="lxr_rc_bits_32",
    feature="lxr_rc_bits_64",
)))]
pub type RcBits = u8;

//original code:

#[cfg(feature="lxr_rc_bits_2")]
pub const LOG_REF_COUNT_BITS: usize = 1;
#[cfg(feature="lxr_rc_bits_4")]
pub const LOG_REF_COUNT_BITS: usize = 2;
#[cfg(feature="lxr_rc_bits_8")]
pub const LOG_REF_COUNT_BITS: usize = 3;
#[cfg(feature="lxr_rc_bits_16")]
pub const LOG_REF_COUNT_BITS: usize = 4;
#[cfg(feature="lxr_rc_bits_32")]
pub const LOG_REF_COUNT_BITS: usize = 5;
#[cfg(feature="lxr_rc_bits_64")]
pub const LOG_REF_COUNT_BITS: usize = 6;

// default
#[cfg(not(any(
    feature="lxr_rc_bits_2",
    feature="lxr_rc_bits_4",
    feature="lxr_rc_bits_8",
    feature="lxr_rc_bits_16",
    feature="lxr_rc_bits_32",
    feature="lxr_rc_bits_64",
)))]
pub const LOG_REF_COUNT_BITS: usize = 3; // default to 8 bits



pub const REF_COUNT_BITS: usize = 1 << LOG_REF_COUNT_BITS;

pub const REF_COUNT_MASK: RcBits = 
    ((1u128 << REF_COUNT_BITS) - 1) as RcBits;

pub const MAX_REF_COUNT: RcBits = REF_COUNT_MASK;


pub const LOG_MIN_OBJECT_SIZE: usize = crate::util::constants::LOG_MIN_OBJECT_SIZE as _;
pub const MIN_OBJECT_SIZE: usize = 1 << LOG_MIN_OBJECT_SIZE;

pub const RC_STRADDLE_LINES: SideMetadataSpec =
    crate::util::metadata::side_metadata::spec_defs::RC_STRADDLE_LINES;

pub const RC_TABLE: SideMetadataSpec = crate::util::metadata::side_metadata::spec_defs::RC_TABLE;

pub const STRONG_RC_TABLE: SideMetadataSpec = crate::util::metadata::side_metadata::spec_defs::STRONG_RC_TABLE;
pub const OBJ_COLOR_TABLE: SideMetadataSpec = crate::util::metadata::side_metadata::spec_defs::OBJ_COLOR_TABLE;
pub const CANDIDATES_STATUS: SideMetadataSpec = crate::util::metadata::side_metadata::spec_defs::CANDIDATES_STATUS;
#[cfg(feature = "sanity")]
pub const SANITY_DEAD_CYCLE_COUNT: SideMetadataSpec = crate::util::metadata::side_metadata::spec_defs::SANITY_DEAD_CYCLE_COUNT;
#[cfg(feature = "graph_project")]
pub const GRAPH_REPORT_MARK : SideMetadataSpec = crate::util::metadata::side_metadata::spec_defs::GRAPH_REPORT_MARK;
pub const BLACK_OUT_OF_STACK: u8 = 0;
pub const BLACK_IN_STACK: u8 = 1;
pub const GREY: u8 = 2;
pub const WHITE: u8 = 3;

pub const RC_NURSERY_OR_DEAD: usize = 0;
/// Transient RC value used only during death processing. Never observed outside that path.
pub const RC_DEATH_TRANSIENT: usize = 1;
/// RC value when an object has exactly 1 real reference (due to the +1 bias).
/// This is the threshold at which a decrement triggers death processing.
pub const RC_DEATH_THRESHOLD: usize = 2;

/// `strong_rc_dec` returned `Ok(STRONG_RC_LAST_BEFORE_ZERO)`: strong RC was 1, now 0.
/// The object just became a cycle candidate.
pub const STRONG_RC_LAST_BEFORE_ZERO: u8 = 1;
/// `strong_rc_dec` returned `Err(STRONG_RC_ALREADY_ZERO)`: strong RC was already 0.
pub const STRONG_RC_ALREADY_ZERO: u8 = 0;



static INC_BUFFER_SIZE: AtomicUsize = AtomicUsize::new(0);

static TOTAL_INCS_PACKETS: AtomicU32 = AtomicU32::new(0);

static TOTAL_INCS: AtomicU32 = AtomicU32::new(0);
static ROOT_INCS: AtomicU32 = AtomicU32::new(0);
static MATURE_INCS: AtomicU32 = AtomicU32::new(0);
static NURSERY_INCS: AtomicU32 = AtomicU32::new(0);
static FAST_NURSERY_INCS: AtomicU32 = AtomicU32::new(0);
static LOS_INCS: AtomicU32 = AtomicU32::new(0);

static PROMOTED_OBJECTS: AtomicU32 = AtomicU32::new(0);
static PROMOTED_SCALARS: [AtomicU32; 3] = [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)];
static PROMOTED_PRIM_ARRAYS: [AtomicU32; 3] =
    [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)];
static PROMOTED_OBJECT_ARRAYS: [AtomicU32; 3] =
    [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)];

#[repr(transparent)]
#[derive(Debug, Copy)]
pub struct RefCountHelper<VM: VMBinding>(PhantomData<VM>);

impl<VM: VMBinding> RefCountHelper<VM> {
    pub const NEW: Self = Self(PhantomData);

    pub fn inc_buffer_size(&self) -> usize {
        INC_BUFFER_SIZE.load(Ordering::Relaxed)
    }

    pub fn increase_inc_buffer_size(&self, delta: usize) {
        if cfg!(feature = "lxr_precise_incs_counter") {
            INC_BUFFER_SIZE.fetch_add(delta, Ordering::Relaxed);
        } else {
            INC_BUFFER_SIZE.store(
                INC_BUFFER_SIZE
                    .load(Ordering::Relaxed)
                    .saturating_add(delta),
                Ordering::Relaxed,
            );
        }
    }

    pub fn reset_inc_buffer_size(&self) {
        INC_BUFFER_SIZE.store(0, Ordering::Relaxed)
    }

    //Eyal change: all u16 was originaly u8
    pub fn fetch_update(
        &self,
        o: ObjectReference,
        f: impl FnMut(RcBits) -> Option<RcBits>,
    ) -> Result<RcBits, RcBits> {
        RC_TABLE.fetch_update_atomic(o.to_raw_address(), Ordering::Relaxed, Ordering::Relaxed, f)
    }

    pub fn is_stuck(&self, o: ObjectReference) -> bool {
        self.count(o) == MAX_REF_COUNT
    }

    //Eyal change: all u16 was originaly u8
    pub fn stick(&self, o: ObjectReference) -> Result<RcBits, RcBits> {
        self.fetch_update(o, |x| {
            debug_assert!(x <= MAX_REF_COUNT);
            if x == MAX_REF_COUNT {
                None
            } else {
                Some(MAX_REF_COUNT)
            }
        })
    }

    //Eyal change: all u16 was originaly u8
    pub fn inc(&self, o: ObjectReference) -> Result<RcBits, RcBits> {
        self.fetch_update(o, |x| {
            debug_assert!(x <= MAX_REF_COUNT);
            //Eyal added this assert to make sure an object doesn't get stuck in debug mode
            //debug_assert!(x < MAX_REF_COUNT - 1);
            if x == MAX_REF_COUNT {
                None
            } else {
                Some(x + 1)
            }
        })
    }
    //Eyal change: all u16 was originaly u8
    pub fn dec(&self, o: ObjectReference) -> Result<RcBits, RcBits> {
        self.fetch_update(o, |x| {
            debug_assert!(x <= MAX_REF_COUNT);
            if x == 0 || x == MAX_REF_COUNT
            /* sticky */
            {
                None
            } else {
                Some(x - 1)
            }
        })
    }

    pub fn dec_unconditionally(&self, o: ObjectReference) -> RcBits {
        RC_TABLE.fetch_sub_atomic(o.to_raw_address(), 1,  Ordering::Relaxed)
    }

    //Eyal change: all u16 was originaly u8
    pub fn set(&self, o: ObjectReference, count: RcBits) {
        RC_TABLE.store_atomic(o.to_raw_address(), count, Ordering::Relaxed)
    }
    //Eyal change: all u16 was originaly u8
    pub fn set_relaxed(&self, o: ObjectReference, count: RcBits) {
        unsafe { RC_TABLE.store(o.to_raw_address(), count) }
    }

    //Eyal change: all u16 was originaly u8
    pub fn count(&self, o: ObjectReference) -> RcBits {
        RC_TABLE.load_atomic(o.to_raw_address(), Ordering::Relaxed)
    }

    pub fn prefetch_read(&self, o: ObjectReference) {
        RC_TABLE.prefetch_read(o.to_raw_address())
    }

    pub fn prefetch_write(&self, o: ObjectReference) {
        RC_TABLE.prefetch_write(o.to_raw_address())
    }

    //Eyal changed this function
    //Originaly was:
    // pub fn object_or_line_is_dead(&self, o: ObjectReference) -> bool {
    //     RC_TABLE.load_byte(o.to_raw_address()) == 0
    // }
    pub fn object_or_line_is_dead(&self, o: ObjectReference) -> bool {
        //RC_TABLE.load_byte(o.to_raw_address()) == 0
        RC_TABLE.load_atomic::<RcBits>(o.to_raw_address(), Ordering::Relaxed) == 0
    }

    pub fn rc_table_range<UInt: Sized>(&self, b: Block) -> &'static [UInt] {
        debug_assert!({
            let log_bits_in_uint: usize =
                (std::mem::size_of::<UInt>() << 3).trailing_zeros() as usize;
            Block::LOG_BYTES - super::rc::LOG_MIN_OBJECT_SIZE + super::rc::LOG_REF_COUNT_BITS
                >= log_bits_in_uint
        });
        let start = address_to_meta_address(&super::rc::RC_TABLE, b.start()).to_ptr::<UInt>();
        let limit = address_to_meta_address(&super::rc::RC_TABLE, b.end()).to_ptr::<UInt>();
        let rc_table = unsafe { std::slice::from_raw_parts(start, limit.offset_from(start) as _) };
        rc_table
    }

    #[allow(unused)]
    //Eyal change: all u16 was originaly u8
    pub fn is_dead(&self, o: ObjectReference) -> bool {
        let v: RcBits = RC_TABLE.load_atomic(o.to_raw_address(), Ordering::Relaxed);
        v == 0
    }
//Eyal change: all u16 was originaly u8
    pub fn is_dead_or_stuck(&self, o: ObjectReference) -> bool {
        let v: RcBits = RC_TABLE.load_atomic(o.to_raw_address(), Ordering::Relaxed);
        v == 0 || v == MAX_REF_COUNT
    }

    pub fn is_straddle_line(&self, line: Line) -> bool {
        let v: u8 = unsafe { RC_STRADDLE_LINES.load::<u8>(line.start()) };
        v != 0
    }

    pub fn address_is_in_straddle_line(&self, a: Address) -> bool {
        let line = Line::from(Line::align(a));
        self.count(a.to_object_reference::<VM>()) != 0 && self.is_straddle_line(line)
    }

    fn mark_straddle_object_with_size(&self, o: ObjectReference, size: usize) {
        debug_assert!(!crate::args::BLOCK_ONLY);
        debug_assert!(size > Line::BYTES);
        let start_line = Line::containing::<VM>(o).next();
        let end_line = Line::from(Line::align(o.to_raw_address() + size));
        let mut line = start_line;
        while line != end_line {
            unsafe { RC_STRADDLE_LINES.store(line.start(), 1u8) };
            self.set_relaxed(line.start().to_object_reference::<VM>(), 1);
            line = line.next();
        }
    }

    pub fn mark_straddle_object(&self, o: ObjectReference) {
        let size = VM::VMObjectModel::get_current_size(o);
        self.mark_straddle_object_with_size(o, size)
    }

    pub fn unmark_straddle_object(&self, o: ObjectReference) {
        debug_assert!(!crate::args::BLOCK_ONLY);
        // debug_assert!(crate::args::RC_NURSERY_EVACUATION);
        let size = VM::VMObjectModel::get_current_size(o);
        if size > Line::BYTES {
            let start_line = Line::containing::<VM>(o).next();
            let end_line = Line::from(Line::align(o.to_raw_address() + size));
            let mut line = start_line;
            while line != end_line {
                self.set_relaxed(line.start().to_object_reference::<VM>(), 0);
                // std::sync::atomic::fence(Ordering::Relaxed);
                unsafe { RC_STRADDLE_LINES.store(line.start(), 0u8) };
                // std::sync::atomic::fence(Ordering::Relaxed);
                line = line.next();
            }
        }
    }

    pub fn assert_zero_ref_count(&self, o: ObjectReference) {
        let size = VM::VMObjectModel::get_current_size(o);
        for i in (0..size).step_by(MIN_OBJECT_SIZE) {
            let a = o.to_raw_address() + i;
            assert_eq!(0, self.count(a.to_object_reference::<VM>()));
        }
    }

    pub fn promote(&self, o: ObjectReference) {
        o.log_start_address::<VM>();
        let size = o.get_size::<VM>();
        if size > Line::BYTES {
            self.mark_straddle_object_with_size(o, size);
        }
    }

    pub fn promote_with_size(&self, o: ObjectReference, size: usize) {
        o.log_start_address::<VM>();
        if size > Line::BYTES {
            self.mark_straddle_object_with_size(o, size);
        }
    }

    //Eyal added this func
    pub fn strong_rc_inc(&self, o: ObjectReference) -> Result<u8, u8> {
        let f = |x: u8| -> Option<u8> {
            if x == MAX_STRONG_REF_COUNT {
                None
            } else {
                Some(x + 1)
            }
        };
        STRONG_RC_TABLE.fetch_update_atomic(o.to_raw_address(), Ordering::Relaxed, Ordering::Relaxed, f)
        //STRONG_RC_TABLE.fetch_add_atomic::<u8>(o.to_raw_address(), 1, Ordering::Relaxed)
    }

    pub fn strong_rc_dec(&self, o: ObjectReference) -> Result<u8, u8> {
        let f = |x: u8| -> Option<u8> {
            if x == 0 {
                None
            } else {
                Some(x - 1)
            }
        };
        STRONG_RC_TABLE.fetch_update_atomic(o.to_raw_address(), Ordering::Relaxed, Ordering::Relaxed, f)
        //STRONG_RC_TABLE.fetch_sub_atomic::<u8>(o.to_raw_address(), 1, Ordering::Relaxed)
    }
}

impl<VM: VMBinding> Clone for RefCountHelper<VM> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}
