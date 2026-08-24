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
    /// Unconditional decrement with no atomic read-modify-write.
    ///
    /// The exclusive-access counterpart of [`Self::dec_unconditionally`].  Like it, this applies
    /// no guard at all — it steps `MAX_REF_COUNT` down to `MAX_REF_COUNT - 1`, which is exactly
    /// what [`Self::dec_exclusive`] refuses to do, and wraps on 0 the same way
    /// `fetch_sub_atomic` does.  It exists for the overflow cache's fallback path, where the RC
    /// table reads `MAX_REF_COUNT` but the map holds no entry, so the true count is exactly
    /// `MAX_REF_COUNT` and must be stepped down.
    ///
    /// # Safety
    ///
    /// As [`Self::dec_exclusive`]: the caller must be the only thread writing this entry — and,
    /// if `RC_TABLE` is sub-byte, any entry in the same byte.
    pub unsafe fn dec_unconditionally_exclusive(&self, o: ObjectReference) -> RcBits {
        let addr = o.to_raw_address();
        let old: RcBits = RC_TABLE.load_atomic(addr, Ordering::Relaxed);
        unsafe {
            RC_TABLE.store_atomic_exclusive(addr, old.wrapping_sub(1), Ordering::Relaxed)
        };
        old
    }

    /// Increment the RC of `o` with no atomic read-modify-write.
    ///
    /// Same semantics as [`Self::inc`] — `MAX_REF_COUNT` is sticky and is not incremented — but
    /// it returns the **previous** value directly instead of a `Result`, matching
    /// [`Self::dec_unconditionally`].  A returned `MAX_REF_COUNT` means "no change was made".
    /// Unlike [`Self::dec_exclusive`] there is no zero guard: incrementing from 0 is how a
    /// nursery or recycled entry becomes live, and is expected.
    ///
    /// # Safety
    ///
    /// The caller must be the only thread writing this entry for the duration of the call.  See
    /// [`Self::dec_exclusive`] for the full argument and for why relaxed atomics are used rather
    /// than a non-atomic store.
    ///
    /// The failure mode differs from the decrement's, and is worth knowing when auditing a call
    /// site.  This function never stores 0 (`old + 1` is at least 1), so it cannot itself hand a
    /// reader the spurious zero that [`Self::dec_exclusive`] can.  What a violated contract costs
    /// here is a **lost increment**, leaving a live object's count too low and exposing it to a
    /// premature free — the same end result, reached one step later.
    pub unsafe fn inc_exclusive(&self, o: ObjectReference) -> RcBits {
        let addr = o.to_raw_address();
        let old: RcBits = RC_TABLE.load_atomic(addr, Ordering::Relaxed);
        // Mirrors the guard in `inc`: MAX_REF_COUNT is sticky.  Once an entry saturates, the true
        // count lives in the overflow cache and must not be disturbed from here.
        if old == MAX_REF_COUNT {
            return old;
        }
        unsafe { RC_TABLE.store_atomic_exclusive(addr, old + 1, Ordering::Relaxed) };
        old
    }

    /// Decrement the RC of `o` with no atomic read-modify-write.
    ///
    /// Same semantics as [`Self::dec`] — 0 is already dead and `MAX_REF_COUNT` is sticky, so
    /// neither is decremented — but it returns the **previous** value directly instead of a
    /// `Result`, matching [`Self::dec_unconditionally`].  A returned `0` or `MAX_REF_COUNT`
    /// therefore means "no change was made".
    ///
    /// # Why this is faster
    ///
    /// [`Self::dec`] goes through `fetch_update_atomic`, i.e. a **CAS loop**.  This does one
    /// relaxed load and one relaxed store, which on x86-64 are both plain `mov`s — no `lock`
    /// prefix, no fence.  That is the entire point of the function.
    ///
    /// # Safety
    ///
    /// **The caller must be the only thread writing this entry for the duration of the call.**
    /// The load and store are individually atomic, but together they are not: a concurrent
    /// writer's update lands between them and is silently lost.  Because a recycled or nursery
    /// entry reads `RC_NURSERY_OR_DEAD` (0), the value most likely to be resurrected by such a
    /// lost update is `0` — publishing a zero RC for a live object.
    ///
    /// Concurrent *readers* are explicitly fine, and that is why the accesses stay relaxed
    /// atomics rather than the non-atomic `SideMetadataSpec::store` behind [`Self::set_relaxed`].
    /// The generated code is the same, but this stays inside the memory model, so the compiler
    /// may not invent, split, merge or vectorise the store.  That matters because the readers are
    /// **wider than one entry** and test for zero: `RCArray::is_dead` (`line.rs`) loads a whole
    /// line's worth of RC entries as one integer, and `Block::rc_dead` (`block.rs`) scans the
    /// block's RC table as `u128`.  A store the compiler was free to tear or synthesise could
    /// make a live line read as empty, and `rc_get_next_available_lines` would hand it to an
    /// allocator.
    ///
    /// Intended caller: the cycle collector, which is a single work packet
    /// (`plan/lxr/gc_work.rs`) and therefore the only mutator of these tables while it runs.
    ///
    /// **The contract widens if `RC_TABLE` ever becomes sub-byte.**  At `lxr_rc_bits_8` and wider
    /// the entry is byte-aligned, `store_atomic_exclusive` is a single plain store, and "no other
    /// writer of this entry" is the whole requirement.  At `lxr_rc_bits_4` or `lxr_rc_bits_2` two
    /// or four objects share a byte, the store becomes a read-merge-write of that byte, and the
    /// requirement strengthens to **no other writer of any entry in the same byte**.  Note the
    /// failure mode there is not merely a lost count: dropping a *neighbour's* increment leaves
    /// that neighbour reading 0, which is exactly the spurious zero the wide readers above turn
    /// into a recycled live line.  The cycle collector satisfies the wider contract as written
    /// today, but a 4-bit build makes the margin much thinner.
    pub unsafe fn dec_exclusive(&self, o: ObjectReference) -> RcBits {
        let addr = o.to_raw_address();
        let old: RcBits = RC_TABLE.load_atomic(addr, Ordering::Relaxed);
        // Mirrors the guard in `dec`: 0 is already dead, MAX_REF_COUNT is sticky (it means the
        // true count lives in the overflow cache), so neither may be decremented here.
        if old == 0 || old == MAX_REF_COUNT {
            return old;
        }
        unsafe { RC_TABLE.store_atomic_exclusive(addr, old - 1, Ordering::Relaxed) };
        old
    }

    /// Increment the strong RC of `o` with no atomic read-modify-write.
    ///
    /// The exclusive-access counterpart of [`Self::strong_rc_inc`], returning the **previous**
    /// value instead of a `Result`.  A returned `MAX_STRONG_REF_COUNT` means no change was made.
    ///
    /// # Safety
    ///
    /// As [`Self::inc_exclusive`]: the caller must be the only thread writing this entry for the
    /// duration of the call.
    ///
    /// **The contract here is stronger than for [`Self::inc_exclusive`].**  `STRONG_RC_TABLE` is
    /// 4 bits per entry (`LOG_STRONG_REF_COUNT_BITS = 2`), so **two adjacent objects share a
    /// byte**, and one nibble cannot be written without rewriting the other object's nibble along
    /// with it.  `SideMetadataSpec::store_atomic_exclusive` does that read-merge-write with no
    /// CAS, so a concurrent write to the *neighbouring* object's strong count would be silently
    /// lost.  The caller must therefore guarantee no other thread writes **any entry in the same
    /// byte** — in practice, that nothing else writes `STRONG_RC_TABLE` at all.
    ///
    /// That is what this saves: `SideMetadataSpec::store_atomic` would take its `bits_num_log < 3`
    /// path, a CAS loop on the containing byte, making [`Self::strong_rc_inc`] (a CAS alone)
    /// *cheaper* than a load-plus-CAS here.  With the exclusive store it is two plain `mov`s.
    ///
    /// In practice the contract holds for the cycle collector: as of the `decs -> sweep -> cc`
    /// reorder the decrement phase (`plan/lxr/rc.rs`, `process_decs`) has fully drained before the
    /// collector starts, the mutator barrier touches only `OBJ_COLOR_TABLE` and `satb_map`, and
    /// the remaining `STRONG_RC_TABLE` sites are either commented-out asserts or `sanity`-gated.
    /// **Re-check that if the phase order changes again, or if decrement work is ever allowed to
    /// overlap cycle collection.**
    ///
    /// Unlike `RC_TABLE`, `STRONG_RC_TABLE` is not consulted by any wide zero-test — the
    /// allocator's hole finder (`RCArray::is_dead`, `Block::rc_dead`) reads `RC_TABLE` only — so
    /// the spurious-zero hazard described on [`Self::dec_exclusive`] does not apply to this table.
    pub unsafe fn strong_rc_inc_exclusive(&self, o: ObjectReference) -> u8 {
        let addr = o.to_raw_address();
        let old: u8 = STRONG_RC_TABLE.load_atomic(addr, Ordering::Relaxed);
        if old == MAX_STRONG_REF_COUNT {
            return old;
        }
        unsafe { STRONG_RC_TABLE.store_atomic_exclusive(addr, old + 1, Ordering::Relaxed) };
        old
    }

    /// Decrement the strong RC of `o` with no atomic read-modify-write.
    ///
    /// The exclusive-access counterpart of [`Self::strong_rc_dec`], returning the **previous**
    /// value instead of a `Result`.  A returned `0` means no change was made.
    ///
    /// Note the guard is `old == 0` only, matching `strong_rc_dec`.  `MAX_STRONG_REF_COUNT` is
    /// **not** treated as sticky on the way down even though `strong_rc_inc` saturates at it —
    /// that asymmetry is pre-existing and is reproduced here deliberately so this function and
    /// `strong_rc_dec` cannot disagree.  If it is wrong, it is wrong in both and should be fixed
    /// in both.
    ///
    /// # Safety
    ///
    /// Identical to [`Self::strong_rc_inc_exclusive`], and note that its contract is the
    /// **stronger** one: because `STRONG_RC_TABLE` packs two objects per byte, the caller must be
    /// the only thread writing the whole table, not just this entry.  See that function for the
    /// full argument and for why the non-atomic accesses are the point rather than an oversight.
    pub unsafe fn strong_rc_dec_exclusive(&self, o: ObjectReference) -> u8 {
        let addr = o.to_raw_address();
        let old: u8 = STRONG_RC_TABLE.load_atomic(addr, Ordering::Relaxed);
        if old == 0 {
            return old;
        }
        unsafe { STRONG_RC_TABLE.store_atomic_exclusive(addr, old - 1, Ordering::Relaxed) };
        old
    }
}

impl<VM: VMBinding> Clone for RefCountHelper<VM> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}
