pub(super) mod barrier;
pub(super) mod gc_work;
pub(super) mod global;
pub(super) mod mutator;

pub use self::global::Immix;
pub use self::global::IMMIX_CONSTRAINTS;

use bytemuck::NoUninit;

#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Copy, Clone, NoUninit)]
pub enum Pause {
    Full = 1,
    FullDefrag,
    RefCount,
    InitialMark,
    FinalMark,
    /// The reference-counting pipeline, run stop-the-world: incs, then decrements, then the
    /// concurrent tail, all inside the pause.  NOT a trace -- nothing here marks, so nothing may
    /// depend on a mark bit.  Distinct from `Full`, which is the mark-and-sweep backup pause, so
    /// that the ~44 `== Pause::Full` tests scattered through the collector keep answering as they
    /// did: every one of them assumes a full-heap trace happened, and for this pause the correct
    /// answer is the `RefCount` one, which is what a distinct variant gives by default.
    /// Appended last: `#[repr(u8)]`, so existing discriminants do not move.
    FullRC,
}

unsafe impl bytemuck::ZeroableInOption for Pause {}
unsafe impl bytemuck::PodInOption for Pause {}

// pub static ACTIVE_BARRIER: BarrierSelector = BarrierSelector::FieldBarrier;
