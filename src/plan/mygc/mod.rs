//! Plan: nogc (allocation-only)

pub(super) mod global;
pub(super) mod mutator;

pub use self::global::MyGC;
pub use self::global::NOGC_CONSTRAINTS;
