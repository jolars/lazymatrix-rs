// Each release needs distinct trait implementations from the shared sources.
#![allow(clippy::duplicate_mod)]

#[cfg(feature = "nalgebra_v0_32")]
mod v0_32;

#[cfg(feature = "nalgebra_v0_33")]
mod v0_33;

#[cfg(feature = "nalgebra_v0_34")]
mod v0_34;

#[cfg(feature = "nalgebra_v0_35")]
mod v0_35;
