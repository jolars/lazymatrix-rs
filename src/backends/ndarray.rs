// Each release needs distinct trait implementations from the shared sources.
#![allow(clippy::duplicate_mod)]

#[cfg(feature = "ndarray_v0_15")]
mod v0_15;

#[cfg(feature = "ndarray_v0_16")]
mod v0_16;

#[cfg(feature = "ndarray_v0_17")]
mod v0_17;
