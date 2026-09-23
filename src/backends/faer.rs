// Each release needs distinct trait implementations from the shared sources.
#![allow(clippy::duplicate_mod)]

#[cfg(feature = "faer_v0_22")]
mod v0_22;

#[cfg(feature = "faer_v0_23")]
mod v0_23;

#[cfg(feature = "faer_v0_24")]
mod v0_24;
