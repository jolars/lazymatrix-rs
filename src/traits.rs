//! Backend-agnostic trait surface for [`LazyMatrix`](crate::LazyMatrix).
//!
//! The traits split into three groups:
//!
//! * [`Scalar`] — the numeric element type, a blanket-implemented bundle of
//!   `num-traits` bounds.
//! * [`MatrixShape`], [`MatVec`] / [`MatTransposeVec`], and their reusable-output
//!   [`MatVecInto`] / [`MatTransposeVecInto`] counterparts — the matrix-free
//!   linear-operator interface, implemented both by concrete backend matrices
//!   and by [`LazyMatrix`](crate::LazyMatrix) itself.
//! * Solver-facing vector algebra ([`DotProduct`], [`L2Norm`],
//!   [`ScaledAddAssign`], and [`ScaleAssign`]).
//! * The five normalization-specific *vector* traits ([`ElemDivAssign`], [`DotSlice`],
//!   [`SubScalarAssign`], [`SumEntries`], [`ScaledSubSlice`]) — the elementwise
//!   primitives that fold the lazy normalization into a backend vector. They are
//!   phrased as a backend vector against a coefficient slice `&[F]`, which is
//!   exactly the shape the centering/scaling math needs.
//! * [`ColumnStats`] — column statistics computed directly over a (possibly
//!   sparse) backend matrix, used by the `normalized` constructor.
//! * [`VectorView`] / [`VectorViewMut`] — storage-independent borrowed vector
//!   access, including strided backend views.
//! * [`RawColumn`] / [`RawColumns`] and [`LogicalColumn`] / [`Columns`] — the
//!   backend and normalized sides of storage-independent column access.
//! * [`SparseColumns`] — the stronger borrowed access capability for
//!   contiguous sparse columns.

/// Numeric scalar element type.
///
/// This is a blanket-implemented alias for the bound bundle the crate relies on,
/// so any floating-point type that satisfies the underlying `num-traits` bounds
/// (notably `f32` and `f64`) is a `Scalar` automatically.
pub trait Scalar:
    num_traits::Float + num_traits::FromPrimitive + std::iter::Sum + std::fmt::Debug + Default + 'static
{
}

mod columns;
mod operator;
mod stats;
mod vectors;

pub use crate::normalization::{Centering, Normalization, Scaling};
pub use columns::{Columns, LogicalColumn, RawColumn, RawColumns, SparseColumns};
pub use operator::{MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixShape};
pub use stats::ColumnStats;
pub use vectors::{
    DotProduct, DotSlice, ElemDivAssign, L2Norm, ScaleAssign, ScaledAddAssign, ScaledSubSlice,
    SubScalarAssign, SumEntries, VectorView, VectorViewMut,
};
