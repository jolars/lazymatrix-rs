//! Lazy normalized design matrices.
//!
//! A [`LazyMatrix`] wraps an underlying (typically sparse) matrix `X` together
//! with optional column **centers** `c` and column **scales** `s`, and presents
//! the *normalized* matrix
//!
//! ```text
//! X̃ = (X − 1cᵀ) S⁻¹,   S = diag(s)
//! ```
//!
//! as a linear operator — **without ever materializing `X − 1cᵀ`**. Centering a
//! sparse matrix turns its structural zeros into nonzeros, destroying sparsity;
//! factoring the normalization into the matrix–vector products avoids that:
//!
//! ```text
//! X̃ v  = X (S⁻¹ v) − 1 · (cᵀ S⁻¹ v)
//! X̃ᵀ u = S⁻¹ (Xᵀ u − c · Σu)
//! ```
//!
//! Both centering and scaling are independently optional, giving the four
//! combinations handled by the `if let Some` branches in the operator impls.
//!
//! # Backends
//!
//! The core is generic over the backend matrix `M` and scalar `F` and pulls in
//! no linear-algebra dependency by itself. Concrete implementations are provided
//! behind feature flags:
//!
//! * `faer` — [`faer::Mat`] and [`faer::sparse::SparseColMat`] over
//!   [`faer::Col`].
//! * `nalgebra` — [`nalgebra::DMatrix`] and [`nalgebra_sparse::CscMatrix`] over
//!   [`nalgebra::DVector`].
//! * `parallel` — parallel column statistics through Rayon and parallel Faer
//!   sparse matrix–vector products when the `faer` feature is also enabled.
//!
//! Any type implementing the [`traits`] surface (a dense matrix, say) works too.
//!
//! # Example
//!
//! ```ignore
//! use lazymatrix::{LazyMatrix, MatVec, Normalization, Centering, Scaling};
//!
//! // `x` is some backend matrix implementing `MatVec`, `MatTransposeVec`,
//! // `ColumnStats`, `MatrixShape`; `v` a backend vector.
//! let spec = Normalization::new(Centering::Mean, Scaling::Sd);
//! let lazy = LazyMatrix::new(x, spec);
//! let y = lazy.matvec(&v); // == ((X − 1cᵀ)S⁻¹) v, sparsity preserved
//! ```

mod backends;
mod column;
mod matrix;
mod normalization;
pub mod traits;

pub use column::{LazyColumn, LazySparseColumn, SparseColumnRef};
pub use matrix::LazyMatrix;
pub use normalization::{Centering, Normalization, Scaling};
pub use traits::{
    ColumnStats, Columns, DotProduct, DotSlice, ElemDivAssign, L2Norm, LogicalColumn,
    MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixShape, RawColumn, RawColumns,
    Scalar, ScaleAssign, ScaledAddAssign, ScaledSubSlice, SparseColumns, SubScalarAssign,
    SumEntries, VectorView, VectorViewMut,
};
