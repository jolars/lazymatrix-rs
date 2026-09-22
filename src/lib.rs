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
//! * `faer` — `faer::Mat` and `faer::sparse::SparseColMat` over `faer::Col`.
//! * `nalgebra` — `nalgebra::DMatrix` and `nalgebra_sparse::CscMatrix` over
//!   `nalgebra::DVector`.
//! * `ndarray` — `ndarray::Array2` and borrowed, strided matrix views over
//!   `ndarray::Array1`.
//! * `parallel` — parallel column statistics through Rayon; also enables the
//!   selected faer release's Rayon support.
//!
//! Unversioned features select the newest supported release. Use a versioned
//! feature to stay on a particular release line:
//!
//! | Backend | Versioned features | Unversioned feature selects |
//! | --- | --- | --- |
//! | faer | `faer_v0_22`, `faer_v0_23`, `faer_v0_24` | 0.24 |
//! | nalgebra | `nalgebra_v0_32`, `nalgebra_v0_33`, `nalgebra_v0_34`, `nalgebra_v0_35` | 0.35 |
//! | ndarray | `ndarray_v0_15`, `ndarray_v0_16`, `ndarray_v0_17` | 0.17 |
//!
//! If Cargo enables several releases of one backend, only the newest enabled
//! release receives trait implementations. Select the same release line in
//! your direct backend dependency. The `*_all` features are internal markers
//! and cannot be enabled without a version feature.
//!
//! The core and older backends require Rust 1.87. The `nalgebra` and
//! `nalgebra_v0_35` features require Rust 1.89. The `nalgebra` feature previously
//! selected 0.34; use `nalgebra_v0_34` to retain that release and Rust 1.87 support.
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
//!
//! With any ndarray version feature, a matrix view borrows the original array:
//!
//! ```
//! # #[cfg(feature = "ndarray_all")]
//! # {
//! # #[cfg(all(feature = "ndarray_v0_15", not(any(feature = "ndarray_v0_16", feature = "ndarray_v0_17"))))]
//! # use ndarray_0_15 as ndarray;
//! # #[cfg(all(feature = "ndarray_v0_16", not(feature = "ndarray_v0_17")))]
//! # use ndarray_0_16 as ndarray;
//! use lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, Scaling};
//! use ndarray::array;
//!
//! let x = array![[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]];
//! let lazy = LazyMatrix::new(
//!     x.view(),
//!     Normalization::new(Centering::Mean, Scaling::Sd),
//! );
//! let y = lazy.matvec(&array![1.0, -1.0]);
//! assert_eq!(y.len(), 3);
//! # }
//! ```
//!
//! Allocating ndarray products use owned `Array1` vectors. Reusable-output
//! products also support mutable, strided destinations. Lazy forward products
//! require a clonable, mutable input, so convert immutable input views with
//! `to_owned()` first. Logical-column operations and reusable transpose products
//! accept immutable vector views directly. Raw columns borrow in O(1) time;
//! dense logical-column operations take O(nrows) time.

// Cargo feature unification may enable several releases of one backend.
// Only the newest enabled release receives trait implementations.

#[cfg(all(
    feature = "nalgebra_v0_32",
    not(any(
        feature = "nalgebra_v0_33",
        feature = "nalgebra_v0_34",
        feature = "nalgebra_v0_35"
    ))
))]
extern crate nalgebra_0_32 as nalgebra;

#[cfg(all(
    feature = "nalgebra_v0_32",
    not(any(
        feature = "nalgebra_v0_33",
        feature = "nalgebra_v0_34",
        feature = "nalgebra_v0_35"
    ))
))]
extern crate nalgebra_sparse_0_9 as nalgebra_sparse;

#[cfg(all(
    feature = "nalgebra_v0_33",
    not(any(feature = "nalgebra_v0_34", feature = "nalgebra_v0_35"))
))]
extern crate nalgebra_0_33 as nalgebra;

#[cfg(all(
    feature = "nalgebra_v0_33",
    not(any(feature = "nalgebra_v0_34", feature = "nalgebra_v0_35"))
))]
extern crate nalgebra_sparse_0_10 as nalgebra_sparse;

#[cfg(all(feature = "nalgebra_v0_34", not(feature = "nalgebra_v0_35")))]
extern crate nalgebra_0_34 as nalgebra;

#[cfg(all(feature = "nalgebra_v0_34", not(feature = "nalgebra_v0_35")))]
extern crate nalgebra_sparse_0_11 as nalgebra_sparse;

#[cfg(feature = "nalgebra_v0_35")]
extern crate nalgebra;

#[cfg(feature = "nalgebra_v0_35")]
extern crate nalgebra_sparse;

#[cfg(all(
    feature = "nalgebra_all",
    not(any(
        feature = "nalgebra_v0_32",
        feature = "nalgebra_v0_33",
        feature = "nalgebra_v0_34",
        feature = "nalgebra_v0_35"
    ))
))]
compile_error!("`nalgebra_all` is internal; enable `nalgebra` or a `nalgebra_v*` feature");

#[cfg(all(
    feature = "ndarray_v0_15",
    not(any(feature = "ndarray_v0_16", feature = "ndarray_v0_17"))
))]
extern crate ndarray_0_15 as ndarray;

#[cfg(all(feature = "ndarray_v0_16", not(feature = "ndarray_v0_17")))]
extern crate ndarray_0_16 as ndarray;

#[cfg(feature = "ndarray_v0_17")]
extern crate ndarray;

#[cfg(all(
    feature = "ndarray_all",
    not(any(
        feature = "ndarray_v0_15",
        feature = "ndarray_v0_16",
        feature = "ndarray_v0_17"
    ))
))]
compile_error!("`ndarray_all` is internal; enable `ndarray` or a `ndarray_v*` feature");

#[cfg(all(
    feature = "faer_v0_22",
    not(any(feature = "faer_v0_23", feature = "faer_v0_24"))
))]
extern crate faer_0_22 as faer;

#[cfg(all(
    feature = "faer_v0_22",
    not(any(feature = "faer_v0_23", feature = "faer_v0_24"))
))]
extern crate faer_traits_0_22 as faer_traits;

#[cfg(all(feature = "faer_v0_23", not(feature = "faer_v0_24")))]
extern crate faer_0_23 as faer;

#[cfg(all(feature = "faer_v0_23", not(feature = "faer_v0_24")))]
extern crate faer_traits_0_23 as faer_traits;

#[cfg(feature = "faer_v0_24")]
extern crate faer;

#[cfg(feature = "faer_v0_24")]
extern crate faer_traits;

#[cfg(all(
    feature = "faer_all",
    not(any(feature = "faer_v0_22", feature = "faer_v0_23", feature = "faer_v0_24"))
))]
compile_error!("`faer_all` is internal; enable `faer` or a `faer_v*` feature");

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
