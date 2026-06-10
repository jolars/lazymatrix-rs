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
//! * `faer` — [`faer::sparse::SparseColMat`] over [`faer::Col`].
//! * `nalgebra` — [`nalgebra_sparse::CscMatrix`] over [`nalgebra::DVector`].
//!
//! Any type implementing the [`traits`] surface (a dense matrix, say) works too.
//!
//! # Example
//!
//! ```ignore
//! use lazymatrix::{LazyMatrix, MatVec, Normalization, Centering, Scaling};
//!
//! // `x` is some backend matrix implementing `MatVec`, `MatTransposeVec`,
//! // `ColumnStats`; `v` a backend vector.
//! let spec = Normalization::new(Centering::Mean, Scaling::Sd);
//! let lazy = LazyMatrix::normalized(x, n, p, spec);
//! let y = lazy.matvec(&v); // == ((X − 1cᵀ)S⁻¹) v, sparsity preserved
//! ```

pub mod traits;

pub use traits::{
    Centering, ColumnStats, DotSlice, ElemDivAssign, MatTransposeVec, MatVec, Normalization,
    Scalar, ScaledSubSlice, Scaling, SubScalarAssign, SumEntries,
};

#[cfg(feature = "faer")]
mod faer_sparse_backend;

#[cfg(feature = "nalgebra")]
mod nalgebra_sparse_backend;

/// A matrix presented with lazy column normalization `X̃ = (X − 1cᵀ)S⁻¹`.
///
/// The underlying matrix `data` is never modified or densified; the centers and
/// scales are folded into the matrix–vector products on the fly. See the
/// [crate-level documentation](crate) for the math.
///
/// `centers` and `scales` are each `None` when that axis of normalization is
/// inactive. When present, each has length `ncols`.
#[derive(Clone, Debug)]
pub struct LazyMatrix<M, F = f64> {
    data: M,
    centers: Option<Vec<F>>,
    scales: Option<Vec<F>>,
    nrows: usize,
    ncols: usize,
}

impl<M, F: Scalar> LazyMatrix<M, F> {
    /// Construct from an explicit center and/or scale vector.
    ///
    /// # Panics
    /// Panics if a provided `centers`/`scales` vector does not have length
    /// `ncols`.
    pub fn new(
        data: M,
        nrows: usize,
        ncols: usize,
        centers: Option<Vec<F>>,
        scales: Option<Vec<F>>,
    ) -> Self {
        if let Some(c) = &centers {
            assert_eq!(c.len(), ncols, "centers length must equal ncols");
        }
        if let Some(s) = &scales {
            assert_eq!(s.len(), ncols, "scales length must equal ncols");
        }
        Self {
            data,
            centers,
            scales,
            nrows,
            ncols,
        }
    }

    /// Wrap a matrix with no normalization (a pure pass-through operator).
    pub fn raw(data: M, nrows: usize, ncols: usize) -> Self {
        Self::new(data, nrows, ncols, None, None)
    }

    /// Wrap a matrix with column centering only.
    ///
    /// # Panics
    /// Panics if `centers.len() != ncols`.
    pub fn with_centers(data: M, nrows: usize, ncols: usize, centers: Vec<F>) -> Self {
        Self::new(data, nrows, ncols, Some(centers), None)
    }

    /// Wrap a matrix with column scaling only.
    ///
    /// # Panics
    /// Panics if `scales.len() != ncols`.
    pub fn with_scales(data: M, nrows: usize, ncols: usize, scales: Vec<F>) -> Self {
        Self::new(data, nrows, ncols, None, Some(scales))
    }

    /// Number of rows of the (logical) normalized matrix.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns of the (logical) normalized matrix.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// The column centers `c`, if centering is active.
    pub fn centers(&self) -> Option<&[F]> {
        self.centers.as_deref()
    }

    /// The column scales `s`, if scaling is active.
    pub fn scales(&self) -> Option<&[F]> {
        self.scales.as_deref()
    }

    /// Borrow the underlying (un-normalized) matrix.
    pub fn data(&self) -> &M {
        &self.data
    }

    /// Consume the wrapper, returning the underlying matrix and the
    /// center/scale vectors.
    pub fn into_parts(self) -> (M, Option<Vec<F>>, Option<Vec<F>>) {
        (self.data, self.centers, self.scales)
    }
}

impl<M, F: Scalar> LazyMatrix<M, F>
where
    M: ColumnStats<F>,
{
    /// Construct by **computing** the centers and scales from `data` according
    /// to `spec`.
    ///
    /// When both centering and scaling are requested, scales are computed from
    /// the *centered* columns (standard deviation is centering-invariant; `L2`
    /// and `MaxAbs` are not, and use the sparse closed-form centered variants).
    ///
    /// Any non-positive scale (e.g. a constant column whose standard deviation
    /// is zero) is floored to `1`, so the resulting operator never divides by
    /// zero.
    pub fn normalized(data: M, nrows: usize, ncols: usize, spec: Normalization) -> Self {
        let centers = match spec.center {
            Centering::None => None,
            Centering::Mean => Some(data.col_means()),
        };

        let scales = match spec.scale {
            Scaling::None => None,
            Scaling::Sd => Some(floor_zeros(data.col_sds())),
            Scaling::MaxAbs => Some(floor_zeros(match &centers {
                Some(c) => data.col_maxabs_centered(c),
                None => data.col_maxabs(),
            })),
            Scaling::L2 => Some(floor_zeros(match &centers {
                Some(c) => data.col_l2_centered(c),
                None => data.col_l2(),
            })),
        };

        Self::new(data, nrows, ncols, centers, scales)
    }
}

/// Replace any non-positive (`<= 0`) entry with `1`, leaving others untouched.
///
/// Mirrors the zero-variance guard used in standard penalized-regression
/// preprocessing: a constant column has scale `0`, which would otherwise produce
/// a division by zero; flooring it to `1` makes that column a no-op under
/// scaling.
fn floor_zeros<F: Scalar>(mut scales: Vec<F>) -> Vec<F> {
    let one = F::one();
    let zero = F::zero();
    for s in &mut scales {
        if *s <= zero {
            *s = one;
        }
    }
    scales
}

impl<M, V, F> MatVec<V> for LazyMatrix<M, F>
where
    F: Scalar,
    M: MatVec<V>,
    V: Clone + ElemDivAssign<F> + DotSlice<F> + SubScalarAssign<F>,
{
    /// `X̃ v = X (S⁻¹ v) − 1 · (cᵀ S⁻¹ v)`.
    fn matvec(&self, v: &V) -> V {
        // The forward op clones `v` because it mutates it into `S⁻¹v`. The
        // transpose op below does NOT clone `u`: it reads `Σu` first, then only
        // reads `u` through the backend product.
        let mut w = v.clone();
        if let Some(s) = &self.scales {
            w.elem_div_assign(s); // w = S⁻¹ v
        }
        let mut y = self.data.matvec(&w); // y = X w   (sparse; no X − 1cᵀ)
        if let Some(c) = &self.centers {
            y.sub_scalar_assign(w.dot_slice(c)); // y −= 1 · (cᵀ w)
        }
        y
    }
}

impl<M, V, F> MatTransposeVec<V> for LazyMatrix<M, F>
where
    F: Scalar,
    M: MatTransposeVec<V>,
    V: SumEntries<F> + ScaledSubSlice<F> + ElemDivAssign<F>,
{
    /// `X̃ᵀ u = S⁻¹ (Xᵀ u − c · Σu)`.
    fn mat_transpose_vec(&self, u: &V) -> V {
        let total = if self.centers.is_some() {
            u.sum_entries()
        } else {
            F::zero()
        };
        let mut t = self.data.mat_transpose_vec(u); // t = Xᵀ u   (sparse)
        if let Some(c) = &self.centers {
            t.scaled_sub_slice(total, c); // t −= Σu · c
        }
        if let Some(s) = &self.scales {
            t.elem_div_assign(s); // t = S⁻¹ t
        }
        t
    }
}
