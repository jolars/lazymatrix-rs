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
//! // `ColumnStats`, `MatrixShape`; `v` a backend vector.
//! let spec = Normalization::new(Centering::Mean, Scaling::Sd);
//! let lazy = LazyMatrix::new(x, spec);
//! let y = lazy.matvec(&v); // == ((X − 1cᵀ)S⁻¹) v, sparsity preserved
//! ```

pub mod traits;

pub use traits::{
    Centering, ColumnStats, DotProduct, DotSlice, ElemDivAssign, L2Norm, MatTransposeVec, MatVec,
    MatrixShape, Normalization, Scalar, ScaleAssign, ScaledAddAssign, ScaledSubSlice, Scaling,
    SparseColumns, SubScalarAssign, SumEntries,
};

#[cfg(feature = "faer")]
mod faer_sparse_backend;

#[cfg(feature = "nalgebra")]
mod nalgebra_sparse_backend;

/// A borrowed view of one lazily normalized sparse column.
///
/// The view exposes the underlying matrix's stored row indices and raw values
/// without copying. Its logical entries are
///
/// ```text
/// stored row:   (value − center) / scale
/// implicit row: (0 − center) / scale
/// ```
///
/// Thus, a centered logical column is generally dense even though the two
/// borrowed slices contain only the underlying matrix's stored entries.
#[derive(Clone, Copy, Debug)]
pub struct LazyColumn<'a, F> {
    row_indices: &'a [usize],
    values: &'a [F],
    len: usize,
    center: F,
    scale: F,
}

impl<'a, F: Scalar> LazyColumn<'a, F> {
    /// Row indices of the raw stored entries.
    pub fn row_indices(&self) -> &'a [usize] {
        self.row_indices
    }

    /// Raw stored values from the underlying matrix.
    pub fn values(&self) -> &'a [F] {
        self.values
    }

    /// Logical length of the column, including structurally absent entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the logical column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Effective column center.
    ///
    /// This is zero when centering is inactive.
    pub fn center(&self) -> F {
        self.center
    }

    /// Effective column scale.
    ///
    /// This is one when scaling is inactive.
    pub fn scale(&self) -> F {
        self.scale
    }

    /// Sum of the logical normalized entries.
    ///
    /// This takes O(nnz) time and does not materialize the column.
    pub fn sum(&self) -> F {
        let raw_sum = self.values.iter().copied().sum::<F>();
        let len = F::from_usize(self.len).unwrap();
        (raw_sum - len * self.center) / self.scale
    }

    /// Squared Euclidean norm of the logical normalized column.
    ///
    /// Structurally absent entries contribute `center² / scale²`. This takes
    /// O(nnz) time and does not materialize the column.
    pub fn norm_squared(&self) -> F {
        let stored_squared_deviations = self
            .values
            .iter()
            .map(|&value| {
                let deviation = value - self.center;
                deviation * deviation
            })
            .sum::<F>();
        let implicit_count = F::from_usize(self.len - self.values.len()).unwrap();
        let centered_norm_squared =
            stored_squared_deviations + implicit_count * self.center * self.center;
        centered_norm_squared / (self.scale * self.scale)
    }

    /// Dot product of the logical normalized column with a dense vector.
    ///
    /// Computing the vector sum takes O(n) time, after which the stored-entry
    /// dot product takes O(nnz). Use [`Self::dot_with_sum`] when the vector sum
    /// is already available.
    ///
    /// # Panics
    ///
    /// Panics if `vector.len() != self.len()`.
    pub fn dot(&self, vector: &[F]) -> F {
        assert_eq!(
            vector.len(),
            self.len,
            "vector length must equal column length"
        );
        let vector_sum = vector.iter().copied().sum();
        self.dot_with_sum(vector, vector_sum)
    }

    /// Dot product using a precomputed sum of the dense vector.
    ///
    /// `vector_sum` must equal `vector.iter().sum()`. Supplying the cached sum
    /// keeps this operation O(nnz), which is useful when repeatedly taking
    /// column products with the same vector.
    ///
    /// # Panics
    ///
    /// Panics if `vector.len() != self.len()`.
    pub fn dot_with_sum(&self, vector: &[F], vector_sum: F) -> F {
        assert_eq!(
            vector.len(),
            self.len,
            "vector length must equal column length"
        );
        let raw_dot = self
            .row_indices
            .iter()
            .zip(self.values)
            .map(|(&row, &value)| value * vector[row])
            .sum::<F>();
        (raw_dot - self.center * vector_sum) / self.scale
    }

    /// Weighted dot product of the logical normalized column with a dense
    /// vector, `sum_i weights[i] * self[i] * vector[i]`.
    ///
    /// Computing the weighted vector sum takes O(n) time, after which the
    /// stored-entry product takes O(nnz). Use [`Self::weighted_dot_with_sum`]
    /// when `sum_i weights[i] * vector[i]` is already available.
    ///
    /// # Panics
    ///
    /// Panics unless `vector.len() == weights.len() == self.len()`.
    pub fn weighted_dot(&self, vector: &[F], weights: &[F]) -> F {
        assert_eq!(
            vector.len(),
            self.len,
            "vector length must equal column length"
        );
        assert_eq!(
            weights.len(),
            self.len,
            "weights length must equal column length"
        );
        let weighted_vector_sum = vector
            .iter()
            .zip(weights)
            .map(|(&value, &weight)| value * weight)
            .sum();
        self.weighted_dot_with_sum(vector, weights, weighted_vector_sum)
    }

    /// Weighted dot product using a precomputed weighted vector sum.
    ///
    /// `weighted_vector_sum` must equal
    /// `sum_i weights[i] * vector[i]`. Supplying it keeps this operation
    /// O(nnz), which is useful for repeatedly computing weighted correlations
    /// against different columns.
    ///
    /// # Panics
    ///
    /// Panics unless `vector.len() == weights.len() == self.len()`.
    pub fn weighted_dot_with_sum(&self, vector: &[F], weights: &[F], weighted_vector_sum: F) -> F {
        assert_eq!(
            vector.len(),
            self.len,
            "vector length must equal column length"
        );
        assert_eq!(
            weights.len(),
            self.len,
            "weights length must equal column length"
        );
        let raw_weighted_dot = self
            .row_indices
            .iter()
            .zip(self.values)
            .map(|(&row, &value)| value * weights[row] * vector[row])
            .sum::<F>();
        (raw_weighted_dot - self.center * weighted_vector_sum) / self.scale
    }

    /// Weighted squared Euclidean norm of the logical normalized column,
    /// `sum_i weights[i] * self[i]^2`.
    ///
    /// Computing the weight sum takes O(n) time, after which the stored-entry
    /// calculation takes O(nnz). Use
    /// [`Self::weighted_norm_squared_with_sum`] when the weight sum is already
    /// available.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != self.len()`.
    pub fn weighted_norm_squared(&self, weights: &[F]) -> F {
        assert_eq!(
            weights.len(),
            self.len,
            "weights length must equal column length"
        );
        self.weighted_norm_squared_with_sum(weights, weights.iter().copied().sum())
    }

    /// Weighted squared norm using a precomputed sum of the weights.
    ///
    /// `weight_sum` must equal `weights.iter().sum()`. Supplying it keeps this
    /// operation O(nnz), which is useful for coordinate-wise Hessian
    /// calculations with a shared weight vector.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != self.len()`.
    pub fn weighted_norm_squared_with_sum(&self, weights: &[F], weight_sum: F) -> F {
        assert_eq!(
            weights.len(),
            self.len,
            "weights length must equal column length"
        );
        let (stored_squared_deviations, stored_weight) =
            self.row_indices.iter().zip(self.values).fold(
                (F::zero(), F::zero()),
                |(squares, total_weight), (&row, &value)| {
                    let deviation = value - self.center;
                    (
                        squares + weights[row] * deviation * deviation,
                        total_weight + weights[row],
                    )
                },
            );
        let implicit_weight = weight_sum - stored_weight;
        let centered_norm_squared =
            stored_squared_deviations + implicit_weight * self.center * self.center;
        centered_norm_squared / (self.scale * self.scale)
    }
}

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
}

impl<M, F: Scalar> LazyMatrix<M, F>
where
    M: MatrixShape,
{
    /// Construct from an explicit center and/or scale vector.
    ///
    /// # Panics
    /// Panics if a provided `centers`/`scales` vector does not have length
    /// `ncols`.
    pub fn from_parts(data: M, centers: Option<Vec<F>>, scales: Option<Vec<F>>) -> Self {
        let ncols = data.ncols();
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
        }
    }

    /// Wrap a matrix with column centering only.
    ///
    /// # Panics
    /// Panics if `centers.len() != ncols`.
    pub fn with_centers(data: M, centers: Vec<F>) -> Self {
        Self::from_parts(data, Some(centers), None)
    }

    /// Wrap a matrix with column scaling only.
    ///
    /// # Panics
    /// Panics if `scales.len() != ncols`.
    pub fn with_scales(data: M, scales: Vec<F>) -> Self {
        Self::from_parts(data, None, Some(scales))
    }

    /// Number of rows of the logical normalized matrix.
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }

    /// Number of columns of the logical normalized matrix.
    pub fn ncols(&self) -> usize {
        self.data.ncols()
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

    /// Borrow one lazily normalized sparse column without copying.
    ///
    /// The returned view contains raw stored entries plus the effective center
    /// and scale needed to interpret both stored and implicit entries. It does
    /// not materialize the generally dense centered column.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    pub fn column(&self, j: usize) -> LazyColumn<'_, F>
    where
        M: SparseColumns<F>,
    {
        assert!(j < self.ncols(), "column index out of bounds");
        let (row_indices, values) = self.data.sparse_column(j);
        LazyColumn {
            row_indices,
            values,
            len: self.nrows(),
            center: self.centers.as_ref().map_or_else(F::zero, |c| c[j]),
            scale: self.scales.as_ref().map_or_else(F::one, |s| s[j]),
        }
    }

    /// Consume the wrapper, returning the underlying matrix and the
    /// center/scale vectors.
    pub fn into_parts(self) -> (M, Option<Vec<F>>, Option<Vec<F>>) {
        (self.data, self.centers, self.scales)
    }
}

impl<M, F: Scalar> LazyMatrix<M, F>
where
    M: ColumnStats<F> + MatrixShape,
{
    /// Construct by **computing** the centers and scales from `data` according
    /// to `spec`.
    ///
    /// When both centering and scaling are requested, scales are computed from
    /// the *centered* columns (standard deviation is centering-invariant; `L2`
    /// and `MaxAbs` are not, and use the sparse closed-form centered variants).
    ///
    /// An exact zero scale (e.g. a constant column whose standard deviation is
    /// zero) is replaced with `1`, so the resulting operator never divides by
    /// zero. Nonfinite statistics retain their IEEE values and propagate through
    /// subsequent operations.
    pub fn new(data: M, spec: Normalization) -> Self {
        let centers = match spec.center {
            Centering::None => None,
            Centering::Mean => Some(data.col_means()),
        };

        let scales = match spec.scale {
            Scaling::None => None,
            Scaling::Sd => Some(replace_zero_scales(data.col_sds())),
            Scaling::MaxAbs => Some(replace_zero_scales(match &centers {
                Some(c) => data.col_maxabs_centered(c),
                None => data.col_maxabs(),
            })),
            Scaling::L2 => Some(replace_zero_scales(match &centers {
                Some(c) => data.col_l2_centered(c),
                None => data.col_l2(),
            })),
        };

        Self::from_parts(data, centers, scales)
    }
}

impl<M, F> MatrixShape for LazyMatrix<M, F>
where
    M: MatrixShape,
{
    fn nrows(&self) -> usize {
        self.data.nrows()
    }

    fn ncols(&self) -> usize {
        self.data.ncols()
    }
}

/// Replace exact zero entries with `1`, leaving nonfinite values untouched.
///
/// Mirrors the zero-variance guard used in standard penalized-regression
/// preprocessing: a constant column has scale `0`, which would otherwise produce
/// a division by zero; replacing it with `1` makes that column a no-op under
/// scaling.
fn replace_zero_scales<F: Scalar>(mut scales: Vec<F>) -> Vec<F> {
    let one = F::one();
    let zero = F::zero();
    for s in &mut scales {
        if *s == zero {
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
