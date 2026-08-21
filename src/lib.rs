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
    Centering, ColumnStats, Columns, DotProduct, DotSlice, ElemDivAssign, L2Norm, LogicalColumn,
    MatTransposeVec, MatVec, MatrixShape, Normalization, RawColumn, RawColumns, Scalar,
    ScaleAssign, ScaledAddAssign, ScaledSubSlice, Scaling, SparseColumns, SubScalarAssign,
    SumEntries, VectorView, VectorViewMut,
};

#[cfg(feature = "faer")]
mod faer_sparse_backend;

#[cfg(feature = "nalgebra")]
mod nalgebra_sparse_backend;

#[cfg(feature = "faer")]
mod faer_dense_backend;

#[cfg(feature = "nalgebra")]
mod nalgebra_dense_backend;

/// A borrowed raw sparse column.
///
/// The view exposes the underlying matrix's stored row indices and raw values
/// without copying. A normalized [`LazyColumn`] wrapping it has logical entries
///
/// ```text
/// stored row:   (value − center) / scale
/// implicit row: (0 − center) / scale
/// ```
///
/// Thus, a centered logical column is generally dense even though the two
/// borrowed slices contain only the underlying matrix's stored entries.
#[derive(Clone, Copy, Debug)]
pub struct SparseColumnRef<'a, F> {
    row_indices: &'a [usize],
    values: &'a [F],
    len: usize,
}

impl<'a, F> SparseColumnRef<'a, F> {
    fn new(row_indices: &'a [usize], values: &'a [F], len: usize) -> Self {
        Self {
            row_indices,
            values,
            len,
        }
    }

    /// Row indices of the raw stored entries.
    pub fn row_indices(&self) -> &'a [usize] {
        self.row_indices
    }

    /// Raw stored values corresponding to [`Self::row_indices`].
    pub fn values(&self) -> &'a [F] {
        self.values
    }
}

impl<F: Scalar> RawColumn<F> for SparseColumnRef<'_, F> {
    fn len(&self) -> usize {
        self.len
    }

    fn stored_len(&self) -> usize {
        self.values.len()
    }

    fn for_each_stored(&self, mut f: impl FnMut(usize, F)) {
        for (&row, &value) in self.row_indices.iter().zip(self.values) {
            f(row, value);
        }
    }
}

/// A borrowed lazily normalized column over an arbitrary raw backend view.
#[derive(Clone, Copy, Debug)]
pub struct LazyColumn<C, F> {
    raw: C,
    center: F,
    scale: F,
}

impl<C, F> LazyColumn<C, F> {
    /// Borrow the backend's raw column view.
    pub fn raw(&self) -> &C {
        &self.raw
    }
}

pub type LazySparseColumn<'a, F> = LazyColumn<SparseColumnRef<'a, F>, F>;

impl<'a, F: Scalar> LazySparseColumn<'a, F> {
    /// Row indices of the raw stored entries.
    pub fn row_indices(&self) -> &'a [usize] {
        self.raw.row_indices()
    }

    /// Raw stored values corresponding to [`Self::row_indices`].
    pub fn values(&self) -> &'a [F] {
        self.raw.values()
    }

    /// Logical value at a structurally absent row, `-center / scale`.
    pub fn implicit_value(&self) -> F {
        -self.center / self.scale
    }

    /// Sum of the raw stored values.
    pub fn raw_sum(&self) -> F {
        self.raw.raw_sum()
    }

    /// Stored corrections to the implicit value as `(row, raw_value / scale)`.
    pub fn stored_corrections(&self) -> impl Iterator<Item = (usize, F)> + '_ {
        self.row_indices()
            .iter()
            .copied()
            .zip(self.values().iter().map(|&value| value / self.scale))
    }
}

impl<C: RawColumn<F>, F: Scalar> LazyColumn<C, F> {
    /// Logical length of the column, including structurally absent entries.
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    /// Whether the logical column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
        let len = F::from_usize(self.len()).unwrap();
        (self.raw.raw_sum() - len * self.center) / self.scale
    }

    /// Squared Euclidean norm of the logical normalized column.
    ///
    /// Structurally absent entries contribute `center² / scale²`. This takes
    /// O(nnz) time and does not materialize the column.
    pub fn norm_squared(&self) -> F {
        let mut stored_squared_deviations = F::zero();
        self.raw.for_each_stored(|_, value| {
            let deviation = value - self.center;
            stored_squared_deviations = stored_squared_deviations + deviation * deviation;
        });
        let implicit_count = F::from_usize(self.len() - self.raw.stored_len()).unwrap();
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
    pub fn dot<V: VectorView<F> + ?Sized>(&self, vector: &V) -> F {
        assert_eq!(
            vector.len(),
            self.len(),
            "vector length must equal column length"
        );
        let vector_sum = vector.sum();
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
    pub fn dot_with_sum<V: VectorView<F> + ?Sized>(&self, vector: &V, vector_sum: F) -> F {
        assert_eq!(
            vector.len(),
            self.len(),
            "vector length must equal column length"
        );
        let mut raw_dot = F::zero();
        self.raw
            .for_each_stored(|row, value| raw_dot = raw_dot + value * vector.get(row));
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
    pub fn weighted_dot<V, W>(&self, vector: &V, weights: &W) -> F
    where
        V: VectorView<F> + ?Sized,
        W: VectorView<F> + ?Sized,
    {
        assert_eq!(
            vector.len(),
            self.len(),
            "vector length must equal column length"
        );
        assert_eq!(
            weights.len(),
            self.len(),
            "weights length must equal column length"
        );
        let weighted_vector_sum = (0..self.len())
            .map(|i| vector.get(i) * weights.get(i))
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
    pub fn weighted_dot_with_sum<V, W>(&self, vector: &V, weights: &W, weighted_vector_sum: F) -> F
    where
        V: VectorView<F> + ?Sized,
        W: VectorView<F> + ?Sized,
    {
        assert_eq!(
            vector.len(),
            self.len(),
            "vector length must equal column length"
        );
        assert_eq!(
            weights.len(),
            self.len(),
            "weights length must equal column length"
        );
        let mut raw_weighted_dot = F::zero();
        self.raw.for_each_stored(|row, value| {
            raw_weighted_dot = raw_weighted_dot + value * weights.get(row) * vector.get(row);
        });
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
    pub fn weighted_norm_squared<W: VectorView<F> + ?Sized>(&self, weights: &W) -> F {
        assert_eq!(
            weights.len(),
            self.len(),
            "weights length must equal column length"
        );
        self.weighted_norm_squared_with_sum(weights, weights.sum())
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
    pub fn weighted_norm_squared_with_sum<W: VectorView<F> + ?Sized>(
        &self,
        weights: &W,
        weight_sum: F,
    ) -> F {
        assert_eq!(
            weights.len(),
            self.len(),
            "weights length must equal column length"
        );
        let mut stored_squared_deviations = F::zero();
        let mut stored_weight = F::zero();
        self.raw.for_each_stored(|row, value| {
            let weight = weights.get(row);
            let deviation = value - self.center;
            stored_squared_deviations = stored_squared_deviations + weight * deviation * deviation;
            stored_weight = stored_weight + weight;
        });
        let implicit_weight = weight_sum - stored_weight;
        let centered_norm_squared =
            stored_squared_deviations + implicit_weight * self.center * self.center;
        centered_norm_squared / (self.scale * self.scale)
    }

    /// Add `alpha` times this logical column to a dense destination.
    pub fn scaled_add_to<V: VectorViewMut<F> + ?Sized>(&self, alpha: F, destination: &mut V) {
        self.raw.affine_add_to(
            alpha / self.scale,
            -alpha * self.center / self.scale,
            destination,
        );
    }
}

impl<C: RawColumn<F>, F: Scalar> LogicalColumn<F> for LazyColumn<C, F> {
    fn len(&self) -> usize {
        self.len()
    }
    fn center(&self) -> F {
        self.center()
    }
    fn scale(&self) -> F {
        self.scale()
    }
    fn sum(&self) -> F {
        self.sum()
    }
    fn norm_squared(&self) -> F {
        self.norm_squared()
    }
    fn dot<V: VectorView<F> + ?Sized>(&self, vector: &V) -> F {
        self.dot(vector)
    }
    fn dot_with_sum<V: VectorView<F> + ?Sized>(&self, vector: &V, vector_sum: F) -> F {
        self.dot_with_sum(vector, vector_sum)
    }
    fn weighted_dot<V, W>(&self, vector: &V, weights: &W) -> F
    where
        V: VectorView<F> + ?Sized,
        W: VectorView<F> + ?Sized,
    {
        self.weighted_dot(vector, weights)
    }
    fn weighted_dot_with_sum<V, W>(&self, vector: &V, weights: &W, sum: F) -> F
    where
        V: VectorView<F> + ?Sized,
        W: VectorView<F> + ?Sized,
    {
        self.weighted_dot_with_sum(vector, weights, sum)
    }
    fn weighted_norm_squared<W: VectorView<F> + ?Sized>(&self, weights: &W) -> F {
        self.weighted_norm_squared(weights)
    }
    fn weighted_norm_squared_with_sum<W: VectorView<F> + ?Sized>(&self, weights: &W, sum: F) -> F {
        self.weighted_norm_squared_with_sum(weights, sum)
    }
    fn scaled_add_to<V: VectorViewMut<F> + ?Sized>(&self, alpha: F, destination: &mut V) {
        self.scaled_add_to(alpha, destination);
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

    /// Borrow one lazily normalized column without copying.
    pub fn column(&self, j: usize) -> LazyColumn<M::Column<'_>, F>
    where
        M: RawColumns<F>,
    {
        assert!(j < self.ncols(), "column index out of bounds");
        LazyColumn {
            raw: self.data.raw_column(j),
            center: self.centers.as_ref().map_or_else(F::zero, |c| c[j]),
            scale: self.scales.as_ref().map_or_else(F::one, |s| s[j]),
        }
    }

    /// Borrow one lazily normalized CSC column with its sparse representation.
    pub fn sparse_column(&self, j: usize) -> LazySparseColumn<'_, F>
    where
        M: SparseColumns<F>,
    {
        assert!(j < self.ncols(), "column index out of bounds");
        let (row_indices, values) = self.data.sparse_column(j);
        LazyColumn {
            raw: SparseColumnRef::new(row_indices, values, self.nrows()),
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
    /// the *centered* columns. Standard deviation and range are
    /// centering-invariant; `L1`, `L2`, and `MaxAbs` use sparse closed-form
    /// centered variants.
    ///
    /// An exact zero scale (e.g. a constant column whose standard deviation is
    /// zero) is replaced with `1`, so the resulting operator never divides by
    /// zero. Nonfinite statistics retain their IEEE values and propagate through
    /// subsequent operations.
    pub fn new(data: M, spec: Normalization) -> Self {
        let centers = match spec.center {
            Centering::None => None,
            Centering::Mean => Some(data.col_means()),
            Centering::Min => Some(data.col_mins()),
        };

        let scales = match spec.scale {
            Scaling::None => None,
            Scaling::Sd => Some(replace_zero_scales(data.col_sds())),
            Scaling::L1 => Some(replace_zero_scales(match &centers {
                Some(c) => data.col_l1_centered(c),
                None => data.col_l1(),
            })),
            Scaling::MaxAbs => Some(replace_zero_scales(match &centers {
                Some(c) => data.col_maxabs_centered(c),
                None => data.col_maxabs(),
            })),
            Scaling::L2 => Some(replace_zero_scales(match &centers {
                Some(c) => data.col_l2_centered(c),
                None => data.col_l2(),
            })),
            Scaling::Range => Some(replace_zero_scales(data.col_ranges())),
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

impl<M, F> Columns<F> for LazyMatrix<M, F>
where
    F: Scalar,
    M: RawColumns<F>,
{
    type Column<'a>
        = LazyColumn<M::Column<'a>, F>
    where
        Self: 'a;

    fn column(&self, j: usize) -> Self::Column<'_> {
        LazyMatrix::column(self, j)
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
