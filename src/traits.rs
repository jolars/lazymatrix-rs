//! Backend-agnostic trait surface for [`LazyMatrix`](crate::LazyMatrix).
//!
//! The traits split into three groups:
//!
//! * [`Scalar`] — the numeric element type, a blanket-implemented bundle of
//!   `num-traits` bounds.
//! * [`MatrixShape`] and [`MatVec`] / [`MatTransposeVec`] — the matrix-free
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

impl<F> Scalar for F where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + std::iter::Sum
        + std::fmt::Debug
        + Default
        + 'static
{
}

/// Computes a population standard deviation from stored values and implicit zeros.
#[cfg(any(feature = "faer", feature = "nalgebra"))]
pub(crate) fn sparse_column_sd<F: Scalar>(values: &[F], nrows: usize) -> F {
    let n = F::from_usize(nrows).unwrap();
    let mean = values.iter().copied().sum::<F>() / n;
    let stored_squared_deviations = values
        .iter()
        .map(|&value| {
            let deviation = value - mean;
            deviation * deviation
        })
        .sum::<F>();
    let implicit_count = F::from_usize(nrows - values.len()).unwrap();
    let variance = (stored_squared_deviations + implicit_count * mean * mean) / n;
    variance.sqrt()
}

/// Returns the maximum of nonnegative values without masking `NaN` entries.
#[cfg(any(feature = "faer", feature = "nalgebra"))]
pub(crate) fn max_or_nan<F: Scalar>(values: impl Iterator<Item = F>) -> F {
    values.fold(F::zero(), |maximum, value| {
        if maximum.is_nan() || value.is_nan() {
            F::nan()
        } else if value > maximum {
            value
        } else {
            maximum
        }
    })
}

/// Returns the minimum value, propagating `NaN` and treating an empty input as
/// undefined.
#[cfg(any(feature = "faer", feature = "nalgebra"))]
pub(crate) fn min_or_nan<F: Scalar>(values: impl Iterator<Item = F>) -> F {
    values
        .fold(None, |minimum: Option<F>, value| {
            Some(match minimum {
                None => value,
                Some(minimum) if minimum.is_nan() || value.is_nan() => F::nan(),
                Some(minimum) => minimum.min(value),
            })
        })
        .unwrap_or_else(F::nan)
}

/// Returns `max - min`, propagating `NaN` and treating an empty input as
/// undefined.
#[cfg(any(feature = "faer", feature = "nalgebra"))]
pub(crate) fn range_or_nan<F: Scalar>(values: impl Iterator<Item = F>) -> F {
    values
        .fold(None, |extrema: Option<(F, F)>, value| {
            Some(match extrema {
                None => (value, value),
                Some((minimum, maximum))
                    if minimum.is_nan() || maximum.is_nan() || value.is_nan() =>
                {
                    (F::nan(), F::nan())
                }
                Some((minimum, maximum)) => (minimum.min(value), maximum.max(value)),
            })
        })
        .map_or_else(F::nan, |(minimum, maximum)| maximum - minimum)
}

/// Dimensions of a matrix or linear operator.
pub trait MatrixShape {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
}

/// Read-only indexed access to a dense logical vector.
///
/// Implementations may be contiguous or strided. Indexing must take O(1)
/// time; algorithms must not assume that the entries form a slice.
pub trait VectorView<F: Scalar> {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> F;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn sum(&self) -> F {
        (0..self.len()).map(|i| self.get(i)).sum()
    }
}

/// Mutable indexed access to a dense logical vector.
pub trait VectorViewMut<F: Scalar>: VectorView<F> {
    fn set(&mut self, index: usize, value: F);
}

impl<F: Scalar> VectorView<F> for [F] {
    fn len(&self) -> usize {
        <[F]>::len(self)
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }
}

impl<F: Scalar> VectorViewMut<F> for [F] {
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

impl<F: Scalar> VectorView<F> for Vec<F> {
    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }
}

impl<F: Scalar> VectorViewMut<F> for Vec<F> {
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

impl<F: Scalar, const N: usize> VectorView<F> for [F; N] {
    fn len(&self) -> usize {
        N
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }
}

impl<F: Scalar, const N: usize> VectorViewMut<F> for [F; N] {
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

/// A borrowed raw column supplied by a matrix backend.
///
/// `for_each_stored` visits every stored entry, including explicitly stored
/// zeros. Dense implementations visit every row. Sparse implementations omit
/// structural zeros.
pub trait RawColumn<F: Scalar> {
    fn len(&self) -> usize;
    fn stored_len(&self) -> usize;
    fn for_each_stored(&self, f: impl FnMut(usize, F));

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn raw_sum(&self) -> F {
        let mut sum = F::zero();
        self.for_each_stored(|_, value| sum = sum + value);
        sum
    }

    /// Add `raw_multiplier * raw_column + offset` to a dense destination.
    ///
    /// Sparse implementations take O(nnz) when `offset == 0`, and O(n + nnz)
    /// otherwise. Dense implementations take O(n).
    fn affine_add_to<V>(&self, raw_multiplier: F, offset: F, destination: &mut V)
    where
        V: VectorViewMut<F> + ?Sized,
    {
        assert_eq!(
            destination.len(),
            self.len(),
            "destination length must equal column length"
        );
        if offset != F::zero() {
            for row in 0..destination.len() {
                destination.set(row, destination.get(row) + offset);
            }
        }
        self.for_each_stored(|row, value| {
            destination.set(row, destination.get(row) + raw_multiplier * value);
        });
    }
}

/// Borrowed raw-column access for a backend matrix.
pub trait RawColumns<F: Scalar>: MatrixShape {
    type Column<'a>: RawColumn<F>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_>;
}

/// Operations on one logical, possibly normalized column.
pub trait LogicalColumn<F: Scalar> {
    fn len(&self) -> usize;
    fn center(&self) -> F;
    fn scale(&self) -> F;
    fn sum(&self) -> F;
    fn norm_squared(&self) -> F;
    fn dot<V: VectorView<F> + ?Sized>(&self, vector: &V) -> F;
    fn dot_with_sum<V: VectorView<F> + ?Sized>(&self, vector: &V, vector_sum: F) -> F;
    fn weighted_dot<V, W>(&self, vector: &V, weights: &W) -> F
    where
        V: VectorView<F> + ?Sized,
        W: VectorView<F> + ?Sized;
    fn weighted_dot_with_sum<V, W>(&self, vector: &V, weights: &W, weighted_vector_sum: F) -> F
    where
        V: VectorView<F> + ?Sized,
        W: VectorView<F> + ?Sized;
    fn weighted_norm_squared<W: VectorView<F> + ?Sized>(&self, weights: &W) -> F;
    fn weighted_norm_squared_with_sum<W: VectorView<F> + ?Sized>(
        &self,
        weights: &W,
        weight_sum: F,
    ) -> F;
    fn scaled_add_to<V: VectorViewMut<F> + ?Sized>(&self, alpha: F, destination: &mut V);

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Borrowed logical-column access for a matrix or matrix-like operator.
pub trait Columns<F: Scalar>: MatrixShape {
    type Column<'a>: LogicalColumn<F>
    where
        Self: 'a;

    fn column(&self, j: usize) -> Self::Column<'_>;
}

/// Matrix–vector product `A x`, returning a freshly allocated vector of length
/// `nrows`.
pub trait MatVec<V>: MatrixShape {
    fn matvec(&self, x: &V) -> V;
}

/// Transposed matrix–vector product `Aᵀ x`, returning a freshly allocated vector
/// of length `ncols`.
pub trait MatTransposeVec<V>: MatrixShape {
    fn mat_transpose_vec(&self, x: &V) -> V;
}

/// Dot product of two backend vectors: `Σ self[i] · other[i]`.
///
/// # Panics
///
/// Panics if the vectors have different lengths.
pub trait DotProduct<F: Scalar> {
    fn dot(&self, other: &Self) -> F;
}

/// Euclidean norm of a backend vector.
pub trait L2Norm<F: Scalar> {
    fn norm_l2(&self) -> F;
}

/// In-place scaled vector addition: `self += alpha · other`.
///
/// # Panics
///
/// Panics if the vectors have different lengths.
pub trait ScaledAddAssign<F: Scalar> {
    fn scaled_add_assign(&mut self, alpha: F, other: &Self);
}

/// In-place vector scaling: `self *= alpha`.
pub trait ScaleAssign<F: Scalar> {
    fn scale_assign(&mut self, alpha: F);
}

/// Borrowed access to columns stored contiguously in sparse form.
///
/// Implement this capability only when a backend can return the stored row
/// indices and values for one column as slices without gathering or copying.
/// Structurally absent entries are implicit zeros.
pub trait SparseColumns<F: Scalar>: MatrixShape {
    /// Return the stored `(row index, raw value)` slices for column `j`.
    ///
    /// The two slices have equal length and corresponding entries. Explicitly
    /// stored zeros remain present.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    fn sparse_column(&self, j: usize) -> (&[usize], &[F]);
}

impl<M: MatrixShape + ?Sized> MatrixShape for &M {
    fn nrows(&self) -> usize {
        (**self).nrows()
    }

    fn ncols(&self) -> usize {
        (**self).ncols()
    }
}

impl<M, V> MatVec<V> for &M
where
    M: MatVec<V> + ?Sized,
{
    fn matvec(&self, x: &V) -> V {
        (**self).matvec(x)
    }
}

impl<M, V> MatTransposeVec<V> for &M
where
    M: MatTransposeVec<V> + ?Sized,
{
    fn mat_transpose_vec(&self, x: &V) -> V {
        (**self).mat_transpose_vec(x)
    }
}

impl<M, F> RawColumns<F> for &M
where
    M: RawColumns<F> + ?Sized,
    F: Scalar,
{
    type Column<'a>
        = M::Column<'a>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        (**self).raw_column(j)
    }
}

impl<M, F> SparseColumns<F> for &M
where
    M: SparseColumns<F> + ?Sized,
    F: Scalar,
{
    fn sparse_column(&self, j: usize) -> (&[usize], &[F]) {
        (**self).sparse_column(j)
    }
}

/// In-place elementwise division by a coefficient slice: `self[i] /= coeffs[i]`.
///
/// Used to apply `S⁻¹` (scaling). Callers guarantee `coeffs[i] != 0`; the
/// `normalized` constructor floors zero scales to `1` so a constant column never
/// triggers a division by zero.
pub trait ElemDivAssign<F: Scalar> {
    fn elem_div_assign(&mut self, coeffs: &[F]);
}

/// Dot product of `self` with a coefficient slice: `Σ self[i] · coeffs[i]`.
///
/// Used to form the scalar `cᵀw` in the forward operator.
pub trait DotSlice<F: Scalar> {
    fn dot_slice(&self, coeffs: &[F]) -> F;
}

/// In-place broadcast subtraction of a scalar: `self[i] -= k`.
///
/// Used to apply the `− 1·(cᵀw)` centering correction in the forward operator.
pub trait SubScalarAssign<F: Scalar> {
    fn sub_scalar_assign(&mut self, k: F);
}

/// Sum of all entries: `Σ self[i]`.
///
/// Used to form `Σu` in the transpose operator.
pub trait SumEntries<F: Scalar> {
    fn sum_entries(&self) -> F;
}

/// In-place scaled subtraction against a coefficient slice: `self[i] -= k · coeffs[i]`.
///
/// Used to apply the `− Σu · c` centering correction in the transpose operator.
pub trait ScaledSubSlice<F: Scalar> {
    fn scaled_sub_slice(&mut self, k: F, coeffs: &[F]);
}

/// Column-wise statistics of a design matrix, returned as plain `Vec<F>` of
/// length `ncols`.
///
/// Implemented per backend so that sparse matrices compute these by walking
/// stored entries — structurally absent entries are treated as zero — without
/// ever densifying a column.
///
/// All means and standard deviations use the **population** convention (divide
/// by `n`, the number of rows), matching the design-matrix normalization used by
/// the R `lazymatrix` package.
///
/// Statistics follow IEEE floating-point behavior. In particular, means,
/// standard deviations, minima, and ranges of a zero-row column are `NaN`;
/// un-centered norms of a zero-row column are zero; and a stored `NaN`
/// propagates through every statistic.
pub trait ColumnStats<F: Scalar> {
    /// Column means `c_j = (Σ_i x_ij) / n`.
    fn col_means(&self) -> Vec<F>;

    /// Column population standard deviations `√(Σ_i (x_ij − c_j)²/n)`.
    ///
    /// Centering-invariant, so this is also the standard deviation of the
    /// centered column. Implementations use a stable two-pass calculation.
    fn col_sds(&self) -> Vec<F>;

    /// Column minima `min_i x_ij` of the un-centered column.
    fn col_mins(&self) -> Vec<F>;

    /// Column ranges `max_i x_ij - min_i x_ij` of the un-centered column.
    fn col_ranges(&self) -> Vec<F>;

    /// Column max-absolute values `max_i |x_ij|` of the un-centered column.
    fn col_maxabs(&self) -> Vec<F>;

    /// Column 1-norms `‖x_j‖₁` of the un-centered column.
    fn col_l1(&self) -> Vec<F>;

    /// Column 2-norms `‖x_j‖₂` of the un-centered column.
    fn col_l2(&self) -> Vec<F>;

    /// Column 2-norms of the **centered** columns, `‖x_j − c_j·1‖₂`.
    ///
    /// Computed sparsely via the closed form
    /// `‖x_j − c_j‖₂² = Σ_stored (v − c_j)² + (n − nnz_j)·c_j²`,
    /// so it never densifies. `centers` must have length `ncols`.
    fn col_l2_centered(&self, centers: &[F]) -> Vec<F>;

    /// Column 1-norms of the **centered** columns, `‖x_j − c_j·1‖₁`.
    ///
    /// Computed sparsely via the closed form
    /// `Σ_stored |v − c_j| + (n − nnz_j)·|c_j|`, so it never densifies.
    /// `centers` must have length `ncols`.
    fn col_l1_centered(&self, centers: &[F]) -> Vec<F>;

    /// Column max-absolute values of the **centered** columns,
    /// `max_i |x_ij − c_j|`.
    ///
    /// The implicit zero entries contribute `|c_j|`, folded in alongside the
    /// stored-entry maxima. `centers` must have length `ncols`.
    fn col_maxabs_centered(&self, centers: &[F]) -> Vec<F>;
}

impl<M, F> ColumnStats<F> for &M
where
    M: ColumnStats<F> + ?Sized,
    F: Scalar,
{
    fn col_means(&self) -> Vec<F> {
        (**self).col_means()
    }

    fn col_sds(&self) -> Vec<F> {
        (**self).col_sds()
    }

    fn col_mins(&self) -> Vec<F> {
        (**self).col_mins()
    }

    fn col_ranges(&self) -> Vec<F> {
        (**self).col_ranges()
    }

    fn col_maxabs(&self) -> Vec<F> {
        (**self).col_maxabs()
    }

    fn col_l1(&self) -> Vec<F> {
        (**self).col_l1()
    }

    fn col_l2(&self) -> Vec<F> {
        (**self).col_l2()
    }

    fn col_l2_centered(&self, centers: &[F]) -> Vec<F> {
        (**self).col_l2_centered(centers)
    }

    fn col_l1_centered(&self, centers: &[F]) -> Vec<F> {
        (**self).col_l1_centered(centers)
    }

    fn col_maxabs_centered(&self, centers: &[F]) -> Vec<F> {
        (**self).col_maxabs_centered(centers)
    }
}

/// How to center each column.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Centering {
    /// No centering.
    #[default]
    None,
    /// Subtract the column mean.
    Mean,
    /// Subtract the column minimum.
    Min,
}

/// How to scale each column.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scaling {
    /// No scaling.
    #[default]
    None,
    /// Divide by the (population) standard deviation.
    Sd,
    /// Divide by the maximum absolute value.
    MaxAbs,
    /// Divide by the 1-norm.
    L1,
    /// Divide by the 2-norm.
    L2,
    /// Divide by the range, `max - min`.
    Range,
}

/// A full normalization specification: an independent [`Centering`] and
/// [`Scaling`] choice.
///
/// When both are active, scales are computed from the **centered** columns
/// Non-translation-invariant scales are computed from centered columns (see
/// [`ColumnStats::col_l1_centered`], [`ColumnStats::col_l2_centered`], and
/// [`ColumnStats::col_maxabs_centered`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Normalization {
    pub center: Centering,
    pub scale: Scaling,
}

impl Normalization {
    /// Build a specification from its two axes.
    pub fn new(center: Centering, scale: Scaling) -> Self {
        Self { center, scale }
    }
}
