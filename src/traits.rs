//! Backend-agnostic trait surface for [`LazyMatrix`](crate::LazyMatrix).
//!
//! The traits split into three groups:
//!
//! * [`Scalar`] — the numeric element type, a blanket-implemented bundle of
//!   `num-traits` bounds.
//! * [`MatVec`] / [`MatTransposeVec`] — the matrix-free linear-operator
//!   interface, implemented both by concrete backend matrices and by
//!   [`LazyMatrix`](crate::LazyMatrix) itself.
//! * The five *vector* traits ([`ElemDivAssign`], [`DotSlice`],
//!   [`SubScalarAssign`], [`SumEntries`], [`ScaledSubSlice`]) — the elementwise
//!   primitives that fold the lazy normalization into a backend vector. They are
//!   phrased as a backend vector against a coefficient slice `&[F]`, which is
//!   exactly the shape the centering/scaling math needs.
//! * [`ColumnStats`] — column statistics computed directly over a (possibly
//!   sparse) backend matrix, used by the `normalized` constructor.

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

/// Matrix–vector product `A x`, returning a freshly allocated vector of length
/// `nrows`.
pub trait MatVec<V> {
    fn matvec(&self, x: &V) -> V;
}

/// Transposed matrix–vector product `Aᵀ x`, returning a freshly allocated vector
/// of length `ncols`.
pub trait MatTransposeVec<V> {
    fn mat_transpose_vec(&self, x: &V) -> V;
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
pub trait ColumnStats<F: Scalar> {
    /// Column means `c_j = (Σ_i x_ij) / n`.
    fn col_means(&self) -> Vec<F>;

    /// Column population standard deviations `√((Σ_i x_ij² )/n − c_j²)`.
    ///
    /// Centering-invariant, so this is also the standard deviation of the
    /// centered column.
    fn col_sds(&self) -> Vec<F>;

    /// Column max-absolute values `max_i |x_ij|` of the un-centered column.
    fn col_maxabs(&self) -> Vec<F>;

    /// Column 2-norms `‖x_j‖₂` of the un-centered column.
    fn col_l2(&self) -> Vec<F>;

    /// Column 2-norms of the **centered** columns, `‖x_j − c_j·1‖₂`.
    ///
    /// Computed sparsely via the closed form
    /// `‖x_j − c_j‖₂² = Σ_stored (v − c_j)² + (n − nnz_j)·c_j²`,
    /// so it never densifies. `centers` must have length `ncols`.
    fn col_l2_centered(&self, centers: &[F]) -> Vec<F>;

    /// Column max-absolute values of the **centered** columns,
    /// `max_i |x_ij − c_j|`.
    ///
    /// The implicit zero entries contribute `|c_j|`, folded in alongside the
    /// stored-entry maxima. `centers` must have length `ncols`.
    fn col_maxabs_centered(&self, centers: &[F]) -> Vec<F>;
}

/// How to center each column.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Centering {
    /// No centering.
    #[default]
    None,
    /// Subtract the column mean.
    Mean,
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
    /// Divide by the 2-norm.
    L2,
}

/// A full normalization specification: an independent [`Centering`] and
/// [`Scaling`] choice.
///
/// When both are active, scales are computed from the **centered** columns
/// (see [`ColumnStats::col_l2_centered`] / [`ColumnStats::col_maxabs_centered`]).
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
