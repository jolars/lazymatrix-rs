use super::Scalar;

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
