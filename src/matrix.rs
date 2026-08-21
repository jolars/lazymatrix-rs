use crate::column::{LazyColumn, LazySparseColumn, SparseColumnRef};
use crate::normalization::{Centering, Normalization, Scaling};
use crate::traits::{
    ColumnStats, Columns, DotSlice, ElemDivAssign, MatTransposeVec, MatTransposeVecInto, MatVec,
    MatVecInto, MatrixShape, RawColumns, Scalar, ScaledSubSlice, SparseColumns, SubScalarAssign,
    SumEntries,
};

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
        LazyColumn::new(
            self.data.raw_column(j),
            self.centers.as_ref().map_or_else(F::zero, |c| c[j]),
            self.scales.as_ref().map_or_else(F::one, |s| s[j]),
        )
    }

    /// Borrow one lazily normalized CSC column with its sparse representation.
    pub fn sparse_column(&self, j: usize) -> LazySparseColumn<'_, F>
    where
        M: SparseColumns<F>,
    {
        assert!(j < self.ncols(), "column index out of bounds");
        let (row_indices, values) = self.data.sparse_column(j);
        LazyColumn::new(
            SparseColumnRef::new(row_indices, values, self.nrows()),
            self.centers.as_ref().map_or_else(F::zero, |c| c[j]),
            self.scales.as_ref().map_or_else(F::one, |s| s[j]),
        )
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

impl<M, X, Y, F> MatVecInto<X, Y> for LazyMatrix<M, F>
where
    F: Scalar,
    M: MatVecInto<X, Y>,
    X: Clone + ElemDivAssign<F> + DotSlice<F>,
    Y: SubScalarAssign<F>,
{
    /// `out = X̃ v = X (S⁻¹ v) − 1 · (cᵀ S⁻¹ v)`.
    fn matvec_into(&self, v: &X, out: &mut Y) {
        if let Some(s) = &self.scales {
            let mut w = v.clone();
            w.elem_div_assign(s);
            self.data.matvec_into(&w, out);
            if let Some(c) = &self.centers {
                out.sub_scalar_assign(w.dot_slice(c));
            }
        } else {
            self.data.matvec_into(v, out);
            if let Some(c) = &self.centers {
                out.sub_scalar_assign(v.dot_slice(c));
            }
        }
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

impl<M, X, Y, F> MatTransposeVecInto<X, Y> for LazyMatrix<M, F>
where
    F: Scalar,
    M: MatTransposeVecInto<X, Y>,
    X: SumEntries<F>,
    Y: ScaledSubSlice<F> + ElemDivAssign<F>,
{
    /// `out = X̃ᵀ u = S⁻¹ (Xᵀ u − c · Σu)`.
    fn mat_transpose_vec_into(&self, u: &X, out: &mut Y) {
        let total = if self.centers.is_some() {
            u.sum_entries()
        } else {
            F::zero()
        };
        self.data.mat_transpose_vec_into(u, out);
        if let Some(c) = &self.centers {
            out.scaled_sub_slice(total, c);
        }
        if let Some(s) = &self.scales {
            out.elem_div_assign(s);
        }
    }
}
