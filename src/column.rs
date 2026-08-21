use crate::traits::{LogicalColumn, RawColumn, Scalar, VectorView, VectorViewMut};

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
    pub(crate) fn new(row_indices: &'a [usize], values: &'a [F], len: usize) -> Self {
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
    pub(crate) fn new(raw: C, center: F, scale: F) -> Self {
        Self { raw, center, scale }
    }

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
