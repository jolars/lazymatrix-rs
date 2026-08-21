//! Dense nalgebra backend over owned matrices and immutable matrix views.

use nalgebra::base::storage::{RawStorage, RawStorageMut};
use nalgebra::{DVector, Dim, Matrix, MatrixView, U1};

use crate::traits::{
    ColumnStats, MatTransposeVec, MatVec, MatrixShape, RawColumn, RawColumns, Scalar, VectorView,
    VectorViewMut, max_or_nan, min_or_nan, range_or_nan,
};

impl<F, R, S> VectorView<F> for Matrix<F, R, U1, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    S: RawStorage<F, R, U1>,
{
    fn len(&self) -> usize {
        self.nrows()
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }
}

impl<F, R, S> VectorViewMut<F> for Matrix<F, R, U1, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    S: RawStorageMut<F, R, U1>,
{
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

impl<F, R, S> RawColumn<F> for Matrix<F, R, U1, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    S: RawStorage<F, R, U1>,
{
    fn len(&self) -> usize {
        self.nrows()
    }

    fn stored_len(&self) -> usize {
        self.nrows()
    }

    fn for_each_stored(&self, mut f: impl FnMut(usize, F)) {
        for row in 0..self.nrows() {
            f(row, self[row]);
        }
    }

    fn affine_add_to<V>(&self, raw_multiplier: F, offset: F, destination: &mut V)
    where
        V: VectorViewMut<F> + ?Sized,
    {
        assert_eq!(
            destination.len(),
            self.nrows(),
            "destination length must equal column length"
        );
        for row in 0..self.nrows() {
            destination.set(
                row,
                destination.get(row) + raw_multiplier * self[row] + offset,
            );
        }
    }
}

impl<F, R, C, S> MatrixShape for Matrix<F, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    fn nrows(&self) -> usize {
        Matrix::nrows(self)
    }

    fn ncols(&self) -> usize {
        Matrix::ncols(self)
    }
}

impl<F, R, C, S> RawColumns<F> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    type Column<'a>
        = MatrixView<'a, F, R, U1, S::RStride, S::CStride>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        assert!(j < self.ncols(), "column index out of bounds");
        self.column(j)
    }
}

impl<F, R, C, S> MatVec<DVector<F>> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    fn matvec(&self, x: &DVector<F>) -> DVector<F> {
        assert_eq!(self.ncols(), x.len(), "matvec: dimension mismatch");
        DVector::from_fn(self.nrows(), |i, _| {
            (0..self.ncols()).map(|j| self[(i, j)] * x[j]).sum()
        })
    }
}

impl<F, R, C, S> MatTransposeVec<DVector<F>> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    fn mat_transpose_vec(&self, x: &DVector<F>) -> DVector<F> {
        assert_eq!(
            self.nrows(),
            x.len(),
            "mat_transpose_vec: dimension mismatch"
        );
        DVector::from_fn(self.ncols(), |j, _| {
            (0..self.nrows()).map(|i| self[(i, j)] * x[i]).sum()
        })
    }
}

impl<F, R, C, S> ColumnStats<F> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    fn col_means(&self) -> Vec<F> {
        let n = F::from_usize(self.nrows()).unwrap();
        (0..self.ncols())
            .map(|j| (0..self.nrows()).map(|i| self[(i, j)]).sum::<F>() / n)
            .collect()
    }

    fn col_sds(&self) -> Vec<F> {
        let centers = self.col_means();
        let n = F::from_usize(self.nrows()).unwrap();
        (0..self.ncols())
            .map(|j| {
                ((0..self.nrows())
                    .map(|i| {
                        let deviation = self[(i, j)] - centers[j];
                        deviation * deviation
                    })
                    .sum::<F>()
                    / n)
                    .sqrt()
            })
            .collect()
    }

    fn col_mins(&self) -> Vec<F> {
        (0..self.ncols())
            .map(|j| min_or_nan((0..self.nrows()).map(|i| self[(i, j)])))
            .collect()
    }

    fn col_ranges(&self) -> Vec<F> {
        (0..self.ncols())
            .map(|j| range_or_nan((0..self.nrows()).map(|i| self[(i, j)])))
            .collect()
    }

    fn col_maxabs(&self) -> Vec<F> {
        (0..self.ncols())
            .map(|j| max_or_nan((0..self.nrows()).map(|i| self[(i, j)].abs())))
            .collect()
    }

    fn col_l1(&self) -> Vec<F> {
        (0..self.ncols())
            .map(|j| (0..self.nrows()).map(|i| self[(i, j)].abs()).sum())
            .collect()
    }

    fn col_l2(&self) -> Vec<F> {
        (0..self.ncols())
            .map(|j| {
                (0..self.nrows())
                    .map(|i| self[(i, j)] * self[(i, j)])
                    .sum::<F>()
                    .sqrt()
            })
            .collect()
    }

    fn col_l2_centered(&self, centers: &[F]) -> Vec<F> {
        assert_eq!(
            centers.len(),
            self.ncols(),
            "col_l2_centered: length mismatch"
        );
        (0..self.ncols())
            .map(|j| {
                (0..self.nrows())
                    .map(|i| {
                        let value = self[(i, j)] - centers[j];
                        value * value
                    })
                    .sum::<F>()
                    .sqrt()
            })
            .collect()
    }

    fn col_l1_centered(&self, centers: &[F]) -> Vec<F> {
        assert_eq!(
            centers.len(),
            self.ncols(),
            "col_l1_centered: length mismatch"
        );
        (0..self.ncols())
            .map(|j| {
                (0..self.nrows())
                    .map(|i| (self[(i, j)] - centers[j]).abs())
                    .sum()
            })
            .collect()
    }

    fn col_maxabs_centered(&self, centers: &[F]) -> Vec<F> {
        assert_eq!(
            centers.len(),
            self.ncols(),
            "col_maxabs_centered: length mismatch"
        );
        (0..self.ncols())
            .map(|j| max_or_nan((0..self.nrows()).map(|i| (self[(i, j)] - centers[j]).abs())))
            .collect()
    }
}
