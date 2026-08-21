//! `faer` sparse backend (feature `faer`).
//!
//! Implements the vector traits on [`faer::Col`] and the operator,
//! statistics, and sparse-column traits on [`faer::sparse::SparseColMat`]. The
//! matrix–vector products delegate to faer's `sparse_dense_matmul`; the
//! transpose uses a transposed *view* (`as_ref().transpose()`) and never
//! materializes a transposed matrix. Column statistics and borrowed column
//! access use the CSC arrays directly, treating absent entries as zero.

use faer::sparse::SparseColMat;
use faer::sparse::linalg::matmul::sparse_dense_matmul;
use faer::{Accum, Col, Par};

use crate::SparseColumnRef;
use crate::backends::support::{
    MaybeSend, MaybeSync, collect_columns, max_or_nan, min_or_nan, range_or_nan, sparse_column_sd,
};
use crate::traits::{
    ColumnStats, MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixShape, RawColumns,
    Scalar, SparseColumns,
};

#[cfg(feature = "parallel")]
fn parallelism() -> Par {
    Par::rayon(0)
}

#[cfg(not(feature = "parallel"))]
fn parallelism() -> Par {
    Par::Seq
}
// --- operator traits on SparseColMat (need the faer ComplexField bound) -------

impl<F> MatrixShape for SparseColMat<usize, F> {
    fn nrows(&self) -> usize {
        self.symbolic().nrows()
    }

    fn ncols(&self) -> usize {
        self.symbolic().ncols()
    }
}

impl<F: Scalar> SparseColumns<F> for SparseColMat<usize, F> {
    fn sparse_column(&self, j: usize) -> (&[usize], &[F]) {
        assert!(j < self.ncols(), "column index out of bounds");
        let start = self.col_ptr()[j];
        let end = self.col_ptr()[j + 1];
        (&self.row_idx()[start..end], &self.val()[start..end])
    }
}

impl<F: Scalar> RawColumns<F> for SparseColMat<usize, F> {
    type Column<'a>
        = SparseColumnRef<'a, F>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        let (rows, values) = self.sparse_column(j);
        SparseColumnRef::new(rows, values, self.nrows())
    }
}

impl<F> MatVec<Col<F>> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn matvec(&self, x: &Col<F>) -> Col<F> {
        let mut y = Col::<F>::zeros(self.nrows());
        self.matvec_into(x, &mut y);
        y
    }
}

impl<F> MatTransposeVec<Col<F>> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn mat_transpose_vec(&self, x: &Col<F>) -> Col<F> {
        let mut y = Col::<F>::zeros(self.ncols());
        self.mat_transpose_vec_into(x, &mut y);
        y
    }
}

impl<F> MatVecInto<Col<F>> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn matvec_into(&self, x: &Col<F>, out: &mut Col<F>) {
        assert_eq!(self.ncols(), x.nrows(), "matvec_into: dimension mismatch");
        assert_eq!(
            self.nrows(),
            out.nrows(),
            "matvec_into: output dimension mismatch"
        );
        sparse_dense_matmul(
            out.as_mat_mut(),
            Accum::Replace,
            self.as_ref(),
            x.as_mat(),
            F::one(),
            parallelism(),
        );
    }
}

impl<F> MatTransposeVecInto<Col<F>> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn mat_transpose_vec_into(&self, x: &Col<F>, out: &mut Col<F>) {
        assert_eq!(
            self.nrows(),
            x.nrows(),
            "mat_transpose_vec_into: dimension mismatch"
        );
        assert_eq!(
            self.ncols(),
            out.nrows(),
            "mat_transpose_vec_into: output dimension mismatch"
        );
        sparse_dense_matmul(
            out.as_mat_mut(),
            Accum::Replace,
            self.as_ref().transpose(),
            x.as_mat(),
            F::one(),
            parallelism(),
        );
    }
}

// --- column statistics over the CSC arrays ------------------------------------

impl<F> ColumnStats<F> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField + MaybeSend + MaybeSync,
{
    fn col_means(&self) -> Vec<F> {
        let n = F::from_usize(self.nrows()).unwrap();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            let sum: F = vals[start..end].iter().copied().sum();
            sum / n
        })
    }

    fn col_sds(&self) -> Vec<F> {
        let nrows = self.nrows();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            sparse_column_sd(&vals[start..end], nrows)
        })
    }

    fn col_mins(&self) -> Vec<F> {
        let nrows = self.nrows();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            min_or_nan(
                vals[start..end]
                    .iter()
                    .copied()
                    .chain((end - start < nrows).then_some(F::zero())),
            )
        })
    }

    fn col_ranges(&self) -> Vec<F> {
        let nrows = self.nrows();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            range_or_nan(
                vals[start..end]
                    .iter()
                    .copied()
                    .chain((end - start < nrows).then_some(F::zero())),
            )
        })
    }

    fn col_maxabs(&self) -> Vec<F> {
        let col_ptr = self.col_ptr();
        let vals = self.val();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            max_or_nan(vals[start..end].iter().map(|v| v.abs()))
        })
    }

    fn col_l1(&self) -> Vec<F> {
        let col_ptr = self.col_ptr();
        let vals = self.val();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            vals[start..end].iter().map(|value| value.abs()).sum()
        })
    }

    fn col_l2(&self) -> Vec<F> {
        let col_ptr = self.col_ptr();
        let vals = self.val();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            let sum_sq: F = vals[start..end].iter().map(|&v| v * v).sum();
            sum_sq.sqrt()
        })
    }

    fn col_l2_centered(&self, centers: &[F]) -> Vec<F> {
        assert_eq!(
            centers.len(),
            self.ncols(),
            "col_l2_centered: length mismatch"
        );
        let nrows = self.nrows();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            let c = centers[j];
            let nnz = end - start;
            // Σ_stored (v − c)² + (n − nnz)·c²
            let stored: F = vals[start..end].iter().map(|&v| (v - c) * (v - c)).sum();
            let implicit = F::from_usize(nrows - nnz).unwrap();
            (stored + implicit * c * c).sqrt()
        })
    }

    fn col_l1_centered(&self, centers: &[F]) -> Vec<F> {
        assert_eq!(
            centers.len(),
            self.ncols(),
            "col_l1_centered: length mismatch"
        );
        let nrows = self.nrows();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            let center = centers[j];
            let stored: F = vals[start..end]
                .iter()
                .map(|&value| (value - center).abs())
                .sum();
            if end - start < nrows {
                stored + F::from_usize(nrows - (end - start)).unwrap() * center.abs()
            } else {
                stored
            }
        })
    }

    fn col_maxabs_centered(&self, centers: &[F]) -> Vec<F> {
        assert_eq!(
            centers.len(),
            self.ncols(),
            "col_maxabs_centered: length mismatch"
        );
        let nrows = self.nrows();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            let c = centers[j];
            // Implicit zeros contribute |0 − c| = |c|.
            if end - start < nrows {
                max_or_nan(
                    vals[start..end]
                        .iter()
                        .map(|&v| (v - c).abs())
                        .chain(std::iter::once(c.abs())),
                )
            } else {
                max_or_nan(vals[start..end].iter().map(|&v| (v - c).abs()))
            }
        })
    }
}
