//! `faer` sparse backend (feature `faer`).
//!
//! Implements the five vector traits on [`faer::Col`] and the operator/stats
//! traits on [`faer::sparse::SparseColMat`]. The matrix–vector products delegate
//! to faer's `sparse_dense_matmul`; the transpose uses a transposed *view*
//! (`as_ref().transpose()`) and never materializes a transposed matrix. Column
//! statistics walk the CSC arrays directly, treating absent entries as zero.

use faer::sparse::SparseColMat;
use faer::sparse::linalg::matmul::sparse_dense_matmul;
use faer::{Accum, Col, Par};

use crate::traits::{
    ColumnStats, DotSlice, ElemDivAssign, MatTransposeVec, MatVec, MatrixShape, Scalar,
    ScaledSubSlice, SubScalarAssign, SumEntries, max_or_nan, sparse_column_sd,
};

// --- vector traits on Col<F> (scalar arithmetic only: bound F: Scalar) -------

impl<F: Scalar> ElemDivAssign<F> for Col<F> {
    fn elem_div_assign(&mut self, coeffs: &[F]) {
        assert_eq!(
            self.nrows(),
            coeffs.len(),
            "elem_div_assign: length mismatch"
        );
        for j in 0..self.nrows() {
            self[j] = self[j] / coeffs[j];
        }
    }
}

impl<F: Scalar> DotSlice<F> for Col<F> {
    fn dot_slice(&self, coeffs: &[F]) -> F {
        assert_eq!(self.nrows(), coeffs.len(), "dot_slice: length mismatch");
        (0..self.nrows()).map(|j| self[j] * coeffs[j]).sum()
    }
}

impl<F: Scalar> SubScalarAssign<F> for Col<F> {
    fn sub_scalar_assign(&mut self, k: F) {
        for j in 0..self.nrows() {
            self[j] = self[j] - k;
        }
    }
}

impl<F: Scalar> SumEntries<F> for Col<F> {
    fn sum_entries(&self) -> F {
        (0..self.nrows()).map(|j| self[j]).sum()
    }
}

impl<F: Scalar> ScaledSubSlice<F> for Col<F> {
    fn scaled_sub_slice(&mut self, k: F, coeffs: &[F]) {
        assert_eq!(
            self.nrows(),
            coeffs.len(),
            "scaled_sub_slice: length mismatch"
        );
        for j in 0..self.nrows() {
            self[j] = self[j] - k * coeffs[j];
        }
    }
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

impl<F> MatVec<Col<F>> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn matvec(&self, x: &Col<F>) -> Col<F> {
        assert_eq!(self.ncols(), x.nrows(), "matvec: dimension mismatch");
        let mut y = Col::<F>::zeros(self.nrows());
        sparse_dense_matmul(
            y.as_mat_mut(),
            Accum::Replace,
            self.as_ref(),
            x.as_mat(),
            F::one(),
            Par::Seq,
        );
        y
    }
}

impl<F> MatTransposeVec<Col<F>> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn mat_transpose_vec(&self, x: &Col<F>) -> Col<F> {
        assert_eq!(
            self.nrows(),
            x.nrows(),
            "mat_transpose_vec: dimension mismatch"
        );
        let mut y = Col::<F>::zeros(self.ncols());
        sparse_dense_matmul(
            y.as_mat_mut(),
            Accum::Replace,
            self.as_ref().transpose(),
            x.as_mat(),
            F::one(),
            Par::Seq,
        );
        y
    }
}

// --- column statistics over the CSC arrays ------------------------------------

impl<F> ColumnStats<F> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn col_means(&self) -> Vec<F> {
        let n = F::from_usize(self.nrows()).unwrap();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        (0..self.ncols())
            .map(|j| {
                let (start, end) = (col_ptr[j], col_ptr[j + 1]);
                let sum: F = vals[start..end].iter().copied().sum();
                sum / n
            })
            .collect()
    }

    fn col_sds(&self) -> Vec<F> {
        let nrows = self.nrows();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        (0..self.ncols())
            .map(|j| {
                let (start, end) = (col_ptr[j], col_ptr[j + 1]);
                sparse_column_sd(&vals[start..end], nrows)
            })
            .collect()
    }

    fn col_maxabs(&self) -> Vec<F> {
        let col_ptr = self.col_ptr();
        let vals = self.val();
        (0..self.ncols())
            .map(|j| {
                let (start, end) = (col_ptr[j], col_ptr[j + 1]);
                max_or_nan(vals[start..end].iter().map(|v| v.abs()))
            })
            .collect()
    }

    fn col_l2(&self) -> Vec<F> {
        let col_ptr = self.col_ptr();
        let vals = self.val();
        (0..self.ncols())
            .map(|j| {
                let (start, end) = (col_ptr[j], col_ptr[j + 1]);
                let sum_sq: F = vals[start..end].iter().map(|&v| v * v).sum();
                sum_sq.sqrt()
            })
            .collect()
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
        (0..self.ncols())
            .map(|j| {
                let (start, end) = (col_ptr[j], col_ptr[j + 1]);
                let c = centers[j];
                let nnz = end - start;
                // Σ_stored (v − c)² + (n − nnz)·c²
                let stored: F = vals[start..end].iter().map(|&v| (v - c) * (v - c)).sum();
                let implicit = F::from_usize(nrows - nnz).unwrap();
                (stored + implicit * c * c).sqrt()
            })
            .collect()
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
        (0..self.ncols())
            .map(|j| {
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
            .collect()
    }
}
