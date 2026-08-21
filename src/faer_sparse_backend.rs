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
use crate::traits::{
    ColumnStats, DotProduct, DotSlice, ElemDivAssign, L2Norm, MatTransposeVec, MatTransposeVecInto,
    MatVec, MatVecInto, MatrixShape, MaybeSend, MaybeSync, RawColumns, Scalar, ScaleAssign,
    ScaledAddAssign, ScaledSubSlice, SparseColumns, SubScalarAssign, SumEntries, collect_columns,
    max_or_nan, min_or_nan, range_or_nan, sparse_column_sd,
};

#[cfg(feature = "parallel")]
fn parallelism() -> Par {
    Par::rayon(0)
}

#[cfg(not(feature = "parallel"))]
fn parallelism() -> Par {
    Par::Seq
}

// --- vector traits on Col<F> (scalar arithmetic only: bound F: Scalar) -------

impl<F: Scalar> DotProduct<F> for Col<F> {
    fn dot(&self, other: &Self) -> F {
        assert_eq!(self.nrows(), other.nrows(), "dot: length mismatch");
        (0..self.nrows()).map(|i| self[i] * other[i]).sum()
    }
}

impl<F> L2Norm<F> for Col<F>
where
    F: Scalar + faer_traits::RealField,
{
    fn norm_l2(&self) -> F {
        self.as_ref().norm_l2()
    }
}

impl<F: Scalar> ScaledAddAssign<F> for Col<F> {
    fn scaled_add_assign(&mut self, alpha: F, other: &Self) {
        assert_eq!(
            self.nrows(),
            other.nrows(),
            "scaled_add_assign: length mismatch"
        );
        for i in 0..self.nrows() {
            self[i] = self[i] + alpha * other[i];
        }
    }
}

impl<F: Scalar> ScaleAssign<F> for Col<F> {
    fn scale_assign(&mut self, alpha: F) {
        for i in 0..self.nrows() {
            self[i] = self[i] * alpha;
        }
    }
}

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
