//! `nalgebra` sparse backend (feature `nalgebra`).
//!
//! Implements the vector traits on [`nalgebra::DVector`] and the operator,
//! statistics, and sparse-column traits on [`nalgebra_sparse::CscMatrix`]. The
//! matrix–vector products delegate to `spmm_csc_dense` (`Op::NoOp` /
//! `Op::Transpose`); the transpose never materializes. Column statistics and
//! borrowed column access use the CSC arrays via [`CscMatrix::csc_data`],
//! treating absent entries as zero.

use nalgebra::{ClosedAddAssign, ClosedMulAssign, DVector};
use nalgebra_sparse::CscMatrix;
use nalgebra_sparse::ops::Op;
use nalgebra_sparse::ops::serial::spmm_csc_dense;

use crate::SparseColumnRef;
use crate::traits::{
    ColumnStats, DotProduct, DotSlice, ElemDivAssign, L2Norm, MatTransposeVec, MatTransposeVecInto,
    MatVec, MatVecInto, MatrixShape, MaybeSend, MaybeSync, RawColumns, Scalar, ScaleAssign,
    ScaledAddAssign, ScaledSubSlice, SparseColumns, SubScalarAssign, SumEntries, collect_columns,
    max_or_nan, min_or_nan, range_or_nan, sparse_column_sd,
};

// --- vector traits on DVector<F> ---------------------------------------------

impl<F> DotProduct<F> for DVector<F>
where
    F: Scalar + nalgebra::RealField,
{
    fn dot(&self, other: &Self) -> F {
        DVector::dot(self, other)
    }
}

impl<F> L2Norm<F> for DVector<F>
where
    F: Scalar + nalgebra::RealField,
{
    fn norm_l2(&self) -> F {
        self.norm()
    }
}

impl<F> ScaledAddAssign<F> for DVector<F>
where
    F: Scalar + nalgebra::RealField,
{
    fn scaled_add_assign(&mut self, alpha: F, other: &Self) {
        self.axpy(alpha, other, F::one());
    }
}

impl<F> ScaleAssign<F> for DVector<F>
where
    F: Scalar + nalgebra::RealField,
{
    fn scale_assign(&mut self, alpha: F) {
        self.scale_mut(alpha);
    }
}

impl<F: Scalar + nalgebra::Scalar> ElemDivAssign<F> for DVector<F> {
    fn elem_div_assign(&mut self, coeffs: &[F]) {
        let s = self.as_mut_slice();
        assert_eq!(s.len(), coeffs.len(), "elem_div_assign: length mismatch");
        for (a, &c) in s.iter_mut().zip(coeffs) {
            *a = *a / c;
        }
    }
}

impl<F: Scalar + nalgebra::Scalar> DotSlice<F> for DVector<F> {
    fn dot_slice(&self, coeffs: &[F]) -> F {
        let s = self.as_slice();
        assert_eq!(s.len(), coeffs.len(), "dot_slice: length mismatch");
        s.iter().zip(coeffs).map(|(&a, &c)| a * c).sum()
    }
}

impl<F: Scalar + nalgebra::Scalar> SubScalarAssign<F> for DVector<F> {
    fn sub_scalar_assign(&mut self, k: F) {
        for a in self.as_mut_slice() {
            *a = *a - k;
        }
    }
}

impl<F: Scalar + nalgebra::Scalar> SumEntries<F> for DVector<F> {
    fn sum_entries(&self) -> F {
        self.as_slice().iter().copied().sum()
    }
}

impl<F: Scalar + nalgebra::Scalar> ScaledSubSlice<F> for DVector<F> {
    fn scaled_sub_slice(&mut self, k: F, coeffs: &[F]) {
        let s = self.as_mut_slice();
        assert_eq!(s.len(), coeffs.len(), "scaled_sub_slice: length mismatch");
        for (a, &c) in s.iter_mut().zip(coeffs) {
            *a = *a - k * c;
        }
    }
}

// --- operator traits on CscMatrix<F> -----------------------------------------

impl<F> MatrixShape for CscMatrix<F> {
    fn nrows(&self) -> usize {
        CscMatrix::nrows(self)
    }

    fn ncols(&self) -> usize {
        CscMatrix::ncols(self)
    }
}

impl<F: Scalar> SparseColumns<F> for CscMatrix<F> {
    fn sparse_column(&self, j: usize) -> (&[usize], &[F]) {
        assert!(j < self.ncols(), "column index out of bounds");
        let (col_offsets, row_indices, values) = self.csc_data();
        let start = col_offsets[j];
        let end = col_offsets[j + 1];
        (&row_indices[start..end], &values[start..end])
    }
}

impl<F: Scalar> RawColumns<F> for CscMatrix<F> {
    type Column<'a>
        = SparseColumnRef<'a, F>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        let (rows, values) = self.sparse_column(j);
        SparseColumnRef::new(rows, values, self.nrows())
    }
}

impl<F> MatVec<DVector<F>> for CscMatrix<F>
where
    F: Scalar + nalgebra::Scalar + ClosedAddAssign + ClosedMulAssign,
{
    fn matvec(&self, x: &DVector<F>) -> DVector<F> {
        let mut out = DVector::zeros(self.nrows());
        self.matvec_into(x, &mut out);
        out
    }
}

impl<F> MatTransposeVec<DVector<F>> for CscMatrix<F>
where
    F: Scalar + nalgebra::Scalar + ClosedAddAssign + ClosedMulAssign,
{
    fn mat_transpose_vec(&self, x: &DVector<F>) -> DVector<F> {
        let mut out = DVector::zeros(self.ncols());
        self.mat_transpose_vec_into(x, &mut out);
        out
    }
}

impl<F> MatVecInto<DVector<F>> for CscMatrix<F>
where
    F: Scalar + nalgebra::Scalar + ClosedAddAssign + ClosedMulAssign,
{
    fn matvec_into(&self, x: &DVector<F>, out: &mut DVector<F>) {
        assert_eq!(self.ncols(), x.len(), "matvec_into: dimension mismatch");
        assert_eq!(
            self.nrows(),
            out.len(),
            "matvec_into: output dimension mismatch"
        );
        spmm_csc_dense(
            F::zero(),
            out.as_view_mut(),
            F::one(),
            Op::NoOp(self),
            Op::NoOp(x.as_view()),
        );
    }
}

impl<F> MatTransposeVecInto<DVector<F>> for CscMatrix<F>
where
    F: Scalar + nalgebra::Scalar + ClosedAddAssign + ClosedMulAssign,
{
    fn mat_transpose_vec_into(&self, x: &DVector<F>, out: &mut DVector<F>) {
        assert_eq!(
            self.nrows(),
            x.len(),
            "mat_transpose_vec_into: dimension mismatch"
        );
        assert_eq!(
            self.ncols(),
            out.len(),
            "mat_transpose_vec_into: output dimension mismatch"
        );
        spmm_csc_dense(
            F::zero(),
            out.as_view_mut(),
            F::one(),
            Op::Transpose(self),
            Op::NoOp(x.as_view()),
        );
    }
}

// --- column statistics over the CSC arrays ------------------------------------

impl<F> ColumnStats<F> for CscMatrix<F>
where
    F: Scalar + nalgebra::Scalar + MaybeSend + MaybeSync,
{
    fn col_means(&self) -> Vec<F> {
        let n = F::from_usize(self.nrows()).unwrap();
        let (col_offsets, _row_idx, values) = self.csc_data();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_offsets[j], col_offsets[j + 1]);
            let sum: F = values[start..end].iter().copied().sum();
            sum / n
        })
    }

    fn col_sds(&self) -> Vec<F> {
        let nrows = self.nrows();
        let (col_offsets, _row_idx, values) = self.csc_data();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_offsets[j], col_offsets[j + 1]);
            sparse_column_sd(&values[start..end], nrows)
        })
    }

    fn col_mins(&self) -> Vec<F> {
        let nrows = self.nrows();
        let (col_offsets, _row_idx, values) = self.csc_data();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_offsets[j], col_offsets[j + 1]);
            min_or_nan(
                values[start..end]
                    .iter()
                    .copied()
                    .chain((end - start < nrows).then_some(F::zero())),
            )
        })
    }

    fn col_ranges(&self) -> Vec<F> {
        let nrows = self.nrows();
        let (col_offsets, _row_idx, values) = self.csc_data();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_offsets[j], col_offsets[j + 1]);
            range_or_nan(
                values[start..end]
                    .iter()
                    .copied()
                    .chain((end - start < nrows).then_some(F::zero())),
            )
        })
    }

    fn col_maxabs(&self) -> Vec<F> {
        let (col_offsets, _row_idx, values) = self.csc_data();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_offsets[j], col_offsets[j + 1]);
            max_or_nan(values[start..end].iter().map(|v| v.abs()))
        })
    }

    fn col_l1(&self) -> Vec<F> {
        let (col_offsets, _row_idx, values) = self.csc_data();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_offsets[j], col_offsets[j + 1]);
            values[start..end].iter().map(|value| value.abs()).sum()
        })
    }

    fn col_l2(&self) -> Vec<F> {
        let (col_offsets, _row_idx, values) = self.csc_data();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_offsets[j], col_offsets[j + 1]);
            let sum_sq: F = values[start..end].iter().map(|&v| v * v).sum();
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
        let (col_offsets, _row_idx, values) = self.csc_data();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_offsets[j], col_offsets[j + 1]);
            let c = centers[j];
            let nnz = end - start;
            let stored: F = values[start..end].iter().map(|&v| (v - c) * (v - c)).sum();
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
        let (col_offsets, _row_idx, values) = self.csc_data();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_offsets[j], col_offsets[j + 1]);
            let center = centers[j];
            let stored: F = values[start..end]
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
        let (col_offsets, _row_idx, values) = self.csc_data();
        collect_columns(self.ncols(), |j| {
            let (start, end) = (col_offsets[j], col_offsets[j + 1]);
            let c = centers[j];
            if end - start < nrows {
                max_or_nan(
                    values[start..end]
                        .iter()
                        .map(|&v| (v - c).abs())
                        .chain(std::iter::once(c.abs())),
                )
            } else {
                max_or_nan(values[start..end].iter().map(|&v| (v - c).abs()))
            }
        })
    }
}
