//! Sparse operators for sprs CSC and CSR matrices, with checked borrowing.

mod csc;
mod csr;
mod stats;

pub use csc::SprsCsc;
pub use csr::SprsCsr;

use std::ops::Deref;

use sprs::{CsMatBase, SpIndex};

use crate::traits::{
    MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixShape, Scalar, VectorView,
    VectorViewMut,
};

impl<F, I, IP, IS, DS, Iptr> MatrixShape for CsMatBase<F, I, IP, IS, DS, Iptr>
where
    I: SpIndex,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
{
    fn nrows(&self) -> usize {
        self.rows()
    }

    fn ncols(&self) -> usize {
        self.cols()
    }
}

impl<F, I, IP, IS, DS, Iptr, X, Y> MatVecInto<X, Y> for CsMatBase<F, I, IP, IS, DS, Iptr>
where
    F: Scalar,
    I: SpIndex,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn matvec_into(&self, x: &X, out: &mut Y) {
        assert_eq!(self.cols(), x.len(), "matvec_into: dimension mismatch");
        assert_eq!(
            self.rows(),
            out.len(),
            "matvec_into: output dimension mismatch"
        );
        if self.is_csr() {
            for (row, values) in self.outer_iterator().enumerate() {
                out.set(row, values.iter().map(|(col, &v)| v * x.get(col)).sum());
            }
        } else {
            for row in 0..out.len() {
                out.set(row, F::zero());
            }
            for (col, values) in self.outer_iterator().enumerate() {
                let coefficient = x.get(col);
                for (row, &value) in values.iter() {
                    out.set(row, out.get(row) + value * coefficient);
                }
            }
        }
    }
}

impl<F, I, IP, IS, DS, Iptr, X, Y> MatTransposeVecInto<X, Y> for CsMatBase<F, I, IP, IS, DS, Iptr>
where
    F: Scalar,
    I: SpIndex,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) {
        assert_eq!(
            self.rows(),
            x.len(),
            "mat_transpose_vec_into: dimension mismatch"
        );
        assert_eq!(
            self.cols(),
            out.len(),
            "mat_transpose_vec_into: output dimension mismatch"
        );
        self.transpose_view().matvec_into(x, out);
    }
}

macro_rules! allocating_products {
    ($vector:ty, $zeros:expr) => {
        impl<F, I, IP, IS, DS, Iptr> MatVec<$vector> for CsMatBase<F, I, IP, IS, DS, Iptr>
        where
            F: Scalar,
            I: SpIndex,
            Iptr: SpIndex,
            IP: Deref<Target = [Iptr]>,
            IS: Deref<Target = [I]>,
            DS: Deref<Target = [F]>,
        {
            fn matvec(&self, x: &$vector) -> $vector {
                let mut out = $zeros(self.rows());
                self.matvec_into(x, &mut out);
                out
            }
        }

        impl<F, I, IP, IS, DS, Iptr> MatTransposeVec<$vector> for CsMatBase<F, I, IP, IS, DS, Iptr>
        where
            F: Scalar,
            I: SpIndex,
            Iptr: SpIndex,
            IP: Deref<Target = [Iptr]>,
            IS: Deref<Target = [I]>,
            DS: Deref<Target = [F]>,
        {
            fn mat_transpose_vec(&self, x: &$vector) -> $vector {
                let mut out = $zeros(self.cols());
                self.mat_transpose_vec_into(x, &mut out);
                out
            }
        }
    };
}

allocating_products!(Vec<F>, |n| vec![F::zero(); n]);
#[cfg(feature = "ndarray_all")]
allocating_products!(ndarray::Array1<F>, ndarray::Array1::zeros);
