use crate::MatrixShape;
use crate::{MatrixWrite, Scalar, VectorView, WeightedGramInto, WeightedGramKernel};
use faer::sparse::SparseColMat;
use faer::{Mat, MatMut};

impl<F> MatrixShape for MatMut<'_, F> {
    fn nrows(&self) -> usize {
        MatMut::nrows(self)
    }
    fn ncols(&self) -> usize {
        MatMut::ncols(self)
    }
}
impl<F: Scalar> MatrixWrite<F> for Mat<F> {
    fn set(&mut self, row: usize, column: usize, value: F) {
        self[(row, column)] = value;
    }
}
impl<F: Scalar> MatrixWrite<F> for MatMut<'_, F> {
    fn set(&mut self, row: usize, column: usize, value: F) {
        self[(row, column)] = value;
    }
}

impl<F> WeightedGramInto<F> for SparseColMat<usize, F>
where
    F: Scalar,
{
    fn weighted_gram_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        self.weighted_gram_normalized_into(weights, None, None, out)
    }
}

impl<F> WeightedGramKernel<F> for SparseColMat<usize, F>
where
    F: Scalar,
{
    fn weighted_gram_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        crate::gram::sparse_gram(self, weights, centers, scales, out);
        Ok(())
    }
}
