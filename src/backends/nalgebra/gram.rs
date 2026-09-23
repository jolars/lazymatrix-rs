use crate::{MatrixWrite, Scalar, VectorView, WeightedGramInto, WeightedGramKernel};
use nalgebra::{Dim, Matrix, RawStorageMut};
use nalgebra_sparse::CscMatrix;

impl<F, R, C, S> MatrixWrite<F> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorageMut<F, R, C>,
{
    fn set(&mut self, row: usize, column: usize, value: F) {
        self[(row, column)] = value;
    }
}

impl<F> WeightedGramInto<F> for CscMatrix<F>
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

impl<F> WeightedGramKernel<F> for CscMatrix<F>
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
