/// Dimensions of a matrix or linear operator.
pub trait MatrixShape {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
}

/// Matrix–vector product `A x`, returning a freshly allocated vector of length
/// `nrows`.
pub trait MatVec<V>: MatrixShape {
    fn matvec(&self, x: &V) -> V;
}

/// Matrix–vector product `out = A x`, overwriting reusable output storage.
///
/// `X` and `Y` may differ so that backends can accept borrowed or strided
/// inputs while writing into an owned output. Implementations must not depend
/// on the previous values in `out`.
///
/// # Panics
///
/// Panics unless `x` has length `ncols` and `out` has length `nrows`.
pub trait MatVecInto<X, Y = X>: MatrixShape {
    fn matvec_into(&self, x: &X, out: &mut Y);
}

/// Transposed matrix–vector product `Aᵀ x`, returning a freshly allocated vector
/// of length `ncols`.
pub trait MatTransposeVec<V>: MatrixShape {
    fn mat_transpose_vec(&self, x: &V) -> V;
}

/// Transposed matrix–vector product `out = Aᵀ x`, overwriting reusable output
/// storage.
///
/// `X` and `Y` may differ so that backends can accept borrowed or strided
/// inputs while writing into an owned output. Implementations must not depend
/// on the previous values in `out`.
///
/// # Panics
///
/// Panics unless `x` has length `nrows` and `out` has length `ncols`.
pub trait MatTransposeVecInto<X, Y = X>: MatrixShape {
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y);
}

impl<M: MatrixShape + ?Sized> MatrixShape for &M {
    fn nrows(&self) -> usize {
        (**self).nrows()
    }

    fn ncols(&self) -> usize {
        (**self).ncols()
    }
}

impl<M, V> MatVec<V> for &M
where
    M: MatVec<V> + ?Sized,
{
    fn matvec(&self, x: &V) -> V {
        (**self).matvec(x)
    }
}

impl<M, X, Y> MatVecInto<X, Y> for &M
where
    M: MatVecInto<X, Y> + ?Sized,
{
    fn matvec_into(&self, x: &X, out: &mut Y) {
        (**self).matvec_into(x, out);
    }
}

impl<M, V> MatTransposeVec<V> for &M
where
    M: MatTransposeVec<V> + ?Sized,
{
    fn mat_transpose_vec(&self, x: &V) -> V {
        (**self).mat_transpose_vec(x)
    }
}

impl<M, X, Y> MatTransposeVecInto<X, Y> for &M
where
    M: MatTransposeVecInto<X, Y> + ?Sized,
{
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) {
        (**self).mat_transpose_vec_into(x, out);
    }
}
