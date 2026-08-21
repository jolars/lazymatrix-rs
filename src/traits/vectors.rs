use super::Scalar;

/// Read-only indexed access to a dense logical vector.
///
/// Implementations may be contiguous or strided. Indexing must take O(1)
/// time; algorithms must not assume that the entries form a slice.
pub trait VectorView<F: Scalar> {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> F;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn sum(&self) -> F {
        (0..self.len()).map(|i| self.get(i)).sum()
    }
}

/// Mutable indexed access to a dense logical vector.
pub trait VectorViewMut<F: Scalar>: VectorView<F> {
    fn set(&mut self, index: usize, value: F);
}

impl<F: Scalar> VectorView<F> for [F] {
    fn len(&self) -> usize {
        <[F]>::len(self)
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }
}

impl<F: Scalar> VectorViewMut<F> for [F] {
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

impl<F: Scalar> VectorView<F> for Vec<F> {
    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }
}

impl<F: Scalar> VectorViewMut<F> for Vec<F> {
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

impl<F: Scalar, const N: usize> VectorView<F> for [F; N] {
    fn len(&self) -> usize {
        N
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }
}

impl<F: Scalar, const N: usize> VectorViewMut<F> for [F; N] {
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

/// Dot product of two backend vectors: `Σ self[i] · other[i]`.
///
/// # Panics
///
/// Panics if the vectors have different lengths.
pub trait DotProduct<F: Scalar> {
    fn dot(&self, other: &Self) -> F;
}

/// Euclidean norm of a backend vector.
pub trait L2Norm<F: Scalar> {
    fn norm_l2(&self) -> F;
}

/// In-place scaled vector addition: `self += alpha · other`.
///
/// # Panics
///
/// Panics if the vectors have different lengths.
pub trait ScaledAddAssign<F: Scalar> {
    fn scaled_add_assign(&mut self, alpha: F, other: &Self);
}

/// In-place vector scaling: `self *= alpha`.
pub trait ScaleAssign<F: Scalar> {
    fn scale_assign(&mut self, alpha: F);
}

/// In-place elementwise division by a coefficient slice: `self[i] /= coeffs[i]`.
///
/// Used to apply `S⁻¹` (scaling). Callers guarantee `coeffs[i] != 0`; the
/// `normalized` constructor floors zero scales to `1` so a constant column never
/// triggers a division by zero.
pub trait ElemDivAssign<F: Scalar> {
    fn elem_div_assign(&mut self, coeffs: &[F]);
}

/// Dot product of `self` with a coefficient slice: `Σ self[i] · coeffs[i]`.
///
/// Used to form the scalar `cᵀw` in the forward operator.
pub trait DotSlice<F: Scalar> {
    fn dot_slice(&self, coeffs: &[F]) -> F;
}

/// In-place broadcast subtraction of a scalar: `self[i] -= k`.
///
/// Used to apply the `− 1·(cᵀw)` centering correction in the forward operator.
pub trait SubScalarAssign<F: Scalar> {
    fn sub_scalar_assign(&mut self, k: F);
}

/// Sum of all entries: `Σ self[i]`.
///
/// Used to form `Σu` in the transpose operator.
pub trait SumEntries<F: Scalar> {
    fn sum_entries(&self) -> F;
}

/// In-place scaled subtraction against a coefficient slice: `self[i] -= k · coeffs[i]`.
///
/// Used to apply the `− Σu · c` centering correction in the transpose operator.
pub trait ScaledSubSlice<F: Scalar> {
    fn scaled_sub_slice(&mut self, k: F, coeffs: &[F]);
}
