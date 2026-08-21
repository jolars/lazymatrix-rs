//! Dense faer backend, including arbitrary-stride immutable matrix views.

use faer::{Col, ColMut, ColRef, Mat, MatRef};

use crate::traits::{
    ColumnStats, MatTransposeVec, MatVec, MatrixShape, RawColumn, RawColumns, Scalar, VectorView,
    VectorViewMut, max_or_nan, min_or_nan, range_or_nan,
};

impl<F: Scalar> VectorView<F> for Col<F> {
    fn len(&self) -> usize {
        self.nrows()
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }
}

impl<F: Scalar> VectorViewMut<F> for Col<F> {
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

impl<F: Scalar> VectorView<F> for ColRef<'_, F> {
    fn len(&self) -> usize {
        self.nrows()
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }
}

impl<F: Scalar> VectorView<F> for ColMut<'_, F> {
    fn len(&self) -> usize {
        self.nrows()
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }
}

impl<F: Scalar> VectorViewMut<F> for ColMut<'_, F> {
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

impl<F: Scalar> RawColumn<F> for ColRef<'_, F> {
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

impl<F> MatrixShape for Mat<F> {
    fn nrows(&self) -> usize {
        Mat::nrows(self)
    }

    fn ncols(&self) -> usize {
        Mat::ncols(self)
    }
}

impl<F> MatrixShape for MatRef<'_, F> {
    fn nrows(&self) -> usize {
        MatRef::nrows(self)
    }

    fn ncols(&self) -> usize {
        MatRef::ncols(self)
    }
}

impl<F: Scalar> RawColumns<F> for Mat<F> {
    type Column<'a>
        = ColRef<'a, F>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        assert!(j < self.ncols(), "column index out of bounds");
        self.col(j)
    }
}

impl<F: Scalar> RawColumns<F> for MatRef<'_, F> {
    type Column<'a>
        = ColRef<'a, F>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        assert!(j < self.ncols(), "column index out of bounds");
        (*self).col(j)
    }
}

fn matvec<F: Scalar>(
    nrows: usize,
    ncols: usize,
    at: impl Fn(usize, usize) -> F,
    x: &Col<F>,
) -> Col<F> {
    assert_eq!(ncols, x.nrows(), "matvec: dimension mismatch");
    Col::from_fn(nrows, |i| (0..ncols).map(|j| at(i, j) * x[j]).sum())
}

fn mat_transpose_vec<F: Scalar>(
    nrows: usize,
    ncols: usize,
    at: impl Fn(usize, usize) -> F,
    x: &Col<F>,
) -> Col<F> {
    assert_eq!(nrows, x.nrows(), "mat_transpose_vec: dimension mismatch");
    Col::from_fn(ncols, |j| (0..nrows).map(|i| at(i, j) * x[i]).sum())
}

macro_rules! impl_dense_ops {
    ($matrix:ty) => {
        impl<F: Scalar> MatVec<Col<F>> for $matrix {
            fn matvec(&self, x: &Col<F>) -> Col<F> {
                matvec(self.nrows(), self.ncols(), |i, j| self[(i, j)], x)
            }
        }

        impl<F: Scalar> MatTransposeVec<Col<F>> for $matrix {
            fn mat_transpose_vec(&self, x: &Col<F>) -> Col<F> {
                mat_transpose_vec(self.nrows(), self.ncols(), |i, j| self[(i, j)], x)
            }
        }
    };
}

impl_dense_ops!(Mat<F>);
impl_dense_ops!(MatRef<'_, F>);

fn means<F: Scalar>(nrows: usize, ncols: usize, at: impl Fn(usize, usize) -> F) -> Vec<F> {
    let n = F::from_usize(nrows).unwrap();
    (0..ncols)
        .map(|j| (0..nrows).map(|i| at(i, j)).sum::<F>() / n)
        .collect()
}

fn sds<F: Scalar>(nrows: usize, ncols: usize, at: impl Fn(usize, usize) -> F) -> Vec<F> {
    let centers = means(nrows, ncols, &at);
    let n = F::from_usize(nrows).unwrap();
    (0..ncols)
        .map(|j| {
            ((0..nrows)
                .map(|i| {
                    let deviation = at(i, j) - centers[j];
                    deviation * deviation
                })
                .sum::<F>()
                / n)
                .sqrt()
        })
        .collect()
}

macro_rules! impl_dense_stats {
    ($matrix:ty) => {
        impl<F: Scalar> ColumnStats<F> for $matrix {
            fn col_means(&self) -> Vec<F> {
                means(self.nrows(), self.ncols(), |i, j| self[(i, j)])
            }

            fn col_sds(&self) -> Vec<F> {
                sds(self.nrows(), self.ncols(), |i, j| self[(i, j)])
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
                    .map(|j| {
                        max_or_nan((0..self.nrows()).map(|i| (self[(i, j)] - centers[j]).abs()))
                    })
                    .collect()
            }
        }
    };
}

impl_dense_stats!(Mat<F>);
impl_dense_stats!(MatRef<'_, F>);
