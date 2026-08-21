//! Dense faer backend, including arbitrary-stride immutable matrix views.

use faer::{Col, ColMut, ColRef, Mat, MatRef};

use crate::traits::{
    ColumnStats, MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixShape, MaybeSend,
    MaybeSync, RawColumn, RawColumns, Scalar, VectorView, VectorViewMut, collect_columns,
    max_or_nan, min_or_nan, range_or_nan,
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

fn matvec_into<F: Scalar>(
    nrows: usize,
    ncols: usize,
    at: impl Fn(usize, usize) -> F,
    x: &Col<F>,
    out: &mut Col<F>,
) {
    assert_eq!(ncols, x.nrows(), "matvec_into: dimension mismatch");
    assert_eq!(nrows, out.nrows(), "matvec_into: output dimension mismatch");
    for i in 0..nrows {
        out[i] = (0..ncols).map(|j| at(i, j) * x[j]).sum();
    }
}

fn mat_transpose_vec_into<F: Scalar>(
    nrows: usize,
    ncols: usize,
    at: impl Fn(usize, usize) -> F,
    x: &Col<F>,
    out: &mut Col<F>,
) {
    assert_eq!(
        nrows,
        x.nrows(),
        "mat_transpose_vec_into: dimension mismatch"
    );
    assert_eq!(
        ncols,
        out.nrows(),
        "mat_transpose_vec_into: output dimension mismatch"
    );
    for j in 0..ncols {
        out[j] = (0..nrows).map(|i| at(i, j) * x[i]).sum();
    }
}

macro_rules! impl_dense_ops {
    ($matrix:ty) => {
        impl<F: Scalar> MatVec<Col<F>> for $matrix {
            fn matvec(&self, x: &Col<F>) -> Col<F> {
                let mut out = Col::from_fn(self.nrows(), |_| F::zero());
                self.matvec_into(x, &mut out);
                out
            }
        }

        impl<F: Scalar> MatTransposeVec<Col<F>> for $matrix {
            fn mat_transpose_vec(&self, x: &Col<F>) -> Col<F> {
                let mut out = Col::from_fn(self.ncols(), |_| F::zero());
                self.mat_transpose_vec_into(x, &mut out);
                out
            }
        }

        impl<F: Scalar> MatVecInto<Col<F>> for $matrix {
            fn matvec_into(&self, x: &Col<F>, out: &mut Col<F>) {
                matvec_into(self.nrows(), self.ncols(), |i, j| self[(i, j)], x, out)
            }
        }

        impl<F: Scalar> MatTransposeVecInto<Col<F>> for $matrix {
            fn mat_transpose_vec_into(&self, x: &Col<F>, out: &mut Col<F>) {
                mat_transpose_vec_into(self.nrows(), self.ncols(), |i, j| self[(i, j)], x, out)
            }
        }
    };
}

impl_dense_ops!(Mat<F>);
impl_dense_ops!(MatRef<'_, F>);

fn means<F>(
    nrows: usize,
    ncols: usize,
    at: impl Fn(usize, usize) -> F + MaybeSend + MaybeSync,
) -> Vec<F>
where
    F: Scalar + MaybeSend + MaybeSync,
{
    let n = F::from_usize(nrows).unwrap();
    collect_columns(ncols, |j| (0..nrows).map(|i| at(i, j)).sum::<F>() / n)
}

fn sds<F>(
    nrows: usize,
    ncols: usize,
    at: impl Fn(usize, usize) -> F + MaybeSend + MaybeSync,
) -> Vec<F>
where
    F: Scalar + MaybeSend + MaybeSync,
{
    let centers = means(nrows, ncols, &at);
    let n = F::from_usize(nrows).unwrap();
    collect_columns(ncols, |j| {
        ((0..nrows)
            .map(|i| {
                let deviation = at(i, j) - centers[j];
                deviation * deviation
            })
            .sum::<F>()
            / n)
            .sqrt()
    })
}

macro_rules! impl_dense_stats {
    ($matrix:ty) => {
        impl<F> ColumnStats<F> for $matrix
        where
            F: Scalar + MaybeSend + MaybeSync,
        {
            fn col_means(&self) -> Vec<F> {
                means(self.nrows(), self.ncols(), |i, j| self[(i, j)])
            }

            fn col_sds(&self) -> Vec<F> {
                sds(self.nrows(), self.ncols(), |i, j| self[(i, j)])
            }

            fn col_mins(&self) -> Vec<F> {
                collect_columns(self.ncols(), |j| {
                    min_or_nan((0..self.nrows()).map(|i| self[(i, j)]))
                })
            }

            fn col_ranges(&self) -> Vec<F> {
                collect_columns(self.ncols(), |j| {
                    range_or_nan((0..self.nrows()).map(|i| self[(i, j)]))
                })
            }

            fn col_maxabs(&self) -> Vec<F> {
                collect_columns(self.ncols(), |j| {
                    max_or_nan((0..self.nrows()).map(|i| self[(i, j)].abs()))
                })
            }

            fn col_l1(&self) -> Vec<F> {
                collect_columns(self.ncols(), |j| {
                    (0..self.nrows()).map(|i| self[(i, j)].abs()).sum()
                })
            }

            fn col_l2(&self) -> Vec<F> {
                collect_columns(self.ncols(), |j| {
                    (0..self.nrows())
                        .map(|i| self[(i, j)] * self[(i, j)])
                        .sum::<F>()
                        .sqrt()
                })
            }

            fn col_l2_centered(&self, centers: &[F]) -> Vec<F> {
                assert_eq!(
                    centers.len(),
                    self.ncols(),
                    "col_l2_centered: length mismatch"
                );
                collect_columns(self.ncols(), |j| {
                    (0..self.nrows())
                        .map(|i| {
                            let value = self[(i, j)] - centers[j];
                            value * value
                        })
                        .sum::<F>()
                        .sqrt()
                })
            }

            fn col_l1_centered(&self, centers: &[F]) -> Vec<F> {
                assert_eq!(
                    centers.len(),
                    self.ncols(),
                    "col_l1_centered: length mismatch"
                );
                collect_columns(self.ncols(), |j| {
                    (0..self.nrows())
                        .map(|i| (self[(i, j)] - centers[j]).abs())
                        .sum()
                })
            }

            fn col_maxabs_centered(&self, centers: &[F]) -> Vec<F> {
                assert_eq!(
                    centers.len(),
                    self.ncols(),
                    "col_maxabs_centered: length mismatch"
                );
                collect_columns(self.ncols(), |j| {
                    max_or_nan((0..self.nrows()).map(|i| (self[(i, j)] - centers[j]).abs()))
                })
            }
        }
    };
}

impl_dense_stats!(Mat<F>);
impl_dense_stats!(MatRef<'_, F>);
