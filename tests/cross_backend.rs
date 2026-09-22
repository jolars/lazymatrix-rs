#![cfg(any(
    all(feature = "faer_all", feature = "nalgebra_all"),
    all(feature = "faer_all", feature = "ndarray_all"),
    all(feature = "nalgebra_all", feature = "ndarray_all"),
    all(
        feature = "sprs_all",
        any(
            feature = "faer_all",
            feature = "nalgebra_all",
            feature = "ndarray_all"
        )
    ),
))]
//! Cross-backend agreement: the same logical matrix, normalized the same way,
//! produces matching operator outputs under each pair of enabled backends.

#[path = "common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::*;

#[path = "common/runner.rs"]
mod common;

use common::{TestMatrix, assert_close, random_matrix, random_vec};
use lazymatrix::{Centering, LazyMatrix, MatTransposeVec, MatVec, Normalization, Scaling};

type Products = (Vec<f64>, Vec<f64>);

#[cfg(feature = "sprs_all")]
fn sprs_products(tm: &TestMatrix, spec: Normalization, v: &[f64], u: &[f64]) -> Products {
    let mut triplets = sprs::TriMat::new((tm.nrows, tm.ncols));
    for &(row, col, value) in &tm.triplets {
        triplets.add_triplet(row, col, value);
    }
    let lazy = LazyMatrix::new(triplets.to_csc::<usize>(), spec);
    (
        lazy.matvec(&v.to_vec()),
        lazy.mat_transpose_vec(&u.to_vec()),
    )
}

#[cfg(feature = "faer_all")]
fn faer_products(tm: &TestMatrix, spec: Normalization, v: &[f64], u: &[f64]) -> Products {
    use faer::Col;
    use faer::sparse::{SparseColMat, Triplet};

    let t: Vec<Triplet<usize, usize, f64>> = tm
        .triplets
        .iter()
        .map(|&(r, c, v)| Triplet::new(r, c, v))
        .collect();
    let matrix = SparseColMat::try_new_from_triplets(tm.nrows, tm.ncols, &t).unwrap();
    let lazy = LazyMatrix::new(matrix, spec);
    let y = lazy.matvec(&Col::from_fn(v.len(), |i| v[i]));
    let z = lazy.mat_transpose_vec(&Col::from_fn(u.len(), |i| u[i]));
    (
        (0..y.nrows()).map(|i| y[i]).collect(),
        (0..z.nrows()).map(|i| z[i]).collect(),
    )
}

#[cfg(feature = "nalgebra_all")]
fn nalgebra_products(tm: &TestMatrix, spec: Normalization, v: &[f64], u: &[f64]) -> Products {
    use nalgebra::DVector;
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    let mut coo = CooMatrix::new(tm.nrows, tm.ncols);
    for &(r, c, v) in &tm.triplets {
        coo.push(r, c, v);
    }
    let lazy = LazyMatrix::new(CscMatrix::from(&coo), spec);
    (
        lazy.matvec(&DVector::from_column_slice(v))
            .as_slice()
            .to_vec(),
        lazy.mat_transpose_vec(&DVector::from_column_slice(u))
            .as_slice()
            .to_vec(),
    )
}

#[cfg(feature = "ndarray_all")]
fn ndarray_products(tm: &TestMatrix, spec: Normalization, v: &[f64], u: &[f64]) -> Products {
    use ndarray::{Array1, Array2};

    let matrix = Array2::from_shape_fn((tm.nrows, tm.ncols), |(i, j)| tm.dense[i][j]);
    let lazy = LazyMatrix::new(matrix, spec);
    (
        lazy.matvec(&Array1::from_vec(v.to_vec())).to_vec(),
        lazy.mat_transpose_vec(&Array1::from_vec(u.to_vec()))
            .to_vec(),
    )
}

fn check_agreement(
    left: impl Fn(&TestMatrix, Normalization, &[f64], &[f64]) -> Products,
    right: impl Fn(&TestMatrix, Normalization, &[f64], &[f64]) -> Products,
) {
    let tm = random_matrix(42, 14, 9, 0.4);
    let v = random_vec(43, tm.ncols);
    let u = random_vec(44, tm.nrows);

    for center in [Centering::None, Centering::Mean, Centering::Min] {
        for scale in [
            Scaling::None,
            Scaling::Sd,
            Scaling::L1,
            Scaling::L2,
            Scaling::MaxAbs,
            Scaling::Range,
        ] {
            let spec = Normalization::new(center, scale);

            let (left_y, left_z) = left(&tm, spec, &v, &u);
            let (right_y, right_z) = right(&tm, spec, &v, &u);
            assert_close(&left_y, &right_y, 1e-10);
            assert_close(&left_z, &right_z, 1e-10);
        }
    }
}

#[cfg(all(feature = "faer_all", feature = "nalgebra_all"))]
#[test]
fn faer_and_nalgebra_agree() {
    check_agreement(faer_products, nalgebra_products);
}

#[cfg(all(feature = "faer_all", feature = "ndarray_all"))]
#[test]
fn faer_and_ndarray_agree() {
    check_agreement(faer_products, ndarray_products);
}

#[cfg(all(feature = "nalgebra_all", feature = "ndarray_all"))]
#[test]
fn nalgebra_and_ndarray_agree() {
    check_agreement(nalgebra_products, ndarray_products);
}

#[cfg(all(feature = "sprs_all", feature = "faer_all"))]
#[test]
fn sprs_and_faer_agree() {
    check_agreement(sprs_products, faer_products);
}

#[cfg(all(feature = "sprs_all", feature = "nalgebra_all"))]
#[test]
fn sprs_and_nalgebra_agree() {
    check_agreement(sprs_products, nalgebra_products);
}

#[cfg(all(feature = "sprs_all", feature = "ndarray_all"))]
#[test]
fn sprs_and_ndarray_agree() {
    check_agreement(sprs_products, ndarray_products);
}
