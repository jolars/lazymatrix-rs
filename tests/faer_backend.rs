#![cfg(feature = "faer")]
//! Verification of the faer sparse backend against the dense oracle.

#[path = "common/runner.rs"]
mod common;

use common::TestMatrix;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Col, Mat};
use lazymatrix::{Centering, LazyMatrix, Normalization, Scaling};

fn build(tm: &TestMatrix) -> SparseColMat<usize, f64> {
    let triplets: Vec<Triplet<usize, usize, f64>> = tm
        .triplets
        .iter()
        .map(|&(r, c, v)| Triplet::new(r, c, v))
        .collect();
    SparseColMat::try_new_from_triplets(tm.nrows, tm.ncols, &triplets).expect("valid triplets")
}

fn to_col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn from_col(c: &Col<f64>) -> Vec<f64> {
    (0..c.nrows()).map(|i| c[i]).collect()
}

fn build_dense(tm: &TestMatrix) -> Mat<f64> {
    Mat::from_fn(tm.nrows, tm.ncols, |i, j| tm.dense[i][j])
}

#[test]
fn faer_backend_suite() {
    common::run_backend_suite(build, to_col, from_col);
    common::run_sparse_columns_suite(build);
    common::run_logical_columns_suite(build);
    common::run_backend_suite(build_dense, to_col, from_col);
    common::run_logical_columns_suite(build_dense);
}

#[test]
fn faer_strided_views_are_borrowed() {
    let design_storage = Mat::from_fn(2, 4, |i, j| (i * 4 + j + 1) as f64);
    let design = design_storage.as_ref().transpose();
    let lazy = LazyMatrix::new(design, Normalization::new(Centering::Mean, Scaling::L2));

    let vector_storage = Mat::from_fn(2, 4, |i, j| (i + j + 1) as f64);
    let vector = vector_storage.row(1).transpose();
    let column = lazy.column(0);
    let expected_dot = (0..4)
        .map(|i| {
            let raw = design_storage[(0, i)];
            let center = 2.5;
            let scale = 5.0_f64.sqrt();
            (raw - center) / scale * vector[i]
        })
        .sum::<f64>();
    approx::assert_abs_diff_eq!(column.dot(&vector), expected_dot, epsilon = 1e-12);

    let mut destination_storage = Mat::zeros(2, 4);
    let mut destination = destination_storage.row_mut(1).transpose_mut();
    column.scaled_add_to(0.5, &mut destination);
    for i in 0..4 {
        let expected = 0.5 * (design_storage[(0, i)] - 2.5) / 5.0_f64.sqrt();
        approx::assert_abs_diff_eq!(destination[i], expected, epsilon = 1e-12);
    }
}
