#![cfg(feature = "faer")]
//! Verification of the faer sparse backend against the dense oracle.

mod common;

use common::TestMatrix;
use faer::Col;
use faer::sparse::{SparseColMat, Triplet};

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

#[test]
fn faer_backend_suite() {
    common::run_backend_suite(build, to_col, from_col);
}
