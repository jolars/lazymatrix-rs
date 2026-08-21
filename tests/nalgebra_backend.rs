#![cfg(feature = "nalgebra")]
//! Verification of the nalgebra sparse backend against the dense oracle.

#[path = "common/runner.rs"]
mod common;

use common::TestMatrix;
use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CscMatrix};

fn build(tm: &TestMatrix) -> CscMatrix<f64> {
    let mut coo = CooMatrix::new(tm.nrows, tm.ncols);
    for &(r, c, v) in &tm.triplets {
        coo.push(r, c, v);
    }
    CscMatrix::from(&coo)
}

fn to_dvec(v: &[f64]) -> DVector<f64> {
    DVector::from_column_slice(v)
}

fn from_dvec(v: &DVector<f64>) -> Vec<f64> {
    v.as_slice().to_vec()
}

#[test]
fn nalgebra_backend_suite() {
    common::run_backend_suite(build, to_dvec, from_dvec);
    common::run_sparse_columns_suite(build);
}
