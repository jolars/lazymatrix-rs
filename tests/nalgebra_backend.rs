#![cfg(feature = "nalgebra")]
//! Verification of the nalgebra sparse backend against the dense oracle.

#[path = "common/runner.rs"]
mod common;

use common::TestMatrix;
use lazymatrix::{Centering, LazyMatrix, Normalization, Scaling};
use nalgebra::{DMatrix, DMatrixView, DVector, DVectorView, DVectorViewMut, Dyn};
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

fn build_dense(tm: &TestMatrix) -> DMatrix<f64> {
    DMatrix::from_fn(tm.nrows, tm.ncols, |i, j| tm.dense[i][j])
}

#[test]
fn nalgebra_backend_suite() {
    common::run_backend_suite(build, to_dvec, from_dvec);
    common::run_sparse_columns_suite(build);
    common::run_logical_columns_suite(build);
    common::run_backend_suite(build_dense, to_dvec, from_dvec);
    common::run_logical_columns_suite(build_dense);
}

#[test]
fn nalgebra_strided_views_are_borrowed() {
    let design_storage = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
    let design = DMatrixView::<_, Dyn, Dyn>::from_slice_with_strides(&design_storage, 4, 2, 2, 1);
    let lazy = LazyMatrix::new(design, Normalization::new(Centering::Mean, Scaling::L2));

    let vector_storage = [1.0, -99.0, 2.0, -99.0, 3.0, -99.0, 4.0];
    let vector = DVectorView::<_, Dyn, Dyn>::from_slice_with_strides(&vector_storage, 4, 2, 1);
    let column = lazy.column(0);
    let expected_dot = (0..4)
        .map(|i| {
            let raw = (i + 1) as f64;
            (raw - 2.5) / 5.0_f64.sqrt() * vector[i]
        })
        .sum::<f64>();
    approx::assert_abs_diff_eq!(column.dot(&vector), expected_dot, epsilon = 1e-12);

    let mut destination_storage = [0.0; 7];
    let mut destination = DVectorViewMut::<_, Dyn, Dyn>::from_slice_with_strides_mut(
        &mut destination_storage,
        4,
        2,
        1,
    );
    column.scaled_add_to(0.5, &mut destination);
    for i in 0..4 {
        let expected = 0.5 * ((i + 1) as f64 - 2.5) / 5.0_f64.sqrt();
        approx::assert_abs_diff_eq!(destination[i], expected, epsilon = 1e-12);
    }
}
