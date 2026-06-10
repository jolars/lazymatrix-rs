#![cfg(all(feature = "faer", feature = "nalgebra"))]
//! Cross-backend agreement: the same logical matrix, normalized the same way,
//! produces matching operator outputs under faer and nalgebra.

#[path = "common/runner.rs"]
mod common;

use common::{TestMatrix, assert_close, random_matrix, random_vec};
use faer::Col;
use faer::sparse::{SparseColMat, Triplet};
use lazymatrix::{Centering, LazyMatrix, MatTransposeVec, MatVec, Normalization, Scaling};
use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CscMatrix};

fn faer_mat(tm: &TestMatrix) -> SparseColMat<usize, f64> {
    let t: Vec<Triplet<usize, usize, f64>> = tm
        .triplets
        .iter()
        .map(|&(r, c, v)| Triplet::new(r, c, v))
        .collect();
    SparseColMat::try_new_from_triplets(tm.nrows, tm.ncols, &t).unwrap()
}

fn nalg_mat(tm: &TestMatrix) -> CscMatrix<f64> {
    let mut coo = CooMatrix::new(tm.nrows, tm.ncols);
    for &(r, c, v) in &tm.triplets {
        coo.push(r, c, v);
    }
    CscMatrix::from(&coo)
}

#[test]
fn faer_and_nalgebra_agree() {
    let tm = random_matrix(42, 14, 9, 0.4);
    let v = random_vec(43, tm.ncols);
    let u = random_vec(44, tm.nrows);

    for center in [Centering::None, Centering::Mean] {
        for scale in [Scaling::None, Scaling::Sd, Scaling::MaxAbs, Scaling::L2] {
            let spec = Normalization::new(center, scale);

            let f = LazyMatrix::normalized(faer_mat(&tm), tm.nrows, tm.ncols, spec);
            let n = LazyMatrix::normalized(nalg_mat(&tm), tm.nrows, tm.ncols, spec);

            let fv = Col::from_fn(v.len(), |i| v[i]);
            let nv = DVector::from_column_slice(&v);
            let fu = Col::from_fn(u.len(), |i| u[i]);
            let nu = DVector::from_column_slice(&u);

            let f_y: Vec<f64> = {
                let y = f.matvec(&fv);
                (0..y.nrows()).map(|i| y[i]).collect()
            };
            let n_y = n.matvec(&nv).as_slice().to_vec();
            assert_close(&f_y, &n_y, 1e-10);

            let f_t: Vec<f64> = {
                let t = f.mat_transpose_vec(&fu);
                (0..t.nrows()).map(|i| t[i]).collect()
            };
            let n_t = n.mat_transpose_vec(&nu).as_slice().to_vec();
            assert_close(&f_t, &n_t, 1e-10);
        }
    }
}
