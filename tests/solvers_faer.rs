#![cfg(feature = "faer")]
//! End-to-end solver check: gradient-descent least squares driven through the
//! lazy operator must recover a known solution.
//!
//! The cross-check is deliberate: the target `y` is built from the **densely
//! materialized** `X̃` (the oracle path), while the solver only ever calls the
//! **lazy** `matvec` / `mat_transpose_vec`. Recovering `β*` therefore confirms
//! the lazy operator equals the materialized one through a full solver loop.

#[path = "common/runner.rs"]
mod common;

use common::{dense_matvec, materialize, random_matrix};
use faer::Col;
use faer::sparse::{SparseColMat, Triplet};
use lazymatrix::{Centering, LazyMatrix, MatTransposeVec, MatVec, Normalization, Scaling};

fn build_faer(tm: &common::TestMatrix) -> SparseColMat<usize, f64> {
    let t: Vec<Triplet<usize, usize, f64>> = tm
        .triplets
        .iter()
        .map(|&(r, c, v)| Triplet::new(r, c, v))
        .collect();
    SparseColMat::try_new_from_triplets(tm.nrows, tm.ncols, &t).unwrap()
}

fn dot(a: &Col<f64>, b: &Col<f64>) -> f64 {
    (0..a.nrows()).map(|i| a[i] * b[i]).sum()
}
fn norm2(a: &Col<f64>) -> f64 {
    dot(a, a).sqrt()
}
fn sub(a: &Col<f64>, b: &Col<f64>) -> Col<f64> {
    Col::from_fn(a.nrows(), |i| a[i] - b[i])
}

fn estimate_lipschitz<Op>(op: &Op, ncols: usize, iters: usize) -> f64
where
    Op: MatVec<Col<f64>> + MatTransposeVec<Col<f64>>,
{
    let mut v = Col::<f64>::from_fn(ncols, |i| 1.0 + (i as f64) * 0.01);
    let mut lambda = 1.0;
    for _ in 0..iters {
        let av = op.mat_transpose_vec(&op.matvec(&v));
        lambda = norm2(&av);
        if lambda == 0.0 {
            break;
        }
        let inv = 1.0 / lambda;
        for i in 0..ncols {
            v[i] = av[i] * inv;
        }
    }
    lambda
}

fn gradient_descent<Op>(
    op: &Op,
    y: &Col<f64>,
    ncols: usize,
    step: f64,
    max_iter: usize,
    tol: f64,
) -> (Col<f64>, f64)
where
    Op: MatVec<Col<f64>> + MatTransposeVec<Col<f64>>,
{
    let mut beta = Col::<f64>::zeros(ncols);
    let mut gnorm = f64::INFINITY;
    for _ in 0..max_iter {
        let resid = sub(&op.matvec(&beta), y);
        let grad = op.mat_transpose_vec(&resid);
        gnorm = norm2(&grad);
        if gnorm < tol {
            break;
        }
        for i in 0..ncols {
            beta[i] -= step * grad[i];
        }
    }
    (beta, gnorm)
}

#[test]
fn gd_recovers_known_solution() {
    let tm = random_matrix(7, 90, 6, 0.5);
    let spec = Normalization::new(Centering::Mean, Scaling::Sd);
    let lazy = LazyMatrix::normalized(build_faer(&tm), tm.nrows, tm.ncols, spec);

    // β*, and a noiseless target produced by the DENSE materialized X̃.
    let beta_star: Vec<f64> = (0..tm.ncols).map(|j| (j as f64 - 2.5) * 0.7).collect();
    let xtilde = materialize(&tm.dense, lazy.centers(), lazy.scales());
    let y_vec = dense_matvec(&xtilde, &beta_star);
    let y = Col::from_fn(tm.nrows, |i| y_vec[i]);

    let l = estimate_lipschitz(&lazy, tm.ncols, 300);
    let (beta, gnorm) = gradient_descent(&lazy, &y, tm.ncols, 1.0 / l, 200_000, 1e-12);

    // First-order optimality of the lazy problem.
    assert!(gnorm < 1e-8, "gradient norm not small enough: {gnorm:e}");
    // Recovery of the known coefficients.
    for j in 0..tm.ncols {
        approx::assert_abs_diff_eq!(beta[j], beta_star[j], epsilon = 1e-6);
    }
}
