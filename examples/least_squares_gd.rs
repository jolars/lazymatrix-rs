//! Least-squares by gradient descent, driven entirely through the lazy
//! normalized operator.
//!
//! Solves `min_β ½‖X̃β − y‖²` where `X̃ = (X − 1cᵀ)S⁻¹` is never materialized.
//! Every matrix touch goes through [`MatVec`] / [`MatTransposeVec`]:
//!
//! * the gradient `X̃ᵀ(X̃β − y)` is one `matvec` then one `mat_transpose_vec`;
//! * the step size is `1/L` with `L ≈ λ_max(X̃ᵀX̃)` estimated by power iteration,
//!   which is itself nothing but repeated `matvec`/`mat_transpose_vec`.
//!
//! This is the cheapest possible end-to-end check that the operator is a
//! faithful linear map: with noiseless `y = X̃β*`, GD must recover `β*`.
//!
//! Run with: `cargo run --example least_squares_gd --features faer`

use faer::Col;
use faer::sparse::{SparseColMat, Triplet};
use lazymatrix::{Centering, LazyMatrix, MatTransposeVec, MatVec, Normalization, Scaling};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

// --- tiny dense-vector helpers on faer Col (the solver's vector algebra) ------

fn dot(a: &Col<f64>, b: &Col<f64>) -> f64 {
    (0..a.nrows()).map(|i| a[i] * b[i]).sum()
}

fn norm2(a: &Col<f64>) -> f64 {
    dot(a, a).sqrt()
}

/// `y += alpha * x`
fn axpy(alpha: f64, x: &Col<f64>, y: &mut Col<f64>) {
    for i in 0..y.nrows() {
        y[i] += alpha * x[i];
    }
}

/// `a - b`
fn sub(a: &Col<f64>, b: &Col<f64>) -> Col<f64> {
    Col::from_fn(a.nrows(), |i| a[i] - b[i])
}

// --- solver pieces, generic over any lazy/backend linear operator -------------

/// Estimate `λ_max(AᵀA)` (the gradient's Lipschitz constant) by power iteration
/// on `AᵀA`, using only `matvec`/`mat_transpose_vec`.
fn estimate_lipschitz<Op>(op: &Op, ncols: usize, iters: usize) -> f64
where
    Op: MatVec<Col<f64>> + MatTransposeVec<Col<f64>>,
{
    let mut v = Col::<f64>::from_fn(ncols, |i| 1.0 + (i as f64) * 0.01);
    let mut lambda = 1.0;
    for _ in 0..iters {
        let av = op.mat_transpose_vec(&op.matvec(&v)); // AᵀA v
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

/// Gradient descent on `½‖Aβ − y‖²`. Returns `(β, iterations, final ‖grad‖)`.
fn gradient_descent<Op>(
    op: &Op,
    y: &Col<f64>,
    ncols: usize,
    step: f64,
    max_iter: usize,
    tol: f64,
) -> (Col<f64>, usize, f64)
where
    Op: MatVec<Col<f64>> + MatTransposeVec<Col<f64>>,
{
    let mut beta = Col::<f64>::zeros(ncols);
    let mut last = f64::INFINITY;
    for k in 0..max_iter {
        let resid = sub(&op.matvec(&beta), y); // Aβ − y
        let grad = op.mat_transpose_vec(&resid); // Aᵀ(Aβ − y)
        let gnorm = norm2(&grad);
        if gnorm < tol {
            return (beta, k, gnorm);
        }
        axpy(-step, &grad, &mut beta);
        last = gnorm;
    }
    (beta, max_iter, last)
}

fn main() {
    let (nrows, ncols, density) = (120, 10, 0.4);
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE);

    // Random sparse design matrix X.
    let mut triplets = Vec::new();
    for i in 0..nrows {
        for j in 0..ncols {
            if rng.random::<f64>() < density {
                triplets.push(Triplet::new(i, j, rng.random_range(-2.0..2.0)));
            }
        }
    }
    let x = SparseColMat::<usize, f64>::try_new_from_triplets(nrows, ncols, &triplets).unwrap();

    // Standardize columns lazily (center + unit sd), never forming X − 1cᵀ.
    let lazy = LazyMatrix::normalized(x, Normalization::new(Centering::Mean, Scaling::Sd));

    // Ground-truth coefficients and a noiseless target y = X̃ β*.
    let beta_star = Col::<f64>::from_fn(ncols, |j| ((j as f64) - 4.5) * 0.5);
    let y = lazy.matvec(&beta_star);

    let l = estimate_lipschitz(&lazy, ncols, 100);
    let step = 1.0 / l;
    println!("estimated Lipschitz L ≈ {l:.4}, step = 1/L = {step:.4e}");

    let (beta, iters, gnorm) = gradient_descent(&lazy, &y, ncols, step, 50_000, 1e-10);

    let err = norm2(&sub(&beta, &beta_star));
    let loss = 0.5 * {
        let r = sub(&lazy.matvec(&beta), &y);
        dot(&r, &r)
    };
    println!("converged in {iters} iters, ‖grad‖ = {gnorm:.3e}");
    println!("‖β − β*‖ = {err:.3e}, ½‖X̃β − y‖² = {loss:.3e}");
    println!("\n   j   β*        β̂");
    for j in 0..ncols {
        println!("  {j:>2}  {:>8.4}  {:>8.4}", beta_star[j], beta[j]);
    }
}
