//! Shared test helpers: a dense oracle and a backend-generic verification suite.
//!
//! The oracle materializes `X̃ = (X − 1cᵀ)S⁻¹` densely and runs naive
//! matrix–vector products; every backend's lazy operator is checked against it.
#![allow(dead_code)]

use lazymatrix::{
    Centering, ColumnStats, DotProduct, DotSlice, ElemDivAssign, L2Norm, LazyColumn, LazyMatrix,
    MatTransposeVec, MatVec, MatrixShape, Normalization, ScaleAssign, ScaledAddAssign,
    ScaledSubSlice, Scaling, SparseColumns, SubScalarAssign, SumEntries,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// A randomly-generated sparse matrix in both dense and triplet form.
pub struct TestMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub dense: Vec<Vec<f64>>,               // n×p, row-major
    pub triplets: Vec<(usize, usize, f64)>, // stored nonzeros (row, col, val)
}

/// Generate a reproducible sparse matrix with the given nonzero `density`.
pub fn random_matrix(seed: u64, nrows: usize, ncols: usize, density: f64) -> TestMatrix {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut dense = vec![vec![0.0; ncols]; nrows];
    let mut triplets = Vec::new();
    for (i, row) in dense.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            if rng.random::<f64>() < density {
                let v: f64 = rng.random_range(-2.0..2.0);
                if v != 0.0 {
                    *cell = v;
                    triplets.push((i, j, v));
                }
            }
        }
    }
    TestMatrix {
        nrows,
        ncols,
        dense,
        triplets,
    }
}

/// A reproducible random vector with entries in `[-1.5, 1.5)`.
pub fn random_vec(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(-1.5..1.5)).collect()
}

/// Materialize `X̃ = (X − 1cᵀ)S⁻¹` densely.
pub fn materialize(
    dense: &[Vec<f64>],
    centers: Option<&[f64]>,
    scales: Option<&[f64]>,
) -> Vec<Vec<f64>> {
    dense
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &x)| {
                    let mut x = x;
                    if let Some(c) = centers {
                        x -= c[j];
                    }
                    if let Some(s) = scales {
                        x /= s[j];
                    }
                    x
                })
                .collect()
        })
        .collect()
}

pub fn dense_matvec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
        .collect()
}

pub fn dense_tmatvec(m: &[Vec<f64>], u: &[f64]) -> Vec<f64> {
    let ncols = m.first().map_or(0, Vec::len);
    (0..ncols)
        .map(|j| m.iter().zip(u).map(|(row, &uu)| row[j] * uu).sum())
        .collect()
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn assert_close(a: &[f64], b: &[f64], eps: f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (x, y) in a.iter().zip(b) {
        approx::assert_abs_diff_eq!(x, y, epsilon = eps);
    }
}

const EPS: f64 = 1e-10;

/// Run the full verification suite against a backend, given closures that build
/// the backend matrix `M` from a [`TestMatrix`] and convert between `Vec<f64>`
/// and the backend vector `V`.
pub fn run_backend_suite<M, V>(
    build: impl Fn(&TestMatrix) -> M,
    to_v: impl Fn(&[f64]) -> V,
    from_v: impl Fn(&V) -> Vec<f64>,
) where
    M: MatVec<V> + MatTransposeVec<V> + ColumnStats<f64>,
    V: Clone
        + DotProduct<f64>
        + L2Norm<f64>
        + ScaledAddAssign<f64>
        + ScaleAssign<f64>
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    vector_algebra(&to_v, &from_v);
    oracle_parity(&build, &to_v, &from_v);
    adjoint_identity(&build, &to_v, &from_v);
    from_parts_passthrough(&build, &to_v, &from_v);
    new_matches_oracle(&build, &to_v, &from_v);
    column_stats_fixed(&build);
    column_sds_large_offset(&build);
    zero_scale_guard(&build, &to_v, &from_v);
    empty_and_nonfinite_stats(&build);
    shape_is_inferred(&build);
}

fn vector_algebra<V>(to_v: &impl Fn(&[f64]) -> V, from_v: &impl Fn(&V) -> Vec<f64>)
where
    V: DotProduct<f64> + L2Norm<f64> + ScaledAddAssign<f64> + ScaleAssign<f64>,
{
    let a = to_v(&[1.0, -2.0, 3.0]);
    let b = to_v(&[4.0, 5.0, -6.0]);
    approx::assert_abs_diff_eq!(a.dot(&b), -24.0, epsilon = EPS);
    approx::assert_abs_diff_eq!(a.norm_l2(), 14.0_f64.sqrt(), epsilon = EPS);

    let mut updated = to_v(&[1.0, -2.0, 3.0]);
    updated.scaled_add_assign(-0.5, &b);
    assert_close(&from_v(&updated), &[-1.0, -4.5, 6.0], EPS);
    updated.scale_assign(-2.0);
    assert_close(&from_v(&updated), &[2.0, 9.0, -12.0], EPS);

    let empty = to_v(&[]);
    approx::assert_abs_diff_eq!(empty.dot(&empty), 0.0, epsilon = EPS);
    approx::assert_abs_diff_eq!(empty.norm_l2(), 0.0, epsilon = EPS);

    let short = to_v(&[1.0]);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.dot(&short))).is_err());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut target = to_v(&[1.0, 2.0]);
            target.scaled_add_assign(1.0, &short);
        }))
        .is_err()
    );
}

/// Run the verification suite for backends with contiguous sparse columns.
///
/// This stays separate from [`run_backend_suite`] because future row-oriented
/// backends should still implement and test the orientation-agnostic traits.
pub fn run_sparse_columns_suite<M>(build: impl Fn(&TestMatrix) -> M)
where
    M: SparseColumns<f64>,
{
    sparse_columns_expose_raw_storage(&build);
    lazy_columns_match_dense_oracle(&build);
    lazy_column_operations_match_dense_oracle(&build);
    lazy_column_operations_handle_nonfinite_values(&build);
    empty_and_out_of_bounds_columns(&build);
}

fn sparse_columns_expose_raw_storage<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: SparseColumns<f64>,
{
    let tm = column_view_matrix();
    let matrix = build(&tm);

    let (rows, values) = matrix.sparse_column(0);
    assert_eq!(rows, &[0, 2, 3]);
    assert_eq!(values, &[1.0, 0.0, -2.0]);

    let (rows, values) = matrix.sparse_column(1);
    assert!(rows.is_empty());
    assert!(values.is_empty());

    let (rows, values) = matrix.sparse_column(2);
    assert_eq!(rows, &[0, 1, 2, 3]);
    assert_eq!(values, &[4.0, 5.0, 6.0, 7.0]);

    let (rows, values) = matrix.sparse_column(3);
    assert_eq!(rows, &[1]);
    assert_eq!(values, &[0.0]);
}

fn lazy_columns_match_dense_oracle<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: SparseColumns<f64>,
{
    let tm = column_view_matrix();
    let centers = vec![0.5, -1.0, 2.0, 3.0];
    let scales = vec![2.0, 4.0, 0.5, 1.5];

    for &use_center in &[false, true] {
        for &use_scale in &[false, true] {
            let active_centers = use_center.then(|| centers.clone());
            let active_scales = use_scale.then(|| scales.clone());
            let lazy =
                LazyMatrix::from_parts(build(&tm), active_centers.clone(), active_scales.clone());
            let dense = materialize(
                &tm.dense,
                active_centers.as_deref(),
                active_scales.as_deref(),
            );

            for j in 0..tm.ncols {
                let column = lazy.column(j);
                assert_eq!(column.len(), tm.nrows);
                assert!(!column.is_empty());
                assert_eq!(
                    column.center(),
                    active_centers.as_ref().map_or(0.0, |c| c[j])
                );
                assert_eq!(column.scale(), active_scales.as_ref().map_or(1.0, |s| s[j]));

                let expected: Vec<f64> = dense.iter().map(|row| row[j]).collect();
                assert_close(&reconstruct_column(column), &expected, EPS);
            }
        }
    }
}

fn lazy_column_operations_match_dense_oracle<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: SparseColumns<f64>,
{
    let tm = column_view_matrix();
    let centers = vec![0.5, -1.0, 2.0, 3.0];
    let scales = vec![2.0, 4.0, 0.5, 1.5];
    let vector = vec![1.5, -2.0, 0.25, 3.0];
    let weights = vec![0.5, 2.0, 1.25, 3.0];
    let vector_sum = vector.iter().sum();
    let weighted_vector_sum = vector
        .iter()
        .zip(&weights)
        .map(|(&value, &weight)| value * weight)
        .sum();
    let weight_sum = weights.iter().sum();

    for &use_center in &[false, true] {
        for &use_scale in &[false, true] {
            let active_centers = use_center.then(|| centers.clone());
            let active_scales = use_scale.then(|| scales.clone());
            let lazy =
                LazyMatrix::from_parts(build(&tm), active_centers.clone(), active_scales.clone());
            let dense = materialize(
                &tm.dense,
                active_centers.as_deref(),
                active_scales.as_deref(),
            );

            for j in 0..tm.ncols {
                let column = lazy.column(j);
                let expected: Vec<f64> = dense.iter().map(|row| row[j]).collect();
                let expected_sum: f64 = expected.iter().sum();
                let expected_norm_squared: f64 = expected.iter().map(|value| value * value).sum();
                let expected_dot = dot(&expected, &vector);
                let expected_weighted_dot: f64 = expected
                    .iter()
                    .zip(&vector)
                    .zip(&weights)
                    .map(|((&column_value, &vector_value), &weight)| {
                        column_value * vector_value * weight
                    })
                    .sum();
                let expected_weighted_norm_squared: f64 = expected
                    .iter()
                    .zip(&weights)
                    .map(|(&column_value, &weight)| weight * column_value * column_value)
                    .sum();

                approx::assert_abs_diff_eq!(column.sum(), expected_sum, epsilon = EPS);
                approx::assert_abs_diff_eq!(
                    column.norm_squared(),
                    expected_norm_squared,
                    epsilon = EPS
                );
                approx::assert_abs_diff_eq!(column.dot(&vector), expected_dot, epsilon = EPS);
                approx::assert_abs_diff_eq!(
                    column.dot_with_sum(&vector, vector_sum),
                    expected_dot,
                    epsilon = EPS
                );
                approx::assert_abs_diff_eq!(
                    column.weighted_dot(&vector, &weights),
                    expected_weighted_dot,
                    epsilon = EPS
                );
                approx::assert_abs_diff_eq!(
                    column.weighted_dot_with_sum(&vector, &weights, weighted_vector_sum),
                    expected_weighted_dot,
                    epsilon = EPS
                );
                approx::assert_abs_diff_eq!(
                    column.weighted_norm_squared(&weights),
                    expected_weighted_norm_squared,
                    epsilon = EPS
                );
                approx::assert_abs_diff_eq!(
                    column.weighted_norm_squared_with_sum(&weights, weight_sum),
                    expected_weighted_norm_squared,
                    epsilon = EPS
                );
            }
        }
    }

    let lazy = LazyMatrix::<_, f64>::from_parts(build(&tm), None, None);
    let column = lazy.column(0);
    let short = &vector[..vector.len() - 1];
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| column.dot(short))).is_err());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            column.dot_with_sum(short, short.iter().sum())
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            column.weighted_dot(short, &weights)
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            column.weighted_dot(&vector, &weights[..weights.len() - 1])
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            column.weighted_dot_with_sum(short, &weights, 0.0)
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            column.weighted_norm_squared(&weights[..weights.len() - 1])
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            column.weighted_norm_squared_with_sum(&weights[..weights.len() - 1], 0.0)
        }))
        .is_err()
    );
}

fn lazy_column_operations_handle_nonfinite_values<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: SparseColumns<f64>,
{
    let tm = TestMatrix {
        nrows: 3,
        ncols: 1,
        dense: vec![vec![f64::NAN], vec![0.0], vec![f64::INFINITY]],
        triplets: vec![(0, 0, f64::NAN), (2, 0, f64::INFINITY)],
    };
    let lazy = LazyMatrix::<_, f64>::from_parts(build(&tm), None, None);
    let column = lazy.column(0);
    let vector = [1.0, 2.0, 3.0];
    let weights = [0.5, 1.5, 2.0];
    let weighted_vector_sum = vector
        .iter()
        .zip(&weights)
        .map(|(&value, &weight)| value * weight)
        .sum();

    assert!(column.sum().is_nan());
    assert!(column.norm_squared().is_nan());
    assert!(column.dot(&vector).is_nan());
    assert!(column.dot_with_sum(&vector, vector.iter().sum()).is_nan());
    assert!(column.weighted_dot(&vector, &weights).is_nan());
    assert!(
        column
            .weighted_dot_with_sum(&vector, &weights, weighted_vector_sum)
            .is_nan()
    );
    assert!(column.weighted_norm_squared(&weights).is_nan());
    assert!(
        column
            .weighted_norm_squared_with_sum(&weights, weights.iter().sum())
            .is_nan()
    );
}

fn empty_and_out_of_bounds_columns<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: SparseColumns<f64>,
{
    let no_rows = TestMatrix {
        nrows: 0,
        ncols: 2,
        dense: Vec::new(),
        triplets: Vec::new(),
    };
    let lazy = LazyMatrix::<_, f64>::from_parts(build(&no_rows), None, None);
    let column = lazy.column(0);
    assert_eq!(column.len(), 0);
    assert!(column.is_empty());
    assert!(column.row_indices().is_empty());
    assert!(column.values().is_empty());
    assert_eq!(column.center(), 0.0);
    assert_eq!(column.scale(), 1.0);
    assert_eq!(column.sum(), 0.0);
    assert_eq!(column.norm_squared(), 0.0);
    assert_eq!(column.dot(&[]), 0.0);
    assert_eq!(column.dot_with_sum(&[], 0.0), 0.0);
    assert_eq!(column.weighted_dot(&[], &[]), 0.0);
    assert_eq!(column.weighted_dot_with_sum(&[], &[], 0.0), 0.0);
    assert_eq!(column.weighted_norm_squared(&[]), 0.0);
    assert_eq!(column.weighted_norm_squared_with_sum(&[], 0.0), 0.0);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lazy.column(2)));
    assert!(result.is_err());
}

fn reconstruct_column(column: LazyColumn<'_, f64>) -> Vec<f64> {
    let mut dense = vec![-column.center() / column.scale(); column.len()];
    for (&row, &value) in column.row_indices().iter().zip(column.values()) {
        dense[row] = (value - column.center()) / column.scale();
    }
    dense
}

fn column_view_matrix() -> TestMatrix {
    TestMatrix {
        nrows: 4,
        ncols: 4,
        dense: vec![
            vec![1.0, 0.0, 4.0, 0.0],
            vec![0.0, 0.0, 5.0, 0.0],
            vec![0.0, 0.0, 6.0, 0.0],
            vec![-2.0, 0.0, 7.0, 0.0],
        ],
        triplets: vec![
            (0, 0, 1.0),
            (2, 0, 0.0),
            (3, 0, -2.0),
            (0, 2, 4.0),
            (1, 2, 5.0),
            (2, 2, 6.0),
            (3, 2, 7.0),
            (1, 3, 0.0),
        ],
    }
}

/// Empty columns use IEEE results where a statistic is undefined, while
/// nonfinite stored values are never hidden by an aggregation.
fn empty_and_nonfinite_stats<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: ColumnStats<f64> + MatrixShape,
{
    let no_rows = TestMatrix {
        nrows: 0,
        ncols: 2,
        dense: Vec::new(),
        triplets: Vec::new(),
    };
    let matrix = build(&no_rows);
    assert!(matrix.col_means().iter().all(|value| value.is_nan()));
    assert!(matrix.col_sds().iter().all(|value| value.is_nan()));
    assert_eq!(matrix.col_maxabs(), vec![0.0, 0.0]);
    assert_eq!(matrix.col_l2(), vec![0.0, 0.0]);

    let lazy = LazyMatrix::new(matrix, Normalization::new(Centering::Mean, Scaling::Sd));
    assert!(lazy.centers().unwrap().iter().all(|value| value.is_nan()));
    assert!(lazy.scales().unwrap().iter().all(|value| value.is_nan()));

    let nan_column = TestMatrix {
        nrows: 2,
        ncols: 1,
        dense: vec![vec![f64::NAN], vec![1.0]],
        triplets: vec![(0, 0, f64::NAN), (1, 0, 1.0)],
    };
    let matrix = build(&nan_column);
    assert!(matrix.col_means()[0].is_nan());
    assert!(matrix.col_sds()[0].is_nan());
    assert!(matrix.col_maxabs()[0].is_nan());
    assert!(matrix.col_l2()[0].is_nan());
    assert!(matrix.col_l2_centered(&[0.0])[0].is_nan());
    assert!(matrix.col_maxabs_centered(&[0.0])[0].is_nan());

    let implicit_zero = TestMatrix {
        nrows: 2,
        ncols: 1,
        dense: vec![vec![1.0], vec![0.0]],
        triplets: vec![(0, 0, 1.0)],
    };
    assert!(build(&implicit_zero).col_maxabs_centered(&[f64::NAN])[0].is_nan());
}

/// Standard deviations retain small variation around a large offset.
fn column_sds_large_offset<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: ColumnStats<f64>,
{
    let offset = 1.0e12;
    let tm = TestMatrix {
        nrows: 3,
        ncols: 1,
        dense: vec![vec![offset + 1.0], vec![offset + 2.0], vec![offset + 3.0]],
        triplets: vec![
            (0, 0, offset + 1.0),
            (1, 0, offset + 2.0),
            (2, 0, offset + 3.0),
        ],
    };

    assert_close(&build(&tm).col_sds(), &[(2.0_f64 / 3.0).sqrt()], EPS);
}

/// The wrapper obtains its dimensions from the backend matrix.
fn shape_is_inferred<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: MatrixShape,
{
    let tm = random_matrix(21, 7, 4, 0.3);
    let lazy = LazyMatrix::<_, f64>::from_parts(build(&tm), None, None);
    assert_eq!(lazy.nrows(), tm.nrows);
    assert_eq!(lazy.ncols(), tm.ncols);

    let no_columns = random_matrix(22, 3, 0, 0.3);
    let lazy = LazyMatrix::<_, f64>::from_parts(build(&no_columns), None, None);
    assert_eq!(lazy.nrows(), no_columns.nrows);
    assert_eq!(lazy.ncols(), 0);
}

/// (1) + (3): all four center×scale combinations match the dense oracle, for
/// both `matvec` and `mat_transpose_vec`.
fn oracle_parity<M, V>(
    build: &impl Fn(&TestMatrix) -> M,
    to_v: &impl Fn(&[f64]) -> V,
    from_v: &impl Fn(&V) -> Vec<f64>,
) where
    M: MatVec<V> + MatTransposeVec<V>,
    V: Clone
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    let tm = random_matrix(1, 11, 7, 0.35);
    let centers = random_vec(2, tm.ncols);
    let scales: Vec<f64> = random_vec(3, tm.ncols)
        .iter()
        .map(|x| x.abs() + 0.5)
        .collect();
    let v = to_v(&random_vec(4, tm.ncols));
    let u = to_v(&random_vec(5, tm.nrows));

    for &use_c in &[false, true] {
        for &use_s in &[false, true] {
            let c = use_c.then(|| centers.clone());
            let s = use_s.then(|| scales.clone());
            let lazy = LazyMatrix::from_parts(build(&tm), c.clone(), s.clone());
            let xtilde = materialize(&tm.dense, c.as_deref(), s.as_deref());

            let got = from_v(&lazy.matvec(&v));
            let want = dense_matvec(&xtilde, &from_v(&v));
            assert_close(&got, &want, EPS);

            let got_t = from_v(&lazy.mat_transpose_vec(&u));
            let want_t = dense_tmatvec(&xtilde, &from_v(&u));
            assert_close(&got_t, &want_t, EPS);
        }
    }
}

/// (2): the adjoint identity ⟨X̃v, u⟩ == ⟨v, X̃ᵀu⟩, independent of the oracle.
fn adjoint_identity<M, V>(
    build: &impl Fn(&TestMatrix) -> M,
    to_v: &impl Fn(&[f64]) -> V,
    from_v: &impl Fn(&V) -> Vec<f64>,
) where
    M: MatVec<V> + MatTransposeVec<V>,
    V: Clone
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    let tm = random_matrix(6, 9, 5, 0.5);
    let centers = random_vec(7, tm.ncols);
    let scales: Vec<f64> = random_vec(8, tm.ncols)
        .iter()
        .map(|x| x.abs() + 0.3)
        .collect();
    let lazy = LazyMatrix::from_parts(build(&tm), Some(centers), Some(scales));
    let v = to_v(&random_vec(9, tm.ncols));
    let u = to_v(&random_vec(10, tm.nrows));

    let xv = from_v(&lazy.matvec(&v));
    let xtu = from_v(&lazy.mat_transpose_vec(&u));
    let lhs = dot(&xv, &from_v(&u));
    let rhs = dot(&from_v(&v), &xtu);
    approx::assert_abs_diff_eq!(lhs, rhs, epsilon = 1e-9);
}

/// (3): `from_parts(_, None, None)` is a bit-exact backend pass-through.
fn from_parts_passthrough<M, V>(
    build: &impl Fn(&TestMatrix) -> M,
    to_v: &impl Fn(&[f64]) -> V,
    from_v: &impl Fn(&V) -> Vec<f64>,
) where
    M: MatVec<V> + MatTransposeVec<V>,
    V: Clone
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    let tm = random_matrix(11, 8, 6, 0.4);
    let v = to_v(&random_vec(12, tm.ncols));
    let u = to_v(&random_vec(13, tm.nrows));

    let bare = build(&tm);
    let bare_y = from_v(&bare.matvec(&v));
    let bare_t = from_v(&bare.mat_transpose_vec(&u));

    let lazy = LazyMatrix::from_parts(build(&tm), None, None);
    assert_eq!(from_v(&lazy.matvec(&v)), bare_y);
    assert_eq!(from_v(&lazy.mat_transpose_vec(&u)), bare_t);
}

/// `new()` with every strategy: read back the computed centers/scales and
/// confirm the operator equals the oracle built from those same vectors.
fn new_matches_oracle<M, V>(
    build: &impl Fn(&TestMatrix) -> M,
    to_v: &impl Fn(&[f64]) -> V,
    from_v: &impl Fn(&V) -> Vec<f64>,
) where
    M: MatVec<V> + MatTransposeVec<V> + ColumnStats<f64>,
    V: Clone
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    let tm = random_matrix(14, 13, 6, 0.45);
    let v = to_v(&random_vec(15, tm.ncols));
    let u = to_v(&random_vec(16, tm.nrows));

    let centerings = [Centering::None, Centering::Mean];
    let scalings = [Scaling::None, Scaling::Sd, Scaling::MaxAbs, Scaling::L2];
    for center in centerings {
        for scale in scalings {
            let spec = Normalization::new(center, scale);
            let lazy = LazyMatrix::new(build(&tm), spec);
            let xtilde = materialize(&tm.dense, lazy.centers(), lazy.scales());

            let got = from_v(&lazy.matvec(&v));
            assert_close(&got, &dense_matvec(&xtilde, &from_v(&v)), EPS);
            let got_t = from_v(&lazy.mat_transpose_vec(&u));
            assert_close(&got_t, &dense_tmatvec(&xtilde, &from_v(&u)), EPS);
        }
    }
}

/// (5): `ColumnStats` against hand-computed values on a fixed tiny matrix,
/// including the centered-l2/maxabs implicit-zero corrections.
///
/// ```text
/// X = | 1  0  5 |
///     | 3  0  5 |
///     | 0  0  5 |
/// ```
fn column_stats_fixed<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: ColumnStats<f64>,
{
    let tm = TestMatrix {
        nrows: 3,
        ncols: 3,
        dense: vec![
            vec![1.0, 0.0, 5.0],
            vec![3.0, 0.0, 5.0],
            vec![0.0, 0.0, 5.0],
        ],
        triplets: vec![
            (0, 0, 1.0),
            (1, 0, 3.0),
            (0, 2, 5.0),
            (1, 2, 5.0),
            (2, 2, 5.0),
        ],
    };
    let m = build(&tm);

    // Column 0 = [1,3,0]; column 1 = [0,0,0]; column 2 = [5,5,5].
    assert_close(&m.col_means(), &[4.0 / 3.0, 0.0, 5.0], EPS);
    // population sd: col0 var = 10/3 − (4/3)² = 14/9; col1 = 0; col2 constant = 0
    assert_close(&m.col_sds(), &[(14.0_f64 / 9.0).sqrt(), 0.0, 0.0], EPS);
    assert_close(&m.col_maxabs(), &[3.0, 0.0, 5.0], EPS);
    assert_close(&m.col_l2(), &[10.0_f64.sqrt(), 0.0, (75.0_f64).sqrt()], EPS);

    let centers = m.col_means();
    // centered l2² of col0 = n·var = 3·14/9 = 14/3; col1 = 0; col2 = 0
    assert_close(
        &m.col_l2_centered(&centers),
        &[(14.0_f64 / 3.0).sqrt(), 0.0, 0.0],
        EPS,
    );
    // centered maxabs col0: max(|1−4/3|, |3−4/3|, implicit |0−4/3|) = 5/3
    assert_close(
        &m.col_maxabs_centered(&centers),
        &[5.0 / 3.0, 0.0, 0.0],
        EPS,
    );
}

/// (6): a constant column (sd 0) is floored to scale 1 → finite output.
fn zero_scale_guard<M, V>(
    build: &impl Fn(&TestMatrix) -> M,
    to_v: &impl Fn(&[f64]) -> V,
    from_v: &impl Fn(&V) -> Vec<f64>,
) where
    M: MatVec<V> + MatTransposeVec<V> + ColumnStats<f64>,
    V: Clone
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    // Middle column is empty (all zero) → sd 0; last column constant → sd 0.
    let tm = TestMatrix {
        nrows: 3,
        ncols: 3,
        dense: vec![
            vec![1.0, 0.0, 4.0],
            vec![2.0, 0.0, 4.0],
            vec![3.0, 0.0, 4.0],
        ],
        triplets: vec![
            (0, 0, 1.0),
            (1, 0, 2.0),
            (2, 0, 3.0),
            (0, 2, 4.0),
            (1, 2, 4.0),
            (2, 2, 4.0),
        ],
    };
    let spec = Normalization::new(Centering::Mean, Scaling::Sd);
    let lazy = LazyMatrix::new(build(&tm), spec);
    let scales = lazy.scales().unwrap();
    assert_eq!(scales[1], 1.0, "empty column scale must be floored to 1");
    assert_eq!(scales[2], 1.0, "constant column scale must be floored to 1");

    let v = to_v(&random_vec(20, tm.ncols));
    let y = from_v(&lazy.matvec(&v));
    assert!(y.iter().all(|x| x.is_finite()), "output must be finite");
}
