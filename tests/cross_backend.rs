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

#[cfg(feature = "ndarray_all")]
#[test]
fn csc_backends_write_f32_grams_directly_into_ndarray() {
    use lazymatrix::{WeightedGramInto, WeightedGramKernel};
    let values = [[1.0_f32, 0.0], [2.0, 3.0]];
    let weights = [2.0_f32, -1.0];
    let dense = ndarray::Array2::from_shape_fn((2, 2), |(i, j)| values[i][j]);
    let lazy = LazyMatrix::from_parts(dense.view(), Some(vec![0.5, 1.0]), Some(vec![-2.0, 3.0]));
    let mut expected = ndarray::Array2::zeros((2, 2));
    lazy.weighted_gram_into(&weights, &mut expected).unwrap();
    fn check<M: WeightedGramKernel<f32>>(
        matrix: M,
        weights: &[f32],
        expected: &ndarray::Array2<f32>,
    ) {
        let lazy = LazyMatrix::from_parts(matrix, Some(vec![0.5, 1.0]), Some(vec![-2.0, 3.0]));
        let mut out = ndarray::Array2::from_elem((2, 2), f32::NAN);
        lazy.weighted_gram_into(weights, &mut out.view_mut())
            .unwrap();
        for (&a, &b) in out.iter().zip(expected) {
            approx::assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
    }
    #[cfg(feature = "faer_all")]
    {
        use faer::sparse::{SparseColMat, Triplet};
        let matrix = SparseColMat::try_new_from_triplets(
            2,
            2,
            &[
                Triplet::new(0, 0, 1.0_f32),
                Triplet::new(1, 0, 2.0),
                Triplet::new(1, 1, 3.0),
            ],
        )
        .unwrap();
        check(&matrix, &weights, &expected);
    }
    #[cfg(feature = "nalgebra_all")]
    {
        let matrix = nalgebra_sparse::CscMatrix::try_from_csc_data(
            2,
            2,
            vec![0, 2, 3],
            vec![0, 1, 1],
            vec![1.0_f32, 2.0, 3.0],
        )
        .unwrap();
        check(&matrix, &weights, &expected);
    }
    #[cfg(feature = "sprs_all")]
    {
        let matrix = sprs::CsMat::new_csc(
            (2, 2),
            vec![0, 2, 3],
            vec![0, 1, 1],
            vec![1.0_f32, 2.0, 3.0],
        );
        check(
            lazymatrix::SprsCsc::try_new(matrix.view()).unwrap(),
            &weights,
            &expected,
        );
    }
}

#[cfg(feature = "sprs_all")]
fn sprs_products(tm: &TestMatrix, spec: Normalization, v: &[f64], u: &[f64]) -> Products {
    let mut triplets = sprs::TriMat::new((tm.nrows, tm.ncols));
    for &(row, col, value) in &tm.triplets {
        triplets.add_triplet(row, col, value);
    }
    let lazy = LazyMatrix::new(triplets.to_csc::<usize>(), spec).unwrap();
    (
        lazy.matvec(&v.to_vec()).unwrap(),
        lazy.mat_transpose_vec(&u.to_vec()).unwrap(),
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
    let lazy = LazyMatrix::new(matrix, spec).unwrap();
    let y = lazy.matvec(&Col::from_fn(v.len(), |i| v[i])).unwrap();
    let z = lazy
        .mat_transpose_vec(&Col::from_fn(u.len(), |i| u[i]))
        .unwrap();
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
    let lazy = LazyMatrix::new(CscMatrix::from(&coo), spec).unwrap();
    (
        lazy.matvec(&DVector::from_column_slice(v))
            .unwrap()
            .as_slice()
            .to_vec(),
        lazy.mat_transpose_vec(&DVector::from_column_slice(u))
            .unwrap()
            .as_slice()
            .to_vec(),
    )
}

#[cfg(feature = "ndarray_all")]
fn ndarray_products(tm: &TestMatrix, spec: Normalization, v: &[f64], u: &[f64]) -> Products {
    use ndarray::{Array1, Array2};

    let matrix = Array2::from_shape_fn((tm.nrows, tm.ncols), |(i, j)| tm.dense[i][j]);
    let lazy = LazyMatrix::new(matrix, spec).unwrap();
    (
        lazy.matvec(&Array1::from_vec(v.to_vec())).unwrap().to_vec(),
        lazy.mat_transpose_vec(&Array1::from_vec(u.to_vec()))
            .unwrap()
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
