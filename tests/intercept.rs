use std::{cell::Cell, fmt};

use lazymatrix::{
    LazyMatrix, MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixErrorType,
    MatrixShape, MatrixWrite, VectorView, VectorViewMut, WeightedColumnSumsInto,
    WeightedColumnSumsKernel, WeightedGramInto, WeightedGramKernel, WithIntercept,
};

#[derive(Debug, PartialEq)]
struct ReadError;
impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("injected read failure")
    }
}
impl std::error::Error for ReadError {}

struct Source {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
    fail: Cell<bool>,
    fail_sums: Cell<bool>,
    calls: Cell<usize>,
}

impl Source {
    fn new(rows: usize, cols: usize, values: Vec<f64>) -> Self {
        Self {
            rows,
            cols,
            values,
            fail: Cell::new(false),
            fail_sums: Cell::new(false),
            calls: Cell::new(0),
        }
    }

    fn begin(&self) -> Result<(), ReadError> {
        self.calls.set(self.calls.get() + 1);
        if self.fail.get() {
            Err(ReadError)
        } else {
            Ok(())
        }
    }

    fn logical(&self, i: usize, j: usize, c: Option<&[f64]>, s: Option<&[f64]>) -> f64 {
        (self.values[i * self.cols + j] - c.map_or(0.0, |c| c[j])) / s.map_or(1.0, |s| s[j])
    }
}

impl MatrixShape for Source {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.cols
    }
}
impl MatrixErrorType for Source {
    type Error = ReadError;
}

impl MatVecInto<Vec<f64>> for Source {
    fn matvec_into(&self, x: &Vec<f64>, out: &mut Vec<f64>) -> Result<(), ReadError> {
        if !out.is_empty() {
            out[0] = 17.0;
        }
        self.begin()?;
        for (i, value) in out.iter_mut().enumerate() {
            *value = (0..self.cols)
                .map(|j| self.values[i * self.cols + j] * x[j])
                .sum();
        }
        Ok(())
    }
}
impl MatTransposeVecInto<Vec<f64>> for Source {
    fn mat_transpose_vec_into(&self, x: &Vec<f64>, out: &mut Vec<f64>) -> Result<(), ReadError> {
        if !out.is_empty() {
            out[0] = 17.0;
        }
        self.begin()?;
        for (j, value) in out.iter_mut().enumerate() {
            *value = (0..self.rows)
                .map(|i| self.values[i * self.cols + j] * x[i])
                .sum();
        }
        Ok(())
    }
}
impl WeightedGramKernel<f64> for Source {
    fn weighted_gram_normalized_into<W, O>(
        &self,
        weights: &W,
        c: Option<&[f64]>,
        s: Option<&[f64]>,
        out: &mut O,
    ) -> Result<(), ReadError>
    where
        W: VectorView<f64> + ?Sized,
        O: MatrixWrite<f64> + ?Sized,
    {
        if self.cols > 0 {
            out.set(0, 0, 17.0);
        }
        self.begin()?;
        for j in 0..self.cols {
            for k in 0..self.cols {
                out.set(
                    j,
                    k,
                    (0..self.rows)
                        .map(|i| {
                            (self.logical(i, j, c, s) * weights.get(i)) * self.logical(i, k, c, s)
                        })
                        .sum(),
                );
            }
        }
        Ok(())
    }
}
impl WeightedColumnSumsKernel<f64> for Source {
    fn weighted_column_sums_normalized_into<W, O>(
        &self,
        weights: &W,
        c: Option<&[f64]>,
        s: Option<&[f64]>,
        out: &mut O,
    ) -> Result<(), ReadError>
    where
        W: VectorView<f64> + ?Sized,
        O: VectorViewMut<f64> + ?Sized,
    {
        self.begin()?;
        if self.fail_sums.get() {
            return Err(ReadError);
        }
        for j in 0..self.cols {
            out.set(
                j,
                (0..self.rows)
                    .map(|i| self.logical(i, j, c, s) * weights.get(i))
                    .sum(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
struct Output([[f64; 3]; 3]);
impl MatrixShape for Output {
    fn nrows(&self) -> usize {
        3
    }
    fn ncols(&self) -> usize {
        3
    }
}
impl MatrixWrite<f64> for Output {
    fn set(&mut self, i: usize, j: usize, value: f64) {
        self.0[i][j] = value;
    }
}

#[test]
fn products_preserve_intercept_after_normalization_and_forward_errors() {
    let source = Source::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let lazy = LazyMatrix::from_parts(&source, Some(vec![1.0, 2.0]), Some(vec![2.0, 4.0]));
    let matrix = WithIntercept::new(&lazy);
    assert_eq!((matrix.nrows(), matrix.ncols()), (2, 3));
    assert!(std::ptr::eq(*matrix.as_inner(), &lazy));
    assert_eq!(matrix.matvec(&vec![3.0, 2.0, 4.0]).unwrap(), [3.0, 7.0]);
    assert_eq!(
        matrix.mat_transpose_vec(&vec![2.0, 4.0]).unwrap(),
        [6.0, 4.0, 2.0]
    );
    source.fail.set(true);
    let mut out = vec![99.0; 2];
    assert_eq!(
        matrix.matvec_into(&vec![3.0, 2.0, 4.0], &mut out),
        Err(ReadError)
    );
    assert_eq!(out, [17.0, 99.0]);
    let mut transpose = vec![99.0; 3];
    assert_eq!(
        matrix.mat_transpose_vec_into(&vec![2.0, 4.0], &mut transpose),
        Err(ReadError)
    );
    assert_eq!(transpose[0], 99.0);
    assert_eq!(matrix.matvec(&vec![3.0, 2.0, 4.0]), Err(ReadError));
    assert_eq!(matrix.mat_transpose_vec(&vec![2.0, 4.0]), Err(ReadError));
    source.fail.set(false);
    matrix.matvec_into(&vec![3.0, 2.0, 4.0], &mut out).unwrap();
    matrix
        .mat_transpose_vec_into(&vec![2.0, 4.0], &mut transpose)
        .unwrap();
    assert_eq!(out, [3.0, 7.0]);
    assert_eq!(transpose, [6.0, 4.0, 2.0]);
    assert!(std::ptr::eq(matrix.into_inner(), &lazy));
}

#[test]
fn invalid_dimensions_panic_before_backend_calls_or_writes() {
    let source = Source::new(2, 2, vec![1.0; 4]);
    let matrix = WithIntercept::<_, f64>::new(&source);
    for (x, size) in [(vec![1.0; 2], 2), (vec![1.0; 3], 3)] {
        let mut out = vec![99.0; size];
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                matrix.matvec_into(&x, &mut out).unwrap();
            }))
            .is_err()
        );
        assert!(out.iter().all(|&x| x == 99.0));
    }
    for (x, size) in [(vec![1.0; 3], 3), (vec![1.0; 2], 2)] {
        let mut out = vec![99.0; size];
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                matrix.mat_transpose_vec_into(&x, &mut out).unwrap();
            }))
            .is_err()
        );
        assert!(out.iter().all(|&x| x == 99.0));
    }
    assert_eq!(source.calls.get(), 0);
}

#[test]
fn gram_preserves_intercept_until_both_predictor_operations_succeed() {
    let source = Source::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let matrix = WithIntercept::new(LazyMatrix::from_parts(&source, Some(vec![1.0, 2.0]), None));
    let mut out = Output([[99.0; 3]; 3]);
    source.fail.set(true);
    assert_eq!(
        matrix.weighted_gram_into(&[1.0, 2.0], &mut out),
        Err(ReadError)
    );
    assert_eq!(out.0[0], [99.0; 3]);
    assert_eq!(out.0[1][1], 17.0);
    assert_eq!(source.calls.get(), 1);
    source.fail.set(false);
    source.fail_sums.set(true);
    assert_eq!(
        matrix.weighted_gram_into(&[1.0, 2.0], &mut out),
        Err(ReadError)
    );
    assert_eq!(out.0[0], [99.0; 3]);
    source.fail_sums.set(false);
    matrix.weighted_gram_into(&[1.0, 2.0], &mut out).unwrap();
    assert_eq!(out.0, [[3.0, 4.0, 4.0], [4.0, 8.0, 8.0], [4.0, 8.0, 8.0]]);
    let calls = source.calls.get();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            matrix.weighted_gram_into(&[1.0], &mut out).unwrap();
        }))
        .is_err()
    );
    assert_eq!(source.calls.get(), calls);
    let nested = WithIntercept::new(&matrix);
    let previous = out.0;
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            nested.weighted_gram_into(&[1.0, 2.0], &mut out).unwrap();
        }))
        .is_err()
    );
    assert_eq!(out.0, previous);
    assert_eq!(source.calls.get(), calls);
    let mut sums = vec![f64::NAN; 4];
    nested
        .weighted_column_sums_into(&[1.0, 2.0], &mut sums)
        .unwrap();
    assert_eq!(sums, [3.0, 3.0, 4.0, 4.0]);
}

#[test]
fn intercept_only_and_empty_products_use_empty_sums() {
    let matrix = WithIntercept::<_, f64>::new(Source::new(3, 0, vec![]));
    assert_eq!(matrix.matvec(&vec![2.0]).unwrap(), [2.0; 3]);
    assert_eq!(
        matrix.mat_transpose_vec(&vec![1.0, -2.0, 3.0]).unwrap(),
        [2.0]
    );
    let matrix = WithIntercept::<_, f64>::new(Source::new(0, 2, vec![]));
    assert!(matrix.matvec(&vec![2.0, 3.0, 4.0]).unwrap().is_empty());
    assert_eq!(matrix.mat_transpose_vec(&vec![]).unwrap(), [0.0; 3]);
}
