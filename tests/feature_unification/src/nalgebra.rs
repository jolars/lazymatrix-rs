#[path = "oracle.rs"]
mod oracle;

/// Exercise this consumer's backend types after Cargo unifies features.
pub fn check() {
    use nalgebra::{DMatrix, DMatrixView, DVector};
    use nalgebra_sparse::CscMatrix;
    let dense = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 2.0, 3.0, 0.0, 6.0]);
    let sparse = CscMatrix::try_from_csc_data(
        3,
        2,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![1.0, 2.0, 3.0, 6.0],
    )
    .unwrap();
    let input = DVector::from_column_slice(&[2.0, -1.0]);
    let rows = DVector::from_column_slice(&[1.0, 2.0, -1.0]);
    let view: DMatrixView<'_, f64> = dense.as_view();
    oracle::check(
        view,
        input.clone(),
        rows.clone(),
        DVector::zeros(3),
        DVector::zeros(2),
    );
    oracle::check(sparse, input, rows, DVector::zeros(3), DVector::zeros(2));
}
