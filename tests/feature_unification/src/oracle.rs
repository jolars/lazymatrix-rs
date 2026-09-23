use lazymatrix::{
    Centering, ColumnStats, LazyMatrix, MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto,
    MatrixShape, Normalization, Scaling, VectorView, VectorViewMut,
};

pub fn check<M, V>(matrix: M, input: V, rows: V, mut out: V, mut transpose_out: V)
where
    M: MatrixShape + ColumnStats<f64>,
    LazyMatrix<M, f64>: MatVec<V> + MatTransposeVec<V> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: VectorView<f64> + VectorViewMut<f64>,
{
    let lazy =
        LazyMatrix::new(matrix, Normalization::new(Centering::Mean, Scaling::Range)).unwrap();
    assert_eq!(lazy.centers().unwrap(), &[1.0, 3.0]);
    assert_eq!(lazy.scales().unwrap(), &[2.0, 6.0]);
    close(&lazy.matvec(&input).unwrap(), &[0.5, 1.0, -1.5]);
    close(&lazy.mat_transpose_vec(&rows).unwrap(), &[1.5, -1.0]);
    lazy.matvec_into(&input, &mut out).unwrap();
    lazy.mat_transpose_vec_into(&rows, &mut transpose_out)
        .unwrap();
    close(&out, &[0.5, 1.0, -1.5]);
    close(&transpose_out, &[1.5, -1.0]);
}

fn close(vector: &impl VectorView<f64>, expected: &[f64]) {
    assert_eq!(vector.len(), expected.len());
    for (i, &value) in expected.iter().enumerate() {
        assert!((vector.get(i) - value).abs() < 1e-12);
    }
}
