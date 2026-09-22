use lazymatrix::{MatrixShape, Scalar, SparseRows};

struct Csr<'a, F> {
    ncols: usize,
    offsets: &'a [usize],
    columns: &'a [usize],
    values: &'a [F],
}

impl<F> MatrixShape for Csr<'_, F> {
    fn nrows(&self) -> usize {
        self.offsets.len() - 1
    }

    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<F: Scalar> SparseRows<F> for Csr<'_, F> {
    fn sparse_row(&self, i: usize) -> (&[usize], &[F]) {
        assert!(i < self.nrows());
        let range = self.offsets[i]..self.offsets[i + 1];
        (&self.columns[range.clone()], &self.values[range])
    }
}

#[test]
fn sparse_rows_supports_borrowed_trait_objects_without_backends() {
    let columns = [2, 0, 1];
    let values = [f32::NAN, f32::INFINITY, -0.0];
    let matrix = Csr {
        ncols: 3,
        offsets: &[0, 1, 3],
        columns: &columns,
        values: &values,
    };
    let erased: &dyn SparseRows<f32> = &matrix;
    let borrowed = &erased;
    let (indices, raw) = <&dyn SparseRows<f32> as SparseRows<f32>>::sparse_row(borrowed, 1);
    assert_eq!(borrowed.nrows(), 2);
    assert_eq!(borrowed.ncols(), 3);
    assert_eq!(indices, &[0, 1]);
    assert_eq!(indices.as_ptr(), columns[1..].as_ptr());
    assert_eq!(raw.as_ptr(), values[1..].as_ptr());
    assert_eq!(raw[0], f32::INFINITY);
    assert_eq!(raw[1].to_bits(), (-0.0_f32).to_bits());
    assert!(borrowed.sparse_row(0).1[0].is_nan());
}
