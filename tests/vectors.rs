#[path = "common/runner.rs"]
mod common;

use lazymatrix::{DotSlice, ElemDivAssign, ScaledSubSlice, SubScalarAssign, SumEntries};

#[test]
fn vec_algebra_and_normalization_work_without_a_backend() {
    common::vector_algebra(&|v| v.to_vec(), &Clone::clone);
    let mut values = vec![6.0, -8.0, 10.0];
    values.elem_div_assign(&[2.0, 4.0, 5.0]);
    assert_eq!(values.dot_slice(&[1.0, -2.0, 3.0]), 13.0);
    values.sub_scalar_assign(1.0);
    values.scaled_sub_slice(0.5, &[2.0, -4.0, 6.0]);
    assert_eq!(values, vec![1.0, -1.0, -2.0]);
    assert_eq!(values.sum_entries(), -2.0);
}
