#[path = "oracle.rs"]
mod oracle;

/// Exercise this consumer's backend types after Cargo unifies features.
pub fn check() {
    use ndarray::{Array1, array};
    let dense = array![[1.0, 0.0], [2.0, 3.0], [0.0, 6.0]];
    oracle::check(
        dense.view(),
        array![2.0, -1.0],
        array![1.0, 2.0, -1.0],
        Array1::zeros(3),
        Array1::zeros(2),
    );
    oracle::check(
        dense,
        array![2.0, -1.0],
        array![1.0, 2.0, -1.0],
        Array1::zeros(3),
        Array1::zeros(2),
    );
}
