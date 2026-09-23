#[path = "oracle.rs"]
mod oracle;

/// Exercise this consumer's backend types after Cargo unifies features.
pub fn check() {
    use faer::sparse::{SparseColMat, Triplet};
    use faer::{Col, Mat};
    let values = [[1.0, 0.0], [2.0, 3.0], [0.0, 6.0]];
    let dense = Mat::from_fn(3, 2, |i, j| values[i][j]);
    let sparse = SparseColMat::try_new_from_triplets(
        3,
        2,
        &[
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 0, 2.0),
            Triplet::new(1, 1, 3.0),
            Triplet::new(2, 1, 6.0),
        ],
    )
    .unwrap();
    let input = Col::from_fn(2, |i| [2.0, -1.0][i]);
    let rows = Col::from_fn(3, |i| [1.0, 2.0, -1.0][i]);
    oracle::check(
        dense.as_ref(),
        input.clone(),
        rows.clone(),
        Col::zeros(3),
        Col::zeros(2),
    );
    oracle::check(sparse, input, rows, Col::zeros(3), Col::zeros(2));
}
