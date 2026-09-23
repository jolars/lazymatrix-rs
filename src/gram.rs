//! Backend-independent validation and sparse Gram accumulation.

use crate::{MatrixWrite, Scalar, VectorView};

pub(crate) fn validate<F, W, O>(
    nrows: usize,
    ncols: usize,
    weights: &W,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &O,
) where
    F: Scalar,
    W: VectorView<F> + ?Sized,
    O: MatrixWrite<F> + ?Sized,
{
    assert_eq!(
        weights.len(),
        nrows,
        "weighted_gram_into: weights length mismatch"
    );
    assert_eq!(
        (out.nrows(), out.ncols()),
        (ncols, ncols),
        "weighted_gram_into: output shape mismatch"
    );
    if let Some(centers) = centers {
        assert_eq!(centers.len(), ncols, "centers length must equal ncols");
    }
    if let Some(scales) = scales {
        assert_eq!(scales.len(), ncols, "scales length must equal ncols");
        for (column, &scale) in scales.iter().enumerate() {
            assert!(
                scale != F::zero(),
                "scale at column {column} must be nonzero"
            );
        }
    }
}

#[cfg(any(
    feature = "ndarray_all",
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "sprs_all"
))]
pub(crate) fn normalize<F: Scalar>(
    mut value: F,
    column: usize,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
) -> F {
    if let Some(c) = centers {
        value = value - c[column];
    }
    if let Some(s) = scales {
        value = value / s[column];
    }
    value
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
mod sparse {
    use super::{normalize, validate};
    use crate::{MatrixWrite, Scalar, SparseColumns, VectorView};

    const TILE: usize = 32;

    struct Panel<F> {
        values: Vec<F>,
        cursors: [usize; TILE],
        first: usize,
        count: usize,
    }

    impl<F: Scalar> Panel<F> {
        fn new() -> Self {
            Self {
                values: vec![F::zero(); TILE * TILE],
                cursors: [0; TILE],
                first: 0,
                count: 0,
            }
        }

        fn reset(&mut self, first: usize, count: usize) {
            self.first = first;
            self.count = count;
            self.cursors.fill(0);
        }

        fn fill<M: SparseColumns<F> + ?Sized>(
            &mut self,
            matrix: &M,
            row: usize,
            rows: usize,
            centers: Option<&[F]>,
            scales: Option<&[F]>,
        ) {
            for a in 0..self.count {
                let column = self.first + a;
                let background = normalize(F::zero(), column, centers, scales);
                for i in 0..rows {
                    self.values[i * TILE + a] = background;
                }
                let (indices, values) = matrix.sparse_column(column);
                let cursor = &mut self.cursors[a];
                while *cursor < indices.len() && indices[*cursor] < row + rows {
                    self.values[(indices[*cursor] - row) * TILE + a] =
                        normalize(values[*cursor], column, centers, scales);
                    *cursor += 1;
                }
            }
        }
    }

    fn tiled_gram<F, M, W, O>(
        matrix: &M,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) where
        F: Scalar,
        M: SparseColumns<F> + ?Sized,
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        let (n, p) = (matrix.nrows(), matrix.ncols());
        let mut left = Panel::new();
        let mut right = Panel::new();
        let mut block = [F::zero(); TILE * TILE];
        for j in (0..p).step_by(TILE) {
            for k in (j..p).step_by(TILE) {
                left.reset(j, (p - j).min(TILE));
                right.reset(k, (p - k).min(TILE));
                block.fill(F::zero());
                for row in (0..n).step_by(TILE) {
                    let rows = (n - row).min(TILE);
                    left.fill(matrix, row, rows, centers, scales);
                    right.fill(matrix, row, rows, centers, scales);
                    for i in 0..rows {
                        let weight = weights.get(row + i);
                        let rhs = &right.values[i * TILE..][..right.count];
                        for a in 0..left.count {
                            let lhs = left.values[i * TILE + a] * weight;
                            let destination = &mut block[a * TILE..][..right.count];
                            // Independent coefficient accumulators allow SIMD
                            // without changing the summation order within a cell.
                            for (value, &y) in destination.iter_mut().zip(rhs) {
                                *value = *value + lhs * y;
                            }
                        }
                    }
                }
                for a in 0..left.count {
                    for b in 0..right.count {
                        if j + a <= k + b {
                            let value = block[a * TILE + b];
                            out.set(j + a, k + b, value);
                            if j + a != k + b {
                                out.set(k + b, j + a, value);
                            }
                        }
                    }
                }
            }
        }
    }

    // Range sums use additions only. Subtracting a nearly equal stored-weight
    // sum from the total could discard the entire implicit-zero contribution.
    struct WeightSums<F> {
        nodes: Vec<F>,
        len: usize,
    }

    impl<F: Scalar> WeightSums<F> {
        fn new<W: VectorView<F> + ?Sized>(weights: &W) -> Self {
            let len = weights.len();
            let mut nodes = vec![F::zero(); 2 * len];
            for i in 0..len {
                nodes[len + i] = weights.get(i);
            }
            for i in (1..len).rev() {
                nodes[i] = nodes[2 * i] + nodes[2 * i + 1];
            }
            Self { nodes, len }
        }

        fn sum(&self, start: usize, end: usize) -> F {
            let (mut l, mut r) = (start + self.len, end + self.len);
            let mut sum = F::zero();
            while l < r {
                if l % 2 == 1 {
                    sum = sum + self.nodes[l];
                    l += 1;
                }
                if r % 2 == 1 {
                    r -= 1;
                    sum = sum + self.nodes[r];
                }
                l /= 2;
                r /= 2;
            }
            sum
        }
    }

    pub(crate) fn sparse_gram<F, M, W, O>(
        matrix: &M,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) where
        F: Scalar,
        M: SparseColumns<F> + ?Sized,
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        let (n, p) = (matrix.nrows(), matrix.ncols());
        validate(n, p, weights, centers, scales, out);
        if p == 0 {
            return;
        }
        let backgrounds: Vec<_> = (0..p)
            .map(|j| normalize(F::zero(), j, centers, scales))
            .collect();
        let finite_weights = (0..n).all(|i| weights.get(i).is_finite());
        let max_weight = (0..n).fold(F::zero(), |m, i| m.max(weights.get(i).abs()));
        let mut safe: Vec<_> = (0..p)
            .map(|j| {
                let (rows, values) = matrix.sparse_column(j);
                finite_weights
                    && (backgrounds[j] * max_weight).is_finite()
                    && rows.windows(2).all(|pair| pair[0] < pair[1])
                    && rows.iter().zip(values).all(|(&i, &x)| {
                        (normalize(x, j, centers, scales) * weights.get(i)).is_finite()
                    })
            })
            .collect();
        let centered = backgrounds.iter().any(|&b| b != F::zero());
        let nnz: usize = (0..p).map(|j| matrix.sparse_column(j).0.len()).sum();
        // At moderate densities, bounded panels avoid a range-sum query for
        // nearly every stored entry. Very sparse matrices retain sparse work.
        if centered
            && n > 0
            && p >= 16
            && safe.iter().all(|&s| s)
            && nnz as f64 / (n as f64 * p as f64) >= 0.005
        {
            tiled_gram(matrix, weights, centers, scales, out);
            return;
        }
        let weight_sums = centered.then(|| WeightSums::new(weights));
        if weight_sums
            .as_ref()
            .is_some_and(|tree| tree.nodes.iter().any(|w| !w.is_finite()))
        {
            safe.fill(false);
        }
        let mut scratch: Option<(Vec<F>, Vec<F>)> = None;
        for j in 0..p {
            for k in j..p {
                let (jr, jv) = matrix.sparse_column(j);
                let (kr, kv) = matrix.sparse_column(k);
                let bj = backgrounds[j];
                let bk = backgrounds[k];
                let mut sum = F::zero();
                let mut fast = safe[j] && safe[k] && ((bj * max_weight) * bk).is_finite();
                if fast && bj == F::zero() && bk == F::zero() {
                    let (mut a, mut b) = (0, 0);
                    while a < jr.len() && b < kr.len() {
                        match jr[a].cmp(&kr[b]) {
                            std::cmp::Ordering::Less => a += 1,
                            std::cmp::Ordering::Greater => b += 1,
                            std::cmp::Ordering::Equal => {
                                sum = sum
                                    + (normalize(jv[a], j, centers, scales) * weights.get(jr[a]))
                                        * normalize(kv[b], k, centers, scales);
                                a += 1;
                                b += 1;
                            }
                        }
                    }
                    fast = sum.is_finite();
                } else if fast {
                    let (mut a, mut b, mut next) = (0, 0, 0);
                    while a < jr.len() || b < kr.len() {
                        let row = jr
                            .get(a)
                            .copied()
                            .unwrap_or(n)
                            .min(kr.get(b).copied().unwrap_or(n));
                        if next < row && bj != F::zero() && bk != F::zero() {
                            sum = sum + (bj * weight_sums.as_ref().unwrap().sum(next, row)) * bk;
                        }
                        let x = if jr.get(a) == Some(&row) {
                            let x = normalize(jv[a], j, centers, scales);
                            a += 1;
                            x
                        } else {
                            bj
                        };
                        let y = if kr.get(b) == Some(&row) {
                            let y = normalize(kv[b], k, centers, scales);
                            b += 1;
                            y
                        } else {
                            bk
                        };
                        sum = sum + (x * weights.get(row)) * y;
                        next = row + 1;
                    }
                    if next < n && bj != F::zero() && bk != F::zero() {
                        sum = sum + (bj * weight_sums.as_ref().unwrap().sum(next, n)) * bk;
                    }
                    fast = sum.is_finite();
                }
                if !fast {
                    // Two working columns retain IEEE operations on implicit
                    // zeros and combine duplicate raw entries before centering.
                    let (x, y) =
                        scratch.get_or_insert_with(|| (vec![F::zero(); n], vec![F::zero(); n]));
                    x.fill(F::zero());
                    y.fill(F::zero());
                    for (&i, &v) in jr.iter().zip(jv) {
                        x[i] = x[i] + v;
                    }
                    for (&i, &v) in kr.iter().zip(kv) {
                        y[i] = y[i] + v;
                    }
                    sum = (0..n)
                        .map(|i| {
                            (normalize(x[i], j, centers, scales) * weights.get(i))
                                * normalize(y[i], k, centers, scales)
                        })
                        .sum();
                }
                out.set(j, k, sum);
                if j != k {
                    out.set(k, j, sum);
                }
            }
        }
    }
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
pub(crate) use sparse::sparse_gram;
