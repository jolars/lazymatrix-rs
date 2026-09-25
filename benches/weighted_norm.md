# Weighted column norm benchmarks

Run the CSC column benchmarks with:

```sh
cargo bench --locked --bench sprs_backend --features sprs -- weighted_norm
```

Each fixture has 100,000 rows and one borrowed CSC column, with stored entries
at strides of 1,000, 100, 10, or 1. Values include explicit zeros. Row weights
are positive and vary by row. Each case runs with centers of zero and 0.5 and
a scale of 2. Matrix construction and the total weight calculation happen
outside the timed loop. `direct` calls `weighted_norm_squared`; `cached_total`
calls `weighted_norm_squared_with_sum`.

Both methods now accumulate normalized contributions from every row, including
implicit zeros. They take O(n + nnz) time and constant scratch space for these
sorted, unique columns. The cached-total method accepts its argument for API
compatibility but ignores it: a rounded total cannot recover small implicit-row
weights by subtracting the stored-weight sum. Unsorted or duplicate indices
need one O(n) working column, which these benchmarks do not exercise.

## Local measurements

Measured on September 25, 2026, with Rust 1.89.0 on an Intel Core Ultra 7 155U.
These short, unpinned runs used 10 samples, a 0.1-second warmup, and a requested
0.2-second measurement window. They illustrate the cost of the scan and are not
a stable performance baseline. The two methods use the same implementation
after the fix; differences between their new timings reflect measurement noise.

Median times in microseconds, with center 0.5:

| Stored fraction | Direct before | Direct after | Cached total before | Cached total after |
| --- | ---: | ---: | ---: | ---: |
| 0.1% | 46.6 | 52.6 | 0.112 | 60.3 |
| 1% | 49.6 | 62.9 | 0.947 | 68.5 |
| 10% | 56.3 | 77.7 | 8.20 | 70.5 |
| 100% | 161 | 221 | 58.2 | 194 |

The old cached-total path was O(nnz), but could discard the entire contribution
from implicit zeros. The old direct path also used that subtraction after an
O(n) sum of the weights. The new calculation additionally normalizes each value
before squaring so that squaring an extreme scale does not overflow or
underflow. Amortizing scans across multiple columns would require more
information than a single cached total, such as a reusable range-sum workspace.
