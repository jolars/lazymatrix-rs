# TODO

This file records design work identified during the initial API audit. The
crate should expose normalized matrices and their storage capabilities; solver
state and solver-specific update logic belong in consuming crates.

## Foundation

- [x] Add an orientation-independent `MatrixShape` trait.
  - Implement `nrows()` and `ncols()` for every backend matrix and for
    `LazyMatrix`.
  - Infer dimensions in `LazyMatrix` constructors instead of accepting caller
    supplied dimensions that can disagree with the backend.
  - Let generic consumers obtain dimensions from the operator.
  - Test rectangular and zero-column matrices; caller/backend dimension
    disagreement is eliminated by construction.

- [x] Make standard-deviation calculations numerically stable.
  - Replace `E[x^2] - E[x]^2` with a stable two-pass or combined-variance
    calculation that accounts for implicit sparse zeros.
  - Add a regression test using a large offset and small variation, such as
    values near `1e12` whose true variance is nonzero.
  - Apply the same algorithm and edge-case policy to every backend.

- [x] Define the policy for empty matrices and nonfinite statistics.
  - Allow normalization of matrices with zero rows. Undefined means and
    standard deviations are `NaN`; zero L2 and max-absolute scales use the
    degenerate-column convention below.
  - Follow IEEE behavior for nonfinite values instead of rejecting them. Ensure
    aggregations propagate `NaN` rather than accidentally masking it.
  - Replace only exact computed zero scales with `1`, leaving nonfinite scales
    untouched. Such degenerate columns are left unscaled.
  - Document and test the behavior for every backend.

- [ ] Validate explicit normalization parameters.
  - Decide whether scales must be finite and strictly positive or merely
    finite and nonzero.
  - Ensure `from_parts` and `with_scales` cannot silently construct an operator
    that divides by zero.
  - Consider a fallible constructor if validation errors should not panic.
  - Ensure the zero-scale guard treats nonfinite values deliberately.

## Sparse access capabilities

- [x] Add `SparseColumns` for contiguous CSC column access.
  - Return borrowed row-index and raw-value slices without copying.
  - Implement it only for storage types that provide efficient contiguous
    column access.
  - Keep it separate from `ColumnStats`, which remains
    orientation-independent.

- [x] Add `LazyMatrix::column` and a borrowed `LazyColumn` view.
  - Gate the method on `M: SparseColumns`.
  - Expose row indices, raw stored values, logical length, center, and scale.
  - Use the canonical `center` and `scale` terminology used by `LazyMatrix`;
    inverse scales and affine background values are derived quantities.
  - Represent inactive centering and scaling as effective values `0` and `1`
    in the view where that simplifies consumers.
  - Do not hide a dense centered-column update behind a method that appears to
    be sparse.
  - Test reconstruction of logical columns against a dense oracle.

- [ ] Add `SparseRows` for contiguous CSR row access.
  - Return borrowed column-index and raw-value slices without copying.
  - Implement it only for storage types that provide efficient contiguous row
    access; do not gather rows from CSC under this trait.

- [ ] Add `LazyMatrix::row` and a borrowed `LazyRow` view.
  - Gate the method on `M: SparseRows`.
  - Expose column indices, raw stored values, logical length, and borrowed
    normalization parameters.
  - Make the sparse-plus-affine structure explicit: a centered logical row is
    generally dense even when its raw row is sparse.
  - Test reconstruction of logical rows against a dense oracle.

- [ ] Add CSR backend support when row access has a concrete consumer.
  - Cover faer `SparseRowMat` and nalgebra-sparse `CsrMatrix` if their APIs
    support the required operations cleanly.
  - Implement `MatrixShape`, `MatVec`, `MatTransposeVec`, `ColumnStats`, and
    `SparseRows`.
  - Reuse the backend-generic oracle and adjoint tests.
  - Do not implement `SparseColumns` by performing an expensive gather.

## Operator performance

- [ ] Allow `LazyMatrix` to wrap a borrowed backend matrix.
  - Add forwarding implementations for the matrix capability traits on `&M`,
    or provide an explicit borrowed wrapper with equivalent ergonomics.
  - Support construction such as `LazyMatrix::new(&x, spec)` so fitting paths,
    cross-validation, and prediction do not need to consume the design matrix.

- [ ] Evaluate reusable-output operator methods before stabilizing the traits.
  - Prototype `matvec_into` and `mat_transpose_vec_into` as additive
    capabilities or as the primitive operator interface.
  - Measure allocation costs in an iterative consumer before designing a
    reusable normalization workspace.
  - Keep allocating convenience methods if they materially improve ergonomics.

- [ ] Avoid cloning the forward input when scaling is inactive.
  - Preserve the direct backend path for raw and center-only products.
  - Benchmark before adding more elaborate scratch-storage machinery.

- [ ] Add multiple-right-hand-side operator capabilities.
  - Prototype `MatMat` and `MatTransposeMat` for multiresponse and multinomial
    consumers rather than requiring one allocation and backend call per
    response.
  - Fold normalization into the batched products using the same identities as
    `MatVec` and `MatTransposeVec`.
  - Keep the capability independent of any response, loss, or solver type.

## SLOPE rewrite support

These items come from comparing the normalization code in `../libslope` with
the current operator and column-view API. The JIT-normalization enum and its
four-way branches should not be ported: optional centers and scales already
represent the same four states.

- [ ] Add weighted logical-column products.
  - Provide a weighted dot product for
    `x_tilde_j^T (weights * vector)` without materializing the elementwise
    product.
  - Provide a weighted squared norm `sum_i weights_i * x_tilde_ij^2` for
    coordinate-wise Hessian calculations.
  - Offer variants accepting cached `sum(weights * vector)` and
    `sum(weights)` so repeated column operations remain O(nnz_j).
  - Accept borrowed inputs without forcing copies of dense matrix columns;
    account explicitly for contiguous versus strided vector views.
  - Test each formula against a dense oracle for all four center/scale
    combinations, including implicit and explicitly stored zeros.

- [ ] Make the sparse-plus-offset decomposition of `LazyColumn` easier to use.
  - Consider accessors such as `implicit_value()` (`-center / scale`),
    `raw_sum()`, and an iterator over stored corrections (`raw_value / scale`).
  - Keep these as representation-level column operations. Residual offsets,
    cached residual sums, and coordinate-update policy remain in the consuming
    solver.
  - Use the coordinate-descent example to verify that a centered residual
    update can stay O(nnz_j) without rederiving normalization formulas.

- [ ] Add the remaining normalization statistics used by `libslope`.
  - Add minimum centering and L1 and range scaling.
  - Extend `ColumnStats` with sparse-aware minima, ranges, L1 norms, and
    centered L1 norms. Implicit zeros must participate in every statistic.
  - Use the sparse closed form
    `sum_stored |value - center| + (n - nnz) * |center|` for centered L1 norms.
  - Preserve the current rule that non-translation-invariant scales such as L1,
    L2, and max-absolute are computed after centering. `libslope` computes its
    scales from raw `X`; exact legacy behavior for unusual combinations can be
    reproduced with `from_parts`.

- [ ] Add dense backends when work on the Rust SLOPE consumer begins.
  - Implement the orientation-independent operator and statistics traits for
    the selected faer and/or nalgebra dense matrix types.
  - Provide borrowed dense-column access without weakening the contract of
    `SparseColumns`.
  - Share logical-column operations across dense and sparse views where their
    complexity contracts remain honest.

- [ ] Benchmark active-set products before adding a restricted-operator API.
  - Full gradients map directly to `MatTransposeVec`, and active-set gradients
    can be computed through `LazyColumn` views.
  - Add a restricted `matvec` or transpose product only if working-set and
    screening benchmarks show that zero-filled full products are a bottleneck.
  - Do not encode flattened feature-response indices or SLOPE working-set
    policy in the matrix API.

- [ ] Pressure-test signed combinations of normalized columns in the SLOPE
      consumer.
  - Build cluster directions from `LazyColumn` views into a consumer-owned
    sparse-plus-offset workspace instead of materializing a normalized sparse
    matrix for every cluster.
  - Promote a general column-combination abstraction into this crate only if a
    second non-SLOPE consumer establishes a reusable contract.

## Submatrices

- [ ] Add submatrix support only after row and column views are established.
  - Define `lazy.submatrix(rows, cols)` as a restriction of the already
    normalized matrix; inherit the selected columns' existing centers and
    scales rather than recomputing them.
  - Start with contiguous ranges, where global-to-local index mapping is cheap.
  - Treat a block as the range-selected form of a submatrix rather than as an
    unrelated abstraction.
  - Use filtering view adapters when restriction breaks slice contiguity; do
    not weaken full CSC/CSR access merely to give every view the same type.
  - Defer arbitrary index selections until their ordering, duplicate-index,
    mapping, allocation, and complexity contracts are clear.
  - Keep selecting raw data and then normalizing it as a distinct operation
    with distinct statistical semantics.

## Tests and maintenance

- [ ] Exercise the public scalar claim with shared `f32` tests.
- [ ] Test explicit stored zeros and fully dense sparse columns/rows.
- [ ] Test every new view through dense reconstruction and relevant algebraic
      identities rather than solver convergence.
- [ ] Consider sharing private CSC statistics helpers between backends to
      prevent their numerical behavior from diverging.
- [ ] Replace the ignored crate-level example with a small compiling doctest
      once the shape-aware constructors settle.

## Explicit non-goals

- Coordinate descent, SGD, residual offsets, coefficient offsets, sampling,
  optimization workspaces, and convergence logic remain in consuming crates.
- Orientation capability traits must not conceal full scans or sparse-to-dense
  materialization.
- Column or row views expose matrix structure and normalization metadata, not
  solver-specific state or update operations.
- Arbitrary affine-expression machinery is deferred until more transformations
  than column centering and scaling require it.
- Intercept fitting, coefficient rescaling, working and strong sets, screening,
  KKT policy, SLOPE clusters, and cluster merging or splitting remain in the
  consuming model crate.
- Dense in-place normalization and a `modify_x` mode are not part of the lazy
  operator. Consumers that deliberately materialize normalized dense data can
  do so outside this crate.
