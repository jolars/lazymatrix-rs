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

- [x] Allow `LazyMatrix` to wrap a borrowed backend matrix.
  - Add forwarding implementations for the matrix capability traits on `&M`,
    or provide an explicit borrowed wrapper with equivalent ergonomics.
  - Support construction such as `LazyMatrix::new(&x, spec)` so fitting paths,
    cross-validation, and prediction do not need to consume the design matrix.

- [ ] Evaluate reusable-output operator methods before stabilizing the traits.
  - [x] Add `matvec_into` and `mat_transpose_vec_into` capabilities, with the
    backend implementations as the primitive path for allocating products.
  - [x] Allow input and output vector types to differ where backend APIs permit
    it, so borrowed or strided inputs can write into owned backend vectors.
  - [x] Specify dimension-checking and overwrite semantics, and implement
    backend-specific fast paths.
  - Measure allocation costs in an iterative consumer before designing a
    reusable normalization workspace.
  - Keep allocating convenience methods if they materially improve ergonomics.

- [ ] Prototype fused scaled operator application.
  - Add forward and transpose capabilities for `y = alpha * A * x + beta * y`.
  - Express overwrite, accumulation, and subtraction through the same primitive
    rather than allocating intermediate vectors.
  - Determine how callers can reuse the `S^-1 x` workspace needed by a scaled
    `LazyMatrix` without exposing backend-specific scratch types in the core
    traits.
  - Test `alpha` and `beta` at zero, one, negative values, and nonfinite values,
    along with empty and rectangular operators.

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
  - Consider reusable-output and fused `alpha`/`beta` variants only after the
    vector forms establish their ownership and workspace contracts.

## Operator algebra

- [ ] Evaluate a borrowed transpose operator view with a concrete consumer.
  - A transpose view should swap `MatVec` and `MatTransposeVec` without copying
    or changing the normalization represented by the original operator.
  - Do not imply that transposed sparse storage has acquired the opposite
    orientation-specific borrowing capability.

- [ ] Evaluate lightweight scaled, sum, and composition operator wrappers.
  - Candidate forms are `Scaled<A>`, `Sum<A, B>`, and `Composition<A, B>`;
    require compatible dimensions and implement products without materializing
    their operands.
  - Prefer named constructors or methods over `Add` and `Mul` until ownership,
    borrowing, scalar-zero behavior, and error messages are settled.
  - Add a wrapper only when a consumer benefits beyond spelling two existing
    operator calls explicitly.
  - Keep broadcast scalar/vector addition out of generic arithmetic: its row
    versus column semantics are ambiguous, and the general case needs a
    low-rank expression rather than altered normalization metadata.

## SLOPE rewrite support

These items come from comparing the normalization code in `../libslope` with
the current operator and column-view API. The JIT-normalization enum and its
four-way branches should not be ported: optional centers and scales already
represent the same four states.

- [x] Add weighted logical-column products.
  - [x] Provide a weighted dot product for
    `x_tilde_j^T (weights * vector)` without materializing the elementwise
    product.
  - [x] Provide a weighted squared norm `sum_i weights_i * x_tilde_ij^2` for
    coordinate-wise Hessian calculations.
  - [x] Offer variants accepting cached `sum(weights * vector)` and
    `sum(weights)` so repeated column operations remain O(nnz_j).
  - [x] Accept borrowed inputs without forcing copies of dense matrix columns;
    account explicitly for contiguous versus strided vector views.
  - [x] Test each formula against a dense oracle for all four center/scale
    combinations, including implicit and explicitly stored zeros.

- [x] Make the sparse-plus-offset decomposition of `LazyColumn` easier to use.
  - Provide `implicit_value()` (`-center / scale`), `raw_sum()`, and an
    iterator over stored corrections (`raw_value / scale`).
  - Keep these as representation-level column operations. Residual offsets,
    cached residual sums, and coordinate-update policy remain in the consuming
    solver.
  - Use the coordinate-descent example to verify that a centered residual
    update can stay O(nnz_j) without rederiving normalization formulas.

- [x] Add the remaining normalization statistics used by `libslope`.
  - Add minimum centering and L1 and range scaling.
  - Extend `ColumnStats` with sparse-aware minima, ranges, L1 norms, and
    centered L1 norms. Implicit zeros must participate in every statistic.
  - Use the sparse closed form
    `sum_stored |value - center| + (n - nnz) * |center|` for centered L1 norms.
  - Preserve the current rule that non-translation-invariant scales such as L1,
    L2, and max-absolute are computed after centering. `libslope` computes its
    scales from raw `X`; exact legacy behavior for unusual combinations can be
    reproduced with `from_parts`.

- [x] Add a generic dense/sparse logical-column interface.
  - Use `RawColumns` with an associated borrowed view for backend storage and
    `Columns` with an associated `LogicalColumn` for generic consumers.
  - Accept contiguous and strided inputs and destinations through
    `VectorView` / `VectorViewMut` without forcing copies.
  - Keep `SparseColumns` as the stronger contiguous-CSC capability and expose
    sparse representation helpers separately from common logical operations.

- [x] Add dense faer and nalgebra backends.
  - Implement operators, statistics, and raw-column access for owned matrices
    and immutable backend-native matrix views.
  - Exercise the same logical-column oracle over dense and sparse storage.
  - Demonstrate SLOPE-shaped weighted derivatives and active-column updates in
    the `slope_primitives` example.

- [ ] Benchmark active-set products before adding a restricted-operator API.
  - Full gradients map directly to `MatTransposeVec`, and active-set gradients
    can be computed through `LazyColumn` views.
  - Add a restricted `matvec` or transpose product only if working-set and
    screening benchmarks show that zero-filled full products are a bottleneck.
  - Do not encode flattened feature-response indices or SLOPE working-set
    policy in the matrix API.

- [ ] Benchmark logical column-pair products before adding a Gram-entry API.
  - Prototype `X_tilde_j^T X_tilde_k` and
    `X_tilde_j^T diag(weights) X_tilde_k` in a consuming coordinate or block
    method.
  - For CSC inputs, merge stored row indices and account for the centered
    background analytically rather than materializing either logical column.
  - Define the capability at matrix level if accepting two backend-specific
    associated column-view types makes a reusable `LogicalColumn` method
    awkward.
  - Document backend-specific complexity; do not promise sparse intersection
    costs for dense storage or for an incompatible sparse orientation.

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

## Surface ownership decisions

- [ ] Revisit whole-operator logical reductions only with a concrete consumer.
  - Column sums and squared norms are already available through `LogicalColumn`;
    do not duplicate them as matrix-wide allocation-returning methods merely
    for symmetry.
  - Spectral norms, bilinear forms such as `u^T A v`, and quadratic forms such
    as `A^T A v` remain compositions of existing primitives unless fusion is
    shown to matter.

- [ ] Keep general vector algebra outside the normalization trait surface.
  - Hadamard products, arbitrary maps and reductions, vector construction, and
    proximal operations belong to backends or solver-facing crates.
  - Promote a vector primitive here only when it is required to implement a
    matrix capability across multiple backends.

## Explicit non-goals

- Coordinate descent, SGD, residual offsets, coefficient offsets, sampling,
  optimization workspaces, and convergence logic remain in consuming crates.
- Orientation capability traits must not conceal full scans or sparse-to-dense
  materialization.
- Column or row views expose matrix structure and normalization metadata, not
  solver-specific state or update operations.
- Arbitrary affine-expression machinery is deferred until more transformations
  than column centering and scaling require it.
- Determinants, inverses, factorizations, direct solves, mutable entrywise
  matrix arithmetic, rank-one updates, and general matrix Hadamard products are
  not linear-operator capabilities and remain out of scope.
- Intercept fitting, coefficient rescaling, working and strong sets, screening,
  KKT policy, SLOPE clusters, and cluster merging or splitting remain in the
  consuming model crate.
- Dense in-place normalization and a `modify_x` mode are not part of the lazy
  operator. Consumers that deliberately materialize normalized dense data can
  do so outside this crate.
