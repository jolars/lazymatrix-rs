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

- [ ] Make standard-deviation calculations numerically stable.
  - Replace `E[x^2] - E[x]^2` with a stable two-pass or combined-variance
    calculation that accounts for implicit sparse zeros.
  - Add a regression test using a large offset and small variation, such as
    values near `1e12` whose true variance is nonzero.
  - Apply the same algorithm and edge-case policy to every backend.

- [ ] Define the policy for empty matrices and nonfinite statistics.
  - Decide whether normalization of a matrix with zero rows is rejected.
  - Decide how `NaN` and infinite input values propagate or fail.
  - Document and test the chosen behavior.

- [ ] Validate explicit normalization parameters.
  - Decide whether scales must be finite and strictly positive or merely
    finite and nonzero.
  - Ensure `new` and `with_scales` cannot silently construct an operator that
    divides by zero.
  - Consider a fallible constructor if validation errors should not panic.
  - Ensure the zero-scale guard treats nonfinite values deliberately.

## Sparse access capabilities

- [ ] Add `SparseColumns` for contiguous CSC column access.
  - Return borrowed row-index and raw-value slices without copying.
  - Implement it only for storage types that provide efficient contiguous
    column access.
  - Keep it separate from `ColumnStats`, which remains
    orientation-independent.

- [ ] Add `LazyMatrix::column` and a borrowed `LazyColumn` view.
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

- [ ] Evaluate reusable-output operator methods before stabilizing the traits.
  - Prototype `matvec_into` and `mat_transpose_vec_into` as additive
    capabilities or as the primitive operator interface.
  - Measure allocation costs in an iterative consumer before designing a
    reusable normalization workspace.
  - Keep allocating convenience methods if they materially improve ergonomics.

- [ ] Avoid cloning the forward input when scaling is inactive.
  - Preserve the direct backend path for raw and center-only products.
  - Benchmark before adding more elaborate scratch-storage machinery.

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
