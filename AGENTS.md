# AGENTS.md

Guidance for AI agents working in this repository.

## What this crate is

`lazymatrix` presents a column-normalized design matrix

```
X̃ = (X − 1cᵀ) S⁻¹      (c = column centers, S = diag(column scales))
```

as a **linear operator**, without ever materializing `X − 1cᵀ`. Centering a
sparse matrix turns its structural zeros into nonzeros and destroys sparsity;
factoring the normalization into the matrix–vector products avoids that. This is
the Rust backend for the R package `lazymatrix` and for Rust regression/ML
crates.

The two operations, folded so the backend only ever multiplies the original
sparse `X`:

```
X̃ v  = X (S⁻¹ v) − 1 · (cᵀ S⁻¹ v)
X̃ᵀ u = S⁻¹ (Xᵀ u − c · Σu)
```

Centering and scaling are each independently optional.

## Architecture

- `src/traits.rs` — the whole trait surface:
  - `Scalar` — blanket-impl bundle of `num-traits` bounds (`f32`/`f64` qualify).
  - `MatVec` / `MatTransposeVec` — the matrix-free operator pair.
  - Five vector primitives (`ElemDivAssign`, `DotSlice`, `SubScalarAssign`,
    `SumEntries`, `ScaledSubSlice`), phrased as a backend vector against a
    coefficient slice `&[F]`.
  - `ColumnStats` — column means/sds/maxabs/l2 (+ centered variants), computed
    over stored sparse entries without densifying.
  - `Centering` / `Scaling` / `Normalization`.
- `src/lib.rs` — `LazyMatrix<M, F = f64>` (generic, dependency-free core),
  constructors, and the two operator impls.
- `src/faer_sparse_backend.rs` — `[feature = "faer"]`, impls on
  `SparseColMat<usize, F>` / `Col<F>`.
- `src/nalgebra_sparse_backend.rs` — `[feature = "nalgebra"]`, impls on
  `CscMatrix<F>` / `DVector<F>`.

With no features the crate is just the traits and `LazyMatrix` — zero deps
beyond `num-traits`. Backends are added by implementing the trait surface on a
matrix/vector pair; no FFI lives in this crate.

## Conventions

- Store centers/scales as `Vec<F>`, never the backend vector type — they only
  appear in elementwise ops against `&[F]`.
- Vector-trait impls do scalar arithmetic only, so they bound on `F: Scalar`
  alone; the heavier backend bounds (`faer_traits::ComplexField`,
  `nalgebra::Closed*Assign`) belong on the matrix/operator impls.
- `Scalar` is `num-traits::Float`-based and does **not** provide `AddAssign`.
  Accumulate with iterator `.sum()` / fold, not `x += y` (clippy's
  `assign_op_pattern` suggestion would not compile here).
- Scales are floored at 1 (`floor_zeros`) so a constant/zero-variance column
  never divides by zero.
- When both centering and scaling are on, scales come from the **centered**
  column. `sd` is centering-invariant; `l2`/`maxabs` use the sparse closed-form
  `*_centered` variants (`‖x_j − c_j‖₂² = Σ_stored (v − c_j)² + (n − nnz_j)·c_j²`).

## Commands

```
cargo build                              # default: zero-dep core
cargo build --all-features
cargo test  --all-features
cargo clippy --all-features --all-targets -- -D warnings
cargo fmt --check
```

The devenv pre-commit hooks run `clippy` (all features) and `rustfmt`; keep both
clean. Backend dependency versions are pinned to match the `basin` crate
(faer 0.24, nalgebra 0.34, nalgebra-sparse 0.11).

## Testing

`tests/common/runner.rs` is a backend-generic suite driven per backend
(`faer_backend.rs`, `nalgebra_backend.rs`, `cross_backend.rs`). The oracle
materializes `X̃` densely and runs naive products; lazy output must match up to
FP noise. New backends get a thin test file supplying build/convert closures and
calling `run_backend_suite`. RNG is seeded (`ChaCha8Rng`) for reproducibility.

Shared test modules live in subdirectories and are pulled in with
`#[path = "common/runner.rs"] mod common;` — do **not** use `mod.rs` filenames
anywhere in this repo.

## Example-driven design

Operator *correctness* is covered by the per-backend operator tests (oracle
parity + the adjoint identity) — not by running solvers. Real algorithms that
*consume* the operator live as runnable `examples/` (e.g. `least_squares_gd`) to
demonstrate usage and pressure-test the API for missing capabilities; the
solvers themselves stay out of the library and out of the test suite. First-order
methods (GD, CG, ISTA/FISTA) need only `matvec`/`mat_transpose_vec`. When a
consuming algorithm reveals a missing *matrix* capability, add it to the
operator; never add the *solver* to the lib.

**Coordinate descent — deferred.** CD wants single-column ops
(`X̃ⱼᵀr = (Xⱼᵀr − cⱼΣr)/sⱼ` and the residual update `r −= Δ·X̃ⱼ`). The dot stays
O(nnzⱼ) if the solver tracks `Σr` incrementally, but the centered residual
update carries an O(n) broadcast term (`r[i] += Δcⱼ/sⱼ` for all i). Keeping CD at
O(nnz) requires the glmnet/sklearn *offset trick*: represent the residual as
`(stored vector + scalar offset)` so the broadcast is O(1). That choice shapes
the column-access API — a simple `col_xt_vec`/`col_axpy` pair is clean but O(n)
when centering, whereas a column view (`(row,value)` entries + `cⱼ`/`sⱼ` +
`‖X̃ⱼ‖²`, solver-managed offset) preserves sparsity. Decide that fork when CD is
actually built.

## Scope (v1)

In: the operator (`matvec` / `mat_transpose_vec`) + `ColumnStats`, over
faer-sparse and nalgebra-sparse. Out (deferred, recoverable without rework):
Gram `X̃ᵀX̃`, norms, SVD, iterative solvers, row normalization, dense/ndarray
backends, and any FFI. Solvers belong in consuming crates — this crate only
provides a faithful linear operator.
