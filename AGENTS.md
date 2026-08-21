# Repository Guidelines

## Project Structure & Module Organization

`src/lib.rs` contains crate documentation, module declarations, and public
re-exports. The main implementation is divided as follows:

- `src/normalization.rs` defines `Centering`, `Scaling`, and `Normalization`.
- `src/matrix.rs` implements `LazyMatrix`, construction, column access, and
  operator behavior.
- `src/column.rs` contains logical and sparse borrowed column views.
- `src/traits/operator.rs` defines matrix shape and allocating or
  reusable-output matrix-vector products.
- `src/traits/vectors.rs`, `stats.rs`, and `columns.rs` define vector algebra,
  sparse-aware statistics, and column capabilities.
- `src/backends/faer/` and `src/backends/nalgebra/` contain feature-gated dense,
  sparse, and vector implementations. Shared helpers live in
  `src/backends/support.rs`.

Keep backend-independent logic out of backend implementation directories. With
no features enabled, the crate provides its traits and `LazyMatrix` with only
`num-traits`; adding a backend means implementing the existing trait surface for
a matrix/vector pair. This crate contains no FFI.

Integration tests are in `tests/`. The backend suites reuse
`tests/common/runner.rs`, while `tests/cross_backend.rs` checks agreement
between implementations. Runnable demonstrations belong in `examples/`, and
Criterion benchmarks belong in `benches/`. See `TODO.md` for planned API work.

## Architecture & Invariants

`LazyMatrix` presents a normalized design matrix as a linear operator:

```text
X_tilde = (X - 1c^T) S^-1
X_tilde v = X(S^-1 v) - 1(c^T S^-1 v)
X_tilde^T u = S^-1(X^T u - c sum(u))
```

Centering and scaling are independently optional. Never materialize the centered
matrix: doing so turns structural zeros into nonzeros and defeats sparse
storage. Backends must multiply the original `X` and fold normalization into the
operator calculation.

Preserve these normalization semantics:

- Replace exact zero scales with one so constant columns do not divide by zero.
- When centering and scaling are both enabled, compute scales from centered
  columns. SD and range are centering-invariant; L1, L2, and max-absolute use
  the sparse closed-form `*_centered` methods, including contributions from
  implicit zeros. Preserve IEEE nonfinite values.
- Store centers and scales as `Vec<F>`, not backend-specific vector types.

## Column Access & Capability Boundaries

Keep storage orientation out of the operator and express contiguous borrowing as
a capability:

- `MatVec`, `MatTransposeVec`, and `ColumnStats` are orientation-agnostic.
- Dense backends and storage with natural column views implement `RawColumns`.
- Only storage that can borrow contiguous CSC slices implements `SparseColumns`.

`LazyMatrix::column(j)` requires `RawColumns`; `sparse_column(j)` retains the
stronger `SparseColumns` bound. Do not hide an O(nnz)-per-column gather behind
`SparseColumns`. An incompatible storage layout should fail at compile time
rather than silently add a gather.

Views borrow rather than copy. `LazyColumn` wraps a backend raw view and the two
normalization scalars; `LazySparseColumn` also exposes CSC slices and the
sparse-plus-offset representation. Put logical column operations—dots, weighted
products, scaled additions, and norms—on the view so callers do not rederive
normalization formulas. Keep raw slices available for specialized algorithms,
and document complexity: a centered dot requiring a dense vector sum is O(n +
nnz), while a cached-sum path is O(nnz). Row views, `SparseRows`, and CSR
backends are prospective work described in `TODO.md`, not current API.

## Example-Driven Design

Keep solver algorithms and state in `examples/` or downstream crates. Examples
such as `least_squares_gd` and `coordinate_descent` demonstrate consumers and
expose missing matrix capabilities, but solvers do not belong in the library or
its correctness suite. Add a reusable operator or logical-column operation when
an example reveals a genuine matrix need; do not add residual policies, cached
sums, update rules, or an entire solver to the crate.

## Build, Test, and Development Commands

- `cargo build --locked` builds the dependency-light core.
- `cargo build --all-features --locked` checks both supported backends.
- `task test` runs the core, each backend, and the all-feature test matrix.
- `task ci` runs formatting, Clippy, documentation, and all tests—the local
  equivalent of GitHub CI.
- `cargo bench --locked --bench column_sds` runs the column-statistics
  benchmarks.
- `cargo run --locked --example coordinate_descent --features faer` runs an
  example.

The repository's devenv supplies Rust 1.87, `go-task`, and the configured
pre-commit hooks.

## Coding Style & Naming Conventions

Use standard `rustfmt` formatting (four-space indentation) and keep
`cargo clippy --all-features --all-targets --locked -- -D warnings` clean. Name
modules, functions, and files in `snake_case`; types and traits use
`UpperCamelCase`. Document public APIs and explain the reason—not the
mechanics—in comments. Keep feature-specific dependencies and trait
implementations behind their matching Cargo feature. `Scalar` does not imply
`AddAssign`; in generic code, accumulate with iterator sums or folds instead of
`+=`, and keep heavier backend bounds on matrix and operator implementations.

## Testing Guidelines

Prefer test-driven changes: add a failing regression or behavior test before the
implementation. Use `#[test]` integration tests with descriptive `snake_case`
names. `tests/common/runner.rs` is a backend-generic suite driven by thin
`faer_backend.rs` and `nalgebra_backend.rs` adapters. Extend the generic runner
when behavior should apply to every backend; use a backend-specific file only
for adapter or representation details. A new backend should provide construction
and conversion closures, then call `run_backend_suite`.

The test oracle materializes the normalized matrix densely and compares naive
products with lazy output up to floating-point noise. Preserve oracle parity and
the adjoint identity; `cross_backend.rs` separately verifies agreement between
implementations. Seed random generators with `ChaCha8Rng`, and compare floats
with `approx`. Import shared modules using
`#[path = "common/runner.rs"] mod common;`; do not introduce `mod.rs` files. Run
`task test` during development and `task ci` before submitting.

## Commit & Pull Request Guidelines

Recent history generally follows Conventional Commits, such as `feat:`,
`refactor:`, `docs:`, and `chore(deps):`. Write short, imperative subjects and
keep each commit focused. Pull requests should explain the motivation and API or
performance impact, link relevant issues (for example, `Closes #123`), and list
the commands run. Include benchmarks when changing hot paths and update examples
or documentation when public behavior changes.
