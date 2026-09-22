# lazymatrix

[![CI](https://github.com/jolars/lazymatrix-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/jolars/lazymatrix-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/lazymatrix.svg)](https://crates.io/crates/lazymatrix)
[![docs.rs](https://img.shields.io/docsrs/lazymatrix)](https://docs.rs/lazymatrix)

Lazy column normalization for design matrices in Rust. `lazymatrix` presents

```text
X̃ = (X − 1cᵀ) S⁻¹
```

as a linear operator without materializing the centered matrix. This matters for
sparse matrices, where subtracting a column center would turn structural zeros
into nonzeros. Matrix–vector products instead use the original matrix:

```text
X̃v  = X(S⁻¹v) − 1(cᵀS⁻¹v)
X̃ᵀu = S⁻¹(Xᵀu − c Σu)
```

Centering and scaling are independently optional. The crate also provides
borrowed logical column views and sparse column access for algorithms such as
coordinate descent.

## Install

The core trait and operator API has no linear algebra dependency beyond
`num-traits`. Enable a backend for ready-made matrix implementations:

```sh
cargo add lazymatrix --features faer
# or
cargo add lazymatrix --features nalgebra
# or, for dense arrays
cargo add lazymatrix --features ndarray
# or, for sprs sparse matrices
cargo add lazymatrix --features sprs
```

Unversioned features select the newest supported release. Use a versioned
feature to stay on a particular release line:

| Backend | Versioned features | Unversioned feature selects |
| --- | --- | --- |
| faer | `faer_v0_22`, `faer_v0_23`, `faer_v0_24` | 0.24 |
| nalgebra | `nalgebra_v0_32`, `nalgebra_v0_33`, `nalgebra_v0_34`, `nalgebra_v0_35` | 0.35 |
| ndarray | `ndarray_v0_15`, `ndarray_v0_16`, `ndarray_v0_17` | 0.17 |
| sprs | `sprs_v0_11` | 0.11 |

For example, `cargo add lazymatrix --features nalgebra_v0_34` enables nalgebra
0.34 and nalgebra-sparse 0.11. Set your direct backend dependency to the same
release line. If Cargo enables multiple releases of one backend, lazymatrix
implements traits only for the newest enabled release. The `*_all` features
are internal markers, not entry points for selecting a backend.

The core and older backends support Rust 1.87. The `nalgebra` feature now selects
nalgebra 0.35, which requires Rust 1.89. To retain the previous release and Rust
1.87 support, replace `nalgebra` with `nalgebra_v0_34` in your feature list.
The sprs backend supports Rust 1.87 with sprs 0.11.4 (used in the lockfile);
sprs 0.11.5 requires Rust 1.88.

## Example

```rust
use lazymatrix::{
    Centering, LazyMatrix, MatVec, Normalization, Scaling,
};
use nalgebra::{DMatrix, DVector};

let x = DMatrix::from_row_slice(
    3,
    2,
    &[1.0, 0.0, 2.0, 3.0, 0.0, 4.0],
);
let x = LazyMatrix::new(
    x,
    Normalization::new(Centering::Mean, Scaling::Sd),
);

let y = x.matvec(&DVector::from_vec(vec![1.0, -1.0]));
```

The same interface works with faer and nalgebra dense matrices, their borrowed
views, and CSC sparse matrices. The ndarray backends support dense arrays,
including borrowed, transposed, and strided views:

```rust
use lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, Scaling};
use ndarray::array;

let x = array![[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]];
let lazy = LazyMatrix::new(
    x.view(),
    Normalization::new(Centering::Mean, Scaling::Sd),
);
let y = lazy.matvec(&array![1.0, -1.0]);
```

Allocating ndarray products use `Array1` vectors. Reusable-output products can
write into mutable, strided vector views. A lazy forward product needs an owned
input because it may clone and scale that input; call `to_owned()` on an
immutable input view first. Logical-column operations and reusable transpose
products accept immutable vector views directly. Borrowed column access preserves
the original strides and takes O(1) time; dense logical-column operations take
O(nrows) time.

The sprs backend supports owned CSC and CSR matrices and borrowed views. Use
`Vec` for allocating products, or enable an ndarray feature to use that
release's `Array1`. Reusable-output products accept any supported dense vector
view, including strided ndarray inputs and destinations. The `sprs` feature
does not select an ndarray backend version.

Wrap CSC storage in `SprsCsc` to borrow logical or sparse columns:

```rust
use lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, Scaling, SprsCsc};
use sprs::CsMat;

let x = CsMat::new_csc(
    (3, 2),
    vec![0, 2, 3],
    vec![0, 2, 1],
    vec![1.0, 3.0, 2.0],
);
let csc = SprsCsc::try_new(x.view()).unwrap();
let lazy = LazyMatrix::new(csc, Normalization::new(Centering::Mean, Scaling::Sd));
let y = lazy.matvec(&vec![1.0, -1.0]);
let column = lazy.sparse_column(0);
assert_eq!(column.row_indices(), &[0, 2]);
```

`SprsCsc::try_new` checks orientation in O(1) time and returns a CSR input
unchanged as `Err`. Convert CSR explicitly with `to_csc()` or `into_csc()` when
column access is needed. Products and statistics also work directly on
`CsMat` or `CsMatView` in either orientation. They visit stored entries without
materializing the normalized matrix. CSC statistics take O(ncols + nnz) time;
CSR statistics take O(nrows + ncols + nnz) time and O(ncols) workspace.

Logical columns borrow sprs vector views for any supported index type.
`SparseColumns` requires `usize` row indices so it can return slices without
copying. A centered column dot takes O(nrows + nnz_column); `dot_with_sum`
uses a caller-supplied vector sum to take O(nnz_column).

Enable `parallel` alongside a backend to compute column statistics with Rayon.
For sprs, CSC columns run independently in parallel; CSR statistics scan rows
serially to accumulate columns without converting storage.
See [`examples/`](examples/) for complete solver examples that consume the
operator.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT license](LICENSE-MIT), at your option.
