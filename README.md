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
```

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
views, and CSC sparse matrices. The `ndarray` feature supports ndarray 0.17 dense
arrays, including borrowed, transposed, and strided views:

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

Enable `parallel` alongside a backend to compute column statistics with Rayon.
See [`examples/`](examples/) for complete solver examples that consume the
operator.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT license](LICENSE-MIT), at your option.
