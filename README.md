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
`num-traits`. Enable a backend for ready-made dense and CSC implementations:

```sh
cargo add lazymatrix --features faer
# or
cargo add lazymatrix --features nalgebra
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
views, and CSC sparse matrices. See [`examples/`](examples/) for complete solver
examples that consume the operator.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT license](LICENSE-MIT), at your option.
