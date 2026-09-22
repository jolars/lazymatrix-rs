use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lazymatrix::{
    Centering, ColumnStats, LazyMatrix, MatTransposeVecInto, MatVecInto, Normalization, Scaling,
};

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::sprs;

fn benchmark_sprs(c: &mut Criterion) {
    let mut group = c.benchmark_group("sprs");
    for (nrows, ncols) in [(1_000, 100), (10_000, 1_000)] {
        let mut triplets = sprs::TriMat::new((nrows, ncols));
        for j in 0..ncols {
            for i in (0..nrows).step_by(100) {
                triplets.add_triplet((i + 17 * j) % nrows, j, (i % 13) as f64 - 6.0);
            }
        }
        let csc = triplets.to_csc::<usize>();
        let csr = csc.to_csr();
        group.throughput(Throughput::Elements(csc.nnz() as u64));
        for (layout, matrix) in [("csc", &csc), ("csr", &csr)] {
            let case = format!("{layout}_{nrows}x{ncols}");
            group.bench_with_input(BenchmarkId::new("col_sds", &case), matrix, |b, matrix| {
                b.iter(|| black_box(matrix).col_sds().unwrap());
            });

            let lazy =
                LazyMatrix::new(matrix, Normalization::new(Centering::Mean, Scaling::Sd)).unwrap();
            let v = vec![1.0; ncols];
            let u = vec![1.0; nrows];
            let mut y = vec![0.0; nrows];
            let mut z = vec![0.0; ncols];
            group.bench_function(BenchmarkId::new("matvec_into", &case), |b| {
                b.iter(|| lazy.matvec_into(black_box(&v), black_box(&mut y)).unwrap());
            });
            group.bench_function(BenchmarkId::new("transpose_into", &case), |b| {
                b.iter(|| {
                    lazy.mat_transpose_vec_into(black_box(&u), black_box(&mut z))
                        .unwrap()
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_sprs);
criterion_main!(benches);
