use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn two_pass(values: &[f64], nrows: usize) -> f64 {
    let n = nrows as f64;
    let mean = values.iter().copied().sum::<f64>() / n;
    let stored = values
        .iter()
        .map(|&value| {
            let deviation = value - mean;
            deviation * deviation
        })
        .sum::<f64>();
    let implicit = (nrows - values.len()) as f64 * mean * mean;
    ((stored + implicit) / n).sqrt()
}

fn combined_variance(values: &[f64], nrows: usize) -> f64 {
    let mut stored_mean = 0.0;
    let mut stored_m2 = 0.0;

    for (index, &value) in values.iter().enumerate() {
        let count = (index + 1) as f64;
        let delta = value - stored_mean;
        stored_mean += delta / count;
        stored_m2 += delta * (value - stored_mean);
    }

    let stored_count = values.len();
    let implicit_count = nrows - stored_count;
    let cross = if stored_count == 0 || implicit_count == 0 {
        0.0
    } else {
        stored_mean * stored_mean * stored_count as f64 * implicit_count as f64 / nrows as f64
    };
    ((stored_m2 + cross) / nrows as f64).sqrt()
}

fn benchmark_column_sds(c: &mut Criterion) {
    let cases = [
        (
            "dense_large_offset",
            100_000,
            (0..100_000)
                .map(|index| 1.0e12 + (index % 17) as f64)
                .collect::<Vec<_>>(),
        ),
        (
            "ten_percent_stored",
            100_000,
            (0..10_000)
                .map(|index| (index % 17) as f64 - 8.0)
                .collect::<Vec<_>>(),
        ),
        (
            "one_percent_stored",
            100_000,
            (0..1_000)
                .map(|index| (index % 17) as f64 - 8.0)
                .collect::<Vec<_>>(),
        ),
    ];

    let mut group = c.benchmark_group("column_sd");
    for (name, nrows, values) in cases {
        group.bench_with_input(BenchmarkId::new("two_pass", name), &values, |b, values| {
            b.iter(|| two_pass(black_box(values), black_box(nrows)));
        });
        group.bench_with_input(
            BenchmarkId::new("combined_variance", name),
            &values,
            |b, values| {
                b.iter(|| combined_variance(black_box(values), black_box(nrows)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_column_sds);
criterion_main!(benches);
