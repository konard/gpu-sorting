//! Criterion benchmarks for CPU vs GPU sorting.
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;

// Import our sorting modules
// Note: This requires the library crate to expose these modules
mod cpu_sort {
    pub fn sort_unstable(data: &mut [u32]) {
        data.sort_unstable();
    }
}

/// Generate random test data of given size
fn generate_random_data(size: usize) -> Vec<u32> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen()).collect()
}

/// Benchmark CPU sorting
fn bench_cpu_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("CPU Sort");

    for size_exp in [10, 12, 14, 16, 18, 20] {
        let size = 1usize << size_exp;
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || generate_random_data(size),
                |mut data| {
                    cpu_sort::sort_unstable(black_box(&mut data));
                    data
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_cpu_sort);
criterion_main!(benches);
