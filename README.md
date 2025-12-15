# GPU Sorting

A proof of concept comparing CPU vs GPU sorting performance on Apple Silicon (M1-M4).

## Overview

This project answers the question: **Will GPU sorting be faster than CPU sorting, and under what conditions?**

It implements:
- **CPU Sorting**: Using Rust's built-in `sort_unstable` (pattern-defeating quicksort)
- **GPU Sorting**: Using Metal compute shaders with bitonic sort algorithm

## Requirements

- macOS with Apple Silicon (M1, M2, M3, or M4)
- Rust 1.70 or later
- Xcode Command Line Tools (for Metal compilation)

## Quick Start

### Install Rust (if not already installed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Clone and Build

```bash
git clone https://github.com/konard/gpu-sorting.git
cd gpu-sorting
cargo build --release
```

### Run Basic Benchmark

```bash
# Default: 1M elements
cargo run --release

# Custom size (will be rounded to nearest power of 2)
cargo run --release -- 1000000

# Full benchmark across multiple sizes
cargo run --release -- 1000000 --benchmark
```

### Run Criterion Benchmarks

For detailed statistical benchmarks:

```bash
cargo bench
```

This generates HTML reports in `target/criterion/`.

## How It Works

### Bitonic Sort on GPU

[Bitonic sort](https://en.wikipedia.org/wiki/Bitonic_sorter) is a parallel sorting algorithm that maps well to GPU architecture because:

1. **Regular structure**: Every comparison follows a predictable pattern
2. **No data dependencies between threads**: Each thread can work independently
3. **O(log²n) parallel steps**: Scales well with large datasets

The algorithm works by:
1. Building bitonic sequences (alternating ascending/descending)
2. Merging sequences pairwise until the entire array is sorted

### Performance Characteristics

**When GPU is faster:**
- Large arrays (typically >100K elements)
- Data already in GPU memory
- Batch sorting operations

**When CPU is faster:**
- Small arrays (<10K elements)
- Single sort operations (GPU initialization overhead)
- Non-power-of-2 sizes (padding overhead)

## Project Structure

```
gpu-sorting/
├── src/
│   ├── main.rs       # Main benchmark runner
│   ├── cpu_sort.rs   # CPU sorting implementation
│   └── gpu_sort.rs   # Metal GPU sorting implementation
├── shaders/
│   └── bitonic_sort.metal  # Metal shader source
├── benches/
│   └── sort_benchmark.rs   # Criterion benchmarks
└── Cargo.toml
```

## Expected Results

On Apple Silicon, you might see results like:

```
Array size: 1048576 elements (4 MB)

--- CPU Sorting (std::sort unstable) ---
CPU sort time: 45.123 ms
CPU sort verified: OK

--- GPU Sorting (Metal Bitonic Sort) ---
Using GPU: Apple M1 Pro
GPU sort time: 12.456 ms
GPU sort verified: OK
Results match CPU sort: OK

--- Performance Comparison ---
GPU is 3.62x faster than CPU
```

*Note: Actual results vary based on your specific Apple Silicon chip.*

## Technical Notes

### Why Bitonic Sort?

While radix sort has better theoretical complexity (O(n)), bitonic sort was chosen because:
1. Simpler implementation for proof of concept
2. Better demonstrates GPU parallelization concepts
3. More consistent performance across data distributions

For production use, consider [Onesweep](https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus) or [FidelityFX Parallel Sort](https://gpuopen.com/fidelityfx-parallel-sort/).

### Power of 2 Requirement

Bitonic sort requires array sizes to be powers of 2. The implementation automatically pads input to the nearest power of 2.

### Memory Model

The implementation uses `StorageModeShared` for CPU-GPU memory sharing, which is efficient on Apple Silicon's unified memory architecture.

## References

- [Metal Programming Guide](https://developer.apple.com/metal/)
- [metal-rs crate](https://crates.io/crates/metal)
- [Bitonic Sort on GPU (NVIDIA)](https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting)
- [Linebender GPU Sorting Wiki](https://linebender.org/wiki/gpu/sorting/)

## License

This project is released into the public domain under the [Unlicense](LICENSE).
