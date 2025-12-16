# GPU Sorting

A proof of concept comparing CPU vs GPU sorting performance on Apple Silicon (M1-M4).

## Overview

This project answers the question: **Will GPU sorting be faster than CPU sorting, and under what conditions?**

It implements:
- **CPU Sorting**: Using Rust's built-in `sort_unstable` (pattern-defeating quicksort) - O(n log n)
- **GPU Bitonic Sort**: Using Metal compute shaders with bitonic sort algorithm - O(n log²n)
- **GPU Radix Sort**: Using Metal compute shaders with DeviceRadixSort algorithm - O(n)

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

# Custom size (radix sort works with any size, bitonic rounds to power of 2)
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

## Sorting Algorithms

### Bitonic Sort (GPU)

[Bitonic sort](https://en.wikipedia.org/wiki/Bitonic_sorter) is a parallel sorting algorithm that maps well to GPU architecture because:

1. **Regular structure**: Every comparison follows a predictable pattern
2. **No data dependencies between threads**: Each thread can work independently
3. **O(log²n) parallel steps**: Scales well with large datasets

**Limitations:**
- Requires power-of-2 array sizes (padding needed for other sizes)
- O(n log²n) complexity is higher than optimal algorithms
- For 134M elements: ~49 billion comparisons vs ~3.6 billion for optimal

### Radix Sort (GPU - DeviceRadixSort)

[Radix sort](https://en.wikipedia.org/wiki/Radix_sort) is a non-comparison sorting algorithm with O(n) complexity. Our implementation uses the **DeviceRadixSort** algorithm with the "reduce-then-scan" approach, which is:

1. **Portable**: Works on Apple Silicon (unlike OneSweep which requires forward progress guarantees)
2. **Linear complexity**: O(n) - constant factor of 4 passes for 32-bit integers
3. **Memory efficient**: Uses threadgroup memory for local operations

**Algorithm passes:**
1. **Histogram**: Count keys in each of 256 buckets per threadgroup
2. **Reduce**: Sum histograms to global counts
3. **Scan**: Compute exclusive prefix sum
4. **Scatter Offsets**: Compute per-threadgroup output positions
5. **Scatter**: Reorder keys to output buffer

This is repeated 4 times (8 bits per pass for 32-bit integers).

**Why not OneSweep?**

OneSweep uses "chained-scan-with-decoupled-lookback" which provides ~10% better performance on NVIDIA GPUs, but **deadlocks on Apple Silicon** due to lack of forward progress guarantees. From [Linebender Wiki](https://linebender.org/wiki/gpu/sorting/):

> "OneSweep tends to run on anything that is not mobile, a software rasterizer, or Apple."

### Performance Characteristics

**When GPU Radix is fastest:**
- Large arrays (typically >10K elements)
- Any data distribution (radix sort is distribution-independent)
- No power-of-2 size requirement

**When GPU Bitonic might be preferred:**
- When stability is not required
- When you need to understand GPU parallelization concepts

**When CPU is faster:**
- Small arrays (<1K elements)
- Single sort operations (GPU initialization overhead)

## Project Structure

```
gpu-sorting/
├── src/
│   ├── main.rs           # Main benchmark runner
│   ├── cpu_sort.rs       # CPU sorting implementation
│   ├── gpu_sort.rs       # Metal GPU bitonic sort implementation
│   └── gpu_radix_sort.rs # Metal GPU radix sort implementation
├── shaders/
│   ├── bitonic_sort.metal  # Metal bitonic sort shaders
│   └── radix_sort.metal    # Metal radix sort shaders
├── benches/
│   └── sort_benchmark.rs   # Criterion benchmarks
├── docs/
│   └── case-studies/       # Performance analysis documentation
└── Cargo.toml
```

## Expected Results

On Apple Silicon, you might see results like:

```
GPU Sorting Proof of Concept
=============================

Array size: 1048576 elements (4 MB)

--- CPU Sorting (std::sort unstable / pdqsort) ---
CPU sort time: 45.123 ms
CPU sort verified: OK

--- GPU Sorting (Metal Bitonic Sort) ---
Using GPU: Device: Apple M3 Pro
GPU bitonic sort time: 52.456 ms
GPU bitonic sort verified: OK

--- GPU Sorting (Metal Radix Sort - DeviceRadixSort) ---
Using GPU: Apple M3 Pro
GPU radix sort time: 8.234 ms
GPU radix sort verified: OK
Results match CPU sort: OK

--- Performance Comparison ---
GPU Bitonic vs CPU: CPU is 1.16x faster
GPU Radix vs CPU: GPU is 5.48x faster
GPU Radix vs GPU Bitonic: Radix is 6.37x faster
```

*Note: Actual results vary based on your specific Apple Silicon chip.*

## Technical Notes

### Complexity Comparison

| Algorithm | Time Complexity | Operations for 134M elements |
|-----------|-----------------|------------------------------|
| CPU pdqsort | O(n log n) | ~3.6 billion |
| GPU Bitonic | O(n log²n) | ~49 billion |
| GPU Radix | O(n) | ~0.5 billion (4 passes) |

### Memory Model

The implementation uses `StorageModeShared` for CPU-GPU memory sharing, which is efficient on Apple Silicon's unified memory architecture.

### Why Bitonic Sort is Slower than CPU

Bitonic sort performs **13x more operations** than CPU's pdqsort. GPU parallelism compensates, but not enough to overcome the algorithmic disadvantage. This is why we implemented radix sort as the superior GPU algorithm.

## References

- [Metal Programming Guide](https://developer.apple.com/metal/)
- [metal-rs crate](https://crates.io/crates/metal)
- [GPUSorting by b0nes164](https://github.com/b0nes164/GPUSorting) - Reference implementation
- [Linebender GPU Sorting Wiki](https://linebender.org/wiki/gpu/sorting/)
- [CUB DeviceRadixSort](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html)

## License

This project is released into the public domain under the [Unlicense](LICENSE).
