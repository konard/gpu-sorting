# GPU Sorting

A proof of concept comparing CPU vs GPU sorting performance on Apple Silicon (M1-M4).

## Overview

This project answers the question: **Will GPU sorting be faster than CPU sorting, and under what conditions?**

It implements:

## Supported Algorithms

This project implements the following GPU-parallelizable sorting algorithms, each with CPU, CPU Parallel, and GPU versions for fair comparison:

| Algorithm | CPU | CPU Parallel | GPU | Complexity | Notes |
|-----------|-----|--------------|-----|------------|-------|
| **Radix Sort** | ✓ | ✓ | ✓ | O(n) | Best for GPU, works with any array size |
| **Bitonic Sort** | ✓ | ✓ | ✓ | O(n log²n) | Requires power-of-2 array sizes |
| **pdqsort** | ✓ | ✓ | N/A | O(n log n) | Not GPU parallelizable (comparison-based) |

### GPU-Parallelizable Algorithms

Based on the [Linebender GPU Sorting Wiki](https://linebender.org/wiki/gpu/sorting/), the main GPU-parallelizable sorting algorithms are:

1. **Radix Sort (DeviceRadixSort)** - Our implementation uses the portable "reduce-then-scan" approach
2. **Bitonic Sort** - Classic parallel sorting network, requires power-of-2 sizes
3. **Merge Sort** - Segmented sorting support, but surpassed by radix approaches
4. **OneSweep** - Fastest LSD radix sort, but deadlocks on Apple Silicon
5. **FidelityFX Parallel Sort** - Tree reduction histogram approach, popular in gamedev

We implement Radix Sort and Bitonic Sort as they are the most practical for Apple Silicon.

### Algorithm Summary

**GPU Algorithms:**
- **GPU Bitonic Sort**: Using Metal compute shaders with bitonic sort algorithm - O(n log²n)
- **GPU Radix Sort**: Using Metal compute shaders with DeviceRadixSort algorithm - O(n)
- **GPU Radix Sort (SIMD)**: SIMD-optimized scatter kernel for faster performance - O(n)

**CPU Algorithms:**
- **CPU pdqsort**: Rust's built-in `sort_unstable` (pattern-defeating quicksort) - O(n log n)
- **CPU Parallel pdqsort**: Multi-threaded pdqsort using rayon - O(n log n)
- **CPU Radix Sort**: For fair comparison with GPU radix sort - O(n)
- **CPU Parallel Radix Sort**: Multi-threaded radix sort - O(n)
- **CPU Bitonic Sort**: For fair comparison with GPU bitonic sort - O(n log²n)
- **CPU Parallel Bitonic Sort**: Multi-threaded bitonic sort using rayon - O(n log²n)

**Reporting:**
- **Links Notation Reports**: Generate benchmark reports in [Links Notation](https://github.com/link-foundation/links-notation) format
- **Markdown Reports**: Auto-generated performance comparison tables
- **Web Visualizer**: Interactive React.js dashboard for viewing benchmark results

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
# Default: 1M elements (2^20 = 1,048,576)
cargo run --release

# Use power of 2 sizes for consistent benchmarking:
cargo run --release -- 1048576      # 2^20 = 1M elements
cargo run --release -- 4194304      # 2^22 = 4M elements
cargo run --release -- 16777216     # 2^24 = 16M elements

# Full benchmark across multiple power-of-2 sizes (1K to 16M)
cargo run --release -- --benchmark

# Generate Links Notation and Markdown reports
cargo run --release -- --report
```

**Note:** Using power of 2 array sizes is recommended for:
- Consistent performance comparison across different runs
- Bitonic sort compatibility (which requires power-of-2 sizes)
- Fair comparison with GPU implementations

### Generate Reports

```bash
# Run benchmark and save reports to data/reports/
cargo run --release -- --report

# Convert existing Lino report to Markdown
cargo run --release --bin lino2md data/reports/benchmark_*.lino output.md
```

### View Reports in Browser

Open `data/index.html` in your browser to visualize benchmark results interactively. The web visualizer can:
- Parse Links Notation benchmark reports
- Display performance charts with logarithmic scale
- Show data tables and fair comparisons
- Calculate GPU vs CPU speedup ratios

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
│   ├── main.rs               # Main benchmark runner with verification & performance tables
│   ├── cpu_sort.rs           # CPU pdqsort (standard library)
│   ├── cpu_radix_sort.rs     # CPU radix sort for fair comparison
│   ├── cpu_bitonic_sort.rs   # CPU bitonic sort for fair comparison
│   ├── cpu_parallel_sort.rs  # CPU parallel sorts (pdqsort, radix, bitonic) using rayon
│   ├── gpu_sort.rs           # Metal GPU bitonic sort implementation
│   ├── gpu_radix_sort.rs     # Metal GPU radix sort implementation
│   ├── lino_report.rs        # Links Notation report generator
│   └── bin/
│       └── lino2md.rs        # CLI tool to convert Lino to Markdown
├── shaders/
│   ├── bitonic_sort.metal    # Metal bitonic sort shaders
│   └── radix_sort.metal      # Metal radix sort shaders
├── data/
│   ├── index.html            # React.js benchmark visualizer
│   └── reports/              # Generated benchmark reports (.lino, .md)
├── benches/
│   └── sort_benchmark.rs     # Criterion benchmarks
├── docs/
│   └── case-studies/         # Performance analysis documentation
└── Cargo.toml
```

## Fair Algorithm Comparisons

A key feature of this project is enabling **fair** performance comparisons:

| Comparison | Why it's Fair |
|------------|---------------|
| GPU Radix vs CPU Radix (single) | Same algorithm, different platform |
| GPU Radix vs CPU Radix (parallel) | Same algorithm, multi-core CPU vs GPU |
| GPU Bitonic vs CPU Bitonic (single) | Same algorithm, different platform |
| GPU Bitonic vs CPU Bitonic (parallel) | Same algorithm, multi-core CPU vs GPU |

This answers the core question: **Does the GPU provide actual performance benefits for the same algorithm?**

### Benchmark Output

The benchmark produces two tables:

1. **Verification Table**: Confirms all algorithms produce correct sorted output (warmup)
2. **Performance Table**: Shows execution times with CPU single-threaded as baseline

Each algorithm shows speedup/slowdown factors relative to its CPU single-threaded baseline:
- **CPU Parallel**: How much faster is multi-threaded CPU vs single-threaded
- **GPU**: How much faster is GPU vs single-threaded CPU

The Performance Ranking section sorts all implementations by speed for each array size.

Cross-algorithm comparisons (GPU Radix vs CPU pdqsort) are also available but should be interpreted differently since they compare different algorithms with different theoretical complexities.

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
- [Links Notation](https://github.com/link-foundation/links-notation) - Data format for benchmark reports

## License

This project is released into the public domain under the [Unlicense](LICENSE).
