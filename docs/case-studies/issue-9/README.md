# Case Study: Can We Really Find a Useful GPU Sorting Algorithm? (Issue #9)

## Executive Summary

This case study analyzes the performance characteristics of GPU vs CPU sorting algorithms on Apple Silicon (M3 Pro), following the implementation of GPU radix sort in Issue #5. Despite adding a theoretically superior O(n) radix sort algorithm, **GPU Radix Sort is still 1.31x slower than CPU Radix Sort on Apple M3 Pro**.

### Key Findings

| Benchmark Result (268M elements) | Performance |
|----------------------------------|-------------|
| GPU Radix vs CPU Radix | CPU is 1.31x faster |
| GPU Bitonic vs CPU Bitonic | GPU is 18.80x faster |
| GPU Radix vs CPU pdqsort | GPU is 1.05x faster |
| CPU Radix vs CPU pdqsort | CPU Radix is 1.37x faster |

### Core Question

> "Bitonic is just completely useless. As GPU version is slower than other CPU methods. GPU Radix is not faster than CPU Radix. Do we have something better still?"

### Bottom Line

The current GPU sorting implementation faces fundamental limitations on Apple Silicon:
1. **Hardware architecture mismatch**: Apple Silicon's unified memory architecture reduces the traditional GPU memory bandwidth advantage
2. **Scatter kernel bottleneck**: The current implementation uses a sequential scatter approach, underutilizing GPU parallelism
3. **Algorithm portability constraints**: The fastest GPU algorithms (OneSweep) cannot run on Apple Silicon due to missing forward progress guarantees

---

## Table of Contents

1. [Timeline of Events](#timeline-of-events)
2. [Observed Performance Data](#observed-performance-data)
3. [Root Cause Analysis](#root-cause-analysis)
4. [Technical Deep Dive](#technical-deep-dive)
5. [Proposed Solutions](#proposed-solutions)
6. [Conclusions](#conclusions)
7. [References](#references)

---

## Timeline of Events

### Project History

| Date | Event | Outcome |
|------|-------|---------|
| Dec 15, 2025 | PR #2: Initial GPU bitonic sort | GPU 2-3x slower than CPU |
| Dec 15, 2025 | PR #4: Threadgroup memory optimization | Gap reduced to 1.04-1.33x |
| Dec 15, 2025 | Issue #5: CPU still faster analysis | Root cause identified: algorithm complexity |
| Dec 15, 2025 | PR #6: GPU Radix Sort implementation | GPU radix added |
| Dec 16, 2025 | Issue #9: GPU Radix still slower | Current investigation |

### Issue #9 Benchmark (December 2025)

The user reported the following results on Apple M3 Pro with 268M elements:

```
GPU Sorting Proof of Concept
=============================

Array size: 268435456 elements (1073 MB)

--- CPU Sorting (std::sort unstable / pdqsort) ---
CPU pdqsort time: 3506.812 ms

--- CPU Sorting (Radix Sort) ---
CPU radix sort time: 2558.773 ms

--- CPU Sorting (Bitonic Sort) ---
CPU bitonic sort time: 87922.123 ms

--- GPU Sorting (Metal Bitonic Sort) ---
GPU bitonic sort time: 4676.231 ms

--- GPU Sorting (Metal Radix Sort - DeviceRadixSort) ---
GPU radix sort time: 3342.749 ms

--- Performance Comparison ---
[Fair Algorithm Comparisons]
GPU Radix vs CPU Radix: CPU is 1.31x faster
GPU Bitonic vs CPU Bitonic: GPU is 18.80x faster
```

---

## Observed Performance Data

### Issue #9 Raw Benchmark Data (268M Elements, Apple M3 Pro)

| Algorithm | Platform | Time (ms) | Elements/sec | Verification |
|-----------|----------|-----------|--------------|--------------|
| pdqsort | CPU | 3506.812 | 76.5M | OK |
| Radix Sort | CPU | 2558.773 | 104.9M | OK |
| Bitonic Sort | CPU | 87922.123 | 3.1M | OK |
| Bitonic Sort | GPU | 4676.231 | 57.4M | OK |
| Radix Sort | GPU | 3342.749 | 80.3M | OK |

### Performance Analysis by Algorithm Comparison

| Comparison | Speedup Factor | Winner |
|------------|----------------|--------|
| GPU Radix vs CPU Radix | 0.76x | CPU (1.31x faster) |
| GPU Bitonic vs CPU Bitonic | 18.80x | GPU |
| GPU Radix vs CPU pdqsort | 1.05x | GPU (marginal) |
| GPU Bitonic vs CPU pdqsort | 0.75x | CPU (1.33x faster) |
| CPU Radix vs CPU pdqsort | 1.37x | CPU Radix |
| GPU Radix vs GPU Bitonic | 1.40x | GPU Radix |

### Key Observations

1. **CPU Radix is the fastest overall** at 2558.773 ms (104.9M elements/sec)
2. **GPU Radix is slower than CPU Radix** by 1.31x despite identical algorithm
3. **GPU excels at parallel-friendly algorithms**: GPU Bitonic is 18.80x faster than CPU Bitonic
4. **Bitonic sort is impractical**: CPU Bitonic takes 87.9 seconds vs 2.6 seconds for CPU Radix

---

## Root Cause Analysis

### Primary Cause: Scatter Kernel Bottleneck

The GPU radix sort scatter kernel in `shaders/radix_sort.metal` uses a sequential approach:

```metal
// Current implementation - only thread 0 does the work!
if (tid == 0) {
    uint key = keys_in[global_idx];
    uint digit = (key >> shift) & RADIX_MASK;
    uint out_idx = local_offsets[digit];
    local_offsets[digit] = out_idx + 1;
    keys_out[out_idx] = key;
}
```

**Impact**: This serializes the scatter operation within each threadgroup, wasting GPU parallelism.

### Secondary Cause: Apple Silicon Unified Memory Architecture

Apple Silicon uses a unified memory architecture where CPU and GPU share the same physical memory:

| Traditional Discrete GPU | Apple Silicon |
|-------------------------|---------------|
| Dedicated high-bandwidth VRAM (400-1000 GB/s) | Shared system memory (~150-200 GB/s) |
| PCIe transfer overhead | No transfer overhead |
| GPU memory-bound algorithms benefit greatly | GPU advantage reduced |

**Impact**: The memory bandwidth advantage that GPUs typically have is significantly reduced on Apple Silicon.

### Tertiary Cause: OneSweep Unavailability

The fastest GPU radix sort algorithm (OneSweep) uses "chained-scan-with-decoupled-lookback" which:

> "On GPUs without a forward progress guarantee (of which Apple Silicon is especially noticeable), the algorithm may deadlock or experience extended stalls."
> — [Linebender GPU Sorting Wiki](https://linebender.org/wiki/gpu/sorting/)

**Impact**: We're forced to use the ~10% slower "reduce-then-scan" approach.

### Algorithm Complexity Analysis

For n = 268,435,456 (268M) elements:

| Algorithm | Complexity | Operations | Relative Work |
|-----------|------------|------------|---------------|
| CPU Radix | O(n × k) | 268M × 4 = 1.07B | 1.0x |
| GPU Radix | O(n × k) | 268M × 4 = 1.07B | 1.0x |
| CPU pdqsort | O(n log n) | 268M × 28 = 7.5B | 7.0x |
| Bitonic | O(n log²n) | 268M × 784 = 210B | 196x |

**Observation**: Despite identical algorithmic complexity, GPU Radix is slower due to implementation and hardware factors.

---

## Technical Deep Dive

### Current GPU Radix Implementation Architecture

```
Pass Structure (4 passes for 32-bit integers, 8 bits per pass):

For each pass:
┌─────────────────────────────────────────────────────────────┐
│ 1. Histogram Kernel    - Count digits per threadgroup       │
│    ↓                                                        │
│ 2. Reduce Kernel       - Sum to global histogram            │
│    ↓                                                        │
│ 3. Scan Kernel         - Exclusive prefix sum (Hillis-Steele)│
│    ↓                                                        │
│ 4. Scatter Offsets     - Compute per-threadgroup positions  │
│    ↓                                                        │
│ 5. Scatter Kernel      - Reorder keys ← BOTTLENECK!         │
└─────────────────────────────────────────────────────────────┘
```

### Scatter Kernel Analysis

**Current Implementation** (`radix_sort.metal:131-175`):

The scatter kernel processes keys sequentially to maintain correctness:
- Only thread 0 writes to output
- Other 255 threads are idle
- ~99.6% of GPU compute resources wasted

**Expected Parallel Implementation**:

A proper parallel scatter would use:
1. **Warp-level multi-split** for ranking keys within a warp
2. **Subgroup ballot operations** for efficient parallel counting
3. **Parallel scatter** where each thread scatters its own keys

### Memory Bandwidth Utilization

For 268M elements (1.07 GB):

| Operation | Read | Write | Total I/O |
|-----------|------|-------|-----------|
| Per radix pass | 1.07 GB | 1.07 GB | 2.14 GB |
| 4 passes total | 4.28 GB | 4.28 GB | 8.56 GB |
| + Histograms | ~50 MB | ~50 MB | ~100 MB |
| **Total** | ~4.3 GB | ~4.3 GB | **~8.6 GB** |

At Apple M3 Pro memory bandwidth (~150 GB/s):
- **Theoretical minimum**: 8.6 GB / 150 GB/s = **57 ms**
- **Observed time**: 3342.749 ms
- **Efficiency**: 57 / 3342.749 = **1.7%**

This massive gap indicates compute-bound bottleneck, not memory-bound.

---

## Proposed Solutions

### Solution 1: Parallel Scatter Implementation (High Priority)

**Description**: Implement proper parallel scatter using warp-level multi-split.

**Expected Improvement**: 10-50x faster scatter kernel

**Implementation Approach**:

```metal
// Proposed parallel scatter with ranking
kernel void radix_scatter_parallel(
    device const uint *keys_in [[buffer(0)]],
    device uint *keys_out [[buffer(1)]],
    device const uint *scatter_offsets [[buffer(2)]],
    constant uint &array_size [[buffer(3)]],
    constant uint &shift [[buffer(4)]],
    threadgroup uint *local_offsets [[threadgroup(0)]],
    threadgroup uint *local_ranks [[threadgroup(1)]],  // NEW: rank per key
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    // 1. Each thread loads its key and computes digit
    uint key = keys_in[global_idx];
    uint digit = (key >> shift) & RADIX_MASK;

    // 2. Use simd_ballot to count matching digits in warp
    // This is the "warp-level multi-split" technique
    simd_vote match = simd_ballot(true); // All threads
    uint rank = simd_prefix_exclusive_sum(match);

    // 3. Compute global output position
    uint output_idx = local_offsets[digit] + rank;

    // 4. Write in parallel
    keys_out[output_idx] = key;
}
```

**References**:
- [FidelityFX Sort](https://github.com/GPUOpen-Effects/FidelityFX-ParallelSort)
- [b0nes164 GPUSorting](https://github.com/b0nes164/GPUSorting)

### Solution 2: Use 4-bit Radix (Medium Priority)

**Description**: Switch from 8-bit to 4-bit digits, reducing histogram size from 256 to 16 buckets.

**Trade-offs**:
- **Pro**: Fewer atomic collisions, better occupancy, simpler ranking
- **Con**: 8 passes instead of 4

**Expected Result**: May be faster on Apple Silicon due to better parallelism within scatter.

**Implementation Reference**: [FidelityFX uses 4-bit digits](https://linebender.org/wiki/gpu/sorting/)

### Solution 3: Parallel CPU Sort with rayon (High Priority)

**Description**: Use Rust's rayon library for parallel CPU sorting.

**Expected Improvement**: 2-4x faster on multi-core CPUs

**Implementation**:

```rust
use rayon::prelude::*;

fn parallel_sort(data: &mut [u32]) {
    data.par_sort_unstable();
}
```

**Why This Helps**: Provides a strong baseline to compare against, may be faster than GPU for all sizes on Apple Silicon.

### Solution 4: IPS4o (In-place Parallel Super Scalar Samplesort)

**Description**: IPS4o is currently the fastest comparison-based parallel sorting algorithm.

**Performance Claims**:
- Outperforms competitors on any number of cores
- Up to 3x faster than alternatives
- Even beats radix sort in many scenarios

**Implementation**: Would require porting to Rust or using FFI to C++ implementation.

**References**:
- [IPS4o GitHub](https://github.com/ips4o/ips4o)
- [ACM Paper](https://dl.acm.org/doi/10.1145/3505286)

### Solution 5: Hybrid Metal/CPU Pipeline (Medium Priority)

**Description**: Use GPU for parallel-friendly preprocessing, CPU for final sort.

**Approach**:
1. GPU: Partition data into buckets (first radix pass)
2. CPU: Sort each bucket in parallel with rayon
3. Combine results

**Expected Benefit**: Leverages GPU for parallel scatter, CPU for efficient comparison.

### Solution Comparison Matrix

| Solution | Effort | Expected Speedup | Apple Silicon Compatible |
|----------|--------|------------------|-------------------------|
| Parallel Scatter | High | 10-50x scatter | Yes |
| 4-bit Radix | Medium | 1.5-2x | Yes |
| rayon CPU | Low | 2-4x | Yes |
| IPS4o | High | 2-3x | Yes |
| Hybrid Pipeline | Medium | 2-4x | Yes |

---

## Conclusions

### Key Insights

1. **GPU Radix Sort underperforms due to implementation bottleneck**, not algorithm design
2. **The scatter kernel is the critical path** - current sequential approach wastes 99.6% of GPU resources
3. **Apple Silicon's unified memory reduces GPU advantage** compared to discrete GPUs
4. **OneSweep unavailability limits maximum performance** on Apple Silicon

### Recommendations

**Short-term (Quick Wins)**:
1. Add parallel CPU sorting with rayon (low effort, immediate benefit)
2. Benchmark against parallel CPU as the new baseline

**Medium-term (High Impact)**:
1. Implement parallel scatter with warp-level multi-split
2. Consider 4-bit radix variant for Apple Silicon optimization

**Long-term (Research)**:
1. Investigate hybrid GPU-CPU pipelines
2. Monitor Apple Silicon evolution for forward progress guarantee support

### Final Answer to the Issue Question

> "Can we really find a useful GPU sorting algorithm?"

**Yes, but with caveats**:

1. **GPU Bitonic Sort is useful** for the specific case when you have a bitonic sort workload (18.8x faster than CPU)
2. **GPU Radix Sort can be made useful** by fixing the scatter kernel implementation
3. **On Apple Silicon specifically**, the unified memory architecture limits GPU advantages
4. **For general-purpose sorting on Apple Silicon**, parallel CPU (rayon) may be the pragmatic choice

The GPU excels when the algorithm is highly parallelizable (Bitonic vs Bitonic comparison shows 18.8x speedup). The Radix Sort comparison (GPU 1.31x slower than CPU) reveals an **implementation problem** rather than a fundamental limitation.

---

## References

### Primary Sources

1. [Linebender GPU Sorting Wiki](https://linebender.org/wiki/gpu/sorting/) - Comprehensive GPU sorting analysis
2. [Onesweep Paper (arXiv:2206.01784)](https://arxiv.org/abs/2206.01784) - State-of-the-art GPU radix sort
3. [b0nes164/GPUSorting](https://github.com/b0nes164/GPUSorting) - Portable GPU sorting implementations
4. [FidelityFX Parallel Sort](https://github.com/GPUOpen-Effects/FidelityFX-ParallelSort) - AMD's GPU sorting solution

### Algorithm References

5. [IPS4o GitHub](https://github.com/ips4o/ips4o) - Fastest parallel comparison sort
6. [Rayon Parallel Sort](https://docs.rs/rayon/latest/rayon/slice/trait.ParallelSliceMut.html) - Rust parallel sorting

### Apple Silicon Specific

7. [Metal Best Practices Guide](https://developer.apple.com/documentation/metal/metal_best_practices_guide) - Apple's optimization guide
8. [Optimize Metal Performance for Apple Silicon (WWDC 2020)](https://developer.apple.com/videos/play/wwdc2020/10632/) - Official Apple optimization video

### Academic Papers

9. [Fast sort on CPUs and GPUs (SIGMOD 2010)](https://dl.acm.org/doi/10.1145/1807167.1807207) - Foundational CPU/GPU comparison
10. [Engineering In-place Sorting Algorithms (ACM TOPC)](https://dl.acm.org/doi/10.1145/3505286) - IPS4o paper

---

## Appendix: Benchmark Reproduction

To reproduce the benchmarks from Issue #9:

```bash
# Clone and build
git clone https://github.com/konard/gpu-sorting.git
cd gpu-sorting
cargo build --release

# Run benchmark with 268M elements
cargo run --release -- 268435456

# Generate full report
cargo run --release -- 268435456 --report
```
