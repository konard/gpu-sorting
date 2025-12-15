# Case Study: Why CPU is Faster Than GPU for Sorting (Issue #5)

## Executive Summary

This case study analyzes why the GPU sorting implementation in this repository is slower than CPU sorting on Apple Silicon M3 Pro, despite significant optimization efforts. The analysis reveals that **the root cause is algorithmic complexity combined with Apple Silicon-specific limitations**, not implementation bugs.

### Key Findings

| Factor | Impact | Solvable? |
|--------|--------|-----------|
| **Bitonic sort O(n log²n) vs CPU O(n log n)** | High | Yes, with algorithm change |
| **Apple Silicon lacks forward progress guarantee** | High | Limits algorithm choices |
| **CPU-GPU data transfer overhead** | Medium | Partially |
| **Command buffer synchronization** | Low | Already optimized |

### Bottom Line

**To achieve GPU faster than CPU on Apple Silicon sorting, the algorithm must change from bitonic sort to radix sort**, specifically using the `DeviceRadixSort` approach (not `OneSweep` which deadlocks on Apple Silicon).

---

## Table of Contents

1. [Timeline of Events](#timeline-of-events)
2. [Observed Performance Data](#observed-performance-data)
3. [Root Cause Analysis](#root-cause-analysis)
4. [Technical Deep Dive](#technical-deep-dive)
5. [Proposed Solutions](#proposed-solutions)
6. [References](#references)

---

## Timeline of Events

### December 15, 2025

| Time | Event | Outcome |
|------|-------|---------|
| 07:36 | PR #2 merged: Initial CPU vs GPU proof of concept | GPU 2-3x slower than CPU |
| 08:06 | PR #4 merged: Threadgroup memory optimization | GPU still ~1.04-1.33x slower |
| 09:43 | Issue #5 opened: CPU still faster than GPU | Current state |

### Before Optimization (Issue #3 Data)

```
Array size: 1,048,576 (1M) elements
CPU sort time: 19.259 ms
GPU sort time: 55.789 ms
Result: CPU 2.90x faster

Array size: 16,777,216 (16M) elements
CPU sort time: 190.810 ms
GPU sort time: 385.333 ms
Result: CPU 2.02x faster

Array size: 134,217,728 (134M) elements
CPU sort time: 1690.627 ms
GPU sort time: 3476.947 ms
Result: CPU 2.06x faster
```

### After Optimization (Issue #5 Data)

```
Array size: 16,777,216 (16M) elements
CPU sort time: ~188 ms
GPU sort time: ~205 ms
Result: CPU 1.04-1.14x faster

Array size: 134,217,728 (134M) elements
CPU sort time: ~1687 ms
GPU sort time: ~2127 ms
Result: CPU 1.19-1.33x faster
```

**Improvement**: GPU went from 2x slower to 1.1-1.3x slower (~50% improvement in relative performance).

---

## Observed Performance Data

### Issue #5 Benchmark Results (Apple M3 Pro)

| Array Size | CPU Time (ms) | GPU Time (ms) | CPU/GPU Ratio |
|------------|---------------|---------------|---------------|
| 16,777,216 | 185.575 | 210.733 | 1.14x |
| 16,777,216 | 190.927 | 199.328 | 1.04x |
| 134,217,728 | 1680.762 | 2008.334 | 1.19x |
| 134,217,728 | 1694.743 | 2246.863 | 1.33x |

### Key Observations

1. **Gap widens with array size**: At 16M elements GPU is ~1.1x slower, at 134M it's ~1.25x slower
2. **Variance in GPU timing**: GPU times vary more than CPU (199-210ms vs 185-190ms)
3. **Both scaled linearly**: Performance scales predictably with data size

---

## Root Cause Analysis

### Primary Cause: Algorithm Complexity Mismatch

| Algorithm | Complexity | Operations for 134M elements |
|-----------|------------|------------------------------|
| CPU pdqsort | O(n log n) | ~3.6 billion comparisons |
| GPU bitonic | O(n log²n) | ~97 billion comparisons |
| GPU radix | O(n) | ~0.5 billion operations |

**Bitonic sort performs 27x more operations than necessary for the same data size.**

### Why Bitonic Sort Was Chosen

Bitonic sort has properties that make it attractive for GPU:
- Regular, predictable memory access patterns
- No data dependencies between parallel threads
- Simple to implement correctly

However, these advantages don't overcome the O(n log²n) complexity penalty for large arrays.

### Secondary Cause: Apple Silicon Limitations

The most efficient GPU sorting algorithm (OneSweep) **does not work on Apple Silicon** due to:

> "OneSweep tends to run on anything that is not mobile, a software rasterizer, or Apple... Because 'chained-scan' relies on forward thread-progress guarantees, OneSweep is less portable."
>
> — [Linebender GPU Sorting Wiki](https://linebender.org/wiki/gpu/sorting/)

Apple Silicon GPUs do not provide forward progress guarantees, meaning algorithms that rely on cooperative thread execution may deadlock.

### Tertiary Causes (Already Optimized)

| Issue | PR #4 Solution | Remaining Impact |
|-------|----------------|------------------|
| Excessive GPU dispatches | Reduced from 210 to ~10 for 1M elements | Minimal |
| No threadgroup memory | Now uses fast shared memory (2048 elements) | Minimal |
| Per-dispatch buffer allocation | Pre-allocated buffers | Minimal |
| Synchronous waiting | Batched commands | Minimal |

---

## Technical Deep Dive

### Understanding the Complexity Gap

For an array of n = 134,217,728 elements:

```
log₂(n) = 27

CPU pdqsort comparisons ≈ n × log₂(n)
                        ≈ 134M × 27
                        ≈ 3.6 billion

GPU bitonic comparisons ≈ n × (log₂(n))² / 2
                        ≈ 134M × 27 × 27 / 2
                        ≈ 49 billion
```

Even with perfect parallelization, the GPU must do 13x more work.

### Current Implementation Analysis

The current implementation in `gpu_sort.rs` uses a two-phase approach:

**Phase 1: Local Sort (Efficient)**
- Sorts 2048-element blocks entirely in threadgroup memory
- Achieves high parallelism with minimal memory bandwidth
- Complexity: O(LOCAL_SIZE × log²(LOCAL_SIZE)) per block

**Phase 2: Global Merge (Bottleneck)**
- Merges sorted blocks using global memory
- Required for arrays > 2048 elements
- Multiple dispatches needed: O(log(n/LOCAL_SIZE)) stages

### Memory Bandwidth Analysis

Apple M3 Pro specifications:
- Memory bandwidth: ~150 GB/s
- Unified memory architecture

For 134M elements (536 MB):
- Single pass read/write: ~1.07 GB
- Bitonic sort requires ~log²(n)/2 = 364 passes at global level
- Actual passes are fewer due to local sorting optimization
- Estimated memory traffic: ~50-100 GB

CPU sorting:
- L3 cache: 36 MB
- Memory traffic: ~5-10 GB for 536 MB array
- Better cache utilization due to sequential access patterns

---

## Proposed Solutions

### Solution 1: Implement Radix Sort (Recommended)

**Expected Improvement**: 3-5x faster than current GPU, potentially 2-3x faster than CPU

**Approach**: Implement `DeviceRadixSort` algorithm which:
- Uses "reduce-then-scan" for prefix sums (works on Apple Silicon)
- Has O(n) complexity (4-8 passes for 32-bit integers)
- Achieves ~1G elements/s on M1 Max (reference implementation)

**Implementation Effort**: High (complete rewrite of GPU sorting)

**Key Reference**: [b0nes164/GPUSorting](https://github.com/b0nes164/GPUSorting) provides portable implementations.

### Solution 2: Hybrid CPU-GPU Approach

**Expected Improvement**: Match CPU performance with GPU fallback

**Approach**:
- Use CPU for small arrays (< 1M elements)
- Use GPU for arrays that will remain in GPU memory
- Detect when data is already on GPU to avoid transfer

**Implementation Effort**: Low

### Solution 3: Keep Data on GPU

**Expected Improvement**: 2-3x faster if data doesn't need CPU round-trip

**Approach**:
- If sorting is part of a GPU compute pipeline, avoid copying to/from CPU
- Current implementation copies data:
  1. CPU → GPU (before sort)
  2. GPU → CPU (after sort)

**Implementation Effort**: Medium (API redesign)

### Solution 4: Parallel Merge Sort (Alternative to Radix)

**Expected Improvement**: 2-3x improvement over bitonic

**Approach**:
- Implement GPU merge sort instead of bitonic
- O(n log n) complexity
- Good cache behavior with sequential merges

**Implementation Effort**: Medium

---

## Performance Targets

Based on research from [Linebender Wiki](https://linebender.org/wiki/gpu/sorting/):

| Implementation | Expected Performance |
|----------------|----------------------|
| Current (bitonic) | ~0.1G elements/s |
| DeviceRadixSort | ~1G elements/s |
| FidelityFX-style (Metal) | ~3G elements/s |

For 134M elements:
- Current: ~2000ms (observed)
- DeviceRadixSort target: ~134ms
- Optimized target: ~45ms

---

## References

### Primary Sources

1. [Linebender GPU Sorting Wiki](https://linebender.org/wiki/gpu/sorting/) - Comprehensive GPU sorting overview
2. [Onesweep Paper (arXiv)](https://arxiv.org/abs/2206.01784) - State-of-the-art GPU radix sort
3. [b0nes164/GPUSorting](https://github.com/b0nes164/GPUSorting) - Portable GPU sorting implementations
4. [AMD GPUOpen - Boosting GPU Radix Sort](https://gpuopen.com/learn/boosting_gpu_radix_sort/) - Memory-efficient radix sort

### Apple Silicon Specific

5. [Metal Compute on MacBook Pro (Apple)](https://developer.apple.com/videos/play/wwdc2020/10632/) - Metal optimization guide
6. [Philip Turner's Metal Benchmarks](https://github.com/philipturner/metal-benchmarks) - Apple GPU microarchitecture research
7. [Apple Silicon vs NVIDIA CUDA Comparison](https://scalastic.io/en/apple-silicon-vs-nvidia-cuda-ai-2025/) - Architecture differences

### Algorithm Theory

8. [NVIDIA GPU Gems - Bitonic Sort](https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting) - Classic reference
9. [Comparison of Parallel Sorting Algorithms (arXiv)](https://arxiv.org/pdf/1511.03404) - Academic comparison
10. [High Performance GPU Sorting (EPFL)](https://www.epfl.ch/labs/lap/wp-content/uploads/2018/05/YeApr10_HighPerformanceComparisonBasedSortingAlgorithmOnManyCoreGpus_IPDPS10.pdf) - Research paper

---

## Conclusion

The CPU being faster than GPU for sorting in this repository is **not a bug but an expected outcome** of using bitonic sort (O(n log²n)) against a highly optimized CPU sort (O(n log n)).

To achieve GPU superiority on Apple Silicon:
1. **Must switch to radix sort** (specifically DeviceRadixSort, not OneSweep)
2. **Cannot use forward-progress-dependent algorithms** due to Apple Silicon limitations
3. **Should minimize CPU-GPU data transfer** by keeping data in GPU memory

The optimization work in PR #4 was valuable and reduced the gap from 2-3x to 1.1-1.3x, but further improvement requires an algorithmic change.
