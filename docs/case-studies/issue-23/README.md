# Case Study: GPU Sorting Performance Analysis and Optimization Opportunities (Issue #23)

## Executive Summary

This case study investigates why GPU sorting on Apple M3 Pro struggles to beat optimized CPU sorting, analyzes the current implementation, and proposes concrete optimizations. The key finding is that **the GPU can beat the CPU for certain algorithms (bitonic sort) at large array sizes, but the GPU radix sort implementation has specific bottlenecks that prevent it from achieving optimal performance**.

### Key Findings

| Finding | Current Status | Potential Improvement |
|---------|---------------|----------------------|
| **GPU Radix Sort is 1.28-650x slower than CPU** | Bottleneck in scatter kernel | 2-10x improvement possible |
| **GPU Bitonic Sort beats CPU at 65K+ elements** | Already optimized | Working as expected |
| **CPU Radix Sort is fastest overall** | N/A | Cannot be GPU-parallelized differently |
| **M3 Pro has 25% less memory bandwidth than M2 Pro** | Hardware limitation | Use more compute, less memory ops |

### Bottom Line

**GPU sorting CAN beat CPU sorting on Apple Silicon**, but only when:
1. Using algorithms that benefit from massive parallelism (bitonic sort at large arrays)
2. Data already resides on GPU (no transfer overhead)
3. Optimal implementations that minimize memory bandwidth bottlenecks

The current GPU radix sort implementation needs optimization to achieve competitive performance.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Benchmark Data Analysis](#benchmark-data-analysis)
3. [Root Cause Analysis](#root-cause-analysis)
4. [Technical Deep Dive](#technical-deep-dive)
5. [Proposed Optimizations](#proposed-optimizations)
6. [Alternative Algorithms](#alternative-algorithms)
7. [Conclusions and Recommendations](#conclusions-and-recommendations)
8. [References](#references)

---

## Problem Statement

The original issue asks:
> "Can we do something better with implementation or find an algorithm that will actually beat CPU sorting using GPU sorting?"

Current observations from Apple M3 Pro:
- **GPU Radix Sort**: 1.28x-650x slower than CPU radix sort depending on array size
- **GPU Bitonic Sort**: Beats CPU at large arrays (23.79x faster at 16M elements)
- **GPU Radix Sort (SIMD)**: Actually slower than basic version (1.22x)

---

## Benchmark Data Analysis

### Performance Crossover Points

| Array Size | GPU Radix vs CPU Radix | GPU Bitonic vs CPU Bitonic | Winner |
|------------|------------------------|---------------------------|--------|
| 1,024 | 650x slower | 6.19x slower | CPU all |
| 4,096 | 215x slower | 2.60x slower | CPU all |
| 16,384 | 62x slower | 1.37x slower | CPU all |
| **65,536** | 14x slower | **1.90x faster** | **GPU Bitonic wins** |
| 262,144 | 5.4x slower | **3.48x faster** | **GPU Bitonic wins** |
| 1,048,576 | 4.1x slower | **6.79x faster** | **GPU Bitonic wins** |
| 4,194,304 | 1.74x slower | **15.72x faster** | **GPU Bitonic wins** |
| 16,777,216 | 1.28x slower | **23.79x faster** | **GPU Bitonic wins** |

### Key Observations

1. **GPU Bitonic Sort Works Well**: The O(n log²n) algorithm actually benefits from GPU parallelism at scale
2. **GPU Radix Sort Has Issues**: Never beats CPU, even though radix sort is theoretically ideal for GPU
3. **Small Array Overhead**: Both GPU algorithms suffer from significant overhead below ~64K elements
4. **SIMD Optimization Backfired**: The SIMD-optimized scatter kernel is 22% slower than basic version

---

## Root Cause Analysis

### Why GPU Radix Sort Underperforms

After analyzing the implementation in `src/gpu_radix_sort.rs` and `shaders/radix_sort.metal`, several bottlenecks were identified:

#### 1. Excessive Synchronization (Primary Bottleneck)

```rust
// Current implementation: 5 separate command buffers per pass × 4 passes = 20 GPU submissions
for pass in 0..4u32 {
    // Pass 1: Histogram - commit and wait
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Pass 2a: Reduce - commit and wait
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // ... continues for all 5 kernels per pass
}
```

**Impact**: Each `commit()` + `wait_until_completed()` incurs ~10-100μs of overhead. With 20 submissions, this adds 0.2-2ms of pure overhead, which dominates for small arrays.

**Solution**: Batch multiple kernels into single command buffers using `MTLBlitCommandEncoder` for barriers instead of CPU-side waits.

#### 2. Suboptimal Scatter Kernel (Secondary Bottleneck)

The scatter kernel has O(n²) behavior within each threadgroup due to the rank computation:

```metal
// Current implementation: O(n) per thread for rank computation
for (uint i = 0; i < tid; i++) {
    if (shared_digits[i] == digit) {
        rank++;
    }
}
```

For 256 threads per threadgroup, this performs:
- Thread 0: 0 iterations
- Thread 255: 255 iterations
- Average: 127.5 iterations per thread × 256 threads = 32,640 iterations per batch

**Solution**: Use parallel prefix sum (scan) to compute ranks in O(log n) steps.

#### 3. Memory Bandwidth Limitations

Apple M3 Pro has only **150 GB/s memory bandwidth** (25% less than M2 Pro's 200 GB/s). Radix sort is memory-bound:

For 16M elements (64 MB):
- 4 passes × 2 read/writes each = 8 memory passes
- 64 MB × 8 = 512 MB transferred
- Theoretical minimum time: 512 MB / 150 GB/s = **3.4ms**
- Actual GPU time: **105ms** → Only **3.2% memory efficiency**

This suggests the bottleneck is NOT memory bandwidth but computation/synchronization overhead.

#### 4. Why SIMD Version is Slower

The SIMD scatter kernel uses `simd_shuffle()` which should be faster, but the implementation has issues:

```metal
// Current SIMD implementation still iterates over all lanes
for (uint lane = 0; lane < simd_size; lane++) {
    uint other_digit = simd_shuffle(digit, lane);
    if (valid && lane < simd_lane && other_digit == digit) {
        rank++;
    }
}
```

This is still O(simd_size) per thread. True SIMD optimization would use `simd_prefix_exclusive_sum` or ballot operations.

---

## Technical Deep Dive

### Comparison with State-of-the-Art

| Implementation | Expected Throughput | Technique |
|----------------|-------------------|-----------|
| Current GPU Radix | ~160M elements/s | DeviceRadixSort (basic) |
| FidelityFX Sort | ~1-3G elements/s | 4-bit digits, tree reduction |
| Onesweep | ~1-2G elements/s | Chained scan (deadlocks on Apple) |
| Hybrid Radix | ~1-2G elements/s | MSD + bandwidth optimization |
| Linebender/Raph (M1 Max) | ~1G elements/s | FidelityFX-style |
| Linebender/Raph (Metal optimized) | ~3G elements/s | Actual subgroups |

Our current implementation achieves only **~16% of the expected throughput** for DeviceRadixSort.

### Apple Silicon Constraints

1. **No Forward Progress Guarantee**: Prevents use of OneSweep algorithm
2. **Reduced Memory Bandwidth (M3 Pro)**: 150 GB/s vs 200 GB/s on M2 Pro
3. **Limited Subgroup Operations**: Not all SIMD operations available in WebGPU/Metal
4. **Dynamic Caching Unpredictability**: New M3 feature may affect threadgroup memory

### Why CPU Radix Sort is So Fast

The CPU implementation benefits from:
1. **Better cache locality**: L1/L2 caches (192KB/24MB on M3 Pro) for small arrays
2. **No synchronization overhead**: Single thread, no barriers needed
3. **Branch prediction**: CPU can speculate effectively
4. **SIMD vectorization**: Rust's auto-vectorization for counting operations

---

## Proposed Optimizations

### Optimization 1: Batch Command Buffers (High Impact)

Instead of 20 separate GPU submissions, batch all kernels into a single command buffer per pass (4 total) or even a single command buffer for all passes.

```rust
// Proposed: Single command buffer with proper barriers
let command_buffer = self.command_queue.new_command_buffer();

for pass in 0..4u32 {
    // Histogram kernel
    let encoder = command_buffer.new_compute_command_encoder();
    // ... dispatch histogram
    encoder.end_encoding();

    // Barrier between kernels (instead of commit+wait)
    let blit_encoder = command_buffer.new_blit_command_encoder();
    blit_encoder.end_encoding();  // implicit barrier

    // Reduce kernel
    let encoder = command_buffer.new_compute_command_encoder();
    // ... dispatch reduce
    encoder.end_encoding();

    // ... continue for all kernels
}

// Single commit at the end
command_buffer.commit();
command_buffer.wait_until_completed();
```

**Expected Improvement**: 2-5x for small-to-medium arrays by eliminating 16 unnecessary round-trips.

### Optimization 2: Parallel Rank Computation (High Impact)

Replace the O(n) rank loop with parallel prefix sum using threadgroup-level operations:

```metal
// Proposed: Use atomic operations for O(1) amortized rank
// Step 1: Count digits per bucket atomically
atomic_fetch_add_explicit(&digit_counts[digit], 1, memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);

// Step 2: Compute prefix sums
// Use Kogge-Stone or Blelloch scan algorithm
for (uint offset = 1; offset < RADIX_SIZE; offset *= 2) {
    if (tid >= offset) {
        temp = prefix[tid - offset];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= offset) {
        prefix[tid] += temp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Step 3: Compute rank using prefix + atomic increment
uint rank = atomic_fetch_add_explicit(&local_rank[digit], 1, memory_order_relaxed);
```

**Expected Improvement**: 3-10x for the scatter phase.

### Optimization 3: Use 4-bit Digits (Medium Impact)

Switch from 8-bit (256 buckets) to 4-bit (16 buckets) like FidelityFX:

| Configuration | Passes | Buckets | Histogram Memory |
|---------------|--------|---------|-----------------|
| 8-bit (current) | 4 | 256 | 1KB per threadgroup |
| 4-bit (proposed) | 8 | 16 | 64B per threadgroup |

While 4-bit requires twice as many passes, each pass is much simpler:
- Smaller histograms fit better in registers
- Fewer bank conflicts in threadgroup memory
- Simpler prefix sum (16 elements vs 256)

**Expected Improvement**: 1.5-2x due to better hardware utilization.

### Optimization 4: Increase Keys Per Thread (Medium Impact)

Currently processing 4 keys per thread (1024 per threadgroup). Increasing to 8-16 keys per thread improves arithmetic intensity:

```rust
const KEYS_PER_THREAD: usize = 8;  // was 4
const KEYS_PER_THREADGROUP: usize = THREADGROUP_SIZE * KEYS_PER_THREAD;  // 2048
```

**Expected Improvement**: 1.3-1.5x by better hiding memory latency.

### Optimization 5: Coalesced Memory Access (Medium Impact)

Ensure memory accesses are coalesced (sequential threads access sequential memory):

```metal
// Current: Stride by threadgroup size
uint idx = block_start + tid + k * tg_size;  // Can cause bank conflicts

// Proposed: Sequential access within warps
uint idx = block_start + k * tg_size + tid;  // Better coalescing
```

**Expected Improvement**: 1.2-1.5x for memory-bound phases.

---

## Alternative Algorithms

### 1. FidelityFX-Style Sort (Recommended)

Implement the AMD FidelityFX approach adapted for Metal:
- 4-bit digits with tree reduction for histograms
- Two 2-bit LSD passes for local ranking
- Better occupancy due to smaller register usage

### 2. Merge Sort (Alternative)

For segmented sorting or when stability is required:
- O(n log n) complexity
- Better cache behavior
- Simpler implementation

### 3. Hybrid CPU-GPU Approach

Use CPU for small arrays, GPU for large:
```rust
fn sort(&self, data: &mut [u32]) {
    if data.len() < 100_000 {
        // CPU radix sort for small arrays
        cpu_radix_sort::sort(data);
    } else {
        // GPU sort for large arrays
        self.gpu_sort_internal(data)?;
    }
}
```

### 4. Keep Data on GPU

If sorting is part of a GPU compute pipeline, avoid CPU round-trip:
```rust
fn sort_buffer(&self, buffer: &Buffer) -> Result<Buffer, String> {
    // Sort directly on GPU memory
    // No CPU-GPU data transfer
}
```

---

## Conclusions and Recommendations

### Is GPU Sorting "Doomed" on Apple Silicon?

**No, but it requires the right approach:**

1. **GPU Bitonic Sort already wins** at 65K+ elements (up to 24x faster than CPU)
2. **GPU Radix Sort CAN be competitive** with proper optimizations
3. **Apple Silicon is uniquely positioned** with unified memory (zero-copy)

### Recommended Actions (Priority Order)

1. **[High] Batch command buffers** to reduce synchronization overhead
2. **[High] Fix scatter kernel** with parallel prefix sum for rank computation
3. **[Medium] Add hybrid threshold** to use CPU for small arrays (<100K)
4. **[Medium] Try 4-bit digits** following FidelityFX approach
5. **[Low] Increase keys per thread** for better latency hiding
6. **[Low] Profile with Metal System Trace** to identify actual bottlenecks

### Expected Results After Optimization

| Scenario | Current | After Optimization |
|----------|---------|-------------------|
| 1M elements | 18.5 ms | ~3-5 ms (target) |
| 16M elements | 105 ms | ~20-40 ms (target) |
| Comparison | 4x slower than CPU | 0.5-1x vs CPU (competitive) |

### Final Answer to Issue Question

> "Is GPU doomed at sorting at all?"

**No.** The GPU bitonic sort already demonstrates that GPU CAN beat CPU significantly (24x at 16M elements). The GPU radix sort implementation needs optimization, but the fundamental approach is sound. With the proposed optimizations, GPU radix sort should be able to compete with or beat CPU radix sort for large arrays.

---

## References

### Academic Papers & Research

1. [Onesweep: A Faster Least Significant Digit Radix Sort for GPUs](https://arxiv.org/abs/2206.01784) - State-of-the-art GPU radix sort
2. [A Memory Bandwidth-Efficient Hybrid Radix Sort on GPUs](https://arxiv.org/pdf/1611.01137) - Memory-efficient approaches
3. [Fast Sort on CPUs and GPUs: A Case for Bandwidth Oblivious SIMD Sort](https://www.researchgate.net/publication/221213255_Fast_sort_on_CPUs_and_GPUs_a_case_for_bandwidth_oblivious_SIMD_sort) - CPU vs GPU analysis
4. [Study on Sorting Performance for Reactor Monte Carlo on Apple GPUs](https://arxiv.org/html/2401.11455) - Apple Silicon specific research

### Industry Resources

5. [Linebender GPU Sorting Wiki](https://linebender.org/wiki/gpu/sorting/) - Comprehensive GPU sorting overview
6. [AMD GPUOpen: Boosting GPU Radix Sort](https://gpuopen.com/learn/boosting_gpu_radix_sort/) - Memory-efficient extensions
7. [AMD FidelityFX Parallel Sort](https://gpuopen.com/fidelityfx-parallel-sort/) - Production-ready implementation
8. [Introduction to GPU Radix Sort (AMD)](https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf) - Fundamentals

### Apple Silicon Specific

9. [Apple M3 Pro Specifications](https://support.apple.com/en-us/117736) - Hardware specs
10. [Apple M3 Pro Memory Bandwidth Analysis](https://www.macrumors.com/2023/10/31/apple-m3-pro-less-memory-bandwidth/) - 25% bandwidth reduction
11. [b0nes164/GPUSorting](https://github.com/b0nes164/GPUSorting) - Portable GPU sorting implementations

### Previous Case Studies

12. [Issue #5 Case Study: Why CPU is Faster Than GPU](../issue-5/README.md) - Previous analysis in this repo

---

## Appendix: Benchmark Data

The complete benchmark data from issue #23 is available in [benchmark_data_m3_pro.txt](./benchmark_data_m3_pro.txt).
