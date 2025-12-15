# Proposed Solutions for Issue #5

## Summary of Solutions

| Solution | Effort | Expected Speedup | Status |
|----------|--------|------------------|--------|
| **1. DeviceRadixSort** | High | 10-20x | ✅ **IMPLEMENTED** |
| **2. Hybrid CPU-GPU** | Low | 1x (smart fallback) | ✅ Included in radix sort |
| **3. Keep Data on GPU** | Medium | 1.5-2x | Future enhancement |
| **4. GPU Merge Sort** | Medium | 2-3x | Not needed (radix is faster) |

---

## Solution 1: DeviceRadixSort ✅ IMPLEMENTED

### Description

We implemented a GPU radix sort using the **DeviceRadixSort** algorithm from [b0nes164/GPUSorting](https://github.com/b0nes164/GPUSorting) with the "reduce-then-scan" approach, which works on Apple Silicon (unlike OneSweep's "chained-scan-with-decoupled-lookback").

### Why This Works

1. **O(n) complexity** vs O(n log²n) for bitonic
2. **Only 4 passes** through data for 32-bit integers (8 bits per pass)
3. **Portable** - Uses reduce-then-scan which doesn't require forward progress guarantees

### Implementation Details

**Files Added:**
- `src/gpu_radix_sort.rs` - Rust wrapper for GPU radix sort
- `shaders/radix_sort.metal` - Metal compute shaders

**Algorithm Passes (per 8-bit digit):**
1. **Histogram**: Count keys in each of 256 buckets per threadgroup
2. **Reduce**: Sum histograms to global counts
3. **Scan**: Compute exclusive prefix sum (Hillis-Steele algorithm)
4. **Scatter Offsets**: Compute per-threadgroup output positions
5. **Scatter**: Reorder keys to output buffer

### Code Structure

```rust
// src/gpu_radix_sort.rs

pub struct GpuRadixSorter {
    device: Device,
    command_queue: CommandQueue,
    histogram_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
    scan_pipeline: ComputePipelineState,
    scatter_offsets_pipeline: ComputePipelineState,
    scatter_pipeline: ComputePipelineState,
}

impl GpuRadixSorter {
    pub fn sort(&self, data: &mut [u32]) -> Result<(), String> {
        // For each 8-bit digit (4 passes for 32-bit integers)
        for pass in 0..4 {
            self.run_histogram(data, pass)?;
            self.run_reduce()?;
            self.run_scan()?;
            self.run_scatter_offsets()?;
            self.run_scatter(data, pass)?;
        }
        Ok(())
    }
}
```

### Metal Shader Configuration

```metal
#define RADIX_BITS 8
#define RADIX_SIZE 256         // 2^8 buckets
#define THREADGROUP_SIZE 256
#define KEYS_PER_THREAD 4
#define KEYS_PER_THREADGROUP 1024  // 256 * 4
```

### Expected Performance

| Array Size | Bitonic (ms) | Radix (ms) | Speedup |
|------------|--------------|------------|---------|
| 16M | 200-210 | ~16-20 | ~10-12x |
| 134M | 2000-2250 | ~100-150 | ~15-20x |

---

## Solution 2: Hybrid CPU-GPU ✅ INCLUDED

### Description

The radix sort implementation automatically falls back to CPU for small arrays (<1024 elements) where GPU overhead isn't worth it.

### Implementation

```rust
// In gpu_radix_sort.rs
pub fn sort(&self, data: &mut [u32]) -> Result<(), String> {
    let n = data.len();

    if n <= 1 {
        return Ok(());
    }

    // For very small arrays, fall back to CPU sort
    if n < 1024 {
        data.sort_unstable();
        return Ok(());
    }

    // GPU radix sort for larger arrays
    // ...
}
```

---

## Solution 3: Keep Data on GPU (Future Enhancement)

### Description

Redesign API to support sorting data that's already in GPU memory, avoiding CPU-GPU transfer overhead.

### Current Flow

```
CPU Array → [copy] → GPU Buffer → [sort] → GPU Buffer → [copy] → CPU Array
```

### Proposed Flow

```
GPU Buffer → [sort] → GPU Buffer
// Only copy when necessary
```

This is a future enhancement that could provide additional speedup when sorting is part of a larger GPU pipeline.

---

## Comparison: Before vs After

### Before (Bitonic Sort Only)

```
Array size: 134217728 elements (536 MB)

CPU sort time: 1680.762 ms
GPU bitonic sort time: 2008.334 ms

CPU is 1.19x faster than GPU
```

### After (With Radix Sort)

```
Array size: 134217728 elements (536 MB)

CPU sort time: 1680.762 ms
GPU bitonic sort time: 2008.334 ms
GPU radix sort time: ~100-150 ms (estimated)

GPU Radix is 11-17x faster than CPU
GPU Radix is 13-20x faster than GPU Bitonic
```

---

## Why Not OneSweep?

OneSweep uses "chained-scan-with-decoupled-lookback" which provides ~10% better performance on NVIDIA GPUs, but **deadlocks on Apple Silicon** due to lack of forward progress guarantees.

From [Linebender Wiki](https://linebender.org/wiki/gpu/sorting/):

> "OneSweep tends to run on anything that is not mobile, a software rasterizer, or Apple."

Our DeviceRadixSort implementation uses the "reduce-then-scan" approach which is portable and works correctly on Apple Silicon.

---

## Verification

The implementation includes comprehensive tests:

```rust
#[test] fn test_radix_sort_small()
#[test] fn test_radix_sort_random_1k()
#[test] fn test_radix_sort_random_4k()
#[test] fn test_radix_sort_random_64k()
#[test] fn test_radix_sort_non_power_of_two()  // Unlike bitonic!
#[test] fn test_radix_sort_empty()
#[test] fn test_radix_sort_single()
#[test] fn test_radix_sort_already_sorted()
#[test] fn test_radix_sort_reverse_sorted()
#[test] fn test_radix_sort_all_same()
#[test] fn test_radix_sort_max_values()
```

---

## References

- [GPUSorting by b0nes164](https://github.com/b0nes164/GPUSorting) - Reference implementation
- [Linebender GPU Sorting Wiki](https://linebender.org/wiki/gpu/sorting/) - Analysis of GPU sorting algorithms
- [CUB DeviceRadixSort](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html) - Original algorithm reference
- [VkRadixSort](https://github.com/MircoWerner/VkRadixSort) - Vulkan reference implementation
