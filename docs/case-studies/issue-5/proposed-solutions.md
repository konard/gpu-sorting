# Proposed Solutions for Issue #5

## Summary of Solutions

| Solution | Effort | Expected Speedup | Risk |
|----------|--------|------------------|------|
| **1. DeviceRadixSort** | High | 10-20x | Medium |
| **2. Hybrid CPU-GPU** | Low | 1x (smart fallback) | Low |
| **3. Keep Data on GPU** | Medium | 1.5-2x | Low |
| **4. GPU Merge Sort** | Medium | 2-3x | Low |

---

## Solution 1: Implement DeviceRadixSort (Recommended)

### Description

Replace bitonic sort with a radix sort implementation that is compatible with Apple Silicon. The `DeviceRadixSort` algorithm from [b0nes164/GPUSorting](https://github.com/b0nes164/GPUSorting) uses "reduce-then-scan" for prefix sums, which works on Apple Silicon (unlike OneSweep's "chained-scan-with-decoupled-lookback").

### Why This Works

1. **O(n) complexity** vs O(n log²n) for bitonic
2. **Only 4-8 passes** through data for 32-bit integers
3. **Proven on Apple Silicon** - Linebender achieved ~1G elem/s on M1 Max

### Implementation Plan

```
Phase 1: Research (1-2 days)
├── Study b0nes164/GPUSorting D3D12 implementation
├── Study Linebender's WebGPU implementation
└── Understand FidelityFX 4-bit digit approach

Phase 2: Core Implementation (3-5 days)
├── Port histogram counting kernel to Metal
├── Port prefix scan kernel (reduce-then-scan approach)
├── Port scatter/gather kernel
└── Integrate with existing GpuSorter interface

Phase 3: Optimization (2-3 days)
├── Tune threadgroup sizes for Apple Silicon
├── Optimize memory access patterns
├── Add support for non-power-of-2 sizes
└── Benchmark across M1/M2/M3 variants

Phase 4: Testing (1-2 days)
├── Verify correctness against CPU sort
├── Test edge cases
└── Performance regression tests
```

### Expected Performance

| Array Size | Current (ms) | Target (ms) | Speedup |
|------------|--------------|-------------|---------|
| 16M | 200-210 | 16 | ~12x |
| 134M | 2000-2250 | 134 | ~15x |

### Risks

- Metal shader differences from HLSL/CUDA may require adjustments
- Apple Silicon's SIMD width (32) may differ from assumptions
- Memory bandwidth may become the bottleneck

### Code Structure (Proposed)

```rust
// src/radix_sort.rs

pub struct GpuRadixSorter {
    device: Device,
    command_queue: CommandQueue,
    histogram_pipeline: ComputePipelineState,
    scan_pipeline: ComputePipelineState,
    scatter_pipeline: ComputePipelineState,
}

impl GpuRadixSorter {
    pub fn sort(&self, data: &mut [u32]) -> Result<(), String> {
        // For each 4-bit digit (8 passes for 32-bit integers)
        for pass in 0..8 {
            self.compute_histogram(data, pass)?;
            self.prefix_scan()?;
            self.scatter_keys(data, pass)?;
        }
        Ok(())
    }
}
```

---

## Solution 2: Hybrid CPU-GPU Approach

### Description

Automatically choose between CPU and GPU based on array size and data location.

### Implementation

```rust
pub fn sort_optimal(data: &mut [u32]) -> Result<(), String> {
    const GPU_THRESHOLD: usize = 1_000_000; // 1M elements

    if data.len() < GPU_THRESHOLD {
        // CPU is faster for small arrays
        data.sort_unstable();
        Ok(())
    } else {
        // Try GPU, fall back to CPU on failure
        match GpuSorter::new() {
            Ok(sorter) => sorter.sort(data),
            Err(_) => {
                data.sort_unstable();
                Ok(())
            }
        }
    }
}
```

### Benefits

- Zero risk - always uses fastest available method
- Graceful degradation on non-Metal systems
- Can be combined with other solutions

---

## Solution 3: Keep Data on GPU

### Description

Redesign API to support sorting data that's already in GPU memory, avoiding CPU-GPU transfer overhead.

### Current Flow (Expensive)

```
CPU Array → [copy] → GPU Buffer → [sort] → GPU Buffer → [copy] → CPU Array
          ~0.5ms                ~2000ms               ~0.5ms
```

### Proposed Flow (Efficient)

```
GPU Buffer → [sort] → GPU Buffer
           ~2000ms

// Only copy when necessary
GPU Buffer → [copy if needed] → CPU Array
                    ~0.5ms
```

### API Design

```rust
pub struct GpuBuffer {
    buffer: metal::Buffer,
    size: usize,
}

impl GpuSorter {
    /// Sort data in CPU memory (current behavior)
    pub fn sort(&self, data: &mut [u32]) -> Result<(), String>;

    /// Sort data already in GPU buffer (new, efficient)
    pub fn sort_gpu_buffer(&self, buffer: &GpuBuffer) -> Result<(), String>;

    /// Create GPU buffer from CPU data
    pub fn create_buffer(&self, data: &[u32]) -> GpuBuffer;

    /// Read GPU buffer back to CPU
    pub fn read_buffer(&self, buffer: &GpuBuffer, dest: &mut [u32]);
}
```

### Use Case

Sorting as part of a GPU compute pipeline (e.g., particle systems, physics simulations) where data lives on GPU.

---

## Solution 4: GPU Merge Sort

### Description

Implement merge sort on GPU instead of bitonic sort. Merge sort has O(n log n) complexity and better cache behavior.

### Algorithm Overview

```
1. Sort small blocks (2048 elements) in threadgroup memory
2. Merge pairs of blocks using global memory
3. Repeat until single sorted array

Passes needed: log₂(n / block_size)
For 134M elements with 2048 block size: ~16 passes
```

### Comparison with Bitonic

| Metric | Bitonic | Merge Sort |
|--------|---------|------------|
| Complexity | O(n log²n) | O(n log n) |
| Memory reads/writes | ~n log²n | ~n log n |
| Implementation | Simpler | More complex |
| Cache efficiency | Poor (strided) | Good (sequential) |

### Expected Performance

| Array Size | Current Bitonic | Estimated Merge |
|------------|-----------------|-----------------|
| 134M | 2000ms | ~800-1000ms |

### Implementation Complexity

Medium - requires:
- Block-level sorting (similar to current local sort)
- Efficient merge kernel with good memory coalescing
- Double-buffering to avoid read/write conflicts

---

## Recommendation

### Short Term (Low effort, immediate benefit)

Implement **Solution 2: Hybrid CPU-GPU Approach**
- Provides optimal performance for all array sizes
- Zero risk, can be done in hours
- Good user experience

### Long Term (High effort, maximum performance)

Implement **Solution 1: DeviceRadixSort**
- Required to achieve GPU faster than CPU
- Substantial engineering effort
- Will provide 10-20x improvement

### Optional Enhancement

Implement **Solution 3: Keep Data on GPU**
- Useful if sorting is part of larger GPU pipeline
- Can be combined with any other solution

---

## Next Steps

1. [ ] Decide on priority: immediate hybrid vs long-term radix
2. [ ] If radix: study reference implementations
3. [ ] Create proof-of-concept Metal radix sort
4. [ ] Benchmark on multiple Apple Silicon chips (M1, M2, M3)
5. [ ] Optimize based on profiling data
