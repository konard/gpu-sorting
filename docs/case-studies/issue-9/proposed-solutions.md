# Proposed Solutions for Issue #9

## Summary of Solutions

| # | Solution | Effort | Expected Impact | Priority |
|---|----------|--------|-----------------|----------|
| 1 | **Parallel Scatter Kernel** | High | 10-50x faster scatter | Critical |
| 2 | **4-bit Radix Variant** | Medium | 1.5-2x faster | High |
| 3 | **Parallel CPU with rayon** | Low | 2-4x CPU speedup | High |
| 4 | **IPS4o Implementation** | High | State-of-art CPU sort | Medium |
| 5 | **Hybrid GPU-CPU Pipeline** | Medium | Best of both worlds | Medium |
| 6 | **Subgroup Operations** | Medium | ~3x improvement | High |

---

## Solution 1: Parallel Scatter Kernel (Critical Priority)

### Problem

The current scatter kernel in `radix_sort.metal` only uses thread 0 to write keys:

```metal
// Current: Only thread 0 works, 255 threads idle!
if (tid == 0) {
    uint key = keys_in[global_idx];
    uint digit = (key >> shift) & RADIX_MASK;
    uint out_idx = local_offsets[digit];
    local_offsets[digit] = out_idx + 1;
    keys_out[out_idx] = key;
}
```

This wastes ~99.6% of GPU compute resources.

### Solution

Implement **warp-level multi-split** for parallel ranking:

```metal
kernel void radix_scatter_parallel(
    device const uint *keys_in [[buffer(0)]],
    device uint *keys_out [[buffer(1)]],
    device atomic_uint *global_offsets [[buffer(2)]],
    constant uint &array_size [[buffer(3)]],
    constant uint &shift [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (gid >= array_size) return;

    // 1. Each thread loads its key
    uint key = keys_in[gid];
    uint digit = (key >> shift) & RADIX_MASK;

    // 2. Count matching digits in SIMD group using ballot
    // Metal provides simd_ballot for this
    simd_vote votes;
    for (uint d = 0; d < RADIX_SIZE; d++) {
        votes = simd_ballot(digit == d);
        if (digit == d) {
            // Count threads with same digit that come before me
            uint rank_in_simd = popcount(votes & ((1u << simd_lane) - 1));

            // Atomically get base offset for this digit
            uint base = atomic_fetch_add_explicit(
                &global_offsets[d], popcount(votes), memory_order_relaxed);

            // Write key to output
            keys_out[base + rank_in_simd] = key;
            break;
        }
    }
}
```

### Expected Impact

- **Current scatter efficiency**: ~1-2%
- **Expected scatter efficiency**: 50-70%
- **Overall speedup**: 10-50x faster scatter kernel
- **End-to-end improvement**: GPU Radix potentially 2-5x faster than CPU

### Implementation References

- [b0nes164/GPUSorting](https://github.com/b0nes164/GPUSorting) - See `OneSweep` and `DeviceRadixSort`
- [FidelityFX ParallelSort](https://github.com/GPUOpen-Effects/FidelityFX-ParallelSort)

---

## Solution 2: 4-bit Radix Variant (High Priority)

### Rationale

Current implementation uses 8-bit digits (256 buckets). FidelityFX uses 4-bit digits (16 buckets):

| Configuration | Buckets | Passes | Pros | Cons |
|---------------|---------|--------|------|------|
| 8-bit | 256 | 4 | Fewer passes | More atomic collisions |
| 4-bit | 16 | 8 | Simpler ranking | More passes |

### Why 4-bit May Be Better for Apple Silicon

1. **Less threadgroup memory**: 16 buckets vs 256
2. **Fewer atomic collisions**: 16 options vs 256
3. **Simpler warp-level ranking**: Can use simple parallel prefix
4. **Better SIMD utilization**: 16 < 32 (typical SIMD width)

### Implementation

```metal
#define RADIX_BITS 4
#define RADIX_SIZE 16
#define RADIX_MASK 0xF

// 8 passes for 32-bit integers
for (uint pass = 0; pass < 8; pass++) {
    uint shift = pass * RADIX_BITS;
    // ... histogram, scan, scatter
}
```

### Expected Impact

- **Trade-off**: 8 passes instead of 4
- **Each pass**: Potentially 2-3x faster due to simpler ranking
- **Net result**: 1.5-2x overall improvement possible

---

## Solution 3: Parallel CPU with rayon (High Priority)

### Rationale

Before further GPU optimization, establish a strong CPU baseline using parallel sorting.

### Implementation

```rust
// Add to Cargo.toml
// [dependencies]
// rayon = "1.10"

use rayon::prelude::*;

pub fn parallel_sort(data: &mut [u32]) {
    data.par_sort_unstable();
}
```

### Integration into Benchmark

```rust
// In main.rs
println!("\n--- CPU Sorting (Parallel Radix with rayon) ---");
let mut cpu_parallel_data = data.clone();
let cpu_parallel_start = Instant::now();
cpu_parallel_data.par_sort_unstable();
let cpu_parallel_duration = cpu_parallel_start.elapsed();
println!(
    "CPU parallel sort time: {:.3} ms",
    cpu_parallel_duration.as_secs_f64() * 1000.0
);
```

### Expected Impact

Apple M3 Pro has ~11 CPU cores (6 performance + 5 efficiency):

- **Current CPU Radix**: Single-threaded, 104.9M elements/sec
- **Expected parallel**: 200-400M elements/sec (2-4x speedup)
- **Result**: New "goal" for GPU to beat

### Alternative: Parallel Radix Sort (rdst crate)

```rust
// [dependencies]
// rdst = "0.20"

use rdst::RadixSort;

pub fn parallel_radix_sort(data: &mut [u32]) {
    data.radix_sort_unstable();
}
```

---

## Solution 4: IPS4o Implementation (Medium Priority)

### Rationale

IPS4o (In-place Parallel Super Scalar Samplesort) is the **fastest known parallel comparison-based sorting algorithm**.

From the [ACM paper](https://dl.acm.org/doi/10.1145/3505286):

> "IPS4o outperforms its closest in-place competitor by a factor of up to 3. Even as a sequential algorithm, it is up to 1.5 times faster than BlockQuicksort."

### Performance Claims

- Outperforms all competitors on any number of cores
- 28.71x faster than IS4o on 32 cores
- Even beats radix sort in many scenarios

### Implementation Options

1. **FFI to C++ implementation**: Use existing [ips4o library](https://github.com/ips4o/ips4o)
2. **Pure Rust port**: Higher effort, better integration
3. **Use existing Rust crate**: Check for Rust ports

### Expected Impact

- 2-3x faster than standard parallel sort
- May be faster than radix sort for certain distributions
- Provides theoretical upper bound for CPU performance

---

## Solution 5: Hybrid GPU-CPU Pipeline (Medium Priority)

### Concept

Use GPU for parallel-friendly operations, CPU for compute-intensive final sort:

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: GPU Partitioning                                   │
│ - Run single radix pass (256 buckets)                       │
│ - Scatter keys into 256 partitions                          │
│ - Output: 256 roughly-equal-sized buckets                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: CPU Parallel Sort                                  │
│ - Sort each bucket in parallel using rayon                  │
│ - 256 independent sorts, each ~1M elements for 268M total   │
│ - Use pdqsort or radix sort per bucket                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Concatenation                                      │
│ - Buckets are already in order (radix property)             │
│ - Just concatenate: bucket_0 ++ bucket_1 ++ ... ++ bucket_255│
└─────────────────────────────────────────────────────────────┘
```

### Expected Benefits

1. **GPU**: Handles parallel scatter efficiently
2. **CPU**: Handles comparison-based sort efficiently
3. **Both**: Work in parallel, potentially overlapping

### Implementation Sketch

```rust
pub fn hybrid_sort(data: &mut [u32]) {
    // Stage 1: GPU partitions data by first byte
    let buckets = gpu_partition_by_byte(data, 0); // Returns Vec<Vec<u32>>

    // Stage 2: CPU sorts each bucket in parallel
    buckets.par_iter_mut().for_each(|bucket| {
        bucket.sort_unstable();
    });

    // Stage 3: Concatenate (already in order)
    let mut offset = 0;
    for bucket in buckets {
        data[offset..offset + bucket.len()].copy_from_slice(&bucket);
        offset += bucket.len();
    }
}
```

### Expected Impact

- Leverages GPU's parallel scatter strength
- Leverages CPU's efficient cache-aware sorting
- 2-4x faster than pure GPU or pure CPU

---

## Solution 6: Subgroup Operations (High Priority)

### Rationale

The [Linebender wiki](https://linebender.org/wiki/gpu/sorting/) notes:

> "Experiments porting this to Metal using actual subgroup ballot operations result in ~3G el/s"

This suggests using Metal's SIMD group functions can provide ~3x improvement.

### Metal SIMD Group Functions

```metal
// Available in Metal Shading Language 2.1+

// Count matching values in SIMD group
simd_vote simd_ballot(bool vote);

// Prefix sum within SIMD group
T simd_prefix_exclusive_sum(T value);
T simd_prefix_inclusive_sum(T value);

// Broadcast value from specific lane
T simd_shuffle(T value, ushort lane);

// Reduce operations
T simd_sum(T value);
T simd_min(T value);
T simd_max(T value);
```

### Implementation for Ranking

```metal
// Efficient warp-level multi-split using subgroup ops
kernel void radix_scatter_simd(
    device const uint *keys_in [[buffer(0)]],
    device uint *keys_out [[buffer(1)]],
    device uint *offsets [[buffer(2)]],
    constant uint &shift [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    uint key = keys_in[gid];
    uint digit = (key >> shift) & 0xFF;

    // For each digit value, count how many threads have matching digit
    // and compute rank within matching threads
    for (uint d = 0; d < 256; d++) {
        bool match = (digit == d);
        simd_vote votes = simd_ballot(match);

        if (match) {
            // Count threads with same digit before me
            uint rank = simd_prefix_exclusive_sum(match ? 1u : 0u);
            uint count = simd_sum(match ? 1u : 0u);

            // Get global offset and write
            uint base_offset = /* ... */;
            keys_out[base_offset + rank] = key;
            break;
        }
    }
}
```

### Expected Impact

- **Current**: ~80M elements/sec
- **With subgroup ops**: ~240M elements/sec (3x improvement)
- **Result**: GPU would be ~2x faster than current CPU

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)

1. Add rayon parallel CPU sort for comparison
2. Benchmark parallel CPU as new baseline
3. Update case study with parallel CPU results

### Phase 2: GPU Optimization (1 week)

1. Implement subgroup-based scatter kernel
2. Test 4-bit radix variant
3. Benchmark and compare

### Phase 3: Advanced Optimization (2+ weeks)

1. Full parallel scatter with warp-level multi-split
2. Hybrid GPU-CPU pipeline
3. Consider IPS4o for CPU baseline

---

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| GPU Radix vs CPU Radix | 0.77x | 1.5x | 3x |
| GPU Radix throughput | 80M el/s | 250M el/s | 500M el/s |
| Memory efficiency | 1.7% | 30% | 60% |

---

## References

### GPU Sorting Implementations

1. [b0nes164/GPUSorting](https://github.com/b0nes164/GPUSorting) - Reference implementation
2. [FidelityFX ParallelSort](https://github.com/GPUOpen-Effects/FidelityFX-ParallelSort) - AMD's 4-bit radix
3. [Linebender GPU Sorting](https://linebender.org/wiki/gpu/sorting/) - Comprehensive analysis

### CPU Sorting

4. [rayon](https://docs.rs/rayon) - Rust parallel sort
5. [rdst](https://lib.rs/crates/rdst) - Rust parallel radix sort
6. [IPS4o](https://github.com/ips4o/ips4o) - Fastest parallel comparison sort

### Metal Optimization

7. [Metal Shading Language Spec](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) - SIMD group functions
8. [Metal Best Practices](https://developer.apple.com/documentation/metal/metal_best_practices_guide) - Apple optimization guide
