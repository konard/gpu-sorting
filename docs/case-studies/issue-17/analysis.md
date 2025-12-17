# Case Study: Issue #17 - GPU Radix Sort (SIMD) Verification Failure

## Executive Summary

This case study documents a critical algorithmic bug in the SIMD-optimized GPU radix sort scatter kernel that causes verification failures. The bug is a **fundamental algorithm error** where only one thread per SIMD group writes digit counts to shared memory, leaving counts for other digits as zero, resulting in incorrect rank computation and output positions.

## Timeline of Events

### Previous Issues and Fixes

1. **Issue #11 / PR #12** (2025-12-17): Introduced SIMD-optimized scatter kernel (`radix_scatter_simd`)
2. **Issue #13 / PR #14** (2025-12-17): Fixed synchronization bug (removed incorrect `simdgroup_barrier`)
3. **Issue #15 / PR #16** (2025-12-17): Fixed variable scoping bug (moved `simd_rank` declaration)

### Current Issue (Issue #17)

- **Date**: 2025-12-17
- **Environment**: Apple M3 Pro, macOS
- **Test command**: `cargo run --release -- 2684354`
- **Symptom**: GPU radix sort (SIMD) fails verification while basic version works correctly
- **Basic GPU Radix Sort**: 25.981 ms, verified OK
- **SIMD GPU Radix Sort**: 14.521 ms, **FAILED VERIFICATION**

## Root Cause Analysis

### The Bug Location

File: `shaders/radix_sort.metal`, lines 325-328 in the `radix_scatter_simd` kernel:

```metal
// Store this SIMD group's count for this digit
if (simd_lane == 0) {
    simd_digit_counts[simd_group_id * RADIX_SIZE + digit] = simd_count;
}
```

### Detailed Problem Analysis

**The Algorithm's Intent**: Track how many keys with each digit value exist in each SIMD group, so threads can compute their global rank within the threadgroup.

**What the Code Does**:
- Only `simd_lane == 0` writes to `simd_digit_counts`
- Lane 0 writes `simd_count` (count of keys with lane 0's digit) at index `simd_group_id * RADIX_SIZE + digit`
- **Critical flaw**: Only the count for lane 0's digit is written

**Why This Is Wrong**:

Consider a concrete example with SIMD group 0 (32 threads):

| Lane | 0 | 1 | 2 | 3 | 4 | 5 | ... |
|------|---|---|---|---|---|---|-----|
| Digit| 3 | 5 | 3 | 7 | 5 | 5 | ... |

After the SIMD shuffle operations:
- Lane 0 computes `simd_count = 2` (two threads have digit 3)
- Lane 1 computes `simd_count = 3` (three threads have digit 5)
- Lane 3 computes `simd_count = 1` (one thread has digit 7)

After the write (only lane 0 executes):
- `simd_digit_counts[0 * 256 + 3] = 2` (digit 3's count written)
- `simd_digit_counts[0 * 256 + 5] = 0` (digit 5's count NOT written, stays 0)
- `simd_digit_counts[0 * 256 + 7] = 0` (digit 7's count NOT written, stays 0)

**Consequence**: When threads compute their prefix sum:

```metal
for (uint sg = 0; sg < simd_group_id; sg++) {
    prefix_from_earlier_simd += simd_digit_counts[sg * RADIX_SIZE + digit];
}
```

Threads with digit 5 or 7 read zeros from earlier SIMD groups, causing incorrect rank values and therefore writing keys to wrong output positions.

### Why This Bug Wasn't Caught Earlier

1. **Previous fixes focused on other bugs**: PR #14 fixed a barrier issue, PR #16 fixed scoping
2. **No comprehensive algorithmic review**: The fundamental algorithm error was present from the start (PR #12)
3. **The SIMD optimization introduced new complexity**: Converting from threadgroup-wide shared memory to SIMD-local operations requires careful handling of per-digit counts
4. **Performance improvements masked the issue**: 1.79x speedup (from 25.981ms to 14.521ms) suggested the implementation was working

## Proposed Solution

### Option 1: Fix the SIMD Count Writing (Recommended)

Change the logic so that **every thread that is the first occurrence of its digit** in the SIMD group writes its count:

```metal
// Store this SIMD group's counts for all digits present
// Each thread that is the "first" for its digit writes the count
if (valid && simd_rank == 0) {  // First thread for this digit in SIMD group
    simd_digit_counts[simd_group_id * RADIX_SIZE + digit] = simd_count;
}
```

This ensures:
- Thread with (digit=3, simd_rank=0) writes count for digit 3
- Thread with (digit=5, simd_rank=0) writes count for digit 5
- Thread with (digit=7, simd_rank=0) writes count for digit 7

### Option 2: Fallback to Basic Scatter

Disable the SIMD scatter kernel entirely and use the working basic scatter kernel. This would sacrifice the 1.79x performance improvement.

**Recommendation**: Option 1 is the correct fix that maintains performance benefits.

## Impact Assessment

### Before Fix
- **SIMD GPU Radix Sort**: Produces incorrect results, fails verification
- **Basic GPU Radix Sort**: Works correctly

### After Fix
- **SIMD GPU Radix Sort**: Should produce correct results, pass verification
- **Performance**: Should maintain ~1.79x speedup over basic kernel

## Testing Strategy

1. **Unit Tests**: Run `cargo test` to verify all existing tests pass
2. **Integration Tests**: Run `cargo run --release -- 2684354` to replicate the original issue
3. **Multiple Array Sizes**: Test with 1K, 1M, 10M, 100M elements
4. **Comparison**: Verify SIMD results match basic kernel results exactly

## References

- [Issue #17](https://github.com/konard/gpu-sorting/issues/17) - The bug report
- [PR #12](https://github.com/konard/gpu-sorting/pull/12) - Original SIMD implementation
- [Linebender GPU Sorting Wiki](https://linebender.org/wiki/gpu/sorting/) - GPU sorting techniques
- [GPUSorting by b0nes164](https://github.com/b0nes164/GPUSorting) - Reference implementations
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) - Apple's Metal documentation

## Conclusion

The root cause is a fundamental algorithmic bug where only one thread per SIMD group writes digit counts, leaving counts for other digits as zero. The fix is straightforward: change `if (simd_lane == 0)` to `if (valid && simd_rank == 0)` so that the first thread for each digit writes its count.

This bug illustrates the importance of:
1. **Careful algorithmic design** when optimizing GPU kernels
2. **Testing with diverse data patterns** (different digits in same SIMD group)
3. **Code review focusing on correctness** before performance optimization
