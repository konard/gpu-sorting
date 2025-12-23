# Case Study: Issue #17 - GPU Radix Sort (SIMD) Verification Failure

## Executive Summary

This case study documents a critical bug in the SIMD-optimized GPU radix sort scatter kernel that causes verification failures. The bug involves **multiple interrelated issues** that were fixed incrementally:

1. **Original Bug (PR #18 fix)**: Only lane 0 wrote digit counts, leaving counts for other digits as zero
2. **Intermediate Fix (PR #19)**: Moved `simd_shuffle` outside `if (valid)` block - still failed
3. **Root Cause (this fix)**: **Variable loop bounds** (`lane < simd_lane`) caused non-uniform execution - different threads executed different numbers of shuffle operations

## Timeline of Events

### Previous Issues and Fixes

1. **Issue #11 / PR #12** (2025-12-17): Introduced SIMD-optimized scatter kernel (`radix_scatter_simd`)
2. **Issue #13 / PR #14** (2025-12-17): Fixed synchronization bug (removed incorrect `simdgroup_barrier`)
3. **Issue #15 / PR #16** (2025-12-17): Fixed variable scoping bug (moved `simd_rank` declaration)
4. **PR #18** (2025-12-17): Changed `if (simd_lane == 0)` to `if (simd_rank == 0)` - partial fix
5. **PR #19** (2025-12-17): Moved simd_shuffle outside `if (valid)` - still failed verification

### Current Issue (Issue #17)

- **Date**: 2025-12-17 to 2025-12-24
- **Environment**: Apple M3 Pro, macOS
- **Test command**: `cargo run --release -- 2684354`
- **Symptom**: GPU radix sort (SIMD) fails verification while basic version works correctly
- **Basic GPU Radix Sort**: 25.981 ms, verified OK
- **SIMD GPU Radix Sort**: 14.521 ms, **FAILED VERIFICATION**

## Root Cause Analysis

### The True Bug: Non-Uniform Loop Bounds for simd_shuffle

The previous fix (PR #19) correctly moved simd_shuffle outside the `if (valid)` block. However, **the loop bounds were still non-uniform across threads**:

#### Problematic Code (after PR #19, still failing):

```metal
// This loop has VARIABLE bounds - each thread loops a different number of times!
for (uint lane = 0; lane < simd_lane; lane++) {  // Thread 0: 0 iterations, Thread 31: 31 iterations
    uint other_digit = simd_shuffle(digit, lane);
    if (valid && other_digit == digit) {
        simd_rank++;
    }
}
```

The issue is subtle but critical:
- **Thread 0 (lane 0)**: Loops 0 times, never calls simd_shuffle
- **Thread 1 (lane 1)**: Loops 1 time, calls simd_shuffle(digit, 0)
- **Thread 31 (lane 31)**: Loops 31 times, calls simd_shuffle(digit, 0) through simd_shuffle(digit, 30)

When Thread 1 calls `simd_shuffle(digit, 0)`, Thread 0 is **NOT** calling simd_shuffle at that moment - it has already exited the loop. This violates SIMD uniform control flow.

### Why This Causes Undefined Behavior

According to [Apple Developer Forums](https://developer.apple.com/forums/thread/703337):

> "For correct behaviour all threads in SIMD group should execute these instructions. For Apple GPUs SIMD size is fixed and equal to 32. So in case of M1 all 32 threads must execute the same code path to produce correct result."

The [Apple G13 GPU Architecture Reference](https://dougallj.github.io/applegpu/docs.html) confirms:

> "SIMD shuffle instructions can read from inactive threads in some cases" - but behavior is undefined.

**Key insight**: SIMD uniform control flow requires not just that all threads call simd_shuffle, but that they call it **with the same lane parameter at the same time**. Variable loop bounds violate this.

### Detailed Problem Analysis

Consider what happens during execution:

| Iteration | Thread 0 | Thread 1 | Thread 31 | Problem |
|-----------|----------|----------|-----------|---------|
| lane=0    | EXIT     | simd_shuffle(digit, 0) | simd_shuffle(digit, 0) | Thread 0 not calling shuffle! |
| lane=1    | EXIT     | EXIT     | simd_shuffle(digit, 1) | Threads 0-1 not calling shuffle! |
| lane=30   | EXIT     | EXIT     | simd_shuffle(digit, 30) | Only Thread 31 calling shuffle! |

## The Correct Fix

### Uniform Loop Bounds with Conditional Logic Inside

The fix ensures ALL threads iterate over ALL lanes (0 to simd_size-1), but use conditional logic INSIDE the loop to count only relevant lanes:

```metal
// CORRECT: Uniform loop bounds - all threads execute same iterations
for (uint lane = 0; lane < simd_size; lane++) {  // All 32 threads loop 32 times
    uint other_digit = simd_shuffle(digit, lane);
    // Count for rank: only lanes BEFORE this thread
    if (valid && lane < simd_lane && other_digit == digit) {
        simd_rank++;
    }
    // Count for total: all lanes
    if (valid && other_digit == digit) {
        simd_count++;
    }
}
```

### Key Changes

1. **Uniform loop bounds**: `lane < simd_size` ensures ALL 32 threads execute the loop 32 times
2. **Conditional inside loop**: `lane < simd_lane` check moved INSIDE the loop body
3. **Single loop for both counts**: Combined simd_rank and simd_count into one loop for efficiency
4. **Invalid threads still excluded**: `valid &&` prefix ensures only valid threads count

### Execution After Fix

| Iteration | Thread 0 | Thread 1 | Thread 31 | Status |
|-----------|----------|----------|-----------|--------|
| lane=0    | simd_shuffle(digit, 0) | simd_shuffle(digit, 0) | simd_shuffle(digit, 0) | ✓ All threads synchronized |
| lane=1    | simd_shuffle(digit, 1) | simd_shuffle(digit, 1) | simd_shuffle(digit, 1) | ✓ All threads synchronized |
| ...       | ...      | ...      | ...       | ✓ |
| lane=31   | simd_shuffle(digit, 31) | simd_shuffle(digit, 31) | simd_shuffle(digit, 31) | ✓ All threads synchronized |

## Impact Assessment

### Before Fix
- **SIMD GPU Radix Sort**: Produces incorrect results due to non-uniform simd_shuffle execution
- **Verification**: FAILS

### After Fix
- **SIMD GPU Radix Sort**: Correct results with uniform SIMD operations
- **Performance**: Slight overhead from extra loop iterations (32 vs average 16), but still faster than basic kernel
- **Verification**: Should PASS

## Testing Strategy

1. **Unit Tests**: Run `cargo test` to verify all existing tests pass
2. **Integration Tests**: Run `cargo run --release -- 2684354` to replicate the original issue
3. **Edge Cases**: Test with array sizes that leave partial SIMD groups (e.g., 2684354 % 32 ≠ 0)
4. **Multiple Array Sizes**: Test with 1K, 1M, 10M elements
5. **Comparison**: Verify SIMD results match basic kernel results exactly

## Lessons Learned

### 1. SIMD Uniform Control Flow Requires Uniform LOOP BOUNDS
Moving simd_shuffle outside conditional blocks is necessary but not sufficient. Loop bounds must also be uniform across all threads in the SIMD group.

### 2. Variable Loop Bounds Are a Subtle Source of Divergence
`for (uint lane = 0; lane < simd_lane; lane++)` looks uniform because all threads enter the loop, but each thread has a different termination condition.

### 3. Move Conditions INSIDE Loops, Not Loop Bounds
Instead of varying the loop bounds, use uniform bounds and conditional logic inside:
- BAD: `for (lane = 0; lane < simd_lane; lane++) { shuffle(lane); }`
- GOOD: `for (lane = 0; lane < simd_size; lane++) { if (lane < simd_lane) count++; shuffle(lane); }`

### 4. GPU Debugging Requires Deep Architecture Understanding
This bug persisted through multiple fix attempts because it required understanding not just WHAT simd_shuffle does, but HOW it must be called (uniformly by all threads).

## References

- [Issue #17](https://github.com/konard/gpu-sorting/issues/17) - The bug report
- [PR #12](https://github.com/konard/gpu-sorting/pull/12) - Original SIMD implementation
- [PR #18](https://github.com/konard/gpu-sorting/pull/18) - Previous fix attempt (write condition)
- [PR #19](https://github.com/konard/gpu-sorting/pull/19) - Previous fix attempt (moved shuffle outside if)
- [Apple Developer Forums - simdgroup issues](https://developer.apple.com/forums/thread/703337) - SIMD uniform control flow requirement
- [Apple G13 GPU Architecture Reference](https://dougallj.github.io/applegpu/docs.html) - Apple Silicon GPU internals
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) - Apple's Metal documentation
- [GPUSorting by b0nes164](https://github.com/b0nes164/GPUSorting) - Reference implementations
- [Linebender GPU Sorting Wiki](https://linebender.org/wiki/gpu/sorting/) - GPU sorting techniques

## Conclusion

The root cause was **non-uniform loop bounds** in the simd_shuffle loop. The loop `for (lane = 0; lane < simd_lane; lane++)` caused each thread to execute a different number of shuffle operations, violating SIMD uniform control flow requirements.

The fix changes the loop to use uniform bounds (`lane < simd_size`) so all 32 threads execute the same number of shuffle operations with the same lane parameters. The conditional logic for counting ranks is moved inside the loop body.

This bug illustrates the importance of:
1. **Understanding that loop bounds must be uniform** for SIMD operations
2. **Moving varying conditions inside loops** rather than using them as loop bounds
3. **Careful analysis of all sources of control flow divergence**, not just explicit conditionals
