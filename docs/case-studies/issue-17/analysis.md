# Case Study: Issue #17 - GPU Radix Sort (SIMD) Verification Failure

## Executive Summary

This case study documents a critical bug in the SIMD-optimized GPU radix sort scatter kernel that causes verification failures. The bug involves **two interrelated issues**:

1. **Original Bug (PR #18 fix)**: Only lane 0 wrote digit counts, leaving counts for other digits as zero
2. **Deeper Root Cause (this fix)**: `simd_shuffle` operations were called inside divergent control flow (`if (valid)`), violating Metal's SIMD uniform control flow requirements

## Timeline of Events

### Previous Issues and Fixes

1. **Issue #11 / PR #12** (2025-12-17): Introduced SIMD-optimized scatter kernel (`radix_scatter_simd`)
2. **Issue #13 / PR #14** (2025-12-17): Fixed synchronization bug (removed incorrect `simdgroup_barrier`)
3. **Issue #15 / PR #16** (2025-12-17): Fixed variable scoping bug (moved `simd_rank` declaration)
4. **PR #18** (2025-12-17): Attempted fix by changing `if (simd_lane == 0)` to `if (simd_rank == 0)` - **DID NOT RESOLVE THE ISSUE**

### Current Issue (Issue #17)

- **Date**: 2025-12-17 to 2025-12-23
- **Environment**: Apple M3 Pro, macOS
- **Test command**: `cargo run --release -- 2684354`
- **Symptom**: GPU radix sort (SIMD) fails verification while basic version works correctly
- **Basic GPU Radix Sort**: 25.981 ms, verified OK
- **SIMD GPU Radix Sort**: 14.521 ms, **FAILED VERIFICATION**

## Root Cause Analysis

### The Real Bug: SIMD Uniform Control Flow Violation

The bug was not just about which thread writes the digit count. The fundamental issue is that **`simd_shuffle` operations were placed inside the `if (valid)` block**, causing non-uniform (divergent) control flow within the SIMD group.

#### Problematic Code (before fix):

```metal
if (valid) {
    // Only valid threads execute this - VIOLATION!
    for (uint lane = 0; lane < simd_lane; lane++) {
        uint other_digit = simd_shuffle(digit, lane);  // Not all threads call this!
        if (other_digit == digit) {
            simd_rank++;
        }
    }
    // ... more simd_shuffle operations inside if (valid) ...
}
```

### Why This Causes Undefined Behavior

According to [Apple Developer Forums](https://developer.apple.com/forums/thread/703337):

> "For correct behaviour all threads in SIMD group should execute these instructions. For Apple GPUs SIMD size is fixed and equal to 32. So in case of M1 all 32 threads must execute the same code path to produce correct result."

The [Apple G13 GPU Architecture Reference](https://dougallj.github.io/applegpu/docs.html) confirms:

> "SIMD shuffle instructions can read from inactive threads in some cases" - but behavior is undefined.

When threads in a SIMD group (32 threads on Apple Silicon) take different code paths, and some call `simd_shuffle` while others don't, the behavior is undefined. This manifests as:
- Incorrect values being shuffled
- Garbage data in computed ranks
- Keys written to wrong positions
- Verification failures

### Detailed Problem Analysis

Consider a batch where only lanes 0-25 are valid (processing elements exist), but lanes 26-31 are invalid:

1. **Valid threads (0-25)**: Enter the `if (valid)` block, call `simd_shuffle`
2. **Invalid threads (26-31)**: Skip the entire block, never call `simd_shuffle`
3. **Result**: The SIMD group has divergent control flow, causing `simd_shuffle` to return undefined values

### Why PR #18's Fix Didn't Work

PR #18 changed `if (simd_lane == 0)` to `if (simd_rank == 0)`. This addressed the symptom (only one digit's count being written) but not the root cause (the `simd_shuffle` calls were still inside the divergent `if (valid)` block).

## The Correct Fix

### Move `simd_shuffle` Outside Divergent Control Flow

The fix moves all `simd_shuffle` operations outside the `if (valid)` block so ALL threads in the SIMD group execute them together:

```metal
// Step 2: Compute rank using SIMD operations
// IMPORTANT: simd_shuffle must be called by ALL threads in the SIMD group
// for uniform control flow. The shuffle operations are placed outside the
// "if (valid)" block to ensure all 32 threads participate. Invalid threads
// have digit = RADIX_SIZE (256) which won't match any valid digit (0-255).
uint rank = 0;
uint simd_count = 0;
uint simd_rank = 0;

// All threads must execute simd_shuffle together (SIMD uniform control flow)
// Count threads with same digit that come before this thread in SIMD group
for (uint lane = 0; lane < simd_lane; lane++) {
    uint other_digit = simd_shuffle(digit, lane);
    if (valid && other_digit == digit) {
        simd_rank++;
    }
}

// Count total threads in this SIMD group with same digit
for (uint lane = 0; lane < simd_size; lane++) {
    uint other_digit = simd_shuffle(digit, lane);
    if (valid && other_digit == digit) {
        simd_count++;
    }
}

// Store this SIMD group's count for this digit
if (valid && simd_rank == 0) {
    simd_digit_counts[simd_group_id * RADIX_SIZE + digit] = simd_count;
}
```

### Key Changes

1. **Loops are outside `if (valid)`**: All 32 threads execute the shuffle loops
2. **Condition moved inside loop**: `if (valid && other_digit == digit)` only increments for valid threads
3. **Invalid threads don't affect counting**: Their digit is `RADIX_SIZE` (256), which won't match any valid digit (0-255)
4. **Write condition includes `valid`**: `if (valid && simd_rank == 0)` ensures only valid threads with rank 0 write

## Impact Assessment

### Before Fix
- **SIMD GPU Radix Sort**: Produces incorrect results due to undefined `simd_shuffle` behavior
- **Verification**: FAILS

### After Fix
- **SIMD GPU Radix Sort**: Should produce correct results with well-defined SIMD operations
- **Performance**: Should maintain ~1.79x speedup over basic kernel
- **Verification**: Should PASS

## Testing Strategy

1. **Unit Tests**: Run `cargo test` to verify all existing tests pass
2. **Integration Tests**: Run `cargo run --release -- 2684354` to replicate the original issue
3. **Edge Cases**: Test with array sizes that leave partial SIMD groups (e.g., 2684354 % 32 ≠ 0)
4. **Multiple Array Sizes**: Test with 1K, 1M, 10M elements
5. **Comparison**: Verify SIMD results match basic kernel results exactly

## Lessons Learned

### 1. SIMD Operations Require Uniform Control Flow
On GPU architectures, SIMD group operations like `simd_shuffle` must be executed by all threads in the group. Placing them inside conditional blocks causes undefined behavior.

### 2. Performance Gains Can Mask Bugs
The 1.79x speedup from the SIMD optimization suggested success, but the algorithm was producing incorrect results.

### 3. Incremental Fixes May Not Address Root Causes
PR #18 fixed a visible symptom (wrong write condition) but the deeper issue (divergent control flow) persisted.

### 4. Edge Cases Reveal Hidden Bugs
Array sizes that don't evenly divide by SIMD group size (32) expose divergent control flow bugs.

## References

- [Issue #17](https://github.com/konard/gpu-sorting/issues/17) - The bug report
- [PR #12](https://github.com/konard/gpu-sorting/pull/12) - Original SIMD implementation
- [PR #18](https://github.com/konard/gpu-sorting/pull/18) - Previous fix attempt (incomplete)
- [Apple Developer Forums - simdgroup issues](https://developer.apple.com/forums/thread/703337) - SIMD uniform control flow requirement
- [Apple G13 GPU Architecture Reference](https://dougallj.github.io/applegpu/docs.html) - Apple Silicon GPU internals
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) - Apple's Metal documentation
- [GPUSorting by b0nes164](https://github.com/b0nes164/GPUSorting) - Reference implementations

## Conclusion

The root cause was a **SIMD uniform control flow violation** where `simd_shuffle` operations were called inside a divergent `if (valid)` block. On Apple Silicon GPUs, all 32 threads in a SIMD group must execute SIMD group operations together.

The fix moves `simd_shuffle` calls outside the conditional block, ensuring all threads participate while still correctly computing ranks only for valid threads.

This bug illustrates the importance of:
1. **Understanding SIMD semantics** on target GPU architectures
2. **Ensuring uniform control flow** for SIMD group operations
3. **Testing edge cases** where array sizes don't align with SIMD group boundaries
