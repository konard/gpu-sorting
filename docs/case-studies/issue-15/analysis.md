# Case Study: Issue #15 - Metal Shader Compilation Error in SIMD Radix Sort

## Executive Summary

This case study documents a critical bug in the Metal shader code for GPU radix sort that caused compilation failures. The bug was a **variable scoping error** introduced during a previous fix (PR #14), where an `if` block was incorrectly closed, leaving subsequent code with undefined variable references.

## Timeline of Events

### Previous Context (Issue #13 / PR #14)
- **Issue #13**: Reported a dead code warning and SIMD verification failure
- **PR #14**: Fixed the SIMD verification failure by removing an incorrect `simdgroup_barrier`
- However, PR #14 appears to have introduced a new bug through incomplete code restructuring

### Current Issue (Issue #15)
- **Date**: Reported immediately after PR #14 was merged
- **Environment**: Apple M3 Pro, macOS
- **Test command**: `cargo run --release -- 2684354`
- **Symptom**: Metal shader fails to compile with multiple "undeclared identifier" errors

## Error Analysis

### Compilation Errors Reported

```
program_source:308:18: warning: unused variable 'my_contribution' [-Wunused-variable]
program_source:346:47: error: use of undeclared identifier 'simd_rank'; did you mean 'simd_lane'?
program_source:350:13: error: use of undeclared identifier 'valid'
program_source:352:57: error: use of undeclared identifier 'digit'
program_source:358:13: error: use of undeclared identifier 'valid'
program_source:359:39: error: use of undeclared identifier 'digit'
program_source:360:35: error: use of undeclared identifier 'rank'
program_source:361:33: error: use of undeclared identifier 'key'
program_source:371:1: error: extraneous closing brace ('}')
```

### Root Cause: Incorrect Brace Structure

The bug was located in `shaders/radix_sort.metal` in the `radix_scatter_simd` kernel function.

**Buggy Code Structure (Before Fix):**
```metal
if (valid) {
    // ... code that declares simd_rank ...
    uint simd_rank = 0;
    for (uint lane = 0; lane < simd_lane; lane++) {
        // ... simd_rank computation ...
    }

    // ... more code ...

    if (simd_lane == 0) {
        simd_digit_counts[simd_group_id * RADIX_SIZE + digit] = simd_count;
    }
}  // <-- if (valid) block closed here prematurely

// Wait for all SIMD groups to write their counts
threadgroup_barrier(mem_flags::mem_threadgroup);

    // This code is OUTSIDE the if (valid) block but tries to use:
    // - simd_rank (declared inside if block, now out of scope)
    // - digit (only meaningful when valid)
    uint prefix_from_earlier_simd = 0;
    for (uint sg = 0; sg < simd_group_id; sg++) {
        prefix_from_earlier_simd += simd_digit_counts[sg * RADIX_SIZE + digit];
    }

    rank = prefix_from_earlier_simd + simd_rank;
}  // <-- Orphan closing brace - doesn't match any opening brace!
```

**Problems identified:**

1. **`simd_rank` out of scope**: Variable declared at line 314 inside the `if (valid)` block, but accessed at line 346 outside the block
2. **Orphan closing brace**: Line 347 had a `}` that didn't match any opening brace
3. **Invalid indentation**: Lines 340-346 were indented as if inside a block, but were actually at function scope

### The Fix

The fix involved two changes:

1. **Move `simd_rank` declaration outside the `if (valid)` block** so it remains in scope
2. **Wrap the prefix sum computation in a new `if (valid)` block** after the barrier

**Fixed Code Structure:**
```metal
uint rank = 0;
uint simd_count = 0;
uint simd_rank = 0;  // <-- Moved outside if (valid) block

if (valid) {
    // ... compute simd_rank inside the block ...
    for (uint lane = 0; lane < simd_lane; lane++) {
        uint other_digit = simd_shuffle(digit, lane);
        if (other_digit == digit) {
            simd_rank++;
        }
    }
    // ... rest of the valid block ...
}

// Wait for all SIMD groups to write their counts
threadgroup_barrier(mem_flags::mem_threadgroup);

if (valid) {  // <-- New if (valid) block for the code after barrier
    // Compute prefix sum across SIMD groups for this digit
    uint prefix_from_earlier_simd = 0;
    for (uint sg = 0; sg < simd_group_id; sg++) {
        prefix_from_earlier_simd += simd_digit_counts[sg * RADIX_SIZE + digit];
    }

    rank = prefix_from_earlier_simd + simd_rank;
}
```

### Additional Cleanup

The fix also removes unused code:
- Removed the unused `match` and `my_contribution` variables that were causing a warning

## Why This Bug Was Introduced

Based on the code structure, this bug likely occurred during the PR #14 fix for Issue #13. The original issue required removing an incorrect `simdgroup_barrier` call. During that edit:

1. The barrier removal may have been done hastily
2. The surrounding code structure was likely modified
3. The `if (valid)` block's closing brace may have been accidentally moved
4. The subsequent code was left "orphaned" with an extra closing brace

This is a common class of bugs when editing code with nested control structures, especially in complex GPU kernels with multiple synchronization points.

## Impact Assessment

### Before Fix
- **GPU Radix Sort (DeviceRadixSort)**: Failed to compile, unusable
- **GPU Radix Sort (SIMD Optimized)**: Failed to compile, unusable
- **Other algorithms**: Unaffected (bitonic sort, CPU sorts all work)

### After Fix
- All GPU radix sort implementations should compile and function correctly
- No performance impact expected (just fixing compilation errors)

## Testing Verification

### Local Tests Run
1. **`cargo fmt -- --check`**: Passed (Rust formatting)
2. **`cargo clippy -- -D warnings`**: Passed (Rust linting)
3. **`cargo test --verbose`**: All 47 tests passed

### CI Verification
The fix needs to be verified on macOS CI runners which have actual Metal GPU support to confirm the shader compiles and runs correctly.

## Lessons Learned

1. **GPU shader code requires careful review**: Metal/GLSL/HLSL don't have the same IDE support as application code, making structural bugs harder to catch
2. **Test on actual hardware**: Shader compilation errors only manifest at runtime on appropriate hardware
3. **Review brace structure carefully**: When editing code with nested blocks, verify brace matching manually
4. **Consider adding shader unit tests**: While challenging, automated shader validation could catch these errors earlier

## References

- [Issue #15](https://github.com/konard/gpu-sorting/issues/15) - The bug report
- [PR #14](https://github.com/konard/gpu-sorting/pull/14) - Previous fix that likely introduced this bug
- [Issue #13](https://github.com/konard/gpu-sorting/issues/13) - Original issue that PR #14 was fixing
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) - Apple's official Metal documentation

## Conclusion

This was a straightforward variable scoping bug caused by incorrect brace placement in a Metal shader kernel. The fix restores the correct code structure by:
1. Moving `simd_rank` declaration to the outer scope
2. Adding a proper `if (valid)` block after the threadgroup barrier
3. Removing the orphan closing brace

The fix maintains the original algorithm's logic while ensuring all variables are properly scoped and accessible where needed.
