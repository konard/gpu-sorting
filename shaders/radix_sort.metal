#include <metal_stdlib>
using namespace metal;

// Radix sort configuration
// Process 8 bits per pass (256 buckets) - standard for GPU radix sort
#define RADIX_BITS 8
#define RADIX_SIZE (1 << RADIX_BITS)  // 256 buckets
#define RADIX_MASK (RADIX_SIZE - 1)   // 0xFF

// Threadgroup configuration
#define THREADGROUP_SIZE 256
#define KEYS_PER_THREAD 4
#define KEYS_PER_THREADGROUP (THREADGROUP_SIZE * KEYS_PER_THREAD)  // 1024

// ===========================================================================
// Pass 1: Histogram Kernel
// ===========================================================================
kernel void radix_histogram(
    device const uint *keys [[buffer(0)]],
    device uint *histogram [[buffer(1)]],
    constant uint &array_size [[buffer(2)]],
    constant uint &shift [[buffer(3)]],
    threadgroup uint *local_histogram [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Initialize local histogram to zero
    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        local_histogram[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint block_start = tgid * KEYS_PER_THREADGROUP;

    for (uint k = 0; k < KEYS_PER_THREAD; k++) {
        uint idx = block_start + tid + k * tg_size;
        if (idx < array_size) {
            uint key = keys[idx];
            uint digit = (key >> shift) & RADIX_MASK;
            atomic_fetch_add_explicit((threadgroup atomic_uint*)&local_histogram[digit],
                                      1, memory_order_relaxed);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        histogram[tgid * RADIX_SIZE + i] = local_histogram[i];
    }
}

// ===========================================================================
// Pass 2a: Reduce Kernel
// ===========================================================================
kernel void radix_reduce(
    device const uint *histogram [[buffer(0)]],
    device uint *global_histogram [[buffer(1)]],
    constant uint &num_threadgroups [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= RADIX_SIZE) return;

    uint sum = 0;
    for (uint tg = 0; tg < num_threadgroups; tg++) {
        sum += histogram[tg * RADIX_SIZE + gid];
    }
    global_histogram[gid] = sum;
}

// ===========================================================================
// Pass 2b: Exclusive Scan Kernel
// ===========================================================================
kernel void radix_scan(
    device uint *global_histogram [[buffer(0)]],
    threadgroup uint *local_data [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (tid < RADIX_SIZE) {
        local_data[tid] = global_histogram[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele inclusive scan
    for (uint offset = 1; offset < RADIX_SIZE; offset *= 2) {
        uint val = 0;
        if (tid >= offset && tid < RADIX_SIZE) {
            val = local_data[tid - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid >= offset && tid < RADIX_SIZE) {
            local_data[tid] += val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Convert to exclusive scan
    if (tid < RADIX_SIZE) {
        uint exclusive = (tid == 0) ? 0 : local_data[tid - 1];
        global_histogram[tid] = exclusive;
    }
}

// ===========================================================================
// Pass 2c: Scatter Offsets Kernel
// ===========================================================================
kernel void radix_scatter_offsets(
    device const uint *histogram [[buffer(0)]],
    device const uint *global_prefix [[buffer(1)]],
    device uint *scatter_offsets [[buffer(2)]],
    constant uint &num_threadgroups [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= RADIX_SIZE) return;

    uint global_offset = global_prefix[gid];
    uint running_sum = 0;

    for (uint tg = 0; tg < num_threadgroups; tg++) {
        scatter_offsets[tg * RADIX_SIZE + gid] = global_offset + running_sum;
        running_sum += histogram[tg * RADIX_SIZE + gid];
    }
}

// ===========================================================================
// Pass 3: Scatter Kernel (Parallel approach using local ranking)
// ===========================================================================
// This kernel processes keys in parallel within each threadgroup.
// All threads participate in loading, ranking, and scattering keys.
//
// Algorithm for each batch of 256 keys (one per thread):
// 1. Each thread loads its key and computes its digit
// 2. Store digits in shared memory for all threads to see
// 3. Each thread counts how many preceding threads have the same digit (rank)
// 4. Each thread writes its key to output at position: base_offset + rank
// 5. Update base offsets for next batch using digit counts
//
// The key optimization is that all threads compute their ranks simultaneously
// in parallel. The ranking loop is O(tid) per thread, averaging O(n/2) total
// work across all threads, which is much better than sequential O(n) per key.
kernel void radix_scatter(
    device const uint *keys_in [[buffer(0)]],
    device uint *keys_out [[buffer(1)]],
    device const uint *scatter_offsets [[buffer(2)]],
    constant uint &array_size [[buffer(3)]],
    constant uint &shift [[buffer(4)]],
    threadgroup uint *local_offsets [[threadgroup(0)]],  // RADIX_SIZE uints for bucket offsets
    threadgroup uint *shared_digits [[threadgroup(1)]],  // THREADGROUP_SIZE uints for digits
    threadgroup uint *digit_counts [[threadgroup(2)]],   // RADIX_SIZE uints for batch counts
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint block_start = tgid * KEYS_PER_THREADGROUP;
    uint block_end = min(block_start + KEYS_PER_THREADGROUP, array_size);
    uint block_size = block_end - block_start;

    // Load scatter offsets for this threadgroup into shared memory
    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        local_offsets[i] = scatter_offsets[tgid * RADIX_SIZE + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process keys in batches of tg_size (256) keys
    // Each batch is processed fully in parallel
    for (uint batch = 0; batch < KEYS_PER_THREAD; batch++) {
        uint local_idx = batch * tg_size + tid;
        bool valid = (local_idx < block_size);

        // Initialize digit counts for this batch
        for (uint d = tid; d < RADIX_SIZE; d += tg_size) {
            digit_counts[d] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 1: Each thread loads its key and computes digit
        uint key = 0;
        uint digit = RADIX_SIZE;  // Invalid marker for out-of-bounds threads
        if (valid) {
            uint global_idx = block_start + local_idx;
            key = keys_in[global_idx];
            digit = (key >> shift) & RADIX_MASK;
        }

        // Step 2: Store digit in shared memory for all threads to see
        shared_digits[tid] = digit;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: Compute rank - count threads with same digit that come before this thread
        // This is the key parallel operation - all threads compute their ranks simultaneously
        // Each thread only looks at threads with lower tid, ensuring stable sort order
        uint rank = 0;
        if (valid) {
            for (uint i = 0; i < tid; i++) {
                if (shared_digits[i] == digit) {
                    rank++;
                }
            }
        }

        // Step 4: Count total keys per digit in this batch using atomics
        if (valid) {
            atomic_fetch_add_explicit(
                (threadgroup atomic_uint*)&digit_counts[digit],
                1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: Write keys - each thread knows exactly where to write
        // Output position = base offset for digit + rank within this batch
        if (valid) {
            uint base = local_offsets[digit];
            uint out_idx = base + rank;
            keys_out[out_idx] = key;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 6: Update local offsets for next batch
        for (uint d = tid; d < RADIX_SIZE; d += tg_size) {
            local_offsets[d] += digit_counts[d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ===========================================================================
// Pass 3 (SIMD-Optimized): Scatter Kernel using SIMD group operations
// ===========================================================================
// This kernel uses Metal's SIMD group functions for O(1) rank computation
// within each SIMD group (32 threads on Apple Silicon).
//
// Key optimization: Uses simd_prefix_exclusive_sum for fast rank computation
// instead of the O(tid) loop in the basic scatter kernel.
//
// Algorithm:
// 1. Each thread loads key and computes digit
// 2. For each possible digit value (0-255):
//    - Use simd_ballot to find which threads in SIMD group have this digit
//    - Use simd_prefix_exclusive_sum to compute rank within SIMD group
//    - Combine ranks across SIMD groups within threadgroup
// 3. Write keys to computed positions
//
// Expected improvement: ~2-3x faster scatter compared to basic kernel
// due to O(1) SIMD operations instead of O(tid) loop.
kernel void radix_scatter_simd(
    device const uint *keys_in [[buffer(0)]],
    device uint *keys_out [[buffer(1)]],
    device const uint *scatter_offsets [[buffer(2)]],
    constant uint &array_size [[buffer(3)]],
    constant uint &shift [[buffer(4)]],
    threadgroup uint *local_offsets [[threadgroup(0)]],    // RADIX_SIZE uints for bucket offsets
    threadgroup uint *simd_digit_counts [[threadgroup(1)]], // RADIX_SIZE * num_simd_groups for counts
    threadgroup uint *digit_counts [[threadgroup(2)]],      // RADIX_SIZE uints for total batch counts
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    uint block_start = tgid * KEYS_PER_THREADGROUP;
    uint block_end = min(block_start + KEYS_PER_THREADGROUP, array_size);
    uint block_size = block_end - block_start;

    // Number of SIMD groups per threadgroup (typically 256/32 = 8 on Apple Silicon)
    uint num_simd_groups = tg_size / simd_size;

    // Load scatter offsets for this threadgroup into shared memory
    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        local_offsets[i] = scatter_offsets[tgid * RADIX_SIZE + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process keys in batches of tg_size (256) keys
    for (uint batch = 0; batch < KEYS_PER_THREAD; batch++) {
        uint local_idx = batch * tg_size + tid;
        bool valid = (local_idx < block_size);

        // Initialize digit counts for this batch
        for (uint d = tid; d < RADIX_SIZE; d += tg_size) {
            digit_counts[d] = 0;
        }
        // Initialize per-SIMD-group counts
        for (uint d = tid; d < RADIX_SIZE * num_simd_groups; d += tg_size) {
            simd_digit_counts[d] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 1: Load key and compute digit
        uint key = 0;
        uint digit = RADIX_SIZE;  // Invalid marker
        if (valid) {
            uint global_idx = block_start + local_idx;
            key = keys_in[global_idx];
            digit = (key >> shift) & RADIX_MASK;
        }

        // Step 2: Compute rank using SIMD operations
        // For each thread, compute how many threads before it (in this SIMD group)
        // have the same digit, plus prefix from earlier SIMD groups
        //
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
        // Each thread that is the FIRST occurrence of its digit in the SIMD group
        // writes the count. This ensures all digits present get their counts written.
        if (valid && simd_rank == 0) {
            simd_digit_counts[simd_group_id * RADIX_SIZE + digit] = simd_count;
        }

        // Wait for all SIMD groups to write their counts
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            // Compute prefix sum across SIMD groups for this digit
            uint prefix_from_earlier_simd = 0;
            for (uint sg = 0; sg < simd_group_id; sg++) {
                prefix_from_earlier_simd += simd_digit_counts[sg * RADIX_SIZE + digit];
            }

            rank = prefix_from_earlier_simd + simd_rank;
        }

        // Step 3: Accumulate total digit counts for this batch
        if (valid) {
            atomic_fetch_add_explicit(
                (threadgroup atomic_uint*)&digit_counts[digit],
                1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 4: Write key to output
        if (valid) {
            uint base = local_offsets[digit];
            uint out_idx = base + rank;
            keys_out[out_idx] = key;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: Update local offsets for next batch
        for (uint d = tid; d < RADIX_SIZE; d += tg_size) {
            local_offsets[d] += digit_counts[d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
