#include <metal_stdlib>
using namespace metal;

// Radix sort configuration
// Process 8 bits per pass (256 buckets) - standard for GPU radix sort
#define RADIX_BITS 8
#define RADIX_SIZE (1 << RADIX_BITS)  // 256 buckets
#define RADIX_MASK (RADIX_SIZE - 1)   // 0xFF

// Threadgroup configuration
// Each threadgroup processes KEYS_PER_THREAD keys per thread
#define THREADGROUP_SIZE 256
#define KEYS_PER_THREAD 4
#define KEYS_PER_THREADGROUP (THREADGROUP_SIZE * KEYS_PER_THREAD)  // 1024

// ===========================================================================
// Pass 1: Histogram Kernel
// ===========================================================================
// Counts how many keys fall into each of the 256 buckets for each threadgroup.
// Output: histogram buffer of size [num_threadgroups * RADIX_SIZE]
//
// Each threadgroup computes a local histogram in shared memory,
// then writes it to the global histogram buffer.
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
    // Each thread clears multiple bins since RADIX_SIZE might be > threadgroup_size
    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        local_histogram[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread processes KEYS_PER_THREAD keys
    uint block_start = tgid * KEYS_PER_THREADGROUP;

    for (uint k = 0; k < KEYS_PER_THREAD; k++) {
        uint idx = block_start + tid + k * tg_size;
        if (idx < array_size) {
            uint key = keys[idx];
            uint digit = (key >> shift) & RADIX_MASK;
            // Use atomic add to avoid race conditions within threadgroup
            atomic_fetch_add_explicit((threadgroup atomic_uint*)&local_histogram[digit],
                                      1, memory_order_relaxed);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write local histogram to global memory
    // Format: histogram[tgid * RADIX_SIZE + digit] = count
    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        histogram[tgid * RADIX_SIZE + i] = local_histogram[i];
    }
}

// ===========================================================================
// Pass 2a: Reduce Kernel (part of reduce-then-scan)
// ===========================================================================
// Reduces per-threadgroup histograms into global digit counts.
// Input: histogram buffer [num_threadgroups * RADIX_SIZE]
// Output: global_histogram [RADIX_SIZE] containing total count per digit
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
// Computes exclusive prefix sum of global histogram.
// Input/Output: global_histogram [RADIX_SIZE]
// After: global_histogram[i] = sum of counts for digits 0..i-1
//
// This uses a simple single-threadgroup parallel scan since RADIX_SIZE=256
// fits entirely in shared memory.
kernel void radix_scan(
    device uint *global_histogram [[buffer(0)]],
    threadgroup uint *local_data [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]])
{
    // Load into shared memory
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

    // Convert to exclusive scan and write back
    if (tid < RADIX_SIZE) {
        uint inclusive = local_data[tid];
        uint exclusive = (tid == 0) ? 0 : local_data[tid - 1];
        // Store both: global_histogram[i] = exclusive prefix sum
        global_histogram[tid] = exclusive;
    }
}

// ===========================================================================
// Pass 2c: Scatter Offsets Kernel
// ===========================================================================
// Computes per-threadgroup scatter offsets using the scanned global histogram.
// For each threadgroup and digit, computes the starting offset in the output.
//
// Output: scatter_offsets[tgid * RADIX_SIZE + digit] =
//         global_offset[digit] + sum of histogram[0..tgid-1][digit]
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
// Pass 3: Scatter Kernel
// ===========================================================================
// Reorders keys based on computed scatter offsets.
// Each threadgroup reads its keys, computes local offsets, and writes to output.
kernel void radix_scatter(
    device const uint *keys_in [[buffer(0)]],
    device uint *keys_out [[buffer(1)]],
    device const uint *scatter_offsets [[buffer(2)]],
    constant uint &array_size [[buffer(3)]],
    constant uint &shift [[buffer(4)]],
    threadgroup uint *local_histogram [[threadgroup(0)]],
    threadgroup uint *local_offsets [[threadgroup(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint block_start = tgid * KEYS_PER_THREADGROUP;

    // Load scatter offsets for this threadgroup into shared memory
    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        local_offsets[i] = scatter_offsets[tgid * RADIX_SIZE + i];
    }

    // Initialize local histogram to zero (used for local offsets within threadgroup)
    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        local_histogram[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process keys one by one to maintain stable sort
    // Use a simpler approach: process in waves
    for (uint k = 0; k < KEYS_PER_THREAD; k++) {
        uint local_idx = tid + k * tg_size;
        uint global_idx = block_start + local_idx;

        if (global_idx < array_size) {
            uint key = keys_in[global_idx];
            uint digit = (key >> shift) & RADIX_MASK;

            // Get position within this digit's bucket for this threadgroup
            uint local_pos = atomic_fetch_add_explicit(
                (threadgroup atomic_uint*)&local_histogram[digit],
                1, memory_order_relaxed);

            // Compute global output position
            uint out_idx = local_offsets[digit] + local_pos;

            keys_out[out_idx] = key;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ===========================================================================
// Optimized Single-Pass Local Sort for Small Arrays
// ===========================================================================
// For arrays that fit in shared memory, we can do the entire sort locally.
// This is useful for very small arrays or the final pass of a multi-pass sort.
kernel void radix_sort_local(
    device uint *data [[buffer(0)]],
    constant uint &array_size [[buffer(1)]],
    threadgroup uint *local_keys [[threadgroup(0)]],
    threadgroup uint *local_counts [[threadgroup(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Load data into shared memory
    if (tid < array_size) {
        local_keys[tid] = data[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process each of 4 passes (8 bits each for 32-bit integers)
    for (uint pass = 0; pass < 4; pass++) {
        uint shift = pass * RADIX_BITS;

        // Clear counts
        for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
            local_counts[i] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Count digits
        if (tid < array_size) {
            uint digit = (local_keys[tid] >> shift) & RADIX_MASK;
            atomic_fetch_add_explicit((threadgroup atomic_uint*)&local_counts[digit],
                                      1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Exclusive prefix sum on counts (single thread for simplicity on small data)
        if (tid == 0) {
            uint sum = 0;
            for (uint i = 0; i < RADIX_SIZE; i++) {
                uint count = local_counts[i];
                local_counts[i] = sum;
                sum += count;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scatter (need temporary storage, reuse part of local_keys)
        // For stable sort, process in order
        if (tid < array_size) {
            uint key = local_keys[tid];
            uint digit = (key >> shift) & RADIX_MASK;
            uint pos = atomic_fetch_add_explicit((threadgroup atomic_uint*)&local_counts[digit],
                                                  1, memory_order_relaxed);
            // Write to second half of buffer temporarily
            local_keys[array_size + pos] = key;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Copy back
        if (tid < array_size) {
            local_keys[tid] = local_keys[array_size + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result back to global memory
    if (tid < array_size) {
        data[tid] = local_keys[tid];
    }
}
