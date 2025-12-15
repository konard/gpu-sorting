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
// Pass 3: Scatter Kernel (Sequential approach for correctness)
// ===========================================================================
// This kernel processes keys sequentially within each threadgroup to ensure
// stable sort. While not the most parallel, it guarantees correctness.
// Performance can be improved later with more sophisticated ranking algorithms.
kernel void radix_scatter(
    device const uint *keys_in [[buffer(0)]],
    device uint *keys_out [[buffer(1)]],
    device const uint *scatter_offsets [[buffer(2)]],
    constant uint &array_size [[buffer(3)]],
    constant uint &shift [[buffer(4)]],
    threadgroup uint *local_offsets [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint block_start = tgid * KEYS_PER_THREADGROUP;
    uint block_end = min(block_start + KEYS_PER_THREADGROUP, array_size);
    uint block_size = block_end - block_start;

    // Load scatter offsets for this threadgroup
    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        local_offsets[i] = scatter_offsets[tgid * RADIX_SIZE + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process keys in chunks - each chunk processes multiple keys in parallel
    // but respects input order by processing thread 0's keys first, then thread 1, etc.
    for (uint k = 0; k < KEYS_PER_THREAD; k++) {
        // Process all threads' k-th keys
        for (uint t = 0; t < tg_size; t++) {
            uint local_idx = k * tg_size + t;
            if (local_idx >= block_size) break;

            uint global_idx = block_start + local_idx;

            // Only the designated thread does the work
            if (tid == 0) {
                uint key = keys_in[global_idx];
                uint digit = (key >> shift) & RADIX_MASK;

                uint out_idx = local_offsets[digit];
                local_offsets[digit] = out_idx + 1;

                keys_out[out_idx] = key;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
