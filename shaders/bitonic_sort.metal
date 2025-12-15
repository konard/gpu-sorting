#include <metal_stdlib>
using namespace metal;

/// Bitonic sort step kernel.
///
/// This kernel performs one step of bitonic sort. The algorithm works by:
/// 1. Comparing pairs of elements at specific distances
/// 2. Swapping elements based on the current sorting direction
///
/// Parameters:
/// - data: The array to sort (in-place)
/// - block_size: Current block size in the bitonic sequence
/// - sub_block_size: Current comparison distance
/// - gid: Thread position in the grid
kernel void bitonic_sort_step(
    device uint *data [[buffer(0)]],
    constant uint &block_size [[buffer(1)]],
    constant uint &sub_block_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    // Calculate the index pair to compare
    uint index = gid;

    // Determine which half of the sub-block this thread is in
    uint half_sub_block = sub_block_size / 2;
    uint sub_block_index = index / half_sub_block;
    uint index_in_sub_block = index % half_sub_block;

    // Calculate the two indices to compare
    uint left_index = sub_block_index * sub_block_size + index_in_sub_block;
    uint right_index = left_index + half_sub_block;

    // Determine sort direction (ascending or descending based on block position)
    // Each block alternates direction in bitonic merge
    uint block_index = left_index / block_size;
    bool ascending = (block_index % 2) == 0;

    // Load values
    uint left_val = data[left_index];
    uint right_val = data[right_index];

    // Compare and swap if needed
    bool should_swap = ascending ? (left_val > right_val) : (left_val < right_val);

    if (should_swap) {
        data[left_index] = right_val;
        data[right_index] = left_val;
    }
}

/// Simple single-pass sort for small arrays that fit in threadgroup memory
/// This is more efficient for small arrays as it avoids multiple kernel dispatches
kernel void bitonic_sort_local(
    device uint *data [[buffer(0)]],
    constant uint &array_size [[buffer(1)]],
    threadgroup uint *shared_data [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[thread_position_in_grid]])
{
    // Load data into threadgroup memory
    if (gid < array_size) {
        shared_data[tid] = data[gid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform bitonic sort in threadgroup memory
    for (uint block_size = 2; block_size <= tg_size; block_size *= 2) {
        for (uint sub_block_size = block_size; sub_block_size >= 2; sub_block_size /= 2) {
            uint half_sub_block = sub_block_size / 2;
            uint sub_block_index = tid / half_sub_block;
            uint index_in_sub_block = tid % half_sub_block;

            uint left_index = sub_block_index * sub_block_size + index_in_sub_block;
            uint right_index = left_index + half_sub_block;

            if (right_index < tg_size) {
                uint block_index = left_index / block_size;
                bool ascending = (block_index % 2) == 0;

                uint left_val = shared_data[left_index];
                uint right_val = shared_data[right_index];

                bool should_swap = ascending ? (left_val > right_val) : (left_val < right_val);

                if (should_swap) {
                    shared_data[left_index] = right_val;
                    shared_data[right_index] = left_val;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write back to global memory
    if (gid < array_size) {
        data[gid] = shared_data[tid];
    }
}
