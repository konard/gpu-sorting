#include <metal_stdlib>
using namespace metal;

// Maximum elements that can be sorted locally (must match THREADGROUP_SIZE * 2)
#define LOCAL_SIZE 2048

/// Perform a compare-and-swap operation for bitonic sort
inline void compare_and_swap(
    thread uint &a,
    thread uint &b,
    bool ascending)
{
    if ((a > b) == ascending) {
        uint temp = a;
        a = b;
        b = temp;
    }
}

/// Local bitonic sort kernel - sorts blocks that fit in threadgroup memory.
/// Each threadgroup sorts LOCAL_SIZE elements entirely in shared memory,
/// dramatically reducing global memory bandwidth.
///
/// This kernel performs the ENTIRE bitonic sort sequence for its local block,
/// executing all log2(LOCAL_SIZE) stages within a single dispatch.
kernel void bitonic_sort_local(
    device uint *data [[buffer(0)]],
    constant uint &array_size [[buffer(1)]],
    threadgroup uint *local_data [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Each thread handles 2 elements
    uint local_size = tg_size * 2;
    uint block_start = tgid * local_size;

    // Load two elements per thread into shared memory
    uint idx1 = tid * 2;
    uint idx2 = tid * 2 + 1;
    uint global_idx1 = block_start + idx1;
    uint global_idx2 = block_start + idx2;

    // Bounds checking for partial blocks
    local_data[idx1] = (global_idx1 < array_size) ? data[global_idx1] : 0xFFFFFFFF;
    local_data[idx2] = (global_idx2 < array_size) ? data[global_idx2] : 0xFFFFFFFF;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform complete bitonic sort in shared memory
    // This executes ALL stages and substages for this local block
    for (uint stage = 0; (1u << stage) < local_size; stage++) {
        uint block_size = 2u << stage;

        for (uint substage = 0; substage <= stage; substage++) {
            uint sub_block_size = block_size >> substage;
            uint half_sub = sub_block_size / 2;

            // Each thread processes one comparison
            uint pair_idx = tid;
            uint sub_block_idx = pair_idx / half_sub;
            uint idx_in_sub = pair_idx % half_sub;

            uint left_idx = sub_block_idx * sub_block_size + idx_in_sub;
            uint right_idx = left_idx + half_sub;

            if (right_idx < local_size) {
                uint block_idx = left_idx / block_size;
                bool ascending = (block_idx % 2) == 0;

                uint left_val = local_data[left_idx];
                uint right_val = local_data[right_idx];

                if ((left_val > right_val) == ascending) {
                    local_data[left_idx] = right_val;
                    local_data[right_idx] = left_val;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write sorted data back to global memory
    if (global_idx1 < array_size) data[global_idx1] = local_data[idx1];
    if (global_idx2 < array_size) data[global_idx2] = local_data[idx2];
}

/// Global bitonic sort step kernel - for cross-threadgroup operations.
/// Used when the comparison distance exceeds threadgroup memory capacity.
///
/// Parameters:
/// - data: The array being sorted
/// - block_size: Current bitonic block size (determines sort direction)
/// - sub_block_size: Current comparison distance
kernel void bitonic_sort_global(
    device uint *data [[buffer(0)]],
    constant uint &block_size [[buffer(1)]],
    constant uint &sub_block_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_sub_block = sub_block_size / 2;
    uint sub_block_index = gid / half_sub_block;
    uint index_in_sub_block = gid % half_sub_block;

    uint left_index = sub_block_index * sub_block_size + index_in_sub_block;
    uint right_index = left_index + half_sub_block;

    uint block_index = left_index / block_size;
    bool ascending = (block_index % 2) == 0;

    uint left_val = data[left_index];
    uint right_val = data[right_index];

    if ((left_val > right_val) == ascending) {
        data[left_index] = right_val;
        data[right_index] = left_val;
    }
}

/// Merge step kernel - performs a single substage of bitonic merge.
/// After local sorting, this kernel handles the remaining global merge stages.
/// Uses the same algorithm as bitonic_sort_global but with proper block alignment.
kernel void bitonic_merge_global(
    device uint *data [[buffer(0)]],
    constant uint &total_block_size [[buffer(1)]],
    constant uint &comparison_distance [[buffer(2)]],
    constant uint &array_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_dist = comparison_distance / 2;
    uint pair_idx = gid;
    uint sub_block_idx = pair_idx / half_dist;
    uint idx_in_sub = pair_idx % half_dist;

    uint left_idx = sub_block_idx * comparison_distance + idx_in_sub;
    uint right_idx = left_idx + half_dist;

    if (right_idx >= array_size) return;

    uint block_idx = left_idx / total_block_size;
    bool ascending = (block_idx % 2) == 0;

    uint left_val = data[left_idx];
    uint right_val = data[right_idx];

    if ((left_val > right_val) == ascending) {
        data[left_idx] = right_val;
        data[right_idx] = left_val;
    }
}

/// Final merge kernel using threadgroup memory for the local portion.
/// After a global comparison step, this performs remaining local comparisons
/// efficiently using shared memory.
kernel void bitonic_merge_local(
    device uint *data [[buffer(0)]],
    constant uint &total_block_size [[buffer(1)]],
    constant uint &max_local_dist [[buffer(2)]],
    constant uint &array_size [[buffer(3)]],
    threadgroup uint *local_data [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint local_size = tg_size * 2;
    uint block_start = tgid * local_size;

    // Load data into shared memory
    uint idx1 = tid * 2;
    uint idx2 = tid * 2 + 1;
    uint global_idx1 = block_start + idx1;
    uint global_idx2 = block_start + idx2;

    local_data[idx1] = (global_idx1 < array_size) ? data[global_idx1] : 0xFFFFFFFF;
    local_data[idx2] = (global_idx2 < array_size) ? data[global_idx2] : 0xFFFFFFFF;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform local merge steps (comparison distances that fit in shared memory)
    for (uint comp_dist = max_local_dist; comp_dist >= 2; comp_dist /= 2) {
        uint half_dist = comp_dist / 2;
        uint pair_idx = tid;
        uint sub_block_idx = pair_idx / half_dist;
        uint idx_in_sub = pair_idx % half_dist;

        uint left_idx = sub_block_idx * comp_dist + idx_in_sub;
        uint right_idx = left_idx + half_dist;

        if (right_idx < local_size) {
            // Calculate block index relative to total_block_size for direction
            uint global_left = block_start + left_idx;
            uint block_idx = global_left / total_block_size;
            bool ascending = (block_idx % 2) == 0;

            uint left_val = local_data[left_idx];
            uint right_val = local_data[right_idx];

            if ((left_val > right_val) == ascending) {
                local_data[left_idx] = right_val;
                local_data[right_idx] = left_val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back to global memory
    if (global_idx1 < array_size) data[global_idx1] = local_data[idx1];
    if (global_idx2 < array_size) data[global_idx2] = local_data[idx2];
}
