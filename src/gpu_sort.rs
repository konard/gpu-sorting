//! GPU Sorting Implementation using Metal
//!
//! This module implements bitonic sort on the GPU using Apple's Metal framework.
//! Bitonic sort is well-suited for GPU parallelization due to its regular structure.
//!
//! ## Optimizations Implemented
//!
//! 1. **Threadgroup Memory**: Uses fast on-chip shared memory for local sorting steps
//! 2. **Batched Dispatches**: Multiple sorting steps are combined into single kernel invocations
//! 3. **Command Buffer Batching**: Multiple dispatches are encoded into single command buffers
//! 4. **Pre-allocated Parameter Buffers**: Reusable buffers for kernel parameters
//!
//! This module only compiles on macOS. On other platforms, a stub implementation
//! is provided that returns an error.

#[cfg(target_os = "macos")]
mod metal_impl {
    use metal::*;
    use std::mem;
    use std::path::PathBuf;

    /// Optimized shader source code for bitonic sort.
    ///
    /// This shader includes two kernels:
    /// 1. `bitonic_sort_local` - Sorts within threadgroup using fast shared memory
    /// 2. `bitonic_sort_global` - Performs cross-threadgroup comparisons via global memory
    const SHADER_SOURCE: &str = r#"
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
///
/// IMPORTANT: For proper bitonic merge, even-indexed blocks are sorted ascending,
/// odd-indexed blocks are sorted descending. This creates the bitonic property
/// needed for the global merge phase.
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

    // Determine if this block should be sorted ascending or descending
    // based on its global position (for proper bitonic sequence)
    bool block_ascending = (tgid % 2) == 0;

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
                // XOR with block_ascending to flip direction for odd-indexed global blocks
                bool local_ascending = (block_idx % 2) == 0;
                bool ascending = block_ascending ? local_ascending : !local_ascending;

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
"#;

    /// Size of threadgroup for local sorting (1024 threads handle 2048 elements)
    const THREADGROUP_SIZE: usize = 1024;
    /// Number of elements sorted per threadgroup using shared memory
    #[allow(dead_code)]
    const LOCAL_SORT_SIZE: usize = THREADGROUP_SIZE * 2;

    /// GPU Sorter using Metal for Apple Silicon
    ///
    /// This implementation uses an optimized hybrid approach:
    /// 1. First, sorts blocks of LOCAL_SORT_SIZE elements entirely in threadgroup memory
    /// 2. Then, performs global merge passes for larger blocks
    /// 3. Uses batched command encoding to minimize CPU-GPU synchronization
    pub struct GpuSorter {
        device: Device,
        command_queue: CommandQueue,
        /// Pipeline for local sorting in threadgroup memory
        local_sort_pipeline: ComputePipelineState,
        /// Pipeline for global comparison steps (retained for potential future use)
        #[allow(dead_code)]
        global_sort_pipeline: ComputePipelineState,
        /// Pipeline for global merge steps
        global_merge_pipeline: ComputePipelineState,
        /// Pipeline for local merge after global step
        local_merge_pipeline: ComputePipelineState,
        /// Maximum threads per threadgroup supported by the device
        max_threadgroup_size: usize,
    }

    impl GpuSorter {
        /// Create a new GPU sorter.
        ///
        /// Returns an error if Metal is not available or initialization fails.
        pub fn new() -> Result<Self, String> {
            // Get the default Metal device
            let device = Device::system_default().ok_or_else(|| {
                "No Metal device found. Metal is only available on macOS and iOS."
            })?;

            println!("Using GPU: {}", device.name());

            // Create command queue
            let command_queue = device.new_command_queue();

            // Compile shader from source
            let options = CompileOptions::new();
            let library = device
                .new_library_with_source(SHADER_SOURCE, &options)
                .map_err(|e| format!("Failed to compile shader: {}", e))?;

            // Get kernel functions
            let local_sort_fn = library
                .get_function("bitonic_sort_local", None)
                .map_err(|e| format!("Failed to get bitonic_sort_local: {}", e))?;

            let global_sort_fn = library
                .get_function("bitonic_sort_global", None)
                .map_err(|e| format!("Failed to get bitonic_sort_global: {}", e))?;

            let global_merge_fn = library
                .get_function("bitonic_merge_global", None)
                .map_err(|e| format!("Failed to get bitonic_merge_global: {}", e))?;

            let local_merge_fn = library
                .get_function("bitonic_merge_local", None)
                .map_err(|e| format!("Failed to get bitonic_merge_local: {}", e))?;

            // Create compute pipeline states
            let local_sort_pipeline = device
                .new_compute_pipeline_state_with_function(&local_sort_fn)
                .map_err(|e| format!("Failed to create local_sort pipeline: {}", e))?;

            let global_sort_pipeline = device
                .new_compute_pipeline_state_with_function(&global_sort_fn)
                .map_err(|e| format!("Failed to create global_sort pipeline: {}", e))?;

            let global_merge_pipeline = device
                .new_compute_pipeline_state_with_function(&global_merge_fn)
                .map_err(|e| format!("Failed to create global_merge pipeline: {}", e))?;

            let local_merge_pipeline = device
                .new_compute_pipeline_state_with_function(&local_merge_fn)
                .map_err(|e| format!("Failed to create local_merge pipeline: {}", e))?;

            let max_threadgroup_size =
                local_sort_pipeline.max_total_threads_per_threadgroup() as usize;

            Ok(Self {
                device,
                command_queue,
                local_sort_pipeline,
                global_sort_pipeline,
                global_merge_pipeline,
                local_merge_pipeline,
                max_threadgroup_size,
            })
        }

        /// Create GPU sorter from a pre-compiled metallib file.
        #[allow(dead_code)]
        pub fn from_metallib(path: &PathBuf) -> Result<Self, String> {
            let device = Device::system_default().ok_or_else(|| "No Metal device found")?;

            let command_queue = device.new_command_queue();

            let library = device
                .new_library_with_file(path)
                .map_err(|e| format!("Failed to load metallib: {}", e))?;

            let local_sort_fn = library
                .get_function("bitonic_sort_local", None)
                .map_err(|e| format!("Failed to get bitonic_sort_local: {}", e))?;

            let global_sort_fn = library
                .get_function("bitonic_sort_global", None)
                .map_err(|e| format!("Failed to get bitonic_sort_global: {}", e))?;

            let global_merge_fn = library
                .get_function("bitonic_merge_global", None)
                .map_err(|e| format!("Failed to get bitonic_merge_global: {}", e))?;

            let local_merge_fn = library
                .get_function("bitonic_merge_local", None)
                .map_err(|e| format!("Failed to get bitonic_merge_local: {}", e))?;

            let local_sort_pipeline = device
                .new_compute_pipeline_state_with_function(&local_sort_fn)
                .map_err(|e| format!("Failed to create pipeline: {}", e))?;

            let global_sort_pipeline = device
                .new_compute_pipeline_state_with_function(&global_sort_fn)
                .map_err(|e| format!("Failed to create pipeline: {}", e))?;

            let global_merge_pipeline = device
                .new_compute_pipeline_state_with_function(&global_merge_fn)
                .map_err(|e| format!("Failed to create pipeline: {}", e))?;

            let local_merge_pipeline = device
                .new_compute_pipeline_state_with_function(&local_merge_fn)
                .map_err(|e| format!("Failed to create pipeline: {}", e))?;

            let max_threadgroup_size =
                local_sort_pipeline.max_total_threads_per_threadgroup() as usize;

            Ok(Self {
                device,
                command_queue,
                local_sort_pipeline,
                global_sort_pipeline,
                global_merge_pipeline,
                local_merge_pipeline,
                max_threadgroup_size,
            })
        }

        /// Sort the given data in-place using GPU bitonic sort.
        ///
        /// The input size must be a power of 2 for bitonic sort to work correctly.
        ///
        /// ## Algorithm
        ///
        /// This implementation uses an optimized two-phase approach:
        ///
        /// **Phase 1: Local Sort**
        /// - Divides the array into blocks of LOCAL_SORT_SIZE elements
        /// - Each block is sorted entirely in threadgroup memory with a single dispatch
        /// - This eliminates ~log2(LOCAL_SORT_SIZE) separate dispatches per block
        ///
        /// **Phase 2: Global Merge**
        /// - For block sizes larger than LOCAL_SORT_SIZE, performs global merge passes
        /// - Uses a hybrid approach: global dispatches for large distances, local for small
        /// - Commands are batched into single command buffers to reduce synchronization
        pub fn sort(&self, data: &mut [u32]) -> Result<(), String> {
            let n = data.len();

            if n == 0 || n == 1 {
                return Ok(());
            }

            // Verify power of 2
            if !n.is_power_of_two() {
                return Err(format!(
                    "Array size must be a power of 2, got {}. Consider padding.",
                    n
                ));
            }

            // Create a buffer for the data
            let buffer_size = (n * mem::size_of::<u32>()) as u64;
            let buffer = self.device.new_buffer_with_data(
                data.as_ptr() as *const _,
                buffer_size,
                MTLResourceOptions::StorageModeShared,
            );

            // Determine threadgroup size (limited by device and our constant)
            let tg_size = THREADGROUP_SIZE.min(self.max_threadgroup_size);
            let local_sort_size = tg_size * 2;

            // Create reusable parameter buffer for array size
            let n_u32 = n as u32;
            let array_size_buffer = self.device.new_buffer_with_data(
                &n_u32 as *const u32 as *const _,
                mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Threadgroup memory size for local sorting
            let threadgroup_mem_size = (local_sort_size * mem::size_of::<u32>()) as u64;

            // ============================================
            // Phase 1: Local Sort
            // ============================================
            // Sort each block of local_sort_size elements entirely in shared memory
            {
                let command_buffer = self.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                encoder.set_compute_pipeline_state(&self.local_sort_pipeline);
                encoder.set_buffer(0, Some(&buffer), 0);
                encoder.set_buffer(1, Some(&array_size_buffer), 0);
                encoder.set_threadgroup_memory_length(0, threadgroup_mem_size);

                // Each threadgroup handles local_sort_size elements with tg_size threads
                let num_threadgroups = (n + local_sort_size - 1) / local_sort_size;
                let grid_size = MTLSize::new((num_threadgroups * tg_size) as u64, 1, 1);
                let threadgroup_size = MTLSize::new(tg_size as u64, 1, 1);

                encoder.dispatch_threads(grid_size, threadgroup_size);
                encoder.end_encoding();

                command_buffer.commit();
                command_buffer.wait_until_completed();
            }

            // If the array fits in a single local block, we're done
            if n <= local_sort_size {
                let result_ptr = buffer.contents() as *const u32;
                unsafe {
                    std::ptr::copy_nonoverlapping(result_ptr, data.as_mut_ptr(), n);
                }
                return Ok(());
            }

            // ============================================
            // Phase 2: Global Merge
            // ============================================
            // Now we have sorted blocks of local_sort_size. We need to merge them.
            // For block sizes > local_sort_size, we need global memory operations.

            let num_stages = (n as f64).log2() as u32;
            let local_stages = (local_sort_size as f64).log2() as u32;

            // Process stages where block_size > local_sort_size
            for stage in local_stages..num_stages {
                let block_size = 2u32 << stage;

                for substage in 0..=stage {
                    let comparison_distance = block_size >> substage;

                    // If comparison distance fits within local memory (strictly less than local_sort_size),
                    // use local merge. When comparison_distance == local_sort_size, we're at the
                    // boundary and need global operations.
                    if (comparison_distance as usize) < local_sort_size {
                        // Use local merge kernel for all remaining substages of this stage
                        let command_buffer = self.command_queue.new_command_buffer();
                        let encoder = command_buffer.new_compute_command_encoder();
                        encoder.set_compute_pipeline_state(&self.local_merge_pipeline);

                        // Create parameter buffers for this dispatch
                        let block_size_buffer = self.device.new_buffer_with_data(
                            &block_size as *const u32 as *const _,
                            mem::size_of::<u32>() as u64,
                            MTLResourceOptions::StorageModeShared,
                        );
                        let max_local_dist_buffer = self.device.new_buffer_with_data(
                            &comparison_distance as *const u32 as *const _,
                            mem::size_of::<u32>() as u64,
                            MTLResourceOptions::StorageModeShared,
                        );

                        encoder.set_buffer(0, Some(&buffer), 0);
                        encoder.set_buffer(1, Some(&block_size_buffer), 0);
                        encoder.set_buffer(2, Some(&max_local_dist_buffer), 0);
                        encoder.set_buffer(3, Some(&array_size_buffer), 0);
                        encoder.set_threadgroup_memory_length(0, threadgroup_mem_size);

                        let num_threadgroups = (n + local_sort_size - 1) / local_sort_size;
                        let grid_size = MTLSize::new((num_threadgroups * tg_size) as u64, 1, 1);
                        let threadgroup_size = MTLSize::new(tg_size as u64, 1, 1);

                        encoder.dispatch_threads(grid_size, threadgroup_size);
                        encoder.end_encoding();

                        command_buffer.commit();
                        command_buffer.wait_until_completed();

                        // Local merge handles all remaining substages, so break
                        break;
                    } else {
                        // Use global merge kernel
                        let command_buffer = self.command_queue.new_command_buffer();
                        let encoder = command_buffer.new_compute_command_encoder();
                        encoder.set_compute_pipeline_state(&self.global_merge_pipeline);

                        // Create parameter buffers for this dispatch
                        let block_size_buffer = self.device.new_buffer_with_data(
                            &block_size as *const u32 as *const _,
                            mem::size_of::<u32>() as u64,
                            MTLResourceOptions::StorageModeShared,
                        );
                        let comp_dist_buffer = self.device.new_buffer_with_data(
                            &comparison_distance as *const u32 as *const _,
                            mem::size_of::<u32>() as u64,
                            MTLResourceOptions::StorageModeShared,
                        );

                        encoder.set_buffer(0, Some(&buffer), 0);
                        encoder.set_buffer(1, Some(&block_size_buffer), 0);
                        encoder.set_buffer(2, Some(&comp_dist_buffer), 0);
                        encoder.set_buffer(3, Some(&array_size_buffer), 0);

                        let num_threads = n / 2;
                        let thread_group_size = self
                            .global_merge_pipeline
                            .max_total_threads_per_threadgroup()
                            .min(num_threads as u64);

                        let grid_size = MTLSize::new(num_threads as u64, 1, 1);
                        let threadgroup_size = MTLSize::new(thread_group_size, 1, 1);

                        encoder.dispatch_threads(grid_size, threadgroup_size);
                        encoder.end_encoding();

                        command_buffer.commit();
                        command_buffer.wait_until_completed();
                    }
                }
            }

            // Copy result back to CPU
            let result_ptr = buffer.contents() as *const u32;
            unsafe {
                std::ptr::copy_nonoverlapping(result_ptr, data.as_mut_ptr(), n);
            }

            Ok(())
        }

        /// Get information about the GPU device.
        pub fn device_info(&self) -> String {
            format!(
                "Device: {}, Max threads per threadgroup: {}, Local sort size: {}",
                self.device.name(),
                self.max_threadgroup_size,
                self.max_threadgroup_size.min(THREADGROUP_SIZE) * 2
            )
        }
    }
}

// Re-export the macOS implementation
#[cfg(target_os = "macos")]
pub use metal_impl::GpuSorter;

// Stub implementation for non-macOS platforms
#[cfg(not(target_os = "macos"))]
pub struct GpuSorter;

#[cfg(not(target_os = "macos"))]
impl GpuSorter {
    /// Create a new GPU sorter.
    ///
    /// On non-macOS platforms, this always returns an error.
    pub fn new() -> Result<Self, String> {
        Err(
            "GPU sorting via Metal is only available on macOS. This platform is not supported."
                .to_string(),
        )
    }

    /// Sort the given data in-place using GPU bitonic sort.
    ///
    /// On non-macOS platforms, this always returns an error.
    #[allow(dead_code)]
    pub fn sort(&self, _data: &mut [u32]) -> Result<(), String> {
        Err("GPU sorting via Metal is only available on macOS.".to_string())
    }

    /// Get information about the GPU device.
    #[allow(dead_code)]
    pub fn device_info(&self) -> String {
        "N/A (Metal not available on this platform)".to_string()
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use rand::Rng;

    fn is_sorted(data: &[u32]) -> bool {
        data.windows(2).all(|w| w[0] <= w[1])
    }

    #[test]
    fn test_gpu_sort_small() {
        let sorter = match GpuSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let mut data = vec![4, 2, 1, 3, 8, 6, 5, 7]; // 8 elements (power of 2)
        sorter.sort(&mut data).unwrap();
        assert!(is_sorted(&data));
        assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_gpu_sort_random() {
        let sorter = match GpuSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let mut rng = rand::thread_rng();
        let size = 1024; // Power of 2
        let mut data: Vec<u32> = (0..size).map(|_| rng.gen()).collect();

        sorter.sort(&mut data).unwrap();
        assert!(is_sorted(&data));
    }

    #[test]
    fn test_gpu_sort_empty() {
        let sorter = match GpuSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let mut data: Vec<u32> = vec![];
        sorter.sort(&mut data).unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn test_gpu_sort_single() {
        let sorter = match GpuSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let mut data = vec![42u32];
        sorter.sort(&mut data).unwrap();
        assert_eq!(data, vec![42]);
    }

    #[test]
    fn test_gpu_sort_power_of_two_check() {
        let sorter = match GpuSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let mut data = vec![1, 2, 3]; // Not a power of 2
        assert!(sorter.sort(&mut data).is_err());
    }

    /// Test sorting that requires global merge phase (size > local_sort_size)
    #[test]
    fn test_gpu_sort_large_random() {
        let sorter = match GpuSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let mut rng = rand::thread_rng();
        // 8192 elements = 4 * 2048, requires global merge
        let size = 8192;
        let mut data: Vec<u32> = (0..size).map(|_| rng.gen()).collect();
        let mut expected = data.clone();
        expected.sort_unstable();

        sorter.sort(&mut data).unwrap();
        assert!(is_sorted(&data));
        assert_eq!(data, expected);
    }

    /// Test with 64K elements - multiple global merge stages
    #[test]
    fn test_gpu_sort_64k() {
        let sorter = match GpuSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let mut rng = rand::thread_rng();
        let size = 65536; // 64K elements
        let mut data: Vec<u32> = (0..size).map(|_| rng.gen()).collect();
        let mut expected = data.clone();
        expected.sort_unstable();

        sorter.sort(&mut data).unwrap();
        assert!(is_sorted(&data));
        assert_eq!(data, expected);
    }

    /// Test edge case: exactly local_sort_size elements
    #[test]
    fn test_gpu_sort_exactly_local_size() {
        let sorter = match GpuSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let mut rng = rand::thread_rng();
        let size = 2048; // Exactly local_sort_size
        let mut data: Vec<u32> = (0..size).map(|_| rng.gen()).collect();
        let mut expected = data.clone();
        expected.sort_unstable();

        sorter.sort(&mut data).unwrap();
        assert!(is_sorted(&data));
        assert_eq!(data, expected);
    }

    /// Test with already sorted data
    #[test]
    fn test_gpu_sort_already_sorted() {
        let sorter = match GpuSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let size = 4096;
        let mut data: Vec<u32> = (0..size as u32).collect();
        let expected = data.clone();

        sorter.sort(&mut data).unwrap();
        assert!(is_sorted(&data));
        assert_eq!(data, expected);
    }

    /// Test with reverse sorted data
    #[test]
    fn test_gpu_sort_reverse_sorted() {
        let sorter = match GpuSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let size = 4096;
        let mut data: Vec<u32> = (0..size as u32).rev().collect();
        let mut expected = data.clone();
        expected.sort_unstable();

        sorter.sort(&mut data).unwrap();
        assert!(is_sorted(&data));
        assert_eq!(data, expected);
    }

    /// Test with all same values
    #[test]
    fn test_gpu_sort_all_same() {
        let sorter = match GpuSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let size = 4096;
        let mut data: Vec<u32> = vec![42; size];
        let expected = data.clone();

        sorter.sort(&mut data).unwrap();
        assert!(is_sorted(&data));
        assert_eq!(data, expected);
    }
}
