//! GPU Sorting Implementation using Metal
//!
//! This module implements bitonic sort on the GPU using Apple's Metal framework.
//! Bitonic sort is well-suited for GPU parallelization due to its regular structure.
//!
//! This module only compiles on macOS. On other platforms, a stub implementation
//! is provided that returns an error.

#[cfg(target_os = "macos")]
mod metal_impl {
    use metal::*;
    use std::mem;
    use std::path::PathBuf;

    /// Shader source code for bitonic sort
    const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

/// Bitonic sort step kernel.
///
/// This kernel performs one step of bitonic sort. The algorithm works by:
/// 1. Comparing pairs of elements at specific distances
/// 2. Swapping elements based on the current sorting direction
kernel void bitonic_sort_step(
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

    bool should_swap = ascending ? (left_val > right_val) : (left_val < right_val);

    if (should_swap) {
        data[left_index] = right_val;
        data[right_index] = left_val;
    }
}
"#;

    /// GPU Sorter using Metal for Apple Silicon
    pub struct GpuSorter {
        device: Device,
        command_queue: CommandQueue,
        pipeline_state: ComputePipelineState,
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

            // Get the kernel function
            let kernel_function = library
                .get_function("bitonic_sort_step", None)
                .map_err(|e| format!("Failed to get kernel function: {}", e))?;

            // Create compute pipeline state
            let pipeline_state = device
                .new_compute_pipeline_state_with_function(&kernel_function)
                .map_err(|e| format!("Failed to create pipeline state: {}", e))?;

            Ok(Self {
                device,
                command_queue,
                pipeline_state,
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

            let kernel_function = library
                .get_function("bitonic_sort_step", None)
                .map_err(|e| format!("Failed to get kernel function: {}", e))?;

            let pipeline_state = device
                .new_compute_pipeline_state_with_function(&kernel_function)
                .map_err(|e| format!("Failed to create pipeline state: {}", e))?;

            Ok(Self {
                device,
                command_queue,
                pipeline_state,
            })
        }

        /// Sort the given data in-place using GPU bitonic sort.
        ///
        /// The input size must be a power of 2 for bitonic sort to work correctly.
        pub fn sort(&self, data: &mut [u32]) -> Result<(), String> {
            let n = data.len();

            if n == 0 {
                return Ok(());
            }

            if n == 1 {
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

            // Bitonic sort requires log2(n) stages
            // Each stage k has k substages
            let num_stages = (n as f64).log2() as u32;

            for stage in 0..num_stages {
                let block_size = 2u32 << stage; // 2, 4, 8, 16, ...

                for substage in 0..=stage {
                    let sub_block_size = block_size >> substage; // block_size, block_size/2, ...

                    // Create parameter buffers
                    let block_size_buffer = self.device.new_buffer_with_data(
                        &block_size as *const u32 as *const _,
                        mem::size_of::<u32>() as u64,
                        MTLResourceOptions::StorageModeShared,
                    );

                    let sub_block_size_buffer = self.device.new_buffer_with_data(
                        &sub_block_size as *const u32 as *const _,
                        mem::size_of::<u32>() as u64,
                        MTLResourceOptions::StorageModeShared,
                    );

                    // Create command buffer and encoder
                    let command_buffer = self.command_queue.new_command_buffer();
                    let encoder = command_buffer.new_compute_command_encoder();

                    encoder.set_compute_pipeline_state(&self.pipeline_state);
                    encoder.set_buffer(0, Some(&buffer), 0);
                    encoder.set_buffer(1, Some(&block_size_buffer), 0);
                    encoder.set_buffer(2, Some(&sub_block_size_buffer), 0);

                    // Calculate thread count: we need n/2 comparisons
                    let num_threads = n / 2;
                    let thread_group_size = self
                        .pipeline_state
                        .max_total_threads_per_threadgroup()
                        .min(num_threads as u64) as u64;

                    let grid_size = MTLSize::new(num_threads as u64, 1, 1);
                    let threadgroup_size = MTLSize::new(thread_group_size, 1, 1);

                    encoder.dispatch_threads(grid_size, threadgroup_size);
                    encoder.end_encoding();

                    command_buffer.commit();
                    command_buffer.wait_until_completed();
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
        #[allow(dead_code)]
        pub fn device_info(&self) -> String {
            format!(
                "Device: {}, Max threads per threadgroup: {}",
                self.device.name(),
                self.pipeline_state.max_total_threads_per_threadgroup()
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
}
