//! GPU Radix Sort Implementation using Metal
//!
//! This module implements a GPU radix sort using Apple's Metal framework.
//! It uses the "reduce-then-scan" approach which is portable across different
//! GPU architectures, including Apple Silicon which lacks forward progress guarantees.
//!
//! ## Algorithm Overview
//!
//! DeviceRadixSort processes 8 bits per pass (4 passes for 32-bit integers):
//! 1. **Histogram**: Count keys in each of 256 buckets per threadgroup
//! 2. **Reduce**: Sum histograms to get global bucket counts
//! 3. **Scan**: Compute exclusive prefix sum of global counts
//! 4. **Scatter Offsets**: Compute per-threadgroup output positions
//! 5. **Scatter**: Reorder keys to output buffer
//!
//! This approach avoids the "chained-scan-with-decoupled-lookback" technique
//! used in OneSweep, which can deadlock on Apple Silicon GPUs.
//!
//! ## Performance
//!
//! Expected complexity: O(n) - linear time regardless of data distribution
//! Expected throughput: 10-20x faster than bitonic sort for large arrays
//!
//! ## References
//!
//! - [GPUSorting by b0nes164](https://github.com/b0nes164/GPUSorting)
//! - [Linebender GPU Sorting Wiki](https://linebender.org/wiki/gpu/sorting/)
//! - [CUB DeviceRadixSort](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html)

#[cfg(target_os = "macos")]
mod metal_impl {
    use metal::*;
    use std::mem;

    /// Radix sort configuration - must match shader constants
    const RADIX_BITS: u32 = 8;
    const RADIX_SIZE: usize = 1 << RADIX_BITS; // 256 buckets
    const THREADGROUP_SIZE: usize = 256;
    const KEYS_PER_THREAD: usize = 4;
    const KEYS_PER_THREADGROUP: usize = THREADGROUP_SIZE * KEYS_PER_THREAD; // 1024

    /// Metal shader source for radix sort
    const SHADER_SOURCE: &str = include_str!("../shaders/radix_sort.metal");

    /// GPU Radix Sorter using Metal for Apple Silicon
    ///
    /// Implements the DeviceRadixSort algorithm with reduce-then-scan approach.
    /// This is portable to all GPU architectures and doesn't rely on forward
    /// progress guarantees.
    pub struct GpuRadixSorter {
        device: Device,
        command_queue: CommandQueue,
        histogram_pipeline: ComputePipelineState,
        reduce_pipeline: ComputePipelineState,
        scan_pipeline: ComputePipelineState,
        scatter_offsets_pipeline: ComputePipelineState,
        scatter_pipeline: ComputePipelineState,
        scatter_simd_pipeline: ComputePipelineState, // SIMD-optimized scatter
        max_threadgroup_size: usize,
        use_simd: bool, // Whether to use SIMD-optimized scatter kernel
    }

    impl GpuRadixSorter {
        /// Create a new GPU radix sorter.
        ///
        /// Returns an error if Metal is not available or initialization fails.
        pub fn new() -> Result<Self, String> {
            let device = Device::system_default()
                .ok_or_else(|| "No Metal device found. Metal is only available on macOS.")?;

            let command_queue = device.new_command_queue();

            // Compile shader from source
            let options = CompileOptions::new();
            let library = device
                .new_library_with_source(SHADER_SOURCE, &options)
                .map_err(|e| format!("Failed to compile radix sort shader: {}", e))?;

            // Get kernel functions
            let histogram_fn = library
                .get_function("radix_histogram", None)
                .map_err(|e| format!("Failed to get radix_histogram: {}", e))?;

            let reduce_fn = library
                .get_function("radix_reduce", None)
                .map_err(|e| format!("Failed to get radix_reduce: {}", e))?;

            let scan_fn = library
                .get_function("radix_scan", None)
                .map_err(|e| format!("Failed to get radix_scan: {}", e))?;

            let scatter_offsets_fn = library
                .get_function("radix_scatter_offsets", None)
                .map_err(|e| format!("Failed to get radix_scatter_offsets: {}", e))?;

            let scatter_fn = library
                .get_function("radix_scatter", None)
                .map_err(|e| format!("Failed to get radix_scatter: {}", e))?;

            let scatter_simd_fn = library
                .get_function("radix_scatter_simd", None)
                .map_err(|e| format!("Failed to get radix_scatter_simd: {}", e))?;

            // Create compute pipeline states
            let histogram_pipeline = device
                .new_compute_pipeline_state_with_function(&histogram_fn)
                .map_err(|e| format!("Failed to create histogram pipeline: {}", e))?;

            let reduce_pipeline = device
                .new_compute_pipeline_state_with_function(&reduce_fn)
                .map_err(|e| format!("Failed to create reduce pipeline: {}", e))?;

            let scan_pipeline = device
                .new_compute_pipeline_state_with_function(&scan_fn)
                .map_err(|e| format!("Failed to create scan pipeline: {}", e))?;

            let scatter_offsets_pipeline = device
                .new_compute_pipeline_state_with_function(&scatter_offsets_fn)
                .map_err(|e| format!("Failed to create scatter_offsets pipeline: {}", e))?;

            let scatter_pipeline = device
                .new_compute_pipeline_state_with_function(&scatter_fn)
                .map_err(|e| format!("Failed to create scatter pipeline: {}", e))?;

            let scatter_simd_pipeline = device
                .new_compute_pipeline_state_with_function(&scatter_simd_fn)
                .map_err(|e| format!("Failed to create scatter_simd pipeline: {}", e))?;

            let max_threadgroup_size =
                histogram_pipeline.max_total_threads_per_threadgroup() as usize;

            Ok(Self {
                device,
                command_queue,
                histogram_pipeline,
                reduce_pipeline,
                scan_pipeline,
                scatter_offsets_pipeline,
                scatter_pipeline,
                scatter_simd_pipeline,
                max_threadgroup_size,
                use_simd: false, // Default to basic scatter; can be enabled
            })
        }

        /// Create a new GPU radix sorter with SIMD-optimized scatter kernel.
        ///
        /// The SIMD-optimized kernel uses Metal's SIMD group operations for
        /// faster rank computation within each 32-thread SIMD group.
        pub fn new_simd() -> Result<Self, String> {
            let mut sorter = Self::new()?;
            sorter.use_simd = true;
            Ok(sorter)
        }

        /// Get the GPU device name
        pub fn device_name(&self) -> String {
            self.device.name().to_string()
        }

        /// Sort the given data in-place using GPU radix sort.
        ///
        /// Unlike bitonic sort, radix sort works on any array size (not just powers of 2).
        /// However, for optimal performance, array size should be at least several thousand
        /// elements to amortize the CPU-GPU transfer overhead.
        ///
        /// ## Algorithm
        ///
        /// Processes 8 bits per pass (4 passes total for 32-bit integers):
        /// 1. Histogram: Count digits per threadgroup
        /// 2. Reduce: Sum to global counts
        /// 3. Scan: Compute prefix sums
        /// 4. Scatter offsets: Compute per-threadgroup offsets
        /// 5. Scatter: Reorder keys
        ///
        /// ## Optimization: Ultra-Batched Command Buffer
        ///
        /// This implementation batches ALL 20 kernel dispatches (4 passes × 5 kernels)
        /// into a SINGLE command buffer, reducing GPU submissions from 20 to just 1.
        /// Metal command buffers execute encoders in order with proper memory synchronization,
        /// allowing us to pipeline all passes without CPU intervention.
        pub fn sort(&self, data: &mut [u32]) -> Result<(), String> {
            let n = data.len();

            if n <= 1 {
                return Ok(());
            }

            // For very small arrays, fall back to CPU sort
            // GPU overhead isn't worth it below ~1000 elements
            if n < 1024 {
                data.sort_unstable();
                return Ok(());
            }

            let buffer_size = (n * mem::size_of::<u32>()) as u64;

            // Create input and output buffers (ping-pong for each pass)
            let buffer_a = self.device.new_buffer_with_data(
                data.as_ptr() as *const _,
                buffer_size,
                MTLResourceOptions::StorageModeShared,
            );

            let buffer_b = self
                .device
                .new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);

            // Calculate number of threadgroups
            let num_threadgroups = (n + KEYS_PER_THREADGROUP - 1) / KEYS_PER_THREADGROUP;

            // Allocate intermediate buffers
            let histogram_size = (num_threadgroups * RADIX_SIZE * mem::size_of::<u32>()) as u64;
            let histogram_buffer = self
                .device
                .new_buffer(histogram_size, MTLResourceOptions::StorageModeShared);

            let global_histogram_size = (RADIX_SIZE * mem::size_of::<u32>()) as u64;
            let global_histogram_buffer = self
                .device
                .new_buffer(global_histogram_size, MTLResourceOptions::StorageModeShared);

            let scatter_offsets_buffer = self
                .device
                .new_buffer(histogram_size, MTLResourceOptions::StorageModeShared);

            // Create parameter buffers
            let n_u32 = n as u32;
            let array_size_buffer = self.device.new_buffer_with_data(
                &n_u32 as *const u32 as *const _,
                mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let num_tg_u32 = num_threadgroups as u32;
            let num_threadgroups_buffer = self.device.new_buffer_with_data(
                &num_tg_u32 as *const u32 as *const _,
                mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Pre-create shift buffers for all 4 passes
            let shift_buffers: Vec<_> = (0..4u32)
                .map(|pass| {
                    let shift = pass * RADIX_BITS;
                    self.device.new_buffer_with_data(
                        &shift as *const u32 as *const _,
                        mem::size_of::<u32>() as u64,
                        MTLResourceOptions::StorageModeShared,
                    )
                })
                .collect();

            // Threadgroup memory sizes
            let histogram_tg_mem = (RADIX_SIZE * mem::size_of::<u32>()) as u64;
            let threadgroup_mem = (THREADGROUP_SIZE * mem::size_of::<u32>()) as u64;

            // Determine actual threadgroup size to use
            let tg_size = THREADGROUP_SIZE.min(self.max_threadgroup_size);

            // =====================================================
            // ULTRA-BATCHED: All 4 passes in a SINGLE command buffer
            // =====================================================
            // This reduces 4 separate GPU submissions to just 1,
            // dramatically reducing CPU-GPU synchronization overhead.
            //
            // Metal command buffers execute encoders in order, and
            // memory operations between encoders are properly synchronized.
            let command_buffer = self.command_queue.new_command_buffer();

            // Process 4 passes (8 bits each for 32-bit integers)
            // Pass 0: buffer_a -> buffer_b (shift 0)
            // Pass 1: buffer_b -> buffer_a (shift 8)
            // Pass 2: buffer_a -> buffer_b (shift 16)
            // Pass 3: buffer_b -> buffer_a (shift 24)
            // After 4 passes, result is in buffer_a

            for pass in 0..4u32 {
                let shift_buffer = &shift_buffers[pass as usize];

                // Determine input/output buffers for this pass
                let (input_buffer, output_buffer) = if pass % 2 == 0 {
                    (&buffer_a, &buffer_b)
                } else {
                    (&buffer_b, &buffer_a)
                };

                // ====== Kernel 1: Histogram ======
                {
                    let encoder = command_buffer.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&self.histogram_pipeline);
                    encoder.set_buffer(0, Some(input_buffer), 0);
                    encoder.set_buffer(1, Some(&histogram_buffer), 0);
                    encoder.set_buffer(2, Some(&array_size_buffer), 0);
                    encoder.set_buffer(3, Some(shift_buffer), 0);
                    encoder.set_threadgroup_memory_length(0, histogram_tg_mem);

                    let grid_size = MTLSize::new((num_threadgroups * tg_size) as u64, 1, 1);
                    let threadgroup_size = MTLSize::new(tg_size as u64, 1, 1);

                    encoder.dispatch_threads(grid_size, threadgroup_size);
                    encoder.end_encoding();
                }

                // ====== Kernel 2: Reduce ======
                {
                    let encoder = command_buffer.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&self.reduce_pipeline);
                    encoder.set_buffer(0, Some(&histogram_buffer), 0);
                    encoder.set_buffer(1, Some(&global_histogram_buffer), 0);
                    encoder.set_buffer(2, Some(&num_threadgroups_buffer), 0);

                    let grid_size = MTLSize::new(RADIX_SIZE as u64, 1, 1);
                    let threadgroup_size =
                        MTLSize::new(RADIX_SIZE.min(self.max_threadgroup_size) as u64, 1, 1);

                    encoder.dispatch_threads(grid_size, threadgroup_size);
                    encoder.end_encoding();
                }

                // ====== Kernel 3: Scan ======
                {
                    let encoder = command_buffer.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&self.scan_pipeline);
                    encoder.set_buffer(0, Some(&global_histogram_buffer), 0);
                    encoder.set_threadgroup_memory_length(0, histogram_tg_mem);

                    // Single threadgroup with RADIX_SIZE threads
                    let grid_size = MTLSize::new(RADIX_SIZE as u64, 1, 1);
                    let threadgroup_size = MTLSize::new(RADIX_SIZE as u64, 1, 1);

                    encoder.dispatch_threads(grid_size, threadgroup_size);
                    encoder.end_encoding();
                }

                // ====== Kernel 4: Scatter Offsets ======
                {
                    let encoder = command_buffer.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&self.scatter_offsets_pipeline);
                    encoder.set_buffer(0, Some(&histogram_buffer), 0);
                    encoder.set_buffer(1, Some(&global_histogram_buffer), 0);
                    encoder.set_buffer(2, Some(&scatter_offsets_buffer), 0);
                    encoder.set_buffer(3, Some(&num_threadgroups_buffer), 0);

                    let grid_size = MTLSize::new(RADIX_SIZE as u64, 1, 1);
                    let threadgroup_size =
                        MTLSize::new(RADIX_SIZE.min(self.max_threadgroup_size) as u64, 1, 1);

                    encoder.dispatch_threads(grid_size, threadgroup_size);
                    encoder.end_encoding();
                }

                // ====== Kernel 5: Scatter ======
                {
                    let encoder = command_buffer.new_compute_command_encoder();

                    // Choose between basic and SIMD-optimized scatter kernel
                    if self.use_simd {
                        encoder.set_compute_pipeline_state(&self.scatter_simd_pipeline);
                    } else {
                        encoder.set_compute_pipeline_state(&self.scatter_pipeline);
                    }

                    encoder.set_buffer(0, Some(input_buffer), 0);
                    encoder.set_buffer(1, Some(output_buffer), 0);
                    encoder.set_buffer(2, Some(&scatter_offsets_buffer), 0);
                    encoder.set_buffer(3, Some(&array_size_buffer), 0);
                    encoder.set_buffer(4, Some(shift_buffer), 0);

                    // Threadgroup memory for scatter kernel:
                    // - local_offsets: RADIX_SIZE = 256 uints
                    // - shared_digits: THREADGROUP_SIZE = 256 uints
                    // - digit_counts: RADIX_SIZE = 256 uints
                    encoder.set_threadgroup_memory_length(0, histogram_tg_mem);
                    encoder.set_threadgroup_memory_length(1, threadgroup_mem);
                    encoder.set_threadgroup_memory_length(2, histogram_tg_mem);

                    let grid_size = MTLSize::new((num_threadgroups * tg_size) as u64, 1, 1);
                    let threadgroup_size = MTLSize::new(tg_size as u64, 1, 1);

                    encoder.dispatch_threads(grid_size, threadgroup_size);
                    encoder.end_encoding();
                }
            }

            // Submit all 20 kernels (4 passes × 5 kernels) at once and wait
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // After 4 passes (even number), result is in buffer_a
            // Pass 0: a -> b, Pass 1: b -> a, Pass 2: a -> b, Pass 3: b -> a
            let result_ptr = buffer_a.contents() as *const u32;
            unsafe {
                std::ptr::copy_nonoverlapping(result_ptr, data.as_mut_ptr(), n);
            }

            Ok(())
        }

        /// Get information about the GPU device and configuration.
        #[allow(dead_code)]
        pub fn device_info(&self) -> String {
            format!(
                "Device: {}, Max threads per threadgroup: {}, Keys per threadgroup: {}",
                self.device.name(),
                self.max_threadgroup_size,
                KEYS_PER_THREADGROUP
            )
        }
    }
}

#[cfg(target_os = "macos")]
pub use metal_impl::GpuRadixSorter;

#[cfg(not(target_os = "macos"))]
pub struct GpuRadixSorter;

#[cfg(not(target_os = "macos"))]
impl GpuRadixSorter {
    pub fn new() -> Result<Self, String> {
        Err("GPU radix sort via Metal is only available on macOS.".to_string())
    }

    pub fn new_simd() -> Result<Self, String> {
        Err("GPU radix sort via Metal is only available on macOS.".to_string())
    }

    #[allow(dead_code)]
    pub fn sort(&self, _data: &mut [u32]) -> Result<(), String> {
        Err("GPU radix sort via Metal is only available on macOS.".to_string())
    }

    #[allow(dead_code)]
    pub fn device_name(&self) -> String {
        "N/A".to_string()
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
    fn test_radix_sort_small() {
        let sorter = match GpuRadixSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let mut data = vec![4, 2, 1, 3, 8, 6, 5, 7];
        sorter.sort(&mut data).unwrap();
        assert!(is_sorted(&data));
        assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_radix_sort_random_1k() {
        let sorter = match GpuRadixSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let mut rng = rand::thread_rng();
        let size = 1024;
        let mut data: Vec<u32> = (0..size).map(|_| rng.gen()).collect();
        let mut expected = data.clone();
        expected.sort_unstable();

        sorter.sort(&mut data).unwrap();
        assert!(is_sorted(&data));
        assert_eq!(data, expected);
    }

    #[test]
    fn test_radix_sort_random_4k() {
        let sorter = match GpuRadixSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let mut rng = rand::thread_rng();
        let size = 4096;
        let mut data: Vec<u32> = (0..size).map(|_| rng.gen()).collect();
        let mut expected = data.clone();
        expected.sort_unstable();

        sorter.sort(&mut data).unwrap();
        assert!(is_sorted(&data));
        assert_eq!(data, expected);
    }

    #[test]
    fn test_radix_sort_random_64k() {
        let sorter = match GpuRadixSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let mut rng = rand::thread_rng();
        let size = 65536;
        let mut data: Vec<u32> = (0..size).map(|_| rng.gen()).collect();
        let mut expected = data.clone();
        expected.sort_unstable();

        sorter.sort(&mut data).unwrap();
        assert!(is_sorted(&data));
        assert_eq!(data, expected);
    }

    #[test]
    fn test_radix_sort_non_power_of_two() {
        let sorter = match GpuRadixSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let mut rng = rand::thread_rng();
        // Non-power-of-2 size - radix sort handles this unlike bitonic sort
        let size = 5000;
        let mut data: Vec<u32> = (0..size).map(|_| rng.gen()).collect();
        let mut expected = data.clone();
        expected.sort_unstable();

        sorter.sort(&mut data).unwrap();
        assert!(is_sorted(&data));
        assert_eq!(data, expected);
    }

    #[test]
    fn test_radix_sort_empty() {
        let sorter = match GpuRadixSorter::new() {
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
    fn test_radix_sort_single() {
        let sorter = match GpuRadixSorter::new() {
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
    fn test_radix_sort_already_sorted() {
        let sorter = match GpuRadixSorter::new() {
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

    #[test]
    fn test_radix_sort_reverse_sorted() {
        let sorter = match GpuRadixSorter::new() {
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

    #[test]
    fn test_radix_sort_all_same() {
        let sorter = match GpuRadixSorter::new() {
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

    #[test]
    fn test_radix_sort_max_values() {
        let sorter = match GpuRadixSorter::new() {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping GPU test: Metal not available");
                return;
            }
        };

        let mut data = vec![u32::MAX, 0, u32::MAX / 2, 1, u32::MAX - 1];
        // Pad to 1024 for GPU execution
        let mut rng = rand::thread_rng();
        while data.len() < 1024 {
            data.push(rng.gen());
        }

        let mut expected = data.clone();
        expected.sort_unstable();

        sorter.sort(&mut data).unwrap();
        assert!(is_sorted(&data));
        assert_eq!(data, expected);
    }
}
