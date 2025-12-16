//! GPU Sorting Proof of Concept
//!
//! This application compares CPU and GPU sorting performance on Apple Silicon (M1-M3).
//! It implements multiple GPU sorting algorithms:
//! - **Bitonic Sort**: O(n log²n) - comparison-based, requires power-of-2 sizes
//! - **Radix Sort**: O(n) - linear time using DeviceRadixSort algorithm
//!
//! The CPU baseline uses Rust's pdqsort (pattern-defeating quicksort).

mod cpu_sort;
mod gpu_radix_sort;
mod gpu_sort;

use rand::Rng;
use std::time::Instant;

/// Default array size for benchmarking
const DEFAULT_ARRAY_SIZE: usize = 1 << 20; // 1 million elements

fn main() {
    println!("GPU Sorting Proof of Concept");
    println!("=============================\n");

    // Parse command line arguments for array size
    let args: Vec<String> = std::env::args().collect();
    let array_size = if args.len() > 1 {
        args[1].parse().unwrap_or(DEFAULT_ARRAY_SIZE)
    } else {
        DEFAULT_ARRAY_SIZE
    };

    // For bitonic sort, we need power of 2, but for radix sort we don't
    let bitonic_array_size = array_size.next_power_of_two();

    println!(
        "Array size: {} elements ({} MB)",
        array_size,
        array_size * 4 / 1_000_000
    );

    if bitonic_array_size != array_size {
        println!(
            "Note: Bitonic sort will use {} elements (rounded to power of 2)",
            bitonic_array_size
        );
    }

    // Generate random data
    println!("\nGenerating random data...");
    let mut rng = rand::thread_rng();
    let data: Vec<u32> = (0..array_size).map(|_| rng.gen()).collect();
    let data_bitonic: Vec<u32> = (0..bitonic_array_size).map(|_| rng.gen()).collect();

    // CPU Sorting
    println!("\n--- CPU Sorting (std::sort unstable / pdqsort) ---");
    let mut cpu_data = data.clone();
    let cpu_start = Instant::now();
    cpu_sort::sort_unstable(&mut cpu_data);
    let cpu_duration = cpu_start.elapsed();
    println!(
        "CPU sort time: {:.3} ms",
        cpu_duration.as_secs_f64() * 1000.0
    );

    // Verify CPU sort
    assert!(cpu_sort::is_sorted(&cpu_data), "CPU sort failed!");
    println!("CPU sort verified: OK");

    // GPU Bitonic Sorting
    println!("\n--- GPU Sorting (Metal Bitonic Sort) ---");
    let gpu_bitonic_duration = match gpu_sort::GpuSorter::new() {
        Ok(sorter) => {
            println!(
                "Using GPU: {}",
                sorter.device_info().split(',').next().unwrap_or("Unknown")
            );
            let mut gpu_data = data_bitonic.clone();
            let gpu_start = Instant::now();
            match sorter.sort(&mut gpu_data) {
                Ok(()) => {
                    let gpu_duration = gpu_start.elapsed();
                    println!(
                        "GPU bitonic sort time: {:.3} ms",
                        gpu_duration.as_secs_f64() * 1000.0
                    );

                    if cpu_sort::is_sorted(&gpu_data) {
                        println!("GPU bitonic sort verified: OK");
                    } else {
                        println!("ERROR: GPU bitonic sort failed verification!");
                    }
                    Some(gpu_duration)
                }
                Err(e) => {
                    println!("GPU bitonic sort error: {}", e);
                    None
                }
            }
        }
        Err(e) => {
            println!("Failed to initialize GPU bitonic sorter: {}", e);
            None
        }
    };

    // GPU Radix Sorting
    println!("\n--- GPU Sorting (Metal Radix Sort - DeviceRadixSort) ---");
    let gpu_radix_duration = match gpu_radix_sort::GpuRadixSorter::new() {
        Ok(sorter) => {
            println!("Using GPU: {}", sorter.device_name());
            let mut gpu_data = data.clone();
            let gpu_start = Instant::now();
            match sorter.sort(&mut gpu_data) {
                Ok(()) => {
                    let gpu_duration = gpu_start.elapsed();
                    println!(
                        "GPU radix sort time: {:.3} ms",
                        gpu_duration.as_secs_f64() * 1000.0
                    );

                    if cpu_sort::is_sorted(&gpu_data) {
                        println!("GPU radix sort verified: OK");

                        // Compare results with CPU
                        if gpu_data == cpu_data {
                            println!("Results match CPU sort: OK");
                        } else {
                            println!("WARNING: Results differ from CPU sort!");
                        }
                    } else {
                        println!("ERROR: GPU radix sort failed verification!");
                    }
                    Some(gpu_duration)
                }
                Err(e) => {
                    println!("GPU radix sort error: {}", e);
                    None
                }
            }
        }
        Err(e) => {
            println!("Failed to initialize GPU radix sorter: {}", e);
            None
        }
    };

    // Performance comparison
    println!("\n--- Performance Comparison ---");

    let cpu_ms = cpu_duration.as_secs_f64() * 1000.0;

    if let Some(bitonic_dur) = gpu_bitonic_duration {
        let bitonic_ms = bitonic_dur.as_secs_f64() * 1000.0;
        let speedup = cpu_ms / bitonic_ms;
        if speedup > 1.0 {
            println!("GPU Bitonic vs CPU: GPU is {:.2}x faster", speedup);
        } else {
            println!("GPU Bitonic vs CPU: CPU is {:.2}x faster", 1.0 / speedup);
        }
    }

    if let Some(radix_dur) = gpu_radix_duration {
        let radix_ms = radix_dur.as_secs_f64() * 1000.0;
        let speedup = cpu_ms / radix_ms;
        if speedup > 1.0 {
            println!("GPU Radix vs CPU: GPU is {:.2}x faster", speedup);
        } else {
            println!("GPU Radix vs CPU: CPU is {:.2}x faster", 1.0 / speedup);
        }
    }

    if let (Some(bitonic_dur), Some(radix_dur)) = (gpu_bitonic_duration, gpu_radix_duration) {
        let speedup = bitonic_dur.as_secs_f64() / radix_dur.as_secs_f64();
        if speedup > 1.0 {
            println!("GPU Radix vs GPU Bitonic: Radix is {:.2}x faster", speedup);
        } else {
            println!(
                "GPU Radix vs GPU Bitonic: Bitonic is {:.2}x faster",
                1.0 / speedup
            );
        }
    }

    // Run multiple sizes for comprehensive benchmark
    if args.len() > 2 && args[2] == "--benchmark" {
        run_benchmark();
    }
}

/// Run benchmarks across multiple array sizes
fn run_benchmark() {
    println!("\n\n====================================");
    println!("Running comprehensive benchmark...");
    println!("====================================\n");

    let sizes: Vec<usize> = vec![
        1 << 10, // 1K
        1 << 12, // 4K
        1 << 14, // 16K
        1 << 16, // 64K
        1 << 18, // 256K
        1 << 20, // 1M
        1 << 22, // 4M
        1 << 24, // 16M
    ];

    let gpu_bitonic_sorter = match gpu_sort::GpuSorter::new() {
        Ok(s) => Some(s),
        Err(e) => {
            println!("GPU Bitonic not available: {}", e);
            None
        }
    };

    let gpu_radix_sorter = match gpu_radix_sort::GpuRadixSorter::new() {
        Ok(s) => Some(s),
        Err(e) => {
            println!("GPU Radix not available: {}", e);
            None
        }
    };

    println!(
        "{:>12} | {:>12} | {:>14} | {:>14} | {:>12} | {:>12}",
        "Size", "CPU (ms)", "GPU Bitonic", "GPU Radix", "Bitonic/CPU", "Radix/CPU"
    );
    println!(
        "{:-<12}-+-{:-<12}-+-{:-<14}-+-{:-<14}-+-{:-<12}-+-{:-<12}",
        "", "", "", "", "", ""
    );

    let mut rng = rand::thread_rng();

    for &size in &sizes {
        let data: Vec<u32> = (0..size).map(|_| rng.gen()).collect();

        // CPU benchmark
        let mut cpu_data = data.clone();
        let cpu_start = Instant::now();
        cpu_sort::sort_unstable(&mut cpu_data);
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

        // GPU Bitonic benchmark
        let (bitonic_ms, bitonic_speedup) = if let Some(ref sorter) = gpu_bitonic_sorter {
            let mut gpu_data = data.clone();
            let gpu_start = Instant::now();
            if sorter.sort(&mut gpu_data).is_ok() && cpu_sort::is_sorted(&gpu_data) {
                let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;
                let speedup = cpu_ms / gpu_ms;
                (format!("{:.3}", gpu_ms), format!("{:.2}x", speedup))
            } else {
                ("ERROR".to_string(), "N/A".to_string())
            }
        } else {
            ("N/A".to_string(), "N/A".to_string())
        };

        // GPU Radix benchmark
        let (radix_ms, radix_speedup) = if let Some(ref sorter) = gpu_radix_sorter {
            let mut gpu_data = data.clone();
            let gpu_start = Instant::now();
            if sorter.sort(&mut gpu_data).is_ok() && cpu_sort::is_sorted(&gpu_data) {
                let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;
                let speedup = cpu_ms / gpu_ms;
                (format!("{:.3}", gpu_ms), format!("{:.2}x", speedup))
            } else {
                ("ERROR".to_string(), "N/A".to_string())
            }
        } else {
            ("N/A".to_string(), "N/A".to_string())
        };

        println!(
            "{:>12} | {:>12.3} | {:>14} | {:>14} | {:>12} | {:>12}",
            size, cpu_ms, bitonic_ms, radix_ms, bitonic_speedup, radix_speedup
        );
    }

    println!("\nNote: Speedup > 1.0x means GPU is faster than CPU");
    println!("      Speedup < 1.0x means CPU is faster than GPU");
}
