//! GPU Sorting Proof of Concept
//!
//! This application compares CPU and GPU sorting performance on Apple Silicon (M1-M3).
//! It uses the Metal framework for GPU compute and implements bitonic sort.

mod cpu_sort;
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

    // Ensure array size is a power of 2 for bitonic sort
    let array_size = array_size.next_power_of_two();

    println!(
        "Array size: {} elements ({} MB)",
        array_size,
        array_size * 4 / 1_000_000
    );

    // Generate random data
    println!("\nGenerating random data...");
    let mut rng = rand::thread_rng();
    let data: Vec<u32> = (0..array_size).map(|_| rng.gen()).collect();

    // CPU Sorting
    println!("\n--- CPU Sorting (std::sort unstable) ---");
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

    // GPU Sorting
    println!("\n--- GPU Sorting (Metal Bitonic Sort) ---");
    match gpu_sort::GpuSorter::new() {
        Ok(sorter) => {
            let mut gpu_data = data.clone();
            let gpu_start = Instant::now();
            match sorter.sort(&mut gpu_data) {
                Ok(()) => {
                    let gpu_duration = gpu_start.elapsed();
                    println!(
                        "GPU sort time: {:.3} ms",
                        gpu_duration.as_secs_f64() * 1000.0
                    );

                    // Verify GPU sort
                    if cpu_sort::is_sorted(&gpu_data) {
                        println!("GPU sort verified: OK");

                        // Compare results
                        if gpu_data == cpu_data {
                            println!("Results match CPU sort: OK");
                        } else {
                            println!("WARNING: Results differ from CPU sort!");
                        }

                        // Performance comparison
                        println!("\n--- Performance Comparison ---");
                        let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
                        if speedup > 1.0 {
                            println!("GPU is {:.2}x faster than CPU", speedup);
                        } else {
                            println!("CPU is {:.2}x faster than GPU", 1.0 / speedup);
                        }
                    } else {
                        println!("ERROR: GPU sort failed verification!");
                    }
                }
                Err(e) => {
                    println!("GPU sort error: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Failed to initialize GPU sorter: {}", e);
            println!(
                "Note: GPU sorting requires macOS with Metal support (Apple Silicon recommended)"
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

    let gpu_sorter = match gpu_sort::GpuSorter::new() {
        Ok(s) => Some(s),
        Err(e) => {
            println!("GPU not available: {}", e);
            None
        }
    };

    println!(
        "{:>12} | {:>12} | {:>12} | {:>10}",
        "Size", "CPU (ms)", "GPU (ms)", "Speedup"
    );
    println!("{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<10}", "", "", "", "");

    let mut rng = rand::thread_rng();

    for &size in &sizes {
        let data: Vec<u32> = (0..size).map(|_| rng.gen()).collect();

        // CPU benchmark
        let mut cpu_data = data.clone();
        let cpu_start = Instant::now();
        cpu_sort::sort_unstable(&mut cpu_data);
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

        // GPU benchmark
        let (gpu_ms, speedup) = if let Some(ref sorter) = gpu_sorter {
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
            "{:>12} | {:>12.3} | {:>12} | {:>10}",
            size, cpu_ms, gpu_ms, speedup
        );
    }
}
