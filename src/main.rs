//! GPU Sorting Proof of Concept
//!
//! This application compares CPU and GPU sorting performance on Apple Silicon (M1-M3).
//! It implements multiple sorting algorithms for fair comparison:
//!
//! **GPU Algorithms:**
//! - **Bitonic Sort**: O(n log²n) - comparison-based, requires power-of-2 sizes
//! - **Radix Sort**: O(n) - linear time using DeviceRadixSort algorithm
//! - **Radix Sort (SIMD)**: O(n) - SIMD-optimized scatter for faster performance
//!
//! **CPU Algorithms:**
//! - **pdqsort**: O(n log n) - Rust's standard library unstable sort
//! - **Parallel pdqsort**: O(n log n) - Multi-threaded using rayon
//! - **Radix Sort**: O(n) - for fair comparison with GPU radix sort
//! - **Parallel Radix Sort**: O(n) - Multi-threaded for fair parallel comparison
//! - **Bitonic Sort**: O(n log²n) - for fair comparison with GPU bitonic sort
//!
//! The benchmark generates reports in Links Notation format for analysis.

mod cpu_bitonic_sort;
mod cpu_parallel_sort;
mod cpu_radix_sort;
mod cpu_sort;
mod gpu_radix_sort;
mod gpu_sort;
mod lino_report;

use lino_report::{BenchmarkReport, BenchmarkResult, SystemInfo};
use rand::Rng;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Default array size for benchmarking
const DEFAULT_ARRAY_SIZE: usize = 1 << 20; // 1 million elements

fn main() {
    println!("GPU Sorting Proof of Concept");
    println!("=============================\n");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let array_size = if args.len() > 1 {
        args[1].parse().unwrap_or(DEFAULT_ARRAY_SIZE)
    } else {
        DEFAULT_ARRAY_SIZE
    };

    // Check for benchmark mode
    let benchmark_mode = args.iter().any(|a| a == "--benchmark");
    let generate_report = args.iter().any(|a| a == "--report");

    // For bitonic sort, we need power of 2
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

    // ========== CPU Sorting (pdqsort - standard library) ==========
    println!("\n--- CPU Sorting (std::sort unstable / pdqsort) ---");
    let mut cpu_data = data.clone();
    let cpu_start = Instant::now();
    cpu_sort::sort_unstable(&mut cpu_data);
    let cpu_duration = cpu_start.elapsed();
    println!(
        "CPU pdqsort time: {:.3} ms",
        cpu_duration.as_secs_f64() * 1000.0
    );
    assert!(cpu_sort::is_sorted(&cpu_data), "CPU pdqsort failed!");
    println!("CPU pdqsort verified: OK");

    // ========== CPU Radix Sort ==========
    println!("\n--- CPU Sorting (Radix Sort) ---");
    let mut cpu_radix_data = data.clone();
    let cpu_radix_start = Instant::now();
    cpu_radix_sort::sort(&mut cpu_radix_data);
    let cpu_radix_duration = cpu_radix_start.elapsed();
    println!(
        "CPU radix sort time: {:.3} ms",
        cpu_radix_duration.as_secs_f64() * 1000.0
    );
    assert!(
        cpu_radix_sort::is_sorted(&cpu_radix_data),
        "CPU radix sort failed!"
    );
    println!("CPU radix sort verified: OK");

    // ========== CPU Parallel Sort (rayon) ==========
    println!("\n--- CPU Sorting (Parallel pdqsort with rayon) ---");
    let mut cpu_parallel_data = data.clone();
    let cpu_parallel_start = Instant::now();
    cpu_parallel_sort::parallel_sort(&mut cpu_parallel_data);
    let cpu_parallel_duration = cpu_parallel_start.elapsed();
    println!(
        "CPU parallel sort time: {:.3} ms",
        cpu_parallel_duration.as_secs_f64() * 1000.0
    );
    assert!(
        cpu_parallel_sort::is_sorted(&cpu_parallel_data),
        "CPU parallel sort failed!"
    );
    println!("CPU parallel sort verified: OK");

    // ========== CPU Parallel Radix Sort ==========
    println!("\n--- CPU Sorting (Parallel Radix Sort) ---");
    let mut cpu_parallel_radix_data = data.clone();
    let cpu_parallel_radix_start = Instant::now();
    cpu_parallel_sort::parallel_radix_sort(&mut cpu_parallel_radix_data);
    let cpu_parallel_radix_duration = cpu_parallel_radix_start.elapsed();
    println!(
        "CPU parallel radix sort time: {:.3} ms",
        cpu_parallel_radix_duration.as_secs_f64() * 1000.0
    );
    assert!(
        cpu_parallel_sort::is_sorted(&cpu_parallel_radix_data),
        "CPU parallel radix sort failed!"
    );
    println!("CPU parallel radix sort verified: OK");

    // ========== CPU Bitonic Sort ==========
    println!("\n--- CPU Sorting (Bitonic Sort) ---");
    let mut cpu_bitonic_data = data_bitonic.clone();
    let cpu_bitonic_start = Instant::now();
    cpu_bitonic_sort::sort(&mut cpu_bitonic_data);
    let cpu_bitonic_duration = cpu_bitonic_start.elapsed();
    println!(
        "CPU bitonic sort time: {:.3} ms",
        cpu_bitonic_duration.as_secs_f64() * 1000.0
    );
    assert!(
        cpu_bitonic_sort::is_sorted(&cpu_bitonic_data),
        "CPU bitonic sort failed!"
    );
    println!("CPU bitonic sort verified: OK");

    // ========== GPU Bitonic Sorting ==========
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

    // ========== GPU Radix Sorting ==========
    println!("\n--- GPU Sorting (Metal Radix Sort - DeviceRadixSort) ---");
    let (gpu_radix_duration, gpu_device_name) = match gpu_radix_sort::GpuRadixSorter::new() {
        Ok(sorter) => {
            let device_name = sorter.device_name();
            println!("Using GPU: {}", device_name);
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
                    (Some(gpu_duration), device_name)
                }
                Err(e) => {
                    println!("GPU radix sort error: {}", e);
                    (None, device_name)
                }
            }
        }
        Err(e) => {
            println!("Failed to initialize GPU radix sorter: {}", e);
            (None, "Unknown".to_string())
        }
    };

    // ========== GPU Radix Sorting (SIMD-Optimized) ==========
    println!("\n--- GPU Sorting (Metal Radix Sort - SIMD Optimized) ---");
    let gpu_radix_simd_duration = match gpu_radix_sort::GpuRadixSorter::new_simd() {
        Ok(sorter) => {
            println!("Using GPU: {} (SIMD scatter)", sorter.device_name());
            let mut gpu_data = data.clone();
            let gpu_start = Instant::now();
            match sorter.sort(&mut gpu_data) {
                Ok(()) => {
                    let gpu_duration = gpu_start.elapsed();
                    println!(
                        "GPU radix sort (SIMD) time: {:.3} ms",
                        gpu_duration.as_secs_f64() * 1000.0
                    );

                    if cpu_sort::is_sorted(&gpu_data) {
                        println!("GPU radix sort (SIMD) verified: OK");

                        // Compare results with CPU
                        if gpu_data == cpu_data {
                            println!("Results match CPU sort: OK");
                        } else {
                            println!("WARNING: Results differ from CPU sort!");
                        }
                    } else {
                        println!("ERROR: GPU radix sort (SIMD) failed verification!");
                    }
                    Some(gpu_duration)
                }
                Err(e) => {
                    println!("GPU radix sort (SIMD) error: {}", e);
                    None
                }
            }
        }
        Err(e) => {
            println!("Failed to initialize GPU radix sorter (SIMD): {}", e);
            None
        }
    };

    // ========== Performance Comparison ==========
    println!("\n--- Performance Comparison ---");

    let cpu_ms = cpu_duration.as_secs_f64() * 1000.0;
    let cpu_radix_ms = cpu_radix_duration.as_secs_f64() * 1000.0;
    let cpu_parallel_ms = cpu_parallel_duration.as_secs_f64() * 1000.0;
    let cpu_parallel_radix_ms = cpu_parallel_radix_duration.as_secs_f64() * 1000.0;
    let cpu_bitonic_ms = cpu_bitonic_duration.as_secs_f64() * 1000.0;

    // Fair comparisons first
    println!("\n[Fair Algorithm Comparisons]");

    // GPU Radix vs CPU Radix (single-threaded)
    if let Some(gpu_radix_dur) = gpu_radix_duration {
        let gpu_radix_ms = gpu_radix_dur.as_secs_f64() * 1000.0;
        let speedup = cpu_radix_ms / gpu_radix_ms;
        if speedup > 1.0 {
            println!(
                "GPU Radix vs CPU Radix (single): GPU is {:.2}x faster",
                speedup
            );
        } else {
            println!(
                "GPU Radix vs CPU Radix (single): CPU is {:.2}x faster",
                1.0 / speedup
            );
        }
    }

    // GPU Radix vs CPU Parallel Radix
    if let Some(gpu_radix_dur) = gpu_radix_duration {
        let gpu_radix_ms = gpu_radix_dur.as_secs_f64() * 1000.0;
        let speedup = cpu_parallel_radix_ms / gpu_radix_ms;
        if speedup > 1.0 {
            println!(
                "GPU Radix vs CPU Radix (parallel): GPU is {:.2}x faster",
                speedup
            );
        } else {
            println!(
                "GPU Radix vs CPU Radix (parallel): CPU is {:.2}x faster",
                1.0 / speedup
            );
        }
    }

    // GPU Radix SIMD vs CPU Parallel Radix
    if let Some(gpu_radix_simd_dur) = gpu_radix_simd_duration {
        let gpu_radix_simd_ms = gpu_radix_simd_dur.as_secs_f64() * 1000.0;
        let speedup = cpu_parallel_radix_ms / gpu_radix_simd_ms;
        if speedup > 1.0 {
            println!(
                "GPU Radix (SIMD) vs CPU Radix (parallel): GPU is {:.2}x faster",
                speedup
            );
        } else {
            println!(
                "GPU Radix (SIMD) vs CPU Radix (parallel): CPU is {:.2}x faster",
                1.0 / speedup
            );
        }
    }

    // GPU Bitonic vs CPU Bitonic
    if let Some(gpu_bitonic_dur) = gpu_bitonic_duration {
        let gpu_bitonic_ms = gpu_bitonic_dur.as_secs_f64() * 1000.0;
        let speedup = cpu_bitonic_ms / gpu_bitonic_ms;
        if speedup > 1.0 {
            println!("GPU Bitonic vs CPU Bitonic: GPU is {:.2}x faster", speedup);
        } else {
            println!(
                "GPU Bitonic vs CPU Bitonic: CPU is {:.2}x faster",
                1.0 / speedup
            );
        }
    }

    println!("\n[Cross-Algorithm Comparisons]");

    // GPU vs CPU standard library
    if let Some(bitonic_dur) = gpu_bitonic_duration {
        let bitonic_ms = bitonic_dur.as_secs_f64() * 1000.0;
        let speedup = cpu_ms / bitonic_ms;
        if speedup > 1.0 {
            println!("GPU Bitonic vs CPU pdqsort: GPU is {:.2}x faster", speedup);
        } else {
            println!(
                "GPU Bitonic vs CPU pdqsort: CPU is {:.2}x faster",
                1.0 / speedup
            );
        }
    }

    if let Some(radix_dur) = gpu_radix_duration {
        let radix_ms = radix_dur.as_secs_f64() * 1000.0;
        let speedup = cpu_ms / radix_ms;
        if speedup > 1.0 {
            println!("GPU Radix vs CPU pdqsort: GPU is {:.2}x faster", speedup);
        } else {
            println!(
                "GPU Radix vs CPU pdqsort: CPU is {:.2}x faster",
                1.0 / speedup
            );
        }
    }

    // GPU Radix vs CPU Parallel
    if let Some(radix_dur) = gpu_radix_duration {
        let radix_ms = radix_dur.as_secs_f64() * 1000.0;
        let speedup = cpu_parallel_ms / radix_ms;
        if speedup > 1.0 {
            println!(
                "GPU Radix vs CPU Parallel pdqsort: GPU is {:.2}x faster",
                speedup
            );
        } else {
            println!(
                "GPU Radix vs CPU Parallel pdqsort: CPU is {:.2}x faster",
                1.0 / speedup
            );
        }
    }

    // CPU algorithm comparisons
    println!("\n[CPU Algorithm Comparisons]");
    let pdq_vs_radix = cpu_ms / cpu_radix_ms;
    if pdq_vs_radix > 1.0 {
        println!(
            "CPU Radix vs CPU pdqsort: Radix is {:.2}x faster",
            pdq_vs_radix
        );
    } else {
        println!(
            "CPU Radix vs CPU pdqsort: pdqsort is {:.2}x faster",
            1.0 / pdq_vs_radix
        );
    }

    // Parallel speedup
    let parallel_speedup = cpu_ms / cpu_parallel_ms;
    if parallel_speedup > 1.0 {
        println!(
            "CPU Parallel vs CPU Single: Parallel is {:.2}x faster",
            parallel_speedup
        );
    } else {
        println!(
            "CPU Parallel vs CPU Single: Single is {:.2}x faster",
            1.0 / parallel_speedup
        );
    }

    // GPU algorithm comparisons
    println!("\n[GPU Algorithm Comparisons]");
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

    // SIMD vs basic scatter comparison
    if let (Some(radix_dur), Some(simd_dur)) = (gpu_radix_duration, gpu_radix_simd_duration) {
        let speedup = radix_dur.as_secs_f64() / simd_dur.as_secs_f64();
        if speedup > 1.0 {
            println!(
                "GPU Radix (SIMD) vs GPU Radix (basic): SIMD is {:.2}x faster",
                speedup
            );
        } else {
            println!(
                "GPU Radix (SIMD) vs GPU Radix (basic): Basic is {:.2}x faster",
                1.0 / speedup
            );
        }
    }

    // Run comprehensive benchmark or generate report
    if benchmark_mode || generate_report {
        run_comprehensive_benchmark(generate_report, &gpu_device_name);
    }
}

/// Run benchmarks across multiple array sizes and optionally generate reports
fn run_comprehensive_benchmark(generate_report: bool, gpu_device: &str) {
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

    // Create report
    let mut report = BenchmarkReport::new("Comprehensive GPU vs CPU sorting benchmark");
    report.system_info = SystemInfo {
        os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        cpu: "Unknown (run on target machine)".to_string(),
        gpu: Some(gpu_device.to_string()),
        ram_gb: None,
    };

    println!(
        "{:>12} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Size", "CPU pdq", "CPU Radix", "CPU Bit", "GPU Radix", "GPU Bit"
    );
    println!(
        "{:-<12}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}",
        "", "", "", "", "", ""
    );

    let mut rng = rand::thread_rng();

    for &size in &sizes {
        let data: Vec<u32> = (0..size).map(|_| rng.gen()).collect();

        // CPU pdqsort benchmark
        let mut cpu_data = data.clone();
        let cpu_start = Instant::now();
        cpu_sort::sort_unstable(&mut cpu_data);
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

        report.add_result(BenchmarkResult {
            algorithm: "cpu_pdqsort".to_string(),
            platform: "cpu".to_string(),
            array_size: size,
            time_ms: cpu_ms,
            verified: cpu_sort::is_sorted(&cpu_data),
            device: None,
        });

        // CPU Radix benchmark
        let mut cpu_radix_data = data.clone();
        let cpu_radix_start = Instant::now();
        cpu_radix_sort::sort(&mut cpu_radix_data);
        let cpu_radix_ms = cpu_radix_start.elapsed().as_secs_f64() * 1000.0;

        report.add_result(BenchmarkResult {
            algorithm: "cpu_radix".to_string(),
            platform: "cpu".to_string(),
            array_size: size,
            time_ms: cpu_radix_ms,
            verified: cpu_radix_sort::is_sorted(&cpu_radix_data),
            device: None,
        });

        // CPU Bitonic benchmark (power of 2 size)
        let mut cpu_bitonic_data = data.clone();
        let cpu_bitonic_start = Instant::now();
        cpu_bitonic_sort::sort(&mut cpu_bitonic_data);
        let cpu_bitonic_ms = cpu_bitonic_start.elapsed().as_secs_f64() * 1000.0;

        report.add_result(BenchmarkResult {
            algorithm: "cpu_bitonic".to_string(),
            platform: "cpu".to_string(),
            array_size: size,
            time_ms: cpu_bitonic_ms,
            verified: cpu_bitonic_sort::is_sorted(&cpu_bitonic_data),
            device: None,
        });

        // GPU Radix benchmark
        let (gpu_radix_ms_str, _gpu_radix_ms) = if let Some(ref sorter) = gpu_radix_sorter {
            let mut gpu_data = data.clone();
            let gpu_start = Instant::now();
            if sorter.sort(&mut gpu_data).is_ok() && cpu_sort::is_sorted(&gpu_data) {
                let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;
                report.add_result(BenchmarkResult {
                    algorithm: "gpu_radix".to_string(),
                    platform: "gpu".to_string(),
                    array_size: size,
                    time_ms: gpu_ms,
                    verified: true,
                    device: Some(gpu_device.to_string()),
                });
                (format!("{:.3}", gpu_ms), Some(gpu_ms))
            } else {
                ("ERROR".to_string(), None)
            }
        } else {
            ("N/A".to_string(), None)
        };

        // GPU Bitonic benchmark
        let (gpu_bitonic_ms_str, _gpu_bitonic_ms) = if let Some(ref sorter) = gpu_bitonic_sorter {
            let mut gpu_data = data.clone();
            let gpu_start = Instant::now();
            if sorter.sort(&mut gpu_data).is_ok() && cpu_sort::is_sorted(&gpu_data) {
                let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;
                report.add_result(BenchmarkResult {
                    algorithm: "gpu_bitonic".to_string(),
                    platform: "gpu".to_string(),
                    array_size: size,
                    time_ms: gpu_ms,
                    verified: true,
                    device: Some(gpu_device.to_string()),
                });
                (format!("{:.3}", gpu_ms), Some(gpu_ms))
            } else {
                ("ERROR".to_string(), None)
            }
        } else {
            ("N/A".to_string(), None)
        };

        println!(
            "{:>12} | {:>10.3} | {:>10.3} | {:>10.3} | {:>10} | {:>10}",
            size, cpu_ms, cpu_radix_ms, cpu_bitonic_ms, gpu_radix_ms_str, gpu_bitonic_ms_str
        );
    }

    println!("\n");

    // Print fair comparison summary
    println!("=== Fair Algorithm Comparisons (Same Algorithm, GPU vs CPU) ===\n");
    println!(
        "{:>12} | {:>20} | {:>22}",
        "Size", "Radix (GPU/CPU)", "Bitonic (GPU/CPU)"
    );
    println!("{:-<12}-+-{:-<20}-+-{:-<22}", "", "", "");

    for &size in &sizes {
        let size_results: Vec<&BenchmarkResult> = report
            .results
            .iter()
            .filter(|r| r.array_size == size)
            .collect();

        let radix_comparison = {
            let gpu = size_results.iter().find(|r| r.algorithm == "gpu_radix");
            let cpu = size_results.iter().find(|r| r.algorithm == "cpu_radix");
            match (gpu, cpu) {
                (Some(g), Some(c)) => {
                    let speedup = c.time_ms / g.time_ms;
                    if speedup > 1.0 {
                        format!("GPU {:.2}x faster", speedup)
                    } else {
                        format!("CPU {:.2}x faster", 1.0 / speedup)
                    }
                }
                _ => "N/A".to_string(),
            }
        };

        let bitonic_comparison = {
            let gpu = size_results.iter().find(|r| r.algorithm == "gpu_bitonic");
            let cpu = size_results.iter().find(|r| r.algorithm == "cpu_bitonic");
            match (gpu, cpu) {
                (Some(g), Some(c)) => {
                    let speedup = c.time_ms / g.time_ms;
                    if speedup > 1.0 {
                        format!("GPU {:.2}x faster", speedup)
                    } else {
                        format!("CPU {:.2}x faster", 1.0 / speedup)
                    }
                }
                _ => "N/A".to_string(),
            }
        };

        println!(
            "{:>12} | {:>20} | {:>22}",
            size, radix_comparison, bitonic_comparison
        );
    }

    // Generate and save reports if requested
    if generate_report {
        // Create data/reports directory
        let reports_dir = Path::new("data/reports");
        if let Err(e) = fs::create_dir_all(reports_dir) {
            eprintln!("Failed to create reports directory: {}", e);
            return;
        }

        // Generate timestamp for filename
        let timestamp = report.timestamp.replace([':', '-', 'T', 'Z'], "_");

        // Save Lino report
        let lino_path = reports_dir.join(format!("benchmark_{}.lino", timestamp));
        match report.save_lino(&lino_path) {
            Ok(()) => println!("\nLinks Notation report saved to: {}", lino_path.display()),
            Err(e) => eprintln!("Failed to save Lino report: {}", e),
        }

        // Save Markdown report
        let md_path = reports_dir.join(format!("benchmark_{}.md", timestamp));
        match report.save_markdown(&md_path) {
            Ok(()) => println!("Markdown report saved to: {}", md_path.display()),
            Err(e) => eprintln!("Failed to save Markdown report: {}", e),
        }
    }

    println!("\nNote: Speedup > 1.0x means GPU is faster than CPU");
    println!("      Speedup < 1.0x means CPU is faster than GPU");
}
