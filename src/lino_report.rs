//! Links Notation (Lino) Report Generator
//!
//! This module provides functionality to generate benchmark reports in Links Notation format.
//! Links Notation is a simple, intuitive format for representing structured data as links
//! between references to links.
//!
//! Format examples:
//! - Simple reference: `name`
//! - Link with id and values: `(id: value1 value2)`
//! - Nested structure using indentation:
//!   ```
//!   parent:
//!     child1
//!     child2
//!   ```

// Allow dead_code since this module provides a public API for external use
// (lino2md binary, future tools, etc.)
#![allow(dead_code)]

use std::fmt::Write;
use std::fs;
use std::io;
use std::path::Path;

/// Represents a single benchmark result for one algorithm at one array size
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the algorithm (e.g., "cpu_pdqsort", "gpu_radix", "cpu_radix")
    pub algorithm: String,
    /// Execution platform ("cpu" or "gpu")
    pub platform: String,
    /// Array size in elements
    pub array_size: usize,
    /// Execution time in milliseconds
    pub time_ms: f64,
    /// Whether the sort was verified correct
    pub verified: bool,
    /// Optional GPU device name
    pub device: Option<String>,
}

/// Represents a complete benchmark report
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Timestamp of when the benchmark was run (ISO 8601 format)
    pub timestamp: String,
    /// Description of the benchmark run
    pub description: String,
    /// System information
    pub system_info: SystemInfo,
    /// All benchmark results
    pub results: Vec<BenchmarkResult>,
}

/// System information for the benchmark report
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// Operating system name and version
    pub os: String,
    /// CPU model/name
    pub cpu: String,
    /// GPU model/name (if available)
    pub gpu: Option<String>,
    /// Amount of RAM in GB
    pub ram_gb: Option<f64>,
}

impl BenchmarkReport {
    /// Create a new empty benchmark report
    pub fn new(description: &str) -> Self {
        let timestamp = chrono_lite_timestamp();
        BenchmarkReport {
            timestamp,
            description: description.to_string(),
            system_info: SystemInfo::default(),
            results: Vec::new(),
        }
    }

    /// Add a benchmark result to the report
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Generate the report in Links Notation format
    pub fn to_lino(&self) -> String {
        let mut output = String::new();

        // Report metadata
        writeln!(output, "benchmark_report:").unwrap();
        writeln!(output, "  timestamp '{}'", self.timestamp).unwrap();
        writeln!(output, "  description '{}'", escape_lino_string(&self.description)).unwrap();

        // System information
        writeln!(output).unwrap();
        writeln!(output, "system_info:").unwrap();
        writeln!(output, "  os '{}'", escape_lino_string(&self.system_info.os)).unwrap();
        writeln!(output, "  cpu '{}'", escape_lino_string(&self.system_info.cpu)).unwrap();
        if let Some(ref gpu) = self.system_info.gpu {
            writeln!(output, "  gpu '{}'", escape_lino_string(gpu)).unwrap();
        }
        if let Some(ram) = self.system_info.ram_gb {
            writeln!(output, "  ram_gb {:.1}", ram).unwrap();
        }

        // Benchmark results grouped by array size
        writeln!(output).unwrap();
        writeln!(output, "results:").unwrap();

        // Group results by array size for better organization
        let mut sizes: Vec<usize> = self.results.iter().map(|r| r.array_size).collect();
        sizes.sort();
        sizes.dedup();

        for size in &sizes {
            writeln!(output, "  size_{}:", size).unwrap();
            for result in self.results.iter().filter(|r| r.array_size == *size) {
                writeln!(output, "    {}:", result.algorithm).unwrap();
                writeln!(output, "      platform {}", result.platform).unwrap();
                writeln!(output, "      time_ms {:.3}", result.time_ms).unwrap();
                writeln!(output, "      verified {}", result.verified).unwrap();
                if let Some(ref device) = result.device {
                    writeln!(output, "      device '{}'", escape_lino_string(device)).unwrap();
                }
            }
        }

        // Add computed comparisons
        writeln!(output).unwrap();
        writeln!(output, "comparisons:").unwrap();

        for size in &sizes {
            let size_results: Vec<&BenchmarkResult> = self
                .results
                .iter()
                .filter(|r| r.array_size == *size)
                .collect();

            if size_results.is_empty() {
                continue;
            }

            writeln!(output, "  size_{}:", size).unwrap();

            // GPU Radix vs CPU Radix (fair comparison)
            let gpu_radix = size_results.iter().find(|r| r.algorithm == "gpu_radix");
            let cpu_radix = size_results.iter().find(|r| r.algorithm == "cpu_radix");
            if let (Some(gpu), Some(cpu)) = (gpu_radix, cpu_radix) {
                let speedup = cpu.time_ms / gpu.time_ms;
                writeln!(output, "    gpu_radix_vs_cpu_radix:").unwrap();
                writeln!(output, "      speedup {:.2}", speedup).unwrap();
                writeln!(output, "      faster {}", if speedup > 1.0 { "gpu" } else { "cpu" }).unwrap();
            }

            // GPU Bitonic vs CPU Bitonic (fair comparison)
            let gpu_bitonic = size_results.iter().find(|r| r.algorithm == "gpu_bitonic");
            let cpu_bitonic = size_results.iter().find(|r| r.algorithm == "cpu_bitonic");
            if let (Some(gpu), Some(cpu)) = (gpu_bitonic, cpu_bitonic) {
                let speedup = cpu.time_ms / gpu.time_ms;
                writeln!(output, "    gpu_bitonic_vs_cpu_bitonic:").unwrap();
                writeln!(output, "      speedup {:.2}", speedup).unwrap();
                writeln!(output, "      faster {}", if speedup > 1.0 { "gpu" } else { "cpu" }).unwrap();
            }

            // GPU Radix vs CPU pdqsort (standard library comparison)
            let cpu_pdqsort = size_results.iter().find(|r| r.algorithm == "cpu_pdqsort");
            if let (Some(gpu), Some(cpu)) = (gpu_radix, cpu_pdqsort) {
                let speedup = cpu.time_ms / gpu.time_ms;
                writeln!(output, "    gpu_radix_vs_cpu_pdqsort:").unwrap();
                writeln!(output, "      speedup {:.2}", speedup).unwrap();
                writeln!(output, "      faster {}", if speedup > 1.0 { "gpu" } else { "cpu" }).unwrap();
            }

            // GPU Bitonic vs CPU pdqsort
            if let (Some(gpu), Some(cpu)) = (gpu_bitonic, cpu_pdqsort) {
                let speedup = cpu.time_ms / gpu.time_ms;
                writeln!(output, "    gpu_bitonic_vs_cpu_pdqsort:").unwrap();
                writeln!(output, "      speedup {:.2}", speedup).unwrap();
                writeln!(output, "      faster {}", if speedup > 1.0 { "gpu" } else { "cpu" }).unwrap();
            }

            // GPU Radix vs GPU Bitonic
            if let (Some(radix), Some(bitonic)) = (gpu_radix, gpu_bitonic) {
                let speedup = bitonic.time_ms / radix.time_ms;
                writeln!(output, "    gpu_radix_vs_gpu_bitonic:").unwrap();
                writeln!(output, "      speedup {:.2}", speedup).unwrap();
                writeln!(output, "      faster {}", if speedup > 1.0 { "radix" } else { "bitonic" }).unwrap();
            }

            // CPU Radix vs CPU pdqsort
            if let (Some(radix), Some(pdqsort)) = (cpu_radix, cpu_pdqsort) {
                let speedup = pdqsort.time_ms / radix.time_ms;
                writeln!(output, "    cpu_radix_vs_cpu_pdqsort:").unwrap();
                writeln!(output, "      speedup {:.2}", speedup).unwrap();
                writeln!(output, "      faster {}", if speedup > 1.0 { "radix" } else { "pdqsort" }).unwrap();
            }
        }

        output
    }

    /// Save the report to a file in Links Notation format
    pub fn save_lino(&self, path: &Path) -> io::Result<()> {
        let content = self.to_lino();
        fs::write(path, content)
    }

    /// Generate a markdown table from the report
    pub fn to_markdown_table(&self) -> String {
        let mut output = String::new();

        writeln!(output, "# GPU Sorting Benchmark Report").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "**Timestamp:** {}", self.timestamp).unwrap();
        writeln!(output, "**Description:** {}", self.description).unwrap();
        writeln!(output).unwrap();

        // System info
        writeln!(output, "## System Information").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "| Property | Value |").unwrap();
        writeln!(output, "|----------|-------|").unwrap();
        writeln!(output, "| OS | {} |", self.system_info.os).unwrap();
        writeln!(output, "| CPU | {} |", self.system_info.cpu).unwrap();
        if let Some(ref gpu) = self.system_info.gpu {
            writeln!(output, "| GPU | {} |", gpu).unwrap();
        }
        if let Some(ram) = self.system_info.ram_gb {
            writeln!(output, "| RAM | {:.1} GB |", ram).unwrap();
        }
        writeln!(output).unwrap();

        // Results table
        writeln!(output, "## Benchmark Results").unwrap();
        writeln!(output).unwrap();
        writeln!(
            output,
            "| Size | CPU pdqsort (ms) | CPU Radix (ms) | CPU Bitonic (ms) | GPU Radix (ms) | GPU Bitonic (ms) |"
        )
        .unwrap();
        writeln!(output, "|------|------------------|----------------|------------------|----------------|------------------|").unwrap();

        // Get unique sizes
        let mut sizes: Vec<usize> = self.results.iter().map(|r| r.array_size).collect();
        sizes.sort();
        sizes.dedup();

        for size in &sizes {
            let size_results: Vec<&BenchmarkResult> = self
                .results
                .iter()
                .filter(|r| r.array_size == *size)
                .collect();

            let cpu_pdqsort = size_results
                .iter()
                .find(|r| r.algorithm == "cpu_pdqsort")
                .map(|r| format!("{:.3}", r.time_ms))
                .unwrap_or_else(|| "N/A".to_string());

            let cpu_radix = size_results
                .iter()
                .find(|r| r.algorithm == "cpu_radix")
                .map(|r| format!("{:.3}", r.time_ms))
                .unwrap_or_else(|| "N/A".to_string());

            let cpu_bitonic = size_results
                .iter()
                .find(|r| r.algorithm == "cpu_bitonic")
                .map(|r| format!("{:.3}", r.time_ms))
                .unwrap_or_else(|| "N/A".to_string());

            let gpu_radix = size_results
                .iter()
                .find(|r| r.algorithm == "gpu_radix")
                .map(|r| format!("{:.3}", r.time_ms))
                .unwrap_or_else(|| "N/A".to_string());

            let gpu_bitonic = size_results
                .iter()
                .find(|r| r.algorithm == "gpu_bitonic")
                .map(|r| format!("{:.3}", r.time_ms))
                .unwrap_or_else(|| "N/A".to_string());

            writeln!(
                output,
                "| {} | {} | {} | {} | {} | {} |",
                format_size(*size),
                cpu_pdqsort,
                cpu_radix,
                cpu_bitonic,
                gpu_radix,
                gpu_bitonic
            )
            .unwrap();
        }

        writeln!(output).unwrap();

        // Fair comparisons table
        writeln!(output, "## Fair Algorithm Comparisons").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "### GPU vs CPU (Same Algorithm)").unwrap();
        writeln!(output).unwrap();
        writeln!(
            output,
            "| Size | Radix (GPU/CPU Speedup) | Bitonic (GPU/CPU Speedup) |"
        )
        .unwrap();
        writeln!(output, "|------|-------------------------|---------------------------|").unwrap();

        for size in &sizes {
            let size_results: Vec<&BenchmarkResult> = self
                .results
                .iter()
                .filter(|r| r.array_size == *size)
                .collect();

            let radix_speedup = {
                let gpu = size_results.iter().find(|r| r.algorithm == "gpu_radix");
                let cpu = size_results.iter().find(|r| r.algorithm == "cpu_radix");
                match (gpu, cpu) {
                    (Some(g), Some(c)) => {
                        let s = c.time_ms / g.time_ms;
                        if s > 1.0 {
                            format!("GPU {:.2}x faster", s)
                        } else {
                            format!("CPU {:.2}x faster", 1.0 / s)
                        }
                    }
                    _ => "N/A".to_string(),
                }
            };

            let bitonic_speedup = {
                let gpu = size_results.iter().find(|r| r.algorithm == "gpu_bitonic");
                let cpu = size_results.iter().find(|r| r.algorithm == "cpu_bitonic");
                match (gpu, cpu) {
                    (Some(g), Some(c)) => {
                        let s = c.time_ms / g.time_ms;
                        if s > 1.0 {
                            format!("GPU {:.2}x faster", s)
                        } else {
                            format!("CPU {:.2}x faster", 1.0 / s)
                        }
                    }
                    _ => "N/A".to_string(),
                }
            };

            writeln!(
                output,
                "| {} | {} | {} |",
                format_size(*size),
                radix_speedup,
                bitonic_speedup
            )
            .unwrap();
        }

        writeln!(output).unwrap();
        writeln!(output, "---").unwrap();
        writeln!(output, "*Report generated by gpu-sorting benchmark tool*").unwrap();

        output
    }

    /// Save the report as a markdown file
    pub fn save_markdown(&self, path: &Path) -> io::Result<()> {
        let content = self.to_markdown_table();
        fs::write(path, content)
    }
}

impl Default for SystemInfo {
    fn default() -> Self {
        SystemInfo {
            os: std::env::consts::OS.to_string(),
            cpu: "Unknown".to_string(),
            gpu: None,
            ram_gb: None,
        }
    }
}

/// Escape a string for use in Links Notation (handle single quotes)
fn escape_lino_string(s: &str) -> String {
    s.replace('\'', "\\'")
}

/// Format a size as a human-readable string (e.g., "1K", "1M")
fn format_size(size: usize) -> String {
    if size >= 1_000_000_000 {
        format!("{}G", size / 1_000_000_000)
    } else if size >= 1_000_000 {
        format!("{}M", size / 1_000_000)
    } else if size >= 1_000 {
        format!("{}K", size / 1_000)
    } else {
        size.to_string()
    }
}

/// Generate a simple timestamp without external dependencies
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let secs = duration.as_secs();

    // Calculate date/time components (simplified, not accounting for leap seconds)
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;

    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Calculate year, month, day from days since epoch
    // This is a simplified calculation
    let mut year = 1970i32;
    let mut remaining_days = days_since_epoch as i32;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let days_in_months: [i32; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for &days in &days_in_months {
        if remaining_days < days {
            break;
        }
        remaining_days -= days;
        month += 1;
    }

    let day = remaining_days + 1;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Parse a Links Notation report file and return benchmark results
/// This is a simplified parser for our specific report format
pub fn parse_lino_report(content: &str) -> Option<BenchmarkReport> {
    let mut report = BenchmarkReport::new("Parsed report");
    let mut current_size: Option<usize> = None;
    let mut _current_algorithm: Option<String> = None;
    let mut current_result: Option<BenchmarkResult> = None;
    let mut in_results_section = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Skip empty lines
        if trimmed.is_empty() {
            continue;
        }

        // Track which section we're in
        if trimmed == "results:" {
            in_results_section = true;
            continue;
        }
        if trimmed == "comparisons:" {
            // Save last result before leaving results section
            if let Some(result) = current_result.take() {
                report.results.push(result);
            }
            in_results_section = false;
            current_size = None;
            continue;
        }

        // Parse timestamp
        if trimmed.starts_with("timestamp '") {
            if let Some(ts) = extract_quoted_value(trimmed, "timestamp") {
                report.timestamp = ts;
            }
        }

        // Parse description
        if trimmed.starts_with("description '") {
            if let Some(desc) = extract_quoted_value(trimmed, "description") {
                report.description = desc;
            }
        }

        // Only parse results when in results section
        if !in_results_section {
            continue;
        }

        // Parse size blocks
        if trimmed.starts_with("size_") && trimmed.ends_with(':') {
            // Save previous result if any
            if let Some(result) = current_result.take() {
                report.results.push(result);
            }

            let size_str = trimmed
                .trim_start_matches("size_")
                .trim_end_matches(':');
            current_size = size_str.parse().ok();
            _current_algorithm = None;
        }

        // Parse algorithm blocks
        if let Some(size) = current_size {
            if (trimmed.starts_with("cpu_") || trimmed.starts_with("gpu_"))
                && trimmed.ends_with(':')
                && !trimmed.contains(' ')
            {
                // Save previous result if any
                if let Some(result) = current_result.take() {
                    report.results.push(result);
                }

                let algorithm = trimmed.trim_end_matches(':').to_string();
                _current_algorithm = Some(algorithm.clone());
                current_result = Some(BenchmarkResult {
                    algorithm,
                    platform: if trimmed.starts_with("cpu_") {
                        "cpu"
                    } else {
                        "gpu"
                    }
                    .to_string(),
                    array_size: size,
                    time_ms: 0.0,
                    verified: false,
                    device: None,
                });
            }
        }

        // Parse result properties
        if let Some(ref mut result) = current_result {
            if trimmed.starts_with("platform ") {
                result.platform = trimmed.trim_start_matches("platform ").to_string();
            } else if trimmed.starts_with("time_ms ") {
                if let Ok(time) = trimmed.trim_start_matches("time_ms ").parse() {
                    result.time_ms = time;
                }
            } else if trimmed.starts_with("verified ") {
                result.verified = trimmed.trim_start_matches("verified ") == "true";
            } else if trimmed.starts_with("device '") {
                result.device = extract_quoted_value(trimmed, "device");
            }
        }
    }

    // Don't forget the last result
    if let Some(result) = current_result {
        report.results.push(result);
    }

    if report.results.is_empty() {
        None
    } else {
        Some(report)
    }
}

fn extract_quoted_value(line: &str, prefix: &str) -> Option<String> {
    let after_prefix = line.trim_start_matches(prefix).trim();
    if after_prefix.starts_with('\'') && after_prefix.ends_with('\'') {
        Some(
            after_prefix[1..after_prefix.len() - 1]
                .replace("\\'", "'")
                .to_string(),
        )
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result_creation() {
        let result = BenchmarkResult {
            algorithm: "cpu_radix".to_string(),
            platform: "cpu".to_string(),
            array_size: 1_000_000,
            time_ms: 42.5,
            verified: true,
            device: None,
        };
        assert_eq!(result.algorithm, "cpu_radix");
        assert_eq!(result.time_ms, 42.5);
    }

    #[test]
    fn test_report_to_lino() {
        let mut report = BenchmarkReport::new("Test benchmark");
        report.add_result(BenchmarkResult {
            algorithm: "cpu_radix".to_string(),
            platform: "cpu".to_string(),
            array_size: 1024,
            time_ms: 1.5,
            verified: true,
            device: None,
        });

        let lino = report.to_lino();
        assert!(lino.contains("benchmark_report:"));
        assert!(lino.contains("cpu_radix:"));
        assert!(lino.contains("time_ms 1.500"));
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500");
        assert_eq!(format_size(1000), "1K");
        assert_eq!(format_size(1024), "1K");
        assert_eq!(format_size(1_000_000), "1M");
        assert_eq!(format_size(1_048_576), "1M");
    }

    #[test]
    fn test_escape_lino_string() {
        assert_eq!(escape_lino_string("hello"), "hello");
        assert_eq!(escape_lino_string("it's"), "it\\'s");
    }

    #[test]
    fn test_parse_lino_roundtrip() {
        let mut report = BenchmarkReport::new("Test benchmark");
        report.add_result(BenchmarkResult {
            algorithm: "cpu_radix".to_string(),
            platform: "cpu".to_string(),
            array_size: 1024,
            time_ms: 1.5,
            verified: true,
            device: None,
        });
        report.add_result(BenchmarkResult {
            algorithm: "gpu_radix".to_string(),
            platform: "gpu".to_string(),
            array_size: 1024,
            time_ms: 0.8,
            verified: true,
            device: Some("Apple M3 Pro".to_string()),
        });

        let lino = report.to_lino();
        let parsed = parse_lino_report(&lino).unwrap();

        assert_eq!(parsed.results.len(), 2);
        assert_eq!(parsed.results[0].algorithm, "cpu_radix");
        assert_eq!(parsed.results[1].algorithm, "gpu_radix");
    }
}
