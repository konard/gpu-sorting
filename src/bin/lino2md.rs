//! Lino to Markdown Converter
//!
//! A simple CLI tool to convert Links Notation benchmark reports to Markdown format.
//!
//! Usage:
//!   lino2md input.lino [output.md]
//!
//! If output is not specified, prints to stdout.

use std::env;
use std::fs;
use std::path::Path;

// Include the main library modules
#[path = "../lino_report.rs"]
mod lino_report;

use lino_report::parse_lino_report;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input.lino> [output.md]", args[0]);
        eprintln!();
        eprintln!("Converts a Links Notation benchmark report to Markdown format.");
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  input.lino    Path to the Links Notation report file");
        eprintln!("  output.md     Optional output path (prints to stdout if not specified)");
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);

    // Read input file
    let content = match fs::read_to_string(input_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading input file: {}", e);
            std::process::exit(1);
        }
    };

    // Parse the Lino report
    let report = match parse_lino_report(&content) {
        Some(r) => r,
        None => {
            eprintln!("Error: Could not parse the Links Notation report");
            eprintln!("Make sure the file contains valid benchmark data");
            std::process::exit(1);
        }
    };

    // Generate markdown
    let markdown = report.to_markdown_table();

    // Output
    if args.len() >= 3 {
        let output_path = Path::new(&args[2]);
        match fs::write(output_path, &markdown) {
            Ok(()) => {
                println!("Markdown report written to: {}", output_path.display());
            }
            Err(e) => {
                eprintln!("Error writing output file: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        print!("{}", markdown);
    }
}
