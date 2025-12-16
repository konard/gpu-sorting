//! CPU Radix Sort Implementation
//!
//! This module provides a CPU-based radix sort implementation for fair comparison
//! with GPU radix sort. It uses LSD (Least Significant Digit) radix sort with
//! counting sort as the stable sort for each digit position.
//!
//! Complexity: O(n * k) where k = 4 (number of passes for 32-bit integers with 8-bit digits)

/// Number of bits per digit (radix)
const RADIX_BITS: usize = 8;
/// Number of buckets (2^RADIX_BITS)
const NUM_BUCKETS: usize = 1 << RADIX_BITS;
/// Mask for extracting a digit
const RADIX_MASK: u32 = (NUM_BUCKETS - 1) as u32;
/// Number of passes needed for 32-bit integers
const NUM_PASSES: usize = 32 / RADIX_BITS;

/// Sort a slice in-place using LSD radix sort.
///
/// This implementation uses 8-bit digits (256 buckets) and performs 4 passes
/// for 32-bit unsigned integers. It achieves O(n) complexity for fixed-width integers.
///
/// # Arguments
/// * `data` - The slice to sort in-place
pub fn sort(data: &mut [u32]) {
    if data.len() <= 1 {
        return;
    }

    let n = data.len();
    let mut temp = vec![0u32; n];
    let mut histogram = [0usize; NUM_BUCKETS];

    // Process each digit position from least significant to most significant
    for pass in 0..NUM_PASSES {
        let shift = pass * RADIX_BITS;

        // Build histogram (count occurrences of each digit)
        histogram.fill(0);
        for &value in data.iter() {
            let digit = ((value >> shift) & RADIX_MASK) as usize;
            histogram[digit] += 1;
        }

        // Convert histogram to prefix sums (exclusive scan)
        let mut sum = 0usize;
        for count in histogram.iter_mut() {
            let c = *count;
            *count = sum;
            sum += c;
        }

        // Scatter elements to their sorted positions
        for &value in data.iter() {
            let digit = ((value >> shift) & RADIX_MASK) as usize;
            let pos = histogram[digit];
            temp[pos] = value;
            histogram[digit] += 1;
        }

        // Copy back to data
        data.copy_from_slice(&temp);
    }
}

/// Check if a slice is sorted in ascending order.
#[inline]
pub fn is_sorted(data: &[u32]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_sort_empty() {
        let mut data: Vec<u32> = vec![];
        sort(&mut data);
        assert!(is_sorted(&data));
    }

    #[test]
    fn test_sort_single() {
        let mut data = vec![42u32];
        sort(&mut data);
        assert!(is_sorted(&data));
    }

    #[test]
    fn test_sort_sorted() {
        let mut data: Vec<u32> = (0..100).collect();
        sort(&mut data);
        assert!(is_sorted(&data));
        assert_eq!(data, (0..100).collect::<Vec<u32>>());
    }

    #[test]
    fn test_sort_reverse() {
        let mut data: Vec<u32> = (0..100).rev().collect();
        sort(&mut data);
        assert!(is_sorted(&data));
        assert_eq!(data, (0..100).collect::<Vec<u32>>());
    }

    #[test]
    fn test_sort_random() {
        let mut rng = rand::thread_rng();
        let mut data: Vec<u32> = (0..1000).map(|_| rng.gen()).collect();
        sort(&mut data);
        assert!(is_sorted(&data));
    }

    #[test]
    fn test_sort_duplicates() {
        let mut data = vec![5, 3, 5, 1, 3, 5, 1, 1];
        sort(&mut data);
        assert!(is_sorted(&data));
        assert_eq!(data, vec![1, 1, 1, 3, 3, 5, 5, 5]);
    }

    #[test]
    fn test_sort_all_same() {
        let mut data = vec![42u32; 100];
        sort(&mut data);
        assert!(is_sorted(&data));
        assert!(data.iter().all(|&x| x == 42));
    }

    #[test]
    fn test_sort_max_values() {
        let mut data = vec![u32::MAX, 0, u32::MAX / 2, 1, u32::MAX - 1];
        sort(&mut data);
        assert!(is_sorted(&data));
        assert_eq!(data, vec![0, 1, u32::MAX / 2, u32::MAX - 1, u32::MAX]);
    }

    #[test]
    fn test_sort_large_array() {
        let mut rng = rand::thread_rng();
        let mut data: Vec<u32> = (0..100_000).map(|_| rng.gen()).collect();
        let mut expected = data.clone();
        expected.sort();

        sort(&mut data);
        assert!(is_sorted(&data));
        assert_eq!(data, expected);
    }

    #[test]
    fn test_is_sorted() {
        assert!(is_sorted(&[1, 2, 3, 4, 5]));
        assert!(is_sorted(&[1, 1, 1, 1]));
        assert!(is_sorted(&[1]));
        assert!(is_sorted(&[]));
        assert!(!is_sorted(&[5, 4, 3, 2, 1]));
        assert!(!is_sorted(&[1, 3, 2]));
    }
}
