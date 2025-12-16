//! CPU Bitonic Sort Implementation
//!
//! This module provides a CPU-based bitonic sort implementation for fair comparison
//! with GPU bitonic sort. Bitonic sort is a comparison-based sorting algorithm that
//! is well-suited for parallel execution.
//!
//! Complexity: O(n * log²n) comparisons

/// Sort a slice in-place using bitonic sort.
///
/// This implementation requires the input length to be a power of 2.
/// If it's not, the slice will be padded internally (not implemented here;
/// caller should handle padding).
///
/// # Arguments
/// * `data` - The slice to sort in-place (must be power of 2 length)
pub fn sort(data: &mut [u32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    // Bitonic sort requires power of 2 length
    assert!(
        n.is_power_of_two(),
        "Bitonic sort requires power of 2 length, got {}",
        n
    );

    // Build up sorted sequences of increasing size
    // k is the size of the bitonic sequence being sorted
    let mut k = 2;
    while k <= n {
        // j is the distance between elements being compared
        let mut j = k / 2;
        while j > 0 {
            // Perform compare-and-swap operations
            for i in 0..n {
                // Calculate which element to compare with
                let ixj = i ^ j;

                // Only compare if ixj > i (avoid double comparison)
                if ixj > i {
                    // Determine sort direction based on position in bitonic sequence
                    // Elements in first half of k-sized block sort ascending
                    // Elements in second half sort descending
                    let ascending = (i & k) == 0;

                    if ascending {
                        // Sort ascending: smaller value goes to lower index
                        if data[i] > data[ixj] {
                            data.swap(i, ixj);
                        }
                    } else {
                        // Sort descending: larger value goes to lower index
                        if data[i] < data[ixj] {
                            data.swap(i, ixj);
                        }
                    }
                }
            }
            j /= 2;
        }
        k *= 2;
    }
}

/// Sort a slice that may not be a power of 2 by padding with MAX values.
///
/// This creates a copy padded to the next power of 2, sorts it, then
/// copies back only the original elements.
///
/// # Arguments
/// * `data` - The slice to sort in-place (any length)
#[allow(dead_code)]
pub fn sort_any_size(data: &mut [u32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    if n.is_power_of_two() {
        sort(data);
        return;
    }

    // Pad to next power of 2
    let padded_len = n.next_power_of_two();
    let mut padded = vec![u32::MAX; padded_len];
    padded[..n].copy_from_slice(data);

    // Sort the padded array
    sort(&mut padded);

    // Copy back only the original number of elements
    // (MAX values will be at the end after sorting)
    data.copy_from_slice(&padded[..n]);
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
    fn test_sort_power_of_2() {
        let mut data: Vec<u32> = (0..16).rev().collect();
        sort(&mut data);
        assert!(is_sorted(&data));
        assert_eq!(data, (0..16).collect::<Vec<u32>>());
    }

    #[test]
    fn test_sort_random_power_of_2() {
        let mut rng = rand::thread_rng();
        let mut data: Vec<u32> = (0..1024).map(|_| rng.gen()).collect();
        sort(&mut data);
        assert!(is_sorted(&data));
    }

    #[test]
    fn test_sort_any_size_non_power_of_2() {
        let mut data: Vec<u32> = (0..100).rev().collect();
        sort_any_size(&mut data);
        assert!(is_sorted(&data));
        assert_eq!(data, (0..100).collect::<Vec<u32>>());
    }

    #[test]
    fn test_sort_any_size_random() {
        let mut rng = rand::thread_rng();
        let mut data: Vec<u32> = (0..1000).map(|_| rng.gen()).collect();
        let mut expected = data.clone();
        expected.sort();

        sort_any_size(&mut data);
        assert!(is_sorted(&data));
        assert_eq!(data, expected);
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
        let mut data = vec![42u32; 64];
        sort(&mut data);
        assert!(is_sorted(&data));
        assert!(data.iter().all(|&x| x == 42));
    }

    #[test]
    fn test_sort_two_elements() {
        let mut data = vec![5u32, 3];
        sort(&mut data);
        assert!(is_sorted(&data));
        assert_eq!(data, vec![3, 5]);
    }

    #[test]
    fn test_sort_four_elements() {
        let mut data = vec![4u32, 3, 2, 1];
        sort(&mut data);
        assert!(is_sorted(&data));
        assert_eq!(data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_sort_large_power_of_2() {
        let mut rng = rand::thread_rng();
        let size = 1 << 14; // 16384 elements
        let mut data: Vec<u32> = (0..size).map(|_| rng.gen()).collect();
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

    #[test]
    #[should_panic(expected = "Bitonic sort requires power of 2 length")]
    fn test_sort_non_power_of_2_panics() {
        let mut data = vec![1u32, 2, 3];
        sort(&mut data);
    }
}
