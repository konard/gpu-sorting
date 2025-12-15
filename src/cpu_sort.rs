//! CPU Sorting Implementation
//!
//! This module provides CPU-based sorting using Rust's standard library.
//! We use the unstable sort as it's typically faster and we don't need stability.

/// Sort a slice in-place using unstable sort (typically faster than stable sort).
///
/// Uses Rust's built-in sorting algorithm which is a pattern-defeating quicksort
/// that falls back to heap sort to guarantee O(n log n) worst case.
#[inline]
pub fn sort_unstable(data: &mut [u32]) {
    data.sort_unstable();
}

/// Sort a slice in-place using stable sort.
///
/// Uses Rust's built-in stable sorting algorithm (TimSort variant).
#[inline]
#[allow(dead_code)]
pub fn sort_stable(data: &mut [u32]) {
    data.sort();
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
    fn test_sort_unstable_empty() {
        let mut data: Vec<u32> = vec![];
        sort_unstable(&mut data);
        assert!(is_sorted(&data));
    }

    #[test]
    fn test_sort_unstable_single() {
        let mut data = vec![42u32];
        sort_unstable(&mut data);
        assert!(is_sorted(&data));
    }

    #[test]
    fn test_sort_unstable_sorted() {
        let mut data: Vec<u32> = (0..100).collect();
        sort_unstable(&mut data);
        assert!(is_sorted(&data));
    }

    #[test]
    fn test_sort_unstable_reverse() {
        let mut data: Vec<u32> = (0..100).rev().collect();
        sort_unstable(&mut data);
        assert!(is_sorted(&data));
    }

    #[test]
    fn test_sort_unstable_random() {
        let mut rng = rand::thread_rng();
        let mut data: Vec<u32> = (0..1000).map(|_| rng.gen()).collect();
        sort_unstable(&mut data);
        assert!(is_sorted(&data));
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
