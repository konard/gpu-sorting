# Benchmark Data for Issue #5

## Raw Data from Issue Reports

### Issue #3 - Before Optimization (December 15, 2025)

**System**: Apple M3 Pro, macOS

#### Run 1: 1M Elements
```
Array size: 1048576 elements (4 MB)
CPU sort time: 19.259 ms
GPU sort time: 55.789 ms
CPU sort verified: OK
GPU sort verified: OK
Results match CPU sort: OK
CPU is 2.90x faster than GPU
```

#### Run 2: 16M Elements
```
Array size: 16777216 elements (67 MB)
CPU sort time: 190.810 ms
GPU sort time: 385.333 ms
CPU sort verified: OK
GPU sort verified: OK
Results match CPU sort: OK
CPU is 2.02x faster than GPU
```

#### Run 3: 134M Elements
```
Array size: 134217728 elements (536 MB)
CPU sort time: 1690.627 ms
GPU sort time: 3476.947 ms
CPU sort verified: OK
GPU sort verified: OK
Results match CPU sort: OK
CPU is 2.06x faster than GPU
```

---

### Issue #5 - After Optimization (December 15, 2025)

**System**: Apple M3 Pro, macOS

#### Run 1: 134M Elements
```
Array size: 134217728 elements (536 MB)
CPU sort time: 1680.762 ms
GPU sort time: 2008.334 ms
CPU sort verified: OK
GPU sort verified: OK
Results match CPU sort: OK
CPU is 1.19x faster than GPU
```

#### Run 2: 134M Elements
```
Array size: 134217728 elements (536 MB)
CPU sort time: 1694.743 ms
GPU sort time: 2246.863 ms
CPU sort verified: OK
GPU sort verified: OK
Results match CPU sort: OK
CPU is 1.33x faster than GPU
```

#### Run 3: 16M Elements
```
Array size: 16777216 elements (67 MB)
CPU sort time: 185.575 ms
GPU sort time: 210.733 ms
CPU sort verified: OK
GPU sort verified: OK
Results match CPU sort: OK
CPU is 1.14x faster than GPU
```

#### Run 4: 16M Elements
```
Array size: 16777216 elements (67 MB)
CPU sort time: 190.927 ms
GPU sort time: 199.328 ms
CPU sort verified: OK
GPU sort verified: OK
Results match CPU sort: OK
CPU is 1.04x faster than GPU
```

---

## Performance Comparison Summary

### Before vs After Optimization

| Array Size | Before: CPU/GPU Ratio | After: CPU/GPU Ratio | Improvement |
|------------|----------------------|---------------------|-------------|
| 1M | 2.90x | N/A | - |
| 16M | 2.02x | 1.04-1.14x | ~47-50% |
| 134M | 2.06x | 1.19-1.33x | ~35-42% |

### Performance in Elements per Second

| Array Size | CPU (M elem/s) | GPU Before (M elem/s) | GPU After (M elem/s) |
|------------|----------------|----------------------|---------------------|
| 1M | 54.4 | 18.8 | - |
| 16M | 87.9 | 43.6 | 79.6-90.6 |
| 134M | 79.4 | 38.6 | 59.7-66.8 |

---

## Variance Analysis

### GPU Timing Variance

For 134M elements:
- Run 1: 2008.334 ms
- Run 2: 2246.863 ms
- Variance: ~238ms (11.9%)

For 16M elements:
- Run 1: 210.733 ms
- Run 2: 199.328 ms
- Variance: ~11ms (5.6%)

### CPU Timing Variance

For 134M elements:
- Run 1: 1680.762 ms
- Run 2: 1694.743 ms
- Variance: ~14ms (0.8%)

For 16M elements:
- Run 1: 185.575 ms
- Run 2: 190.927 ms
- Variance: ~5ms (2.9%)

**Observation**: GPU has higher timing variance than CPU, possibly due to:
- Thermal throttling
- Background GPU processes
- Memory contention

---

## Theoretical Analysis

### Complexity Comparison

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Rust pdqsort (CPU) | O(n log n) | O(log n) |
| Bitonic sort (GPU) | O(n log²n) | O(1) |
| Radix sort (GPU) | O(nk) ≈ O(n) | O(n) |

### Expected Operations for 134M Elements

```
n = 134,217,728

log₂(n) = 27

pdqsort:   n × log₂(n)      = 3.6 × 10⁹ operations
bitonic:   n × (log₂(n))²/2 = 48.6 × 10⁹ operations
radix:     n × k (k=4-8)    = 0.5-1.1 × 10⁹ operations

Ratio: bitonic/pdqsort ≈ 13.5x more work
Ratio: bitonic/radix ≈ 44-97x more work
```

### Observed vs Theoretical Performance

| Metric | Theoretical | Observed |
|--------|-------------|----------|
| Bitonic/pdqsort work ratio | 13.5x | 1.19-1.33x time ratio |
| Parallelism benefit | ~10-13x | ~10x (balances complexity) |

The GPU's massive parallelism (~10 compute units × ~1000s of threads) compensates for the higher complexity, bringing performance close to CPU despite doing 13x more work.
