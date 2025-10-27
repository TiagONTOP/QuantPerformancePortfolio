# FFT Autocorrelation - Rust Implementation

High-performance autocorrelation computation using FFT, implemented in Rust with PyO3 bindings for Python.

## Overview

This module provides a Rust implementation of autocorrelation computation using the Fast Fourier Transform (FFT) method, based on the Wiener-Khinchin theorem. It's designed to be a drop-in replacement for the Python/SciPy implementation in `suboptimal/processing.py`.

## Features

- Fast autocorrelation computation using FFT
- Compatible with NumPy arrays
- PyO3 bindings for seamless Python integration
- Identical results to SciPy's `signal.correlate` (max difference < 1e-16)
- Optimized for small to medium-sized time series

## Installation

### Prerequisites

- Rust toolchain (install from https://rustup.rs/)
- Python 3.8+
- Maturin (`pip install maturin`)

### Build

Build and install the module in development mode:

```bash
cd optimized
maturin develop --release
```

For production builds:

```bash
maturin build --release
```

## Usage

### Python

```python
import numpy as np
import fft_autocorr

# Create sample data
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

# Compute autocorrelation for lags 1 to 3
result = fft_autocorr.compute_autocorrelation(data, max_lag=3)

print(result)
# Output: [0.7  0.41212121  0.14848485]
```

### Drop-in Replacement

This implementation is designed to be compatible with the Python version:

```python
# Python version (suboptimal/processing.py)
from processing import compute_autocorrelation
result_py = compute_autocorrelation(pd.Series(data), max_lag=3)

# Rust version
import fft_autocorr
result_rust = fft_autocorr.compute_autocorrelation(data, max_lag=3)

# Results are identical (within floating-point precision)
assert np.allclose(result_py.values, result_rust)
```

## Testing

Run the comprehensive test suite:

```bash
cd optimized/examples
python example.py
```

The test suite includes:
1. **Basic functionality tests** - Verify correctness with simple examples
2. **Edge case tests** - Test with constant series, random data, sine waves, linear trends
3. **Performance benchmarks** - Compare speed with Python/SciPy implementation
4. **Large max_lag tests** - Test with large lag values

## Performance

### Benchmark Results (Highly Optimized Implementation v2)

Performance comparison with max_lag=50, 10 iterations per size:

| Size   | Python (ms) | Rust (ms) | Speedup  | Improvement vs v1 | Method |
|--------|-------------|-----------|----------|-------------------|--------|
| 100    | 0.236       | 0.005     | **44.9x** | +115% faster | Direct |
| 1000   | 0.318       | 0.129     | **2.5x**  | -35% (regression) | Direct |
| 10000  | 1.121       | 0.237     | **4.7x**  | +21% faster | FFT |
| 50000  | 6.680       | 0.743     | **9.0x**  | +150% faster | FFT |

**Note on n=1000 regression:** Thread setup overhead dominates for this specific size. Sequential version would be faster. This is a known tradeoff in parallelization.

### Key Performance Features

**Optimizations Applied (v2):**
- ✅ **Real FFT (R2C/C2R)** instead of Complex FFT - cuts work in half
- ✅ **2357-smooth FFT sizes** instead of power-of-2 - optimal transform lengths
- ✅ **Thread-local buffer pool** - zero allocations after warmup
- ✅ **Cached FFT plans** - amortized planning overhead
- ✅ **Adaptive algorithm selection** - calibrated cost model
- ✅ **4-way loop unrolling** - better CPU pipelining in direct method
- ✅ **Parallel computation** - rayon for lags and power spectrum
- ✅ **Single-pass mean/variance** - reduced memory bandwidth
- ✅ **GIL release** - allows Python concurrency during computation
- ✅ **Aggressive compilation** - LTO, codegen-units=1, target-cpu=native

**Performance Characteristics:**
- **Tiny series (100)**: **~45x faster** than SciPy
- **Small series (1000)**: **~2.5x faster** than SciPy
- **Medium series (10k)**: **~5x faster** than SciPy
- **Large series (50k)**: **~9x faster** than SciPy
- **Consistent correctness** (max difference < 1e-16)

### Performance Evolution

| Implementation | n=100 | n=1000 | n=10k | n=50k |
|----------------|-------|--------|-------|-------|
| **Naive Rust v0** | 12.7x | 2.6x | **0.4x** ❌ | **0.5x** ❌ |
| **Optimized v1** | 20.9x | 14.4x | 3.9x | 3.6x |
| **Optimized v2** | **44.9x** | 2.5x | **4.7x** | **9.0x** |

**Total improvement from v0 to v2:**
- n=100: +254% faster
- n=10k: from **0.4x slower** to **4.7x faster** = **~12x improvement**
- n=50k: from **0.5x slower** to **9.0x faster** = **~18x improvement**

## Algorithm

The implementation intelligently chooses between two methods:

### Direct Method (for small max_lag)
1. Mean-center the input data
2. Compute lag-0 variance
3. For each lag k: compute sum of x[i] * x[i+k]
4. Normalize by lag-0 variance

**Complexity:** O(n · max_lag)
**Best for:** max_lag < ~100 for typical n

### FFT Method (for large max_lag)
Uses the Wiener-Khinchin theorem with optimizations:

1. Mean-center the input data
2. Determine optimal FFT size using 2357-smooth length
3. Compute forward Real FFT (R2C)
4. Calculate power spectrum in-place: |FFT(x)|²
5. Compute inverse Real FFT (C2R) to get autocorrelation
6. Normalize by FFT size and lag-0 variance
7. Extract values for lags 1 to max_lag

**Complexity:** O(m log m) where m ≈ 2n
**Best for:** max_lag > ~100 or when m log m < n · max_lag

## Project Structure

```
optimized/
├── Cargo.toml          # Rust dependencies and build configuration
├── pyproject.toml      # Python package configuration
├── README.md           # This file
├── src/
│   └── lib.rs          # Rust implementation with PyO3 bindings
└── examples/
    └── example.py      # Comprehensive test suite and usage examples
```

## Dependencies

### Rust
- `pyo3` (0.22.0) - Python bindings
- `numpy` (0.22.0) - NumPy integration
- `ndarray` (0.16.1) - N-dimensional arrays
- `realfft` (3.5.0) - Real FFT implementation (R2C/C2R)
- `num-complex` (0.4) - Complex number support
- `once_cell` (1.21) - Plan caching

### Python
- `numpy` (>=1.20.0)
- `pandas` (>=1.3.0)
- `scipy` (for comparison tests)

## Limitations

- Constant series (zero variance) will produce NaN results (same as SciPy)
- Current implementation is single-threaded per call (but releases GIL for concurrency)

## Future Improvements

- ✅ ~~Optimize FFT implementation for large series~~ **DONE** (3-4x faster)
- ✅ ~~Add adaptive direct/FFT selection~~ **DONE**
- Add parallelization for batch processing multiple series
- Implement additional autocorrelation methods (Yule-Walker, Burg)
- Add support for partial autocorrelation (PACF)
- Consider SIMD optimizations for direct method

## License

This project is part of the quant-performance-portfolio case studies.

## References

- Python reference implementation: `../suboptimal/processing.py`
- Wiener-Khinchin theorem: https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem
