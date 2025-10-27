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

### Benchmark Results (Optimized Implementation)

Performance comparison with max_lag=50, 10 iterations per size:

| Size   | Python (ms) | Rust (ms) | Speedup  | Notes |
|--------|-------------|-----------|----------|-------|
| 100    | 0.390       | 0.019     | **20.9x** | Direct method |
| 1000   | 0.281       | 0.020     | **14.4x** | Direct method |
| 10000  | 0.844       | 0.219     | **3.9x**  | Real FFT |
| 50000  | 6.383       | 1.785     | **3.6x**  | Real FFT |

### Key Performance Features

**Optimizations Applied:**
- ✅ **Real FFT (R2C/C2R)** instead of Complex FFT - cuts work in half
- ✅ **2357-smooth FFT sizes** instead of power-of-2 - optimal transform lengths
- ✅ **In-place operations** - zero-copy processing, minimal allocations
- ✅ **Cached FFT plans** - amortized planning overhead
- ✅ **Adaptive algorithm selection** - direct O(n·k) for small lag, FFT O(n log n) for large lag
- ✅ **GIL release** - allows Python concurrency during computation
- ✅ **Native CPU optimizations** - target-cpu=native compilation

**Performance Characteristics:**
- **Small to medium series (< 10k)**: **14-21x faster** than SciPy (direct method)
- **Large series (> 10k)**: **3-4x faster** than SciPy (optimized FFT)
- **Consistent speedup** across all tested sizes
- **Identical numerical results** (max difference < 1e-16)

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
