# FFT Autocorrelation: Rust + Python Performance Case Study

## Project Objective

This project demonstrates the power of **Rust-Python integration** via **PyO3** and **Maturin** to create ultra-performant Python extensions that **significantly outperform** pure Python implementations, even those using optimized libraries like **SciPy**.

### The Challenge

Implement FFT (Fast Fourier Transform) based autocorrelation computation to **decisively beat** SciPy's reference implementation, which is already highly optimized and uses performant C/Fortran backends.

### The Solution

Combine:
- **The power of Rust**: native performance, aggressive optimizations, memory safety
- **The simplicity of Python**: ease of use, rich ecosystem, universal deployment
- **PyO3**: Rust ↔ Python bindings with minimal overhead
- **Maturin**: automatic packaging and publication of Python wheels

### The Results

**Final Performance vs SciPy (optimized Python implementation):**

| Size   | SciPy (ms) | Rust (ms) | **Speedup** |
|--------|------------|-----------|-------------|
| 100    | 0.236      | 0.005     | **44.9x** |
| 1,000  | 0.318      | 0.129     | **2.5x**  |
| 10,000 | 1.121      | 0.237     | **4.7x** |
| 50,000 | 6.680      | 0.743     | **9.0x** |

**Conclusion: From 2.5x to 45x faster than SciPy!**

---

## Project Structure

```
03_fft_autocorrelation/
├── README.md                      # This file
├── STRUCTURE.md                   # Detailed architecture documentation
├── TESTS.md                       # Unit test documentation
├── BENCHMARKS.md                  # Detailed benchmark results
│
├── suboptimal/                    # Python reference implementation
│   ├── __init__.py
│   └── processing.py              # Python version with SciPy (optimized)
│
├── optimized/                     # Rust + PyO3 implementation
│   ├── Cargo.toml                 # Rust configuration
│   ├── pyproject.toml             # Python/Maturin configuration
│   ├── src/
│   │   └── lib.rs                 # Optimized Rust code (315 lines)
│   ├── README.md                  # Rust module documentation
│   ├── OPTIMIZATION_SUMMARY.md    # Optimization history v1
│   ├── OPTIMIZATION_V2_SUMMARY.md # Optimization details v2
│   └── BUILD_AND_RUN.md           # Compilation instructions
│
└── tests/                         # Tests and benchmarks
    ├── test_unit.py               # Unit tests (correctness)
    └── test_benchmark.py          # Performance tests
```

---

## Technologies Used

### Rust
- **rustfft / realfft**: Pure Rust FFT implementation
- **PyO3**: Rust ↔ Python bindings
- **numpy crate**: Integration with NumPy arrays
- **rayon**: Data-parallel processing
- **once_cell**: Thread-safe cache for FFT plans

### Python
- **Maturin**: Build system for Rust extensions
- **NumPy**: Numerical arrays
- **Pandas**: Time series manipulation
- **SciPy**: Reference implementation (signal.correlate)

---

## Quick Start

### Prerequisites

```bash
# Rust (https://rustup.rs/)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python 3.10+
python --version

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### Installation

```bash
# 1. Navigate to the optimized folder
cd optimized

# 2. Compile and install the Rust module
maturin develop --release --strip

# 3. Test
cd ../tests
python test_unit.py        # Unit tests
python test_benchmark.py   # Performance benchmarks
```

### Usage

```python
import fft_autocorr
import numpy as np

# Generate data
data = np.random.randn(10000)

# Compute autocorrelation
result = fft_autocorr.compute_autocorrelation(data, max_lag=50)

print(f"Shape: {result.shape}")  # (50,)
print(f"First 5 values: {result[:5]}")
```

---

## Why Rust + PyO3?

### Rust Advantages

1. **Native Performance**
   - Ahead-of-time compilation
   - Aggressive optimizations (LTO, inlining, vectorization)
   - Zero interpretation overhead

2. **Memory Control**
   - Manual management without GC
   - Explicit allocations
   - Cache-friendly data structures

3. **Safety**
   - No segfaults
   - No data races
   - Compile-time verification

4. **Parallelism**
   - Rayon for easy data-parallelism
   - Thread-safe by design

### PyO3 Advantages

1. **Zero-copy**
   - Direct access to NumPy buffers
   - No Python ↔ Rust conversion

2. **Ergonomic API**
   - Macros to expose Rust functions
   - Python types mapped automatically

3. **GIL release**
   - Computations without blocking Python
   - Native concurrency

4. **Simple Packaging**
   - Maturin builds wheels automatically
   - PyPI compatible

### Maturin Advantages

1. **Automated Build**
   - Rust toolchain detection
   - Optimized compilation by default

2. **Easy Distribution**
   - Multi-platform wheels
   - Installation via `pip install`

3. **Rapid Development**
   - `maturin develop` for rapid iteration
   - Hot-reload in dev mode

---

## Optimization Methodology

### Phase 1: Naive Implementation (v0)

**Problem**: Slower than SciPy for large arrays (0.4-0.5x)

**Causes**:
- Complex FFT (C2C) instead of real FFT (R2C)
- Power-of-2 FFT sizes (too large)
- Multiple allocations and copies
- No FFT plan caching

### Phase 2: Algorithmic Optimization (v1)

**Optimizations**:
1. Real FFT (R2C/C2R) → 2x gain
2. 2357-smooth sizes → 1.6x gain
3. FFT plan caching → 10-20% gain
4. Adaptive direct/FFT selection → 10-20x gain (small max_lag)

**Result**: 3.6-21x faster than SciPy ✓

### Phase 3: Micro-Optimization (v2)

**Additional Optimizations**:
1. Thread-local buffer pool → zero allocation after warmup
2. LTO + codegen-units=1 → better inlining
3. 4-way loop unrolling → better CPU pipelining
4. Parallelization (rayon) → multi-core exploitation
5. Single-pass mean/variance → -33% memory bandwidth

**Final Result**: 2.5-45x faster than SciPy ✓✓

---

## Lessons Learned

### 1. Rust is not magic
- A naive implementation can be **slower** than Python+C
- You must **understand the problem** and optimize intelligently

### 2. Algorithm beats implementation
- Direct O(n·k) beats FFT O(n log n) for small max_lag
- Adaptive selection is crucial

### 3. Allocations kill performance
- Buffer reuse → massive gain
- Thread-local storage avoids contention

### 4. Parallelization has a cost
- Overhead visible for small problems
- Threshold calibration essential

### 5. Profiling is indispensable
- Measure before optimizing
- Benchmarks on real hardware
- Warmup to eliminate cache bias

---

## Complete Documentation

- **[STRUCTURE.md](STRUCTURE.md)**: Detailed architecture and implementation analysis
- **[TESTS.md](TESTS.md)**: Unit tests, validation, results
- **[BENCHMARKS.md](BENCHMARKS.md)**: Detailed benchmarks, comparisons, analysis
- **[optimized/README.md](optimized/README.md)**: Module user documentation
- **[optimized/OPTIMIZATION_SUMMARY.md](optimized/OPTIMIZATION_SUMMARY.md)**: v1 optimizations
- **[optimized/OPTIMIZATION_V2_SUMMARY.md](optimized/OPTIMIZATION_V2_SUMMARY.md)**: v2 optimizations
- **[optimized/BUILD_AND_RUN.md](optimized/BUILD_AND_RUN.md)**: Build instructions

---

## Future Improvements

### Short Term
- [ ] Explicit SIMD with `std::simd` (nightly) → +10-30%
- [ ] Automatic threshold calibration via profiling
- [ ] Batch API to process multiple series → +2-5x

### Medium Term
- [ ] Multi-threaded FFT backend (FFTW, MKL) → +1.5-3x for large arrays
- [ ] GPU support via cuFFT → +10-100x for very large arrays
- [ ] PACF (partial autocorrelation) implementation

### Long Term
- [ ] Architecture-optimized wheel distribution (AVX2, AVX-512, ARM NEON)
- [ ] Async support for integration in concurrent workflows
- [ ] Bindings for other languages (Julia, R, Node.js)

---

## License

This project is part of the quant-performance-portfolio.

---

## Acknowledgments

- **SciPy** for the reference implementation
- **PyO3** and **Maturin** for making Rust accessible to Python
- **rustfft** for a performant pure Rust FFT implementation

---

## Contact & Contributions

This project is a demonstrative case study. For questions or suggestions:
- Open an issue on the repository
- Contribute via pull request

**Summary: This project proves that with Rust + PyO3, we can create Python extensions that not only match, but significantly outperform optimized C/Fortran implementations, while remaining simple to use from Python!**
