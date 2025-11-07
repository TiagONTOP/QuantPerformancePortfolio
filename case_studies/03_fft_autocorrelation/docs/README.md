# Case Study 03 — FFT Autocorrelation Optimization (Rust vs Python)

## 1\. Overview

This project is a case study in production-grade optimization for a common quantitative finance operation: the autocorrelation of time series.

The goal is to compare a high-level yet efficient **Python baseline** using **SciPy** against a **Rust module** built for maximum performance. We aim to quantify the significant, order-of-magnitude speedups achievable through hardware-aware engineering in Rust, exposed to Python via **PyO3** and **Maturin**.

The Python reference (`scipy.signal.correlate(method='fft')`) is already a strong $O(n \log n)$ implementation based on the Wiener–Khinchin theorem. This project demonstrates how deeper system-level optimizations — such as adaptive algorithm selection, explicit memory management, and FFT plan caching — can systematically outperform this baseline.

-----

## 2\. Compared Implementations

### • Suboptimal (Baseline)

  - **File:** `suboptimal/processing.py`
  - **Method:** Uses `scipy.signal.correlate(method='fft')`
  - **Analysis:**
    Relies entirely on SciPy’s precompiled FFT engine (likely MKL or PocketFFT). It is efficient but incurs Python-level overhead (memory allocation, type conversions) and is not specialized for cases where `max_lag` is small.

### • Optimized (Rust)

  - **File:** `optimized/src/lib.rs`
  - **Method:** Native Rust module leveraging `realfft` for transforms, `rayon` for parallelism, and custom caching and pooling strategies.
  - **Analysis:**
    Designed for maximum throughput by addressing bottlenecks at both the algorithmic and memory-management levels.

-----

## 3\. Key Optimizations in the Rust Implementation

Performance gains stem not merely from Rust’s speed, but from **targeted algorithmic and memory-level engineering**:

1.  **Adaptive Algorithm Selection**
    The main function `autocorr_adaptive()` dynamically switches between two methods using a cost heuristic:

      - **Direct Method ($O(n \cdot k)$):** For small `max_lag` (`k`), a direct, parallelized computation using `rayon` and 4-way loop unrolling is implemented. This avoids FFT overhead entirely and is cache-friendly, proving significantly faster for low `k`.
      - **FFT Method ($O(n \log n)$):** For larger `max_lag`, it uses the standard FFT-based Wiener–Khinchin approach (via `realfft`).

2.  **FFT Plan Caching**
    FFT “plans” (precomputed transform setup) are expensive to create. A global, thread-safe `PLAN_CACHE` (guarded by a `Mutex` and initialized via `OnceCell`) stores and reuses plans, amortizing setup costs across all calls in the process.

3.  **Thread-Local Buffer Pooling (Zero-Allocation)**
    Instead of allocating new working buffers (for time-domain, frequency-domain, and scratch space) on every call, a `thread_local!` `BUFFER_POOL` provides per-thread reusable `Vec`s. To **guarantee zero heap allocations** during computation, the implementation uses a **"loan pattern"** (`with_buffers`). This function "lends" a mutable `BufferSet` to a closure containing the FFT logic, ensuring the `Vec`s' capacity is reused without any new allocation or cloning.

4.  **Prime-Factor FFT Sizing (2,3,5,7-smooth)**
    FFTs are fastest for lengths with small prime factors (not just powers-of-two). The code uses `next_fast_len` to pad vectors to the next efficient "smooth" length $m \ge 2n-1$, which is often faster than naïve power-of-two padding.

5.  **GIL-Free Parallelism (`rayon`)**
    Both the power-spectrum computation ($|X|^2$) in the FFT method and the lag loop in the direct method are fully parallelized using `rayon`. The main entry point `compute_autocorrelation()` releases Python’s Global Interpreter Lock (GIL) via `py.allow_threads(...)`, allowing true parallelism with other Python threads.

-----

## 4\. Benchmark Results

Benchmarks were run to compare the Python baseline against the final adaptive Rust implementation. The analysis demonstrates the effectiveness of the adaptive strategy.

### 4.1. Heuristic & Crossover Point Analysis

The adaptive strategy's goal is to **always pick the optimal algorithm**. The heuristic in `autocorr_adaptive()` calculates the crossover point between the Direct $O(n \cdot k)$ and FFT $O(n \log n)$ methods.

  - `fft_cost_estimate` $\approx m \log_2(m)$
  - `direct_cost_estimate` $\approx n \cdot k \cdot 0.25$ (due to 4-way unrolling)

For `n = 10,000`:

  - The FFT size `m` is `next_fast_len(19999) = 20000`.
  - The `fft_total_cost` (including overhead) is `(20000 * log2(20000)) + 1000 ≈ 286,754`.
  - The `direct_cost` (with safety margin) is `(10000 * k * 0.25) * 1.2 = 3000 * k`.
  - **Crossover Point:** The code switches from Direct to FFT when `3000 * k \ge 286,754`, which occurs at **`k \approx 95.6`**.

This heuristic is robust and validated by the benchmarks:

  - At `k=50` (n=10k), the **Direct** path is correctly chosen (0.175 ms).
  - At `k=100` (n=10k), the heuristic correctly switches to the **FFT** path (0.326 ms).

### 4.2. Adaptive Strategy Validation

This table shows the final performance of the `compute_autocorrelation` function, which automatically selects the best path.

| Test Case | Size (n) | Max Lag (k) | Python (SciPy) | Rust (Adaptive) | **Speedup** | Method Selected |
| :--- | ---: | ---: | ---: | ---: | :---: | :--- |
| Small n, Small k | 100 | 50 | 0.300 ms | 0.004 ms | **70.69×** | **Direct** |
| Medium n, Small k | 1,000 | 50 | 0.357 ms | 0.038 ms | **9.31×** | **Direct** |
| Large n, Small k | 10,000 | 50 | 0.968 ms | 0.175 ms | **5.54×** | **Direct** (k \< 96) |
| Large n, Crossover | 10,000 | 100 | 0.890 ms | 0.326 ms | **2.73×** | **FFT** (k \> 96) |
| Large n, Large k | 10,000 | 500 | 0.965 ms | 0.372 ms | **2.60×** | **FFT** (k \> 96) |
| Very Large n, Small k| 50,000 | 50 | 7.088 ms | 0.731 ms | **9.70×** | **Direct** |
| Repeated Calls¹ | 10,000 | 50 | 1.886 ms/call| 0.190 ms/call| **9.91×** | **Direct** |

¹ *Confirms the sustained benefit of caching (plans) and pooling (buffers).*

**Performance Conclusion:**
The adaptive strategy successfully combines the strengths of both methods, achieving **2.60×–70.69× speedups** over SciPy. It correctly identifies the crossover point, leveraging the highly efficient Direct path for small `k` (a very common case) and the robust FFT path for large `k`.

-----

## 5\. Validation and Testing

Performance is meaningless without correctness.
The project includes a full validation suite (`tests/test_correctness.py`) that compares Rust and SciPy outputs across a range of inputs.

  - **Result:** The full suite **passes 14/14 tests**, confirming numerical equivalence.
  - **Numerical Accuracy:** Maximum absolute difference $< 1 \times 10^{-8}$ between implementations (tolerance set to account for different computational paths).
  - **Edge Cases:** Constant or short series producing `NaN` are handled correctly and identically.

-----

## 6\. Installation and Usage

This project uses [**Poetry**](https://python-poetry.org/) for Python dependency management and [**Maturin**](https://www.maturin.rs/) for Rust module compilation.

### Initial Setup

```bash
# Install Poetry (if not already)
pip install poetry

# Configure Poetry to create a local virtual environment
poetry config virtualenvs.in-project true

# Install dependencies (SciPy, NumPy, PyTest, etc.)
poetry install

# Activate virtual environment
.venv\Scripts\activate
```

### Compile the Rust Module

```bash
# Navigate to the Rust package
cd optimized

# Build and install Rust module into the active Python env
maturin develop --release

# Return to project root
cd ..
```

### Run Tests

```bash
# Validate numerical correctness
python -m pytest tests/test_correctness.py -v

# Run performance benchmarks
python -m pytest tests/test_benchmark.py -v -s
```

-----

## 7\. Benchmark Hardware Configuration

All results were obtained on the following system:

| Component | Specification |
| :--- | :--- |
| **CPU** | Intel Core i7-4770 (Haswell) OC @ 4.1 GHz |
| **Motherboard** | ASUS Z87 |
| **RAM** | 16 GB DDR3 @ 2400 MHz |
| **OS** | Windows 10 (64-bit) |
| **Compiler** | Rust 1.70+ (stable) |

This configuration represents a mature 2010s setup, ideal for evaluating CPU-bound computational performance where **memory bandwidth**, **cache efficiency**, and **core-level parallelism (including AVX2 SIMD capabilities)** are the primary bottlenecks.

-----

## 8\. Conclusion

The adaptive Rust implementation achieves consistent **2.6×–70× speedups** over SciPy's FFT baseline while maintaining **near bit-level numerical accuracy**.

This performance is not magic; it is the result of systematic engineering:

1.  **Algorithmic Adaptivity:** Correctly identifying that the problem has two distinct computational regimes ($O(n \cdot k)$ vs. $O(n \log n)$) and implementing a robust heuristic to select the optimal path.
2.  **Memory Hierarchy Management:** **Eliminating heap allocations** via a `thread_local!` buffer pool and a "loan pattern", ensuring `Vec` capacity is reused without cost.
3.  **Amortized Setup Cost:** Reusing FFT plans via a global, thread-safe cache.
4.  **GIL-Free Parallelism:** Leveraging `rayon` to scale computation across all available CPU cores.

This case study exemplifies how hardware-aware, memory-conscious engineering in Rust can push scientific computing workloads far beyond the reach of standard high-level vectorized libraries.