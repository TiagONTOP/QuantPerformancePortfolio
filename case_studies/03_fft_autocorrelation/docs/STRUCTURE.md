# STRUCTURE.md — Technical and Architectural Analysis

This document provides a detailed analysis of the code structure, algorithmic choices, and low-level optimizations implemented in this project.

-----

## 1\. Project Overview

The project is divided into three main directories, reflecting the comparative methodology:

```
03_fft_autocorrelation/
├── .pytest_cache/
├── .venv/
├── docs/
│   ├── BENCHMARKS.md
│   ├── README.md
│   ├── STRUCTURE.md
│   └── TESTS.md
├── optimized/
│   ├── .cargo/
│   ├── .github/
│   ├── .pytest_cache/
│   ├── examples/
│   ├── src/
│   │   └── lib.rs
│   ├── target/
│   ├── .gitignore
│   ├── Cargo.lock
│   ├── Cargo.toml
│   └── pyproject.toml
├── suboptimal/
│   ├── __pycache__/
│   ├── __init__.py
│   └── processing.py
├── tests/
│   ├── __pycache__/
│   ├── .benchmarks/
│   ├── .pytest_cache/
│   ├── __init__.py
│   ├── pytest.ini
│   ├── test_benchmark.py
│   └── test_correctness.py
├── poetry.lock
└── pyproject.toml
```

  - **`suboptimal/`** — Python reference implementation for correctness and performance baselining.
  - **`optimized/`** — Rust crate compiled into a native Python module via [Maturin](https://www.maturin.rs/).
  - **`tests/`** — `pytest` scripts importing both implementations for correctness validation and performance benchmarking.

-----

## 2\. Baseline Analysis — `suboptimal/processing.py`

The "suboptimal" version is not naïve: it already uses SciPy’s FFT-based autocorrelation, which is fast for Python code.

### 2.1. Method: Wiener–Khinchin Theorem

The baseline relies on `scipy.signal.correlate(x, x, method='fft')`, implementing the **Wiener–Khinchin theorem**, which states that a signal’s autocorrelation equals the inverse Fourier transform of its power spectrum.

Algorithm:

1.  Center the data: `x = x - np.mean(x)`
2.  Compute the full correlation via FFT:
    $\mathcal{F}^{-1}(|\mathcal{F}(x_p)|^2)$ where $x_p$ is zero-padded.
3.  Keep the positive-lag half.
4.  Normalize by variance (lag 0): `autocorr = autocorr / autocorr[0]`.

### 2.2. Bottlenecks — Why Optimize?

Despite its $O(n \log n)$ complexity, several performance penalties remain inherent to Python/SciPy:

1.  **Call and Type Overhead** – Every `compute_autocorrelation_python` call converts `pd.Series → np.array`, runs `np.mean`, allocates temporary arrays, and calls `scipy.signal.correlate`.
2.  **Memory Allocations** – SciPy allocates FFT buffers (including zero-padding) on *every* call, wasting time on repeated workloads of identical size.
3.  **No Adaptive Strategy** – FFT is *always* used. When `max_lag` is small (e.g., 5) and `n` is large (e.g., 1,000,000), the $O(n \log n)$ FFT cost far exceeds a simple $O(n \cdot k)$ direct method.
4.  **GIL Boundaries** – While SciPy’s FFT likely releases the GIL, surrounding Python operations (mean, normalization) do not.

-----

## 3\. Optimized Implementation — `optimized/src/lib.rs`

The Rust implementation addresses each of the above inefficiencies systematically.

### 3.1. Interface: PyO3 and GIL Management

The module is exposed through `#[pymodule] fn fft_autocorr`.
Its core function is `compute_autocorrelation`.

```rust
#[pyfunction]
#[pyo3(signature = (series, max_lag=1))]
fn compute_autocorrelation<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<'py, f64>,
    max_lag: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Validation
    let x = series.as_slice()?; // <-- Zero-copy read access to NumPy memory

    // V-- GIL released
    let result = py.allow_threads(|| autocorr_adaptive(x, max_lag));

    Ok(PyArray1::from_vec_bound(py, result)) // Final copy back to NumPy
}
```

  - **`PyReadonlyArray1`** – Critical optimization: NumPy data is read directly as a Rust slice (`&[f64]`) with **zero copy**.
  - **`py.allow_threads(...)`** – Fully releases Python’s Global Interpreter Lock (GIL) during computation, allowing:
    1.  True parallel Rust execution without blocking Python.
    2.  Internal `rayon` multithreading without contention.

-----

### 3.2. Optimization \#1 — Adaptive Strategy (`autocorr_adaptive`)

The module’s “brain”. Instead of assuming FFT is always optimal, it models both algorithms’ costs dynamically.

```rust
fn autocorr_adaptive(x: &[f64], max_lag: usize) -> Vec<f64> {
    let fft_cost_estimate = (m as f64) * (m as f64).log2();
    let direct_cost_estimate = (n as f64) * (max_lag as f64) * 0.25; // 4-way unrolling
    let fft_total_cost = fft_cost_estimate + 1000.0; // Fixed setup overhead

    if direct_cost_estimate * 1.2 < fft_total_cost {
        autocorr_direct_norm(x, max_lag)
    } else {
        autocorr_fft_norm(x, max_lag)
    }
}
```

  - **Direct cost:** $O(n \cdot k)$ scaled by 0.25, modeling **4-way loop unrolling** efficiency (≈4 multiplies per cycle).
  - **FFT cost:** $O(m \log m)$ plus a fixed overhead for plan creation.
  - **Decision rule:** Direct method is used if its estimated cost (with a 20% margin) is lower.
    → Explains the strong speedups for small `max_lag`.

-----

### 3.3. Optimization \#2 — Parallel Direct Algorithm (`autocorr_direct_norm`)

Used when `max_lag` is small; tuned for CPU throughput.

```rust
let use_parallel = (available_lags as u64) * (n as u64) > 100_000;

let result = if use_parallel && available_lags > 10 {
    (1..=available_lags)
        .into_par_iter() // Rayon parallelism
        .map(|k| {
            // compute correlation for lag k
        })
        .collect()
} else {
    // Sequential fallback
};
```

Within each iteration, the scalar product loop is **manually unrolled**:

```rust
while i + 4 <= limit {
    s += xx[i] * xx[i + k]
       + xx[i + 1] * xx[i + k + 1]
       + xx[i + 2] * xx[i + k + 2]
       + xx[i + 3] * xx[i + k + 3];
    i += 4;
}
while i < limit {
    s += xx[i] * xx[i + k];
    i += 1;
}
```

  - **`rayon::into_par_iter`** distributes lag computations across all CPU cores.
  - **Loop unrolling** improves **Instruction-Level Parallelism (ILP)** and allows multiple Fused Multiply-Add (FMA) operations per cycle, making it a prime target for auto-vectorization (SIMD) by the compiler.

-----

### 3.4. Optimization \#3 — Optimized FFT Algorithm (`autocorr_fft_norm`)

For large `max_lag`, the FFT path dominates.
This version aggressively optimizes memory and planning behavior.

#### 3.4.1. Smooth FFT Lengths

FFT algorithms (e.g., [FFTW](http://www.fftw.org/)) are fastest for *smooth* transform sizes — numbers with small prime factors (2, 3, 5, 7).

```rust
fn is_smooth_2357(mut n: usize) -> bool { ... }

fn next_fast_len(mut n: usize) -> usize {
    while !is_smooth_2357(n) { n += 1; }
    n
}
```

Instead of padding to the next power of two, the code uses
`let m = next_fast_len(2 * n - 1)`, finding the **smallest efficient FFT size ≥ required length**, minimizing wasted computation.

-----

#### 3.4.2. FFT Plan Caching (`PLAN_CACHE`)

Creating an FFT “plan” (the computation schedule for a given size) is expensive.
This cache stores and reuses plans per transform length.

```rust
struct Plan {
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
}

static PLAN_CACHE: OnceCell<Mutex<HashMap<usize, Plan>>> = OnceCell::new();

fn get_plan(m: usize) -> Plan {
    let cache = PLAN_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(p) = cache.lock().unwrap().get(&m) {
        return p.clone(); // Cheap Arc clone
    }

    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(m);
    let c2r = planner.plan_fft_inverse(m);
    let plan = Plan { r2c, c2r };

    cache.lock().unwrap().insert(m, plan.clone());
    plan
}
```

  - **`static PLAN_CACHE`** — Global cache persisting through the Python process.
  - **`OnceCell`** — Thread-safe, one-time initialization.
  - **`Mutex`** — Synchronizes concurrent plan creation.
  - **`Arc`** — Shared, atomic reference-counted pointer for thread-safe reuse.
  - **`RealToComplex` (R2C)** — Exploits Hermitian symmetry, \~2× faster and \~2× smaller than complex FFTs.

-----

#### 3.4.3. Buffer Pool Analysis (`BUFFER_POOL`) — A Flawed Implementation

The *intent* of the `BUFFER_POOL` is to eliminate heap allocation overhead on repeated calls by reusing large temporary `Vec`s.

```rust
thread_local! {
    static BUFFER_POOL: RefCell<HashMap<usize, BufferSet>> = RefCell::new(HashMap::new());
}
```

**Analysis of the Flaw:**
The implementation *partially* works, but contains a critical performance defect:

1.  **Correct Reuse:** The `BUFFER_POOL.with(...)` block correctly uses `map.entry(m).or_insert_with(...)` to create a `BufferSet` only once per thread per size `m`. On subsequent calls, the `bufset.time.resize(m, 0.0)` calls **correctly reuse the `Vec`'s existing capacity**, avoiding reallocation *inside* the pool.
2.  **The Flaw:** The function `get_or_create_buffers` concludes by returning a *new* `BufferSet` struct populated with **`Vec::clone()`** (e.g., `time: bufset.time.clone()`).
3.  **Impact:** `Vec::clone()` performs a **new heap allocation** and a **full `memcpy`** of the *entire* buffer's contents (even if it's all zeros).

**Conclusion:** The primary goal of eliminating heap allocation is defeated by the `clone()` operation at the end. The optimization is *not* non-functional—it successfully amortizes the cost of *growing* the pooled `Vec`s—but it fails to prevent a new, large allocation and a costly copy on *every* function call.

-----

## 4\. Summary

  * The Python baseline is already efficient ($O(n \log n)$) but suffers from call overhead, non-adaptive logic, and redundant allocations.
  * The Rust implementation eliminates these inefficiencies through:
      * Zero-copy memory access (`PyReadonlyArray1`)
      * GIL-free multithreading (`py.allow_threads`)
      * Adaptive algorithm selection (`autocorr_adaptive`)
      * Global, thread-safe FFT plan caching (`PLAN_CACHE`)
      * A **flawed buffer pool** that correctly reuses `Vec` *capacity* but still triggers a costly `Vec::clone()` on every call.
      * ILP-aware loop unrolling (`autocorr_direct_norm`)

Refactoring the `BUFFER_POOL` to return *mutable borrows* (`&mut Vec<f64>`) inside a closure or via a RAII guard—instead of *clones*—would eliminate this final allocation bottleneck and likely yield significant additional gains on repeated workloads.

-----

## 5\. Hardware and SIMD Context

All performance measurements were run on the following configuration:

| Component | Specification |
| :--- | :--- |
| **CPU** | Intel Core i7-4770 (Haswell) overclocked to 4.1 GHz |
| **Motherboard** | ASUS Z87 |
| **RAM** | 16 GB DDR3-2400 MHz |
| **GPU** | NVIDIA GTX 980 Ti (OC) |
| **OS** | Windows 10 ×64 |

This configuration is a critical part of the analysis. The **Intel Haswell** architecture was the first to introduce the **AVX2** (Advanced Vector Extensions 2) instruction set, a 256-bit SIMD engine.

This benchmark is **not** "free from SIMD bias"; it is a **direct competition between heavily SIMD-optimized libraries**:

1.  **Rust (`realfft`)**: This crate depends on `rustfft`, which has explicit SIMD optimizations for AVX/AVX2 and FMA (Fused Multiply-Add), providing massive acceleration.
2.  **Rust (`rayon`)**: Parallel iterators and reducers are often implemented using SIMD instructions.
3.  **Rust (Compiler)**: The manually unrolled loop in `autocorr_direct_norm` is a classic optimization pattern that strongly hints to `rustc`'s auto-vectorizer to use AVX2 instructions, performing multiple `f64` operations per cycle.
4.  **SciPy (Baseline)**: SciPy's FFT backend (e.g., Intel MKL or FFTW) is itself aggressively optimized using AVX2.

Therefore, this benchmark provides a realistic evaluation of performance where memory management, algorithmic strategy, and efficient **leveraging of modern CPU SIMD capabilities** are the deciding factors.