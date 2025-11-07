# `BENCHMARKS.MD`: Quantitative Performance Analysis

## 1. Objective

This document presents and analyzes the performance benchmark results comparing the `suboptimal` (Python/Scipy) reference implementation against the `optimized` (Rust/PyO3) implementation.

The objective is to quantify the speedup across different scenarios and link these gains to the specific architectural optimizations (adaptive strategy, parallelism, memory management) described in `README.MD`.

## 2. Test Configuration

* **Hardware:**
    * **CPU:** Intel Core i7 4770 @ 4.1 GHz (Overclocked)
    * **RAM:** 16 GB DDR3 @ 2400 MHz
* **Software:**
    * Python 3.12+
    * `pytest`
    * `scipy` (used by the baseline)
    * `rustc` (used to compile the optimized module in `release` mode)
* **Methodology:**
    * Tests are executed using `pytest`.
    * Duration is measured using `time.perf_counter()`.
    * To reduce variance, each benchmark is run multiple times. The reported time is the **median** of these executions.
    * **Speedup** is calculated as `Speedup = Python_Time / Rust_Time`.

---

## 3. Benchmark Results

### 3.1. Test 1: Performance by Series Size (Variable `n`, fixed `max_lag = 50`)

This test evaluates the impact of the input series size (`n`) for a small, fixed `max_lag`.

| Size (n) | Python (Scipy) | Rust (Optimized) | **Speedup** | Rust Method (Analysis) |
| ---: | ---: | ---: | :---: | :--- |
| 100 | 0.300 ms | 0.004 ms | **70.69x** | `Direct` |
| 1,000 | 0.357 ms | 0.038 ms | **9.31x** | `Direct` |
| 10,000 | 0.968 ms | 0.175 ms | **5.54x** | `Direct` |
| 50,000 | 7.088 ms | 0.731 ms | **9.70x** | `Direct` |

#### Analysis (Test 1)

1.  **70.69x Speedup (n=100):** This result is the clearest proof of **Python overhead**. For such a small series, the computational cost is negligible. Scipy's time (0.300 ms) is almost entirely dominated by function calls, NumPy type conversions, and internal buffer allocations. The Rust version, being pre-compiled and using the `Direct` algorithm (a simple loop), has near-zero overhead (4 microseconds).

2.  **Adaptive Strategy in Action:** The heuristic analysis (`autocorr_adaptive`) shows that for *all* sizes in this test, the estimated $O(nk)$ cost of the `Direct` method is lower than the $O(m \log m)$ cost of the FFT.
    * Ex: For `n=50000, k=50`, the direct cost (with margin) is $\approx (50000 \cdot 50 \cdot 0.25) \cdot 1.2 = 750,000$ units.
    * The FFT cost (for $m = \text{next\_fast\_len}(99999) = 100000$) is $\approx 100000 \cdot \log_2(100000) \approx 1,661,000$.
    * `750,000 < 1,661,000`. Rust therefore **correctly** selects the `Direct` method (parallelized with `rayon`) and vastly outperforms Scipy, which uses the FFT (unnecessarily costly here).

3.  **Optimal 9.3x-9.7x Speedup:** These two points (`n=1000` and `n=50000`) show the best balance between Python overhead and computational load. The ~9-10x speedup represents the "pure" gain of Rust for the parallelized `Direct` algorithm, without being dominated by overhead (n=100) or cache effects.

---

### 3.2. Test 2: Performance by Lag (Fixed `n = 10,000`, variable `max_lag`)

This test is crucial as it challenges the **crossover point** of the adaptive heuristic.

| Max Lag (k) | Python (Scipy) | Rust (Optimized) | **Speedup** | Rust Method (Analysis) |
| ---: | ---: | ---: | :---: | :--- |
| 10 | 0.837 ms | 0.101 ms | **8.24x** | `Direct` |
| 50 | 0.968 ms | 0.175 ms | **5.54x** | `Direct` |
| 100 | 0.890 ms | 0.326 ms | **2.73x** | **`FFT`** |
| 200 | 1.158 ms | 0.344 ms | **3.36x** | **`FFT`** |
| 500 | 0.965 ms | 0.372 ms | **2.60x** | **`FFT`** |

#### Analysis (Test 2)

1.  **Scipy Stability:** Python/Scipy's time is almost constant (between 0.83ms and 1.16ms). This is expected: it *always* uses the $O(m \log m)$ FFT, where $m \approx 2n$. The `max_lag` has almost no impact on its compute time.

2.  **Rust's Crossover Point:** Rust's behavior is radically different and proves the heuristic's effectiveness.
    * **For k=10 to 50:** Rust's time increases (from 0.101ms to 0.175ms). This is the expected behavior of the `Direct` algorithm ($O(nk)$): compute time is proportional to `max_lag`.
    * **The Crossover (k=100):** The `autocorr_adaptive` heuristic detects that the cost of `Direct` (for `k=100`) now exceeds the cost of `FFT`.
        * **Heuristic Calculation (`n=10k`):**
        * FFT size `m = next_fast_len(19999) = 20000`.
        * `FFT` cost $\approx (20000 \cdot \log_2(20000)) + 1000 \approx 286,754$
        * `Direct` cost $\approx (10000 \cdot k \cdot 0.25) \cdot 1.2 = 3000 \cdot k$
        * **Crossover point:** `3000 * k >= 286,754` $\implies$ `k >= 95.6`.
    * The heuristic therefore **correctly** switches to **`FFT`** at `k=100`, because `3000 * 100 > 286,754`.
    * **For k=100 to 500:** Rust's time (0.326ms to 0.372ms) becomes stable again, just like Scipy's, because it is now also using the FFT.

3.  **Rust-FFT vs. Scipy-FFT Performance:** Even when both implementations use the FFT (k=100, 200, and 500), the Rust version is **2.60x to 3.36x faster**. This gain is explained by:
    * The **`PLAN_CACHE`** (amortizes FFT setup cost).
    * The **`BUFFER_POOL`** (guarantees **zero-allocation** for working buffers via the "loan pattern").
    * The use of **`realfft`** (R2C), ~2x more efficient than complex FFT.
    * The **`rayon`** parallelism on the power spectrum calculation.

---

### 3.3. Test 3: Repeated Calls (Cache)

This test measures efficiency by calling the function 100 times with the *same* parameters (`n=10,000`, `max_lag=50`).

| Metric | Python (Scipy) | Rust (Optimized) | Speedup |
| :--- | ---: | ---: | :---: |
| Total Time (100 calls) | 188.6 ms | 19.0 ms | 9.91x |
| Average Time per Call | 1.886 ms/call | 0.190 ms/call | **9.91x** |

#### Analysis (Test 3)

* The parameters (`n=10000`, `k=50`) force the use of the **`Direct`** method (since `k < 95.6`).
* The `Direct` method does *not* use the `PLAN_CACHE` or `BUFFER_POOL` (which are FFT-specific).
* The sustained speedup of **9.91x** is even better than the `5.54x` speedup from Test 3.1 (for `n=10k, k=50`), indicating that the Rust version maintains perfect cache performance on repeated calls, while Python's overhead accumulates.
* This gain comes from the raw efficiency of `rayon` (parallelism), loop unrolling, and low call overhead.

## 4. Overall Conclusion

1.  **Total Superiority:** The Rust module is faster than the Scipy baseline in *all tested scenarios*, with a speedup ranging from **2.60x** (worst-case, FFT vs. FFT) to **70.69x** (best-case, minimal Python overhead).
2.  **The Adaptive Algorithm is Key:** The most significant performance gain (5x to 70x) comes from using `autocorr_adaptive`. The parallelized `Direct` ($O(nk)$) implementation massively outperforms Scipy's FFT method when `max_lag` is small, which is a very common use case.
3.  **Effective FFT Optimizations:** Even when Rust must use the FFT, its implementation (plan caching, zero-alloc buffer pooling, R2C, `rayon`) is **2.60x to 3.36x** faster than Scipy's already-optimized version.