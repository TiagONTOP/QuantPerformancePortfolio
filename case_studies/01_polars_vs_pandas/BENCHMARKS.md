# Performance Analysis: Backtest Vectorization (Pandas vs. Polars)

## 1\. Executive Summary

This analysis compares three implementations of a backtesting strategy to evaluate the impact of vectorization and the choice of libraries (Pandas vs. Polars/NumPy).

The optimized (vectorized) versions demonstrate a **drastic performance improvement**, achieving a **speedup of 615x** compared to the reference implementation (Python loop). This acceleration is not merely due to the removal of Python loops, but to a **fundamental change in algorithmic complexity**.

The hybrid **Polars/NumPy** implementation proved to be the most performant in all configurations, outperforming the pure Pandas version by 1.1x to 2.6x, while also being more memory-efficient (on the Python side).

## 2\. Benchmark Methodology

### Test Configurations

The benchmarks were executed on three distinct data scales to evaluate the scalability of the solutions.

| Configuration | `sample_size` (Days, $T$) | `n_backtest` (Assets, $N$) | `window` (Window, $W$) |
| :--- | :--- | :--- | :--- |
| **SMALL** | 500 | 10 | 50 |
| **MEDIUM** | 1500 | 50 | 100 |
| **LARGE** | 3000 | 100 | 100 |

### Measurement Protocol

  * **Time:** Measured via `time.perf_counter()` for high resolution.
  * **Memory:** Measured via `tracemalloc` to trace the peak allocation *on the Python side*.

#### Technical Caveat on Memory Measurement

It is essential to note that `tracemalloc` **only captures memory allocated by the Python API**. It **does not trace** direct allocations made by the C (NumPy) or Rust (Polars) backends. The memory figures reported below therefore represent the **Python interfacing overhead** and underestimate the true total memory footprint of the process.

## 3\. Detailed Results

The following results are extracted from the full execution of the `test_benchmark_summary` suite.

| Config | Implementation | Time (ms) | Memory (MB) | Speedup (vs. Suboptimal) |
| :--- | :--- | :--- | :--- | :--- |
| **SMALL** | Suboptimal | 1029.77 | 0.28 | - |
| (500×10) | Pandas | 22.31 | 0.56 | **46.16x** |
| | **Polars (Hybrid)** | **8.62** | **0.45** | **119.42x** |
| | | | | |
| **MEDIUM** | Suboptimal | 9731.33 | 1.84 | - |
| (1500×50) | Pandas | 34.73 | 7.67 | **280.24x** |
| | **Polars (Hybrid)** | **30.80** | **6.23** | **315.91x** |
| | | | | |
| **LARGE** | Suboptimal | 35238.49 | 7.11 | - |
| (3000×100) | Pandas | 79.15 | 29.61 | **445.23x** |
| | **Polars (Hybrid)** | **57.26** | **23.89** | **615.43x** |

## 4\. In-Depth Technical Analysis

### 4.1. Source of Inefficiency: The $O(T \times W \times N)$ Algorithmic Complexity

The `suboptimal` implementation suffers from an algorithmic complexity problem. Its code structure is as follows:

```python
# Complexity: O(T)
for t in range(window, n_obs - 1):
    
    # Complexity: O(W * N)
    # This operation is recalculated on *every* T iteration
    sigmas = sig.iloc[t - window:t].std() 
    
    # Complexity: O(N)
    for j in range(n_assets):
        ...
```

The total complexity is not $O(T \times N)$, but **$O(T \times (W \times N + N))$**, which simplifies to **$O(T \times W \times N)$**. The execution time grows linearly with the window size ($W$), in addition to the $T$ and $N$ dimensions. This is the primary cause of the execution times of several seconds (or even minutes).

### 4.2. Source of Optimization: The $O(T \times N)$ Algorithmic Complexity

Both optimized implementations (`pandas` and `polars`) solve this problem by replacing the $T \times W \times N$ loop with a single "rolling window" operation, which uses an efficient online algorithm (e.g., Welford):

```python
# Complexity: O(T * N)
sigma = signal.rolling(window=window, min_periods=1).std().shift(1)
```

All subsequent operations (cost calculation, position masks, `cumprod`) are also vectorized and execute in **$O(T \times N)$** complexity.

**The performance gain (up to 615x) stems primarily from this change in complexity from $O(T \times W \times N)$ to $O(T \times N)$.**

### 4.3. Pandas vs. Polars (Hybrid): The Choice of Tool

Both optimized versions share the same $O(T \times N)$ complexity, but their performance differs.

1.  **Optimized Pandas:** Relies entirely on Pandas operations, which use a Cython/NumPy backend.
2.  **Optimized Polars (Hybrid):** Uses a targeted strategy:
      * **Polars (Rust):** Only for the `rolling_std` operation, which is its strength (a multi-threaded and efficient Rust implementation).
      * **NumPy (C/Fortran):** The rest of the logic (boolean masks, matrix `cumprod`) is performed in NumPy, which is the optimal tool for dense matrix operations.

The Polars implementation is faster because it uses the Rust engine (Polars) for the windowing operation (the bottleneck compared to NumPy) and the C engine (NumPy) for matrix operations (where NumPy excels). It also allocates less memory *on the Python side* (see Caveat 2.1).

### 4.4. Note on Numerical Parity (FP Drift)

The optimized implementations are **mathematically equivalent** to the reference, but they are **not numerically identical (bit-for-bit)**.

  * The **reference (loop)** accumulates capital via *repeated additions* ($cap_t = cap_{t-1} + pnl - cost$).
  * The **vectorized versions** calculate capital via a *cumulative product* of growth factors ($cap_T = cap_0 \times \prod G_t$).

These two distinct computational paths accumulate floating-point (FP) errors differently. This is expected and correct behavior. For this reason, the parity tests (`test_correctness.py`) use a larger absolute tolerance (e.g., `atol=6e-8`) for the capital (`equity`), while verifying that the daily returns (`strategy_returns`) maintain near-machine precision (`atol \approx 1e-12`).

## 5\. Conclusion

1.  **Vectorization is Non-Negotiable:** The shift from $O(T \times W \times N)$ to $O(T \times N)$ complexity is the primary source of the 46x to 615x performance gain, transforming an operation from seconds to milliseconds.
2.  **The reference implementation (loop)** is validated as a logical "source of truth" but is entirely unsuitable for any use beyond semantic validation.
3.  **The hybrid Polars/NumPy approach** is the highest-performing solution. It demonstrates mature technical understanding by using each library for its primary strength (Polars for columnar rolling operations, NumPy for dense matrix algebra).