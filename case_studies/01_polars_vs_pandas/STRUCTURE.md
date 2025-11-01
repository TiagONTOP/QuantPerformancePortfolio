# Project Architecture: Backtest Vectorization

## 1. Overview

This document details the technical architecture of the case study **`01_polars_vs_pandas`**.  
The project compares a reference backtest implementation (Python loop) with two optimized and vectorized implementations.

The main goal is to demonstrate the migration from a sub-optimal algorithmic complexity **O(T × W × N)** to a more efficient **O(T × N)**, and to benchmark the performance of *rolling window* engines in **Pandas (Cython)** versus **Polars (Rust)**.

---

## 2. File Structure

```

01_polars_vs_pandas/
├── suboptimal/
│   └── backtest.py          # Impl. 0: Reference (loop-based) – O(T*W*N)
│
├── optimized/
│   └── backtest.py          # Impl. 1: Pandas/NumPy hybrid – O(T*N)
│                            # Impl. 2: Polars/NumPy hybrid – O(T*N)
│
├── tests/
│   ├── test_correctness.py  # Numerical parity tests (vs reference)
│   ├── test_edge_cases.py   # Robustness tests (NaNs, zeros, etc.)
│   └── test_benchmark.py    # Performance tests (time & memory)
│
├── tools/
│   ├── utils.py             # Data generation, metrics, and testing utilities
│   └── exceptions.py        # Custom exceptions (InvalidParameterError)
│
├── README.md                # Project summary
├── STRUCTURE.md             # This file (architecture description)
├── BENCHMARKS.md            # Performance analysis
├── pyproject.toml           # Project dependencies
└── pytest.ini               # Pytest configuration and markers (e.g., benchmark)

````

---

## 3. Implementation Components

### 3.1 Implementation 0: Reference (Loop-Based)

- **File:** `suboptimal/backtest.py`  
- **Function:** `suboptimal_backtest_strategy`

**Architecture:**  
This serves as the *logical ground truth*.  
It uses two nested Python loops: one over time (`t`) and one over assets (`j`).

```python
for t in range(window, n_obs - 1):
    # Outer loop (Time)
    sigmas = sig.iloc[t - window:t].std()  # Bottleneck: O(W*N)
    
    for j in range(n_assets):
        # Inner loop (Assets)
        # if/else logic for position, cost, and capital update
        cap[j] = cap[j] + pnl_j - cost_amt
````

**Performance Analysis:**

* The loop `for t...` runs approximately `T` times.
* The `.std()` operation is recomputed at each step: `O(W*N)`.
* **Total Complexity: O(T × W × N)**.

---

### 3.2 Optimized Implementations (Hybrid)

* **File:** `optimized/backtest.py`

The two optimized implementations (`pandas` and `polars`) share a **hybrid NumPy architecture** for ~90% of their logic.
They differ only in the method used to compute the `rolling_std`.

#### Shared Architecture (NumPy Matrix Logic)

1. **Conversion to NumPy:** All DataFrames (signals, sigmas, returns) are converted to `np.ndarray`.
2. **Position Logic:** Boolean masking (`signal_vals > thr_long_vals`).
3. **Transaction Costs:** Vectorized arithmetic (`change = (desired != pos_prev)` …).
4. **Growth Factor `G`:** Computed as a NumPy matrix.
5. **Capital Path:** Calculated via `np.cumprod(G, axis=0)`.
6. **Floor (Dead State):** Implemented with `np.maximum.accumulate`.
7. **Equity:** `cap_path.sum(axis=1)`.
8. **Output:** Final slice reconverted to `pd.Series`.

#### Impl. 1: Pandas/NumPy Hybrid (`optimal_backtest_strategy_pandas`)

* **Rolling Window (Pandas Engine):**

  ```python
  sigma = signal.rolling(window).std().shift(1)
  ```
* **Backend:** Pandas (Cython) for rolling computation, NumPy for all matrix logic.

#### Impl. 2: Polars/NumPy Hybrid (`optimal_backtest_strategy_polars`)

* **Rolling Window (Polars Engine):**

  ```python
  signal_pl = pl.from_pandas(signal_pd)
  sigma_exprs = [pl.col(c).rolling_std(...) for c in signal_pl.columns]
  sigma_pl = signal_pl.select(sigma_exprs)
  ```
* **Backend:** Polars (Rust) for rolling computation, NumPy for matrix logic.

---

## 4. Supporting Components

### 4.1 `tools/utils.py`

The central support library of the project — separates data generation, metric computation, and testing utilities from the backtest logic.

* **`generate_synthetic_df`**: Generates test signals and returns with a controlled Information Coefficient (IC).
* **`parity_assert`**: Core of `test_correctness`; compares two Pandas series with absolute tolerance (`atol`).
* **Financial Metrics:** Includes `sharpe_ratio`, `capm_alpha_beta_tstats`, `max_drawdown`, `sortino_ratio`, `calmar_ratio`.
* **`numeric_sort_cols`**: Ensures columns (`signal_1`, `signal_10`, …) are sorted numerically, not lexicographically.

### 4.2 `tools/exceptions.py`

Defines custom error types for robust parameter validation.

* **`InvalidParameterError`**: Raised by all backtest implementations when parameters are inconsistent
  (e.g., `signal_sigma_thr_long < signal_sigma_thr_short`) to enforce a fail-fast design.

### 4.3 `tests/`

* **`test_correctness.py`**: Runs the reference and optimized implementations on identical synthetic data and checks numerical parity (with expected FP drift tolerance, e.g., `atol=6e-8` for equity).
* **`test_edge_cases.py`**: Stress-tests all implementations on edge scenarios (NaNs, single asset, zero returns, extreme window sizes).
* **`test_benchmark.py`**: Benchmarks execution time (`time.perf_counter`) and Python-side memory (`tracemalloc`) for SMALL, MEDIUM, and LARGE datasets.

---

## 5. Core Architectural Insights

1. **Primary Speedup is Algorithmic:**
   The ~600× improvement comes from the reduction of complexity from **O(T × W × N)** to **O(T × N)** — removing the redundant rolling window term `W`.

2. **Real Comparison: Polars (Rust) vs Pandas (Cython):**
   This project does *not* compare “Pandas vs Polars” globally.
   It compares two **identical NumPy hybrid** implementations differing only in their rolling engine.
   The 1.1×–2.6× difference directly reflects the performance gap between Polars’ (Rust) and Pandas’ (Cython) rolling window backends.

3. **NumPy as the Matrix Engine:**
   Both optimized implementations conclude — correctly — that for dense matrix operations (boolean masking, `cumprod`, etc.), NumPy remains optimal.

---