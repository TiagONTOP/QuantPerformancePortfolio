## 1. Overview

This project is supported by a comprehensive and rigorous test suite designed to validate both the **semantic correctness** and **robustness** of all backtest implementations.

The main objective is to **prove** that the optimized implementations (`pandas` and `polars`) are numerically identical to the reference implementation (`suboptimal`), while properly handling a wide range of data scenarios and edge cases.

**Current CI/CD Status:**

```bash
> python -m pytest -v -m "not benchmark"

collected 41 items / 4 deselected / 37 selected
...
================= 37 passed, 4 deselected in 82.64s =================
````

* **37 logic and robustness tests:** All passing successfully
* **4 performance tests (benchmarks):** Intentionally deselected by the `-m "not benchmark"` filter

---

## 2. Configuration (`pytest.ini` and Markers)

To distinguish *logic* tests (which must always pass) from *performance* tests (intended for analysis only), the suite uses custom Pytest markers:

* **`@pytest.mark.benchmark`** – marks slow tests measuring execution time and memory (`test_benchmark.py`)
* **`@pytest.mark.slow`** – synonym for `benchmark`

The CI/CD command

```bash
pytest -m "not benchmark"
```

runs all tests *except* those marked as benchmarks, ensuring a fast and stable logical validation (the 37 logic tests).

---

## 3. `tests/test_correctness.py` — Numerical Parity Tests

**Goal:** Prove that `optimized == reference` under normal conditions.

**Total Tests:** 19

### 3.1 Methodology

This suite uses Pytest parametrization for combinatorial “stress testing”:

* **3 Data Configurations:** `SMALL`, `MEDIUM`, `LARGE`
* **3 Random Seeds:** `42`, `43`, `100` for reproducible data generation
* **2 Implementations:** `pandas` vs. `suboptimal`, `polars` vs. `suboptimal`

This produces `3 × 3 × 2 = 18` parity tests.
A final test (`test_pandas_polars_parity`) ensures both optimized versions are also numerically identical — for a total of **19 tests**.

### 3.2 Critical Analysis: Tolerance (`atol`) and Floating-Point Drift (FP)

The key challenge in this suite is numerical tolerance handling.
We don’t just assert results are “close” — we validate *why* any difference exists.

The helper `parity_assert` (from `tools/utils.py`) is used with different tolerances (`atol`):

#### 1. `strategy_returns`: `atol = 1e-12` (Near machine precision)

* **Why?** The return is defined as $ R_t = (E_t - E_{t-1}) / E_{t-1} $.
* Even if capital (`equity`, $E_t$) accumulates FP rounding drift, both $E_t$ and $E_{t-1}$ share almost identical FP noise.
  The subtraction cancels it out, leaving extremely small differences.

#### 2. `portfolio_equity`: `atol` up to `6e-8` (Expected drift)

* **Why?** Capital evolves cumulatively over `T` timesteps:

  * The `suboptimal` loop performs *additive accumulation*.
  * The `optimized` versions use *multiplicative accumulation* via `cumprod`.
* Though mathematically equivalent, they accumulate FP rounding errors differently.
* The test confirms that drift remains small and **grows predictably with dataset size** (`1e-8` for SMALL → `6e-8` for LARGE) — the expected and correct FP behavior.

---

## 4. `tests/test_edge_cases.py` — Robustness Tests

**Goal:** Ensure all three implementations behave identically and correctly under abnormal data, extreme parameters, and error conditions.

**Total Tests:** 18

This suite ensures that the vectorized logic fully reproduces all `if/else` branches and edge conditions from the reference loop.

### 4.1 Degenerate Data Tests

* **Oversized Window (`test_window_too_large...`):**
  Tests `window >= n_obs`. All implementations must return empty series.

* **No Trades (`test_no_trade_high_thresholds...`):**
  Uses unreachable thresholds (`thr = 1e9`). All positions remain at 0; returns are all zeros.

* **Single Asset (`test_single_asset...`):**
  Tests `n_assets = 1` to ensure correct 1D handling in NumPy logic.

* **Zero Returns (`test_zero_returns...`):**
  Sets all returns `r = 0`.
  P&L must remain zero (except for small decreases from transaction costs, identical across all versions).

### 4.2 Robustness and Invariance Tests

* **Column Order Invariance (`test_column_order_invariance...`):**
  Confirms that a DataFrame with shuffled columns (e.g., `signal_10`, `signal_1`) yields identical results, since the code must sort them numerically internally.

* **Handling of `NaN` Values (`test_nan_in_signals...`):**
  Confirms that NaN signals are gracefully handled — producing zero positions.

### 4.3 Validation and Metadata Tests

* **Invalid Parameters (`test_invalid_threshold_params...`):**
  Ensures all implementations raise `AssertionError` or `ValueError` when `thr_long < thr_short`.

* **Output Metadata (`test_output_metadata...`):**
  Validates that returned `pd.Series` objects have correct `name`, `dtype` (`float64`), and sorted `DatetimeIndex`.

* **Metadata Parity (`test_metadata_parity_all_implementations`):**
  Confirms metadata consistency across all three implementations.

---

## 5. Test Execution

### Run the CI/CD Suite (Fast – Logic Only)

```bash
# Runs 37 logic and robustness tests
python -m pytest -v -m "not benchmark"
```

### Run a Specific File

```bash
# Run only numerical parity tests
python -m pytest tests/test_correctness.py -v

# Run only edge-case tests
python -m pytest tests/test_edge_cases.py -v
```

### Run the Full Suite (Including Benchmarks)

```bash
# Runs all 41 tests (37 logic + 4 performance)
python -m pytest -v
```
