# TESTS.md — Validation and Numerical Robustness

This project includes a complete testing suite managed with `pytest`, ensuring **numerical correctness**, **backend parity**, and **robust behavior** across all modules.

**Main conclusion:** all test suites (Correctness, Parity, Statistical, and Asian Option Logic) pass successfully on both CPU and GPU implementations.

---

## 🚀 1. Running the Full Test Suite

A dedicated orchestration script is provided to run **all test suites and benchmarks**, compiling results into a unified log file.

```bash
# Run all tests and benchmarks
python tests/run_all_tests_and_benchmarks.py
````

This script produces two outputs:

1. **Console Output** — real-time `pytest` progress, showing PASSED/FAILED/SKIPPED for each individual test.
2. **`tests/benchmark_results.txt`** — a full timestamped log archiving all `pytest` output for each suite.

---

## ✅ 2. Results — Full Success

**All six test suites executed successfully (`[OK] PASSED`).**

Excerpt from `tests/benchmark_results.txt`:

```
================================================================================
                        TEST AND BENCHMARK SUMMARY
================================================================================

Detailed Results:
--------------------------------------------------------------------------------
Gpu Correctness                 [OK] PASSED
Correctness                     [OK] PASSED
Asian Correctness               [OK] PASSED
Gpu Benchmarks                  [OK] PASSED
New Benchmarks                  [OK] PASSED
Asian Benchmarks                [OK] PASSED

================================================================================
TOTAL: 6/6 test suites passed
================================================================================
[SUCCESS] ALL TEST SUITES PASSED!
================================================================================
```

This confirms that the optimized GPU implementation is not only faster, but also **numerically correct, stable, and reliable**.

---

## 🗂️ 3. Test Categories

The testing suite is divided into multiple files, each targeting specific validation objectives.

---

### `tests/test_correctness.py` and `tests/test_correctness_gpu.py`

These validate the **core GBM simulation logic** for both CPU (NumPy) and GPU (CuPy) backends.

* **Shape Tests:** Ensure `paths` and `time_grid` have the correct dimensions (`(n_steps + 1, n_paths)`).
* **Value Tests:** Verify no `NaN` or `Inf`, all prices are strictly positive (`> 0`), and the simulation starts exactly at `s0` (`paths[0, :] == s0`).
* **Reproducibility Tests (Seed):** Confirm that using the same seed yields numerically identical results across repeated runs.
* **Statistical Moment Tests:** Verify that the mean and standard deviation of simulated log-returns match theoretical parameters ($\mu$, $\sigma$) over large samples.
* **Functional Tests:** Validate that `dividend_yield` correctly decreases the drift, and `antithetic=True` effectively reduces variance.

---

### `tests/test_asian_option_correctness.py`

This suite focuses on validating the **Asian option pricing logic**, using simulated paths as input.

* **Payoff Logic Tests:** Use *deterministic* price paths (e.g. `[100, 101, 102, 103, 104]`) to verify that computed call/put prices (ITM, ATM, OTM) match exact analytical expectations.
* **Edge Case Tests:** Validate behavior under zero volatility ($\sigma = 0$), where the price should match the deterministic analytical solution.
* **Statistical Parity Tests:** Confirm that CPU (NumPy) and GPU (CuPy), although based on different random number generators, converge to statistically equivalent option prices (`rel_diff < 0.05`).

---

### `tests/test_benchmark_*.py` and `tests/test_asian_option_benchmark.py`

While primarily designed for performance analysis (`BENCHMARKS.md`), these files also serve as **integration tests**.

They run full end-to-end simulations on both backends across different scales, validating that:

* The code handles varying workloads without memory errors.
* VRAM chunking (`max_paths_per_chunk`) works correctly.
* Precision conversions (`float32` / `float64`) remain consistent.

---

## 📊 4. Summary

| Category              | Purpose                      | Backend     | Status   |
| :-------------------- | :--------------------------- | :---------- | :------- |
| **Correctness**       | Core GBM validation          | CPU (NumPy) | ✅ PASSED |
| **GPU Correctness**   | Core GBM validation          | GPU (CuPy)  | ✅ PASSED |
| **Asian Logic**       | Option payoff correctness    | CPU & GPU   | ✅ PASSED |
| **Statistical Tests** | Drift/volatility consistency | CPU & GPU   | ✅ PASSED |
| **Benchmarks**        | Performance & scalability    | CPU & GPU   | ✅ PASSED |
| **Integration**       | Full pipeline validation     | CPU & GPU   | ✅ PASSED |

---

### ✅ Final Verdict

All test suites confirm:

* **Functional correctness:** GBM simulation and Asian payoff logic are accurate.
* **Numerical stability:** No floating-point anomalies or divergence between backends.
* **Hardware parity:** Both implementations are mathematically equivalent.
* **Production readiness:** The optimized GPU version is robust for high-throughput Monte Carlo simulations.

> **Conclusion:**
> The project achieves both *quantitative validation* and *engineering-grade robustness*.
> Every computation tested — from path generation to payoff valuation — passes strict parity and stability checks on both CPU and GPU.