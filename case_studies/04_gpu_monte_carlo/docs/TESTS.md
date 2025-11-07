# TESTS.md — Validation and Numerical Robustness

This document outlines the complete testing suite (58 tests) for the GPU Monte Carlo optimization case study, ensuring **numerical correctness**, **backend parity**, and **robust behavior** across CPU (NumPy) and GPU (CuPy) implementations.

**Conclusion:** The test suite passes successfully, confirming the GPU (CuPy) implementation is not only fast but also numerically correct, stable, and produces equivalent results to the CPU (NumPy) implementation.

-----

## 🚀 Running the Test Suite

Tests are consolidated into two primary files:

### 1\. Correctness Tests (Robustness & Accuracy)

This suite validates that the code produces correct results and handles errors.

```bash
# Run all 44 correctness tests
python -m pytest tests/test_correctness.py -v
```

**Result:** ✅ **43 Passed, 1 Skipped**

-----

### 2\. Performance Tests (Benchmarks)

This suite measures speedup and validates optimization strategies.

```bash
# Run all 14 performance benchmarks
python -m pytest tests/test_benchmark.py -v -s
```

**Result:** ✅ **14 Passed**

-----

## 🗂️ Test Coverage

### ✅ Correctness Suite (`test_correctness.py`)

The **44 tests** in this suite confirm the code is numerically accurate and robust. Coverage includes:

  * **Basic Logic (CPU & GPU):**

      * Verifies correct output `shape`.
      * Ensures no `NaN` or `Inf` values.
      * Confirms all paths start at `s0`.

  * **Backend Parity (CPU vs. GPU):**

      * Proves that CPU and GPU give **numerically identical results** when fed the same random shock data.
      * Validates simulation reproducibility with a fixed `seed`.

  * **Robustness & Edge Cases:**

      * Tests error handling for invalid inputs (e.g., `s0 <= 0`, `sigma < 0`).
      * Validates the VRAM chunking mechanism.

  * **Feature Validation:**

      * Validates the Asian option payoff logic with deterministic paths.
      * Confirms correct behavior of *antithetic variates* and *dividend yield*.

  * **Statistical Consistency:**

      * Verifies that the mean and volatility of simulated paths match GBM theory.

-----

### ⚡ Performance Suite (`test_benchmark.py`)

The **14 benchmarks** in this suite quantify performance gains and validate architectural choices. The analysis covers:

  * **CPU vs. GPU Speedup:**

      * Measures end-to-end performance on **Small, Medium, and Large** problem sizes.

  * **Precision Impact:**

      * Quantifies the performance difference between `float32` and `float64` on the GPU.

  * **Zero-Copy Analysis:**

      * Measures the *additional* speedup gained by avoiding CPU-GPU memory transfers (`device_output=True`).
      * Analyzes memory transfer overhead to prove its bottleneck status.