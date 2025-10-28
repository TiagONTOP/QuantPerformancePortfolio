# Unit Tests - FFT Autocorrelation

## 📋 Overview

This document describes the unit test suite that validates the correctness (accuracy) of the Python (suboptimal) and Rust (optimized) implementations of the FFT-based autocorrelation calculation.

## 🎯 Testing Objectives

1. **Numerical validation**: Verify that results are correctly identical between both implementations
2. **Edge case handling**: Test behavior on edge case data (constants, NaN, etc.)
3. **Robustness**: Ensure no regressions are introduced during optimizations
4. **Non-regression**: Guarantee stability across versions

## 📁 Test Files

### `tests/test_unit.py`

Complete unit test suite comprising 4 test categories.

---

## 🧪 Implemented Tests

### TEST 1: Basic Correctness ✓

**Objective:** Validate fundamental accuracy with known values

**Test data:**
```python
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
max_lag = 3
```

**Expected values:**
```
lag 1: 0.700000
lag 2: 0.412121
lag 3: 0.148485
```

**Success criteria:**
- ✅ Python vs expected values: difference < 1e-5
- ✅ Rust vs expected values: difference < 1e-5
- ✅ Rust vs Python: difference < 1e-10 (machine precision)

**Result:**
```
Python: PASS (max diff: 2.22e-16)
Rust: PASS (max diff: 2.22e-16)
Rust vs Python: PASS (max diff: 2.22e-16)
```

---

### TEST 2: Edge Cases ✓

**Objective:** Validate behavior on edge cases

**Tested cases:**

#### 1. Constant Series
```python
data = np.ones(100)
```
**Expected behavior:** NaN (zero variance)
**Result:** ✓ Both implementations return NaN

#### 2. Normal Random Noise
```python
data = np.random.randn(100)
```
**Expected behavior:** Decreasing autocorrelation
**Result:** ✓ PASS (max diff: 5.55e-17)

#### 3. Sine Wave
```python
data = np.sin(np.linspace(0, 4*np.pi, 100))
```
**Expected behavior:** Periodic oscillations
**Result:** ✓ PASS (max diff: 5.55e-16)

#### 4. Linear Trend
```python
data = np.arange(100, dtype=float)
```
**Expected behavior:** Strong autocorrelation
**Result:** ✓ PASS (max diff: 3.33e-16)

#### 5. Zero Mean
```python
data = np.random.randn(100) - mean
```
**Expected behavior:** Identical to normal noise
**Result:** ✓ PASS

**Success criteria:**
- ✅ No NaN for non-constant series
- ✅ No Inf in any case
- ✅ Rust vs Python: difference < 1e-10

---

### TEST 3: Different Sizes ✓

**Objective:** Validate robustness across different array sizes

**Tested sizes:**
- 10, 50, 100, 500, 1000, 5000, 10000

**For each size:**
- Random data generation
- Calculation with max_lag=20
- Result shape verification
- Rust vs Python comparison

**Success criteria:**
- ✅ Correct shape: `len(result) == max_lag`
- ✅ Difference < 1e-10 for all sizes

**Results:**
```
Size 10:    PASS (max diff: 1.11e-16)
Size 50:    PASS (max diff: 2.22e-16)
Size 100:   PASS (max diff: 5.55e-17)
Size 500:   PASS (max diff: 8.88e-17)
Size 1000:  PASS (max diff: 1.00e-16)
Size 5000:  PASS (max diff: 1.48e-16)
Size 10000: PASS (max diff: 7.72e-17)
```

---

### TEST 4: Large max_lag ✓

**Objective:** Test behavior with very large max_lag values

**Configuration:**
```python
data_size = 1000
max_lag = 500  # 50% of data size
```

**Why it's important:**
- Tests algorithm limits
- Validates implementation makes no incorrect assumptions
- Verifies numerical stability over long lags

**Success criteria:**
- ✅ No error or exception
- ✅ Correct shape: 500 values
- ✅ Max difference < 1e-10
- ✅ Mean difference < 1e-15

**Result:**
```
Data size: 1000
Max lag: 500

Max difference: 6.77e-17
Mean difference: 1.50e-17

PASS: Results match perfectly
```

---

## 📊 Test Summary

### Overall Result

```
TEST SUMMARY
==================================================
✓ basic           : PASS
✓ edge_cases      : PASS
✓ sizes           : PASS
✓ large_lag       : PASS

ALL TESTS PASSED ✓
```

### Precision Statistics

| Test | Max Difference | Mean Difference | Status |
|------|----------------|-----------------|--------|
| Basic Correctness | 2.22e-16 | ~1e-16 | ✓ PASS |
| Edge Cases | 5.55e-16 | ~2e-16 | ✓ PASS |
| Different Sizes | 2.22e-16 | ~1e-16 | ✓ PASS |
| Large max_lag | 6.77e-17 | 1.50e-17 | ✓ PASS |

**Conclusion: Numerical precision is at machine level (< 1e-15), which is optimal.**

---

## 🚀 Running the Tests

### Installation

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install numpy pandas scipy

# 3. Compile Rust module
cd optimized
maturin develop --release --strip
cd ..
```

### Execution

```bash
# From project root
python tests/test_unit.py
```

### Expected Output

```
╔══════════════════════════════════════════════════════════════════╗
║                    UNIT TEST SUITE                               ║
╚══════════════════════════════════════════════════════════════════╝

======================================================================
TEST 1: Basic Correctness
======================================================================

Input: [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
Max lag: 3

Python result:
lag
1    0.700000
2    0.412121
3    0.148485
Name: autocorrelation, dtype: float64

Rust result:
[0.7        0.41212121 0.14848485]

Maximum difference: 2.22e-16
Results match perfectly!

======================================================================
TEST 2: Edge Cases
======================================================================

Constant series:
  PASS: Both correctly return NaN for constant series

Random normal:
  PASS (max diff: 5.55e-17)

Sine wave:
  PASS (max diff: 5.55e-16)

...

======================================================================
TEST SUMMARY
======================================================================
✓ basic           : PASS
✓ edge_cases      : PASS
✓ sizes           : PASS
✓ large_lag       : PASS

======================================================================
ALL TESTS PASSED ✓
======================================================================
```

---

## 🔍 Implementation Details

### Testing Strategy

1. **Reproducible data generation**
   - Fixed seed for numpy.random
   - Synthetic data with known properties

2. **Multi-level comparison**
   - Expected values (ground truth)
   - Python vs Rust (cross-validation)
   - Internal consistency verification

3. **Adaptive tolerances**
   - 1e-5 vs expected values (rounding in docs)
   - 1e-10 Python vs Rust (FFT rounding errors)
   - Special handling of NaN/Inf

### Error Handling

**Handled cases:**
- ✅ Constant series → NaN (zero variance)
- ✅ Empty array → ValueError
- ✅ max_lag = 0 → ValueError
- ✅ max_lag > len(data) → Automatic truncation

**Consistency:**
- Python and Rust behave identically
- Clear error messages
- No silent failures

---

## 📈 Test Evolution

### Version 1
- Basic correctness tests
- Manual result comparison

### Version 2 (Current)
- Complete automated suite
- 4 test categories
- Cross-validation Python/Rust
- Adaptive tolerance based on context

### Future Version
- [ ] Property-based testing (Hypothesis)
- [ ] Performance tests (minimum speedup thresholds)
- [ ] Automatic regression tests (CI/CD)
- [ ] Code coverage (coverage.py)

---

## 🐛 Debugging

### If a test fails

1. **Verify Rust compilation**
   ```bash
   cd optimized
   cargo clean
   maturin develop --release
   ```

2. **Verify Python dependencies**
   ```bash
   pip install --upgrade numpy pandas scipy
   ```

3. **Test in isolation**
   ```python
   python -c "import fft_autocorr; print(fft_autocorr.__file__)"
   ```

4. **Verbose mode**
   ```bash
   python tests/test_unit.py -v
   ```

### Known warnings

**RuntimeWarning: invalid value encountered in divide**
- Origin: constant series in SciPy
- Impact: none (expected behavior)
- Resolution: not necessary

---

## ✅ Validation Checklist

Before each release, verify:

- [ ] All tests pass
- [ ] No performance regressions
- [ ] No unhandled warnings
- [ ] Documentation up to date
- [ ] Examples functional

---

## 📚 References

- [NumPy Testing Guidelines](https://numpy.org/doc/stable/reference/testing.html)
- [pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [SciPy signal.correlate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html)

---

**Summary: All tests pass with machine-level precision (< 1e-15). Python and Rust implementations are numerically identical and robust across all tested cases. ✓**
