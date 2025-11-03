"""
Unit tests for FFT Autocorrelation implementations.

This module tests both the Python (suboptimal) and Rust (optimized) implementations
to ensure correctness and numerical accuracy.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "suboptimal"))

# Import implementations
from processing import compute_autocorrelation as compute_autocorrelation_python

try:
    import fft_autocorr
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


@pytest.fixture(scope="session", autouse=True)
def check_rust_availability():
    """Display warning if Rust module is not available."""
    if not RUST_AVAILABLE:
        print("\nWARNING: Rust module not available. Run 'maturin develop --release' in optimized/")
        print("Running tests with Python implementation only...\n")


def test_basic_correctness():
    """Test basic correctness with known values."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Correctness")
    print("=" * 70)

    # Simple test case
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    max_lag = 3

    print(f"\nInput: {data}")
    print(f"Max lag: {max_lag}")

    # Python implementation
    series = pd.Series(data)
    result_python = compute_autocorrelation_python(series, max_lag)

    print(f"\nPython result:\n{result_python}")

    # Expected values (approximately)
    expected = np.array([0.7, 0.412121, 0.148485])

    # Check Python
    diff_python = np.abs(result_python.values - expected)
    max_diff_python = np.max(diff_python)

    assert max_diff_python < 1e-5, f"Python result differs too much: {max_diff_python}"
    print(f"Python: PASS (max diff: {max_diff_python:.2e})")

    # Rust implementation
    if RUST_AVAILABLE:
        result_rust = fft_autocorr.compute_autocorrelation(data, max_lag)
        print(f"\nRust result:\n{result_rust}")

        # Check Rust
        diff_rust = np.abs(result_rust - expected)
        max_diff_rust = np.max(diff_rust)

        assert max_diff_rust < 1e-5, f"Rust result differs too much: {max_diff_rust}"
        print(f"Rust: PASS (max diff: {max_diff_rust:.2e})")

        # Check Rust vs Python
        diff_rust_python = np.abs(result_python.values - result_rust)
        max_diff = np.max(diff_rust_python)

        assert max_diff < 1e-10, f"Rust and Python differ too much: {max_diff}"
        print(f"\nRust vs Python: PASS (max diff: {max_diff:.2e})")


@pytest.mark.parametrize("name,data_func", [
    ("Constant series", lambda: np.ones(100)),
    ("Random normal", lambda: np.random.randn(100)),
    ("Sine wave", lambda: np.sin(np.linspace(0, 4*np.pi, 100))),
    ("Linear trend", lambda: np.arange(100, dtype=float)),
    ("Zero mean", lambda: np.random.randn(100) - np.random.randn(100).mean()),
])
def test_edge_cases(name, data_func):
    """Test edge cases and special scenarios."""
    print(f"\n{name}:")

    max_lag = 5
    data = data_func()
    series = pd.Series(data)

    result_python = compute_autocorrelation_python(series, max_lag)

    # Check for NaN or Inf
    if np.any(np.isnan(result_python.values)) and name != "Constant series":
        pytest.fail(f"Python produced NaN for non-constant series")

    if np.any(np.isinf(result_python.values)):
        pytest.fail(f"Python produced Inf")

    if RUST_AVAILABLE:
        result_rust = fft_autocorr.compute_autocorrelation(data, max_lag)

        # Check for NaN or Inf
        if np.any(np.isnan(result_rust)) and name != "Constant series":
            pytest.fail(f"Rust produced NaN for non-constant series")

        if np.any(np.isinf(result_rust)):
            pytest.fail(f"Rust produced Inf")

        # Compare Rust vs Python
        # For constant series, both should be NaN
        if name == "Constant series":
            assert np.all(np.isnan(result_python.values)) and np.all(np.isnan(result_rust)), \
                "Inconsistent handling of constant series"
            print(f"  PASS: Both correctly return NaN for constant series")
        else:
            diff = np.abs(result_python.values - result_rust)
            max_diff = np.max(diff)

            assert max_diff < 1e-10, f"Results differ by {max_diff:.2e}"
            print(f"  PASS (max diff: {max_diff:.2e})")
    else:
        print(f"  PASS (Python only, first 3 lags: {result_python.values[:3]})")


@pytest.mark.parametrize("size", [10, 50, 100, 500, 1000, 5000, 10000])
def test_different_sizes(size):
    """Test various input sizes."""
    print(f"\nSize {size}:")

    max_lag = 20
    data = np.random.randn(size)
    series = pd.Series(data)

    result_python = compute_autocorrelation_python(series, max_lag)

    # Check shape
    assert len(result_python) == max_lag, \
        f"Python returned wrong shape: {len(result_python)} != {max_lag}"

    if RUST_AVAILABLE:
        result_rust = fft_autocorr.compute_autocorrelation(data, max_lag)

        # Check shape
        assert len(result_rust) == max_lag, \
            f"Rust returned wrong shape: {len(result_rust)} != {max_lag}"

        # Compare - handle NaN values properly
        # For small series, some lags may be NaN (when lag >= series length)
        nan_mask_python = np.isnan(result_python.values)
        nan_mask_rust = np.isnan(result_rust)

        # Both should have NaN in the same positions
        assert np.array_equal(nan_mask_python, nan_mask_rust), \
            "NaN positions differ between Python and Rust"

        # Compare non-NaN values
        valid_mask = ~nan_mask_python
        if np.any(valid_mask):
            diff = np.abs(result_python.values[valid_mask] - result_rust[valid_mask])
            max_diff = np.max(diff)
            assert max_diff < 1e-10, f"Results differ by {max_diff:.2e}"
            print(f"  PASS (max diff: {max_diff:.2e}, {np.sum(valid_mask)}/{max_lag} valid lags)")
        else:
            print(f"  PASS (all NaN - series too small)")
    else:
        print(f"  PASS (Python only)")


def test_large_max_lag():
    """Test with large max_lag values."""
    print("\n" + "=" * 70)
    print("TEST 4: Large max_lag")
    print("=" * 70)

    data = np.random.randn(1000)
    series = pd.Series(data)
    max_lag = 500

    print(f"\nData size: {len(data)}")
    print(f"Max lag: {max_lag}")

    result_python = compute_autocorrelation_python(series, max_lag)

    if RUST_AVAILABLE:
        result_rust = fft_autocorr.compute_autocorrelation(data, max_lag)

        diff = np.abs(result_python.values - result_rust)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"\nMax difference: {max_diff:.2e}")
        print(f"Mean difference: {mean_diff:.2e}")

        assert max_diff < 1e-10, f"Results differ by {max_diff:.2e}"
        print("PASS: Results match perfectly")
    else:
        print(f"PASS (Python only, first 5 lags: {result_python.values[:5]})")
