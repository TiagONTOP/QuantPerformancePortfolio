"""
Unit tests for FFT Autocorrelation implementations.

This module tests both the Python (suboptimal) and Rust (optimized) implementations
to ensure correctness and numerical accuracy.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "suboptimal"))

# Import implementations
from processing import compute_autocorrelation as compute_autocorrelation_python

try:
    import fft_autocorr
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("WARNING: Rust module not available. Run 'maturin develop --release' in optimized/")


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

    return True


def test_edge_cases():
    """Test edge cases and special scenarios."""
    print("\n" + "=" * 70)
    print("TEST 2: Edge Cases")
    print("=" * 70)

    test_cases = [
        ("Constant series", np.ones(100)),
        ("Random normal", np.random.randn(100)),
        ("Sine wave", np.sin(np.linspace(0, 4*np.pi, 100))),
        ("Linear trend", np.arange(100, dtype=float)),
        ("Zero mean", np.random.randn(100) - np.random.randn(100).mean()),
    ]

    max_lag = 5
    all_pass = True

    for name, data in test_cases:
        print(f"\n{name}:")
        series = pd.Series(data)

        try:
            result_python = compute_autocorrelation_python(series, max_lag)

            # Check for NaN or Inf
            if np.any(np.isnan(result_python.values)) and name != "Constant series":
                print(f"  FAIL: Python produced NaN for non-constant series")
                all_pass = False
                continue

            if np.any(np.isinf(result_python.values)):
                print(f"  FAIL: Python produced Inf")
                all_pass = False
                continue

            if RUST_AVAILABLE:
                result_rust = fft_autocorr.compute_autocorrelation(data, max_lag)

                # Check for NaN or Inf
                if np.any(np.isnan(result_rust)) and name != "Constant series":
                    print(f"  FAIL: Rust produced NaN for non-constant series")
                    all_pass = False
                    continue

                if np.any(np.isinf(result_rust)):
                    print(f"  FAIL: Rust produced Inf")
                    all_pass = False
                    continue

                # Compare Rust vs Python
                # For constant series, both should be NaN
                if name == "Constant series":
                    if np.all(np.isnan(result_python.values)) and np.all(np.isnan(result_rust)):
                        print(f"  PASS: Both correctly return NaN for constant series")
                    else:
                        print(f"  FAIL: Inconsistent handling of constant series")
                        all_pass = False
                else:
                    diff = np.abs(result_python.values - result_rust)
                    max_diff = np.max(diff)

                    if max_diff < 1e-10:
                        print(f"  PASS (max diff: {max_diff:.2e})")
                    else:
                        print(f"  FAIL (max diff: {max_diff:.2e})")
                        all_pass = False
            else:
                print(f"  PASS (Python only, first 3 lags: {result_python.values[:3]})")

        except Exception as e:
            print(f"  FAIL: Exception raised: {e}")
            all_pass = False

    return all_pass


def test_different_sizes():
    """Test various input sizes."""
    print("\n" + "=" * 70)
    print("TEST 3: Different Sizes")
    print("=" * 70)

    sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    max_lag = 20
    all_pass = True

    for size in sizes:
        print(f"\nSize {size}:")
        data = np.random.randn(size)
        series = pd.Series(data)

        try:
            result_python = compute_autocorrelation_python(series, max_lag)

            # Check shape
            if len(result_python) != max_lag:
                print(f"  FAIL: Python returned wrong shape: {len(result_python)} != {max_lag}")
                all_pass = False
                continue

            if RUST_AVAILABLE:
                result_rust = fft_autocorr.compute_autocorrelation(data, max_lag)

                # Check shape
                if len(result_rust) != max_lag:
                    print(f"  FAIL: Rust returned wrong shape: {len(result_rust)} != {max_lag}")
                    all_pass = False
                    continue

                # Compare
                diff = np.abs(result_python.values - result_rust)
                max_diff = np.max(diff)

                if max_diff < 1e-10:
                    print(f"  PASS (max diff: {max_diff:.2e})")
                else:
                    print(f"  FAIL (max diff: {max_diff:.2e})")
                    all_pass = False
            else:
                print(f"  PASS (Python only)")

        except Exception as e:
            print(f"  FAIL: Exception raised: {e}")
            all_pass = False

    return all_pass


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

    try:
        result_python = compute_autocorrelation_python(series, max_lag)

        if RUST_AVAILABLE:
            result_rust = fft_autocorr.compute_autocorrelation(data, max_lag)

            diff = np.abs(result_python.values - result_rust)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            print(f"\nMax difference: {max_diff:.2e}")
            print(f"Mean difference: {mean_diff:.2e}")

            if max_diff < 1e-10:
                print("PASS: Results match perfectly")
                return True
            else:
                print(f"FAIL: Results differ by {max_diff:.2e}")
                return False
        else:
            print(f"PASS (Python only, first 5 lags: {result_python.values[:5]})")
            return True

    except Exception as e:
        print(f"FAIL: Exception raised: {e}")
        return False


def run_all_tests():
    """Run all unit tests."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "UNIT TEST SUITE" + " " * 33 + "║")
    print("╚" + "═" * 68 + "╝")

    if not RUST_AVAILABLE:
        print("\nWARNING: Rust module not available!")
        print("Build it with: cd optimized && maturin develop --release")
        print("\nRunning tests with Python implementation only...\n")

    results = {}

    # Run all tests
    results['basic'] = test_basic_correctness()
    results['edge_cases'] = test_edge_cases()
    results['sizes'] = test_different_sizes()
    results['large_lag'] = test_large_max_lag()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name:20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
