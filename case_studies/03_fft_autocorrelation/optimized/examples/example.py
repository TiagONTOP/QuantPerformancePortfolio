"""
Example usage and benchmarking of the Rust FFT-based autocorrelation implementation.

This script demonstrates:
1. Basic usage of the fft_autocorr.compute_autocorrelation function
2. Comparison with the Python scipy implementation
3. Performance benchmarking
"""

import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Add parent directories to path to import both implementations
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "suboptimal"))

try:
    import fft_autocorr
    RUST_AVAILABLE = True
except ImportError:
    print("Warning: fft_autocorr module not built. Run 'maturin develop' first.")
    RUST_AVAILABLE = False

from processing import compute_autocorrelation as compute_autocorrelation_python


def test_basic_functionality():
    """Test basic functionality with simple examples."""
    print("=" * 70)
    print("TEST 1: Basic Functionality")
    print("=" * 70)

    # Test case 1: Simple sequence
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    max_lag = 3

    print(f"\nInput data: {data}")
    print(f"Max lag: {max_lag}")

    # Python implementation
    series = pd.Series(data)
    result_python = compute_autocorrelation_python(series, max_lag)
    print(f"\nPython result:\n{result_python}")

    # Rust implementation
    if RUST_AVAILABLE:
        result_rust = fft_autocorr.compute_autocorrelation(data, max_lag)
        print(f"\nRust result:\n{result_rust}")

        # Compare results
        diff = np.abs(result_python.values - result_rust)
        max_diff = np.max(diff)
        print(f"\nMaximum difference: {max_diff:.2e}")

        if max_diff < 1e-10:
            print("Results match perfectly!")
        else:
            print(f"Results differ by {max_diff:.2e}")


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
    ]

    for name, data in test_cases:
        print(f"\n{name}:")
        series = pd.Series(data)
        max_lag = 5

        result_python = compute_autocorrelation_python(series, max_lag)

        if RUST_AVAILABLE:
            result_rust = fft_autocorr.compute_autocorrelation(data, max_lag)
            diff = np.abs(result_python.values - result_rust)
            max_diff = np.max(diff)
            print(f"  Max difference: {max_diff:.2e}")

            if max_diff < 1e-10:
                print("  Match")
            else:
                print(f"  Differ by {max_diff:.2e}")
        else:
            print(f"  Python result (first 3 lags): {result_python.values[:3]}")


def benchmark_performance():
    """Benchmark performance comparison."""
    print("\n" + "=" * 70)
    print("TEST 3: Performance Benchmark")
    print("=" * 70)

    if not RUST_AVAILABLE:
        print("Rust module not available. Skipping benchmark.")
        return

    sizes = [100, 1000, 10000, 50000]
    max_lag = 50
    n_iterations = 10

    print(f"\nBenchmarking with max_lag={max_lag}, {n_iterations} iterations per size\n")
    print(f"{'Size':<10} {'Python (ms)':<15} {'Rust (ms)':<15} {'Speedup':<10}")
    print("-" * 55)

    for size in sizes:
        data = np.random.randn(size)
        series = pd.Series(data)

        # Benchmark Python
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = compute_autocorrelation_python(series, max_lag)
        python_time = (time.perf_counter() - start) * 1000 / n_iterations

        # Benchmark Rust
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = fft_autocorr.compute_autocorrelation(data, max_lag)
        rust_time = (time.perf_counter() - start) * 1000 / n_iterations

        speedup = python_time / rust_time
        print(f"{size:<10} {python_time:<15.3f} {rust_time:<15.3f} {speedup:<10.2f}x")


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

        print(f"\nMax difference: {max_diff:.2e}")
        print(f"Mean absolute difference: {np.mean(diff):.2e}")

        if max_diff < 1e-10:
            print("Results match perfectly!")
        else:
            print(f"Results differ by {max_diff:.2e}")
    else:
        print(f"Python result (first 5 lags): {result_python.values[:5]}")


def main():
    """Run all tests."""
    print("\n")
    print("=" * 70)
    print(" " * 15 + "FFT AUTOCORRELATION TESTING SUITE")
    print("=" * 70)

    if not RUST_AVAILABLE:
        print("\nWARNING: Rust module not available!")
        print("Build it with: maturin develop --release")
        print("\nRunning tests with Python implementation only...\n")

    test_basic_functionality()
    test_edge_cases()
    benchmark_performance()
    test_large_max_lag()

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()