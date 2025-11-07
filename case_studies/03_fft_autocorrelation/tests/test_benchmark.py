"""
Performance benchmark tests comparing Python (suboptimal) and Rust (optimized) implementations.

This module measures execution time and speedup across various input sizes and max_lag values.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
import pytest

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "suboptimal"))

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


@pytest.mark.parametrize("size", [100, 1000, 10000, 50000])
def test_benchmark_sizes(size):
    """Benchmark different array sizes with fixed max_lag."""
    if not RUST_AVAILABLE:
        pytest.skip("Rust module not available")

    max_lag = 50
    n_iterations = 10

    print(f"\nBenchmarking size {size} with max_lag={max_lag}")

    # Warmup
    data_warmup = np.random.randn(size)
    series_warmup = pd.Series(data_warmup)
    _ = compute_autocorrelation_python(series_warmup, max_lag)
    _ = fft_autocorr.compute_autocorrelation(data_warmup, max_lag)

    # Pre-generate data for all iterations to avoid contaminating benchmarks
    test_data = [np.random.randn(size) for _ in range(n_iterations)]
    test_series = [pd.Series(data) for data in test_data]

    # Benchmark Python
    times_python = []
    for series in test_series:
        start = time.perf_counter()
        _ = compute_autocorrelation_python(series, max_lag)
        times_python.append(time.perf_counter() - start)

    python_time = np.median(times_python) * 1000

    # Benchmark Rust
    times_rust = []
    for data in test_data:
        start = time.perf_counter()
        _ = fft_autocorr.compute_autocorrelation(data, max_lag)
        times_rust.append(time.perf_counter() - start)

    rust_time = np.median(times_rust) * 1000

    speedup = python_time / rust_time

    print(f"  Python: {python_time:.3f} ms")
    print(f"  Rust:   {rust_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Verify that Rust is at least competitive (not slower)
    assert rust_time <= python_time * 1.5, f"Rust unexpectedly slow: {speedup:.2f}x"


@pytest.mark.parametrize("max_lag", [10, 50, 100, 200, 500])
def test_benchmark_max_lags(max_lag):
    """Benchmark different max_lag values with fixed array size."""
    if not RUST_AVAILABLE:
        pytest.skip("Rust module not available")

    n = 10000
    n_iterations = 10

    print(f"\nBenchmarking max_lag={max_lag} with n={n}")

    # Warmup
    data_warmup = np.random.randn(n)
    series_warmup = pd.Series(data_warmup)
    _ = compute_autocorrelation_python(series_warmup, max_lag)
    _ = fft_autocorr.compute_autocorrelation(data_warmup, max_lag)

    # Pre-generate data for all iterations to avoid contaminating benchmarks
    test_data = [np.random.randn(n) for _ in range(n_iterations)]
    test_series = [pd.Series(data) for data in test_data]

    # Benchmark Python
    times_python = []
    for series in test_series:
        start = time.perf_counter()
        _ = compute_autocorrelation_python(series, max_lag)
        times_python.append(time.perf_counter() - start)

    python_time = np.median(times_python) * 1000

    # Benchmark Rust
    times_rust = []
    for data in test_data:
        start = time.perf_counter()
        _ = fft_autocorr.compute_autocorrelation(data, max_lag)
        times_rust.append(time.perf_counter() - start)

    rust_time = np.median(times_rust) * 1000

    speedup = python_time / rust_time

    print(f"  Python: {python_time:.3f} ms")
    print(f"  Rust:   {rust_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Verify that Rust is at least competitive (not slower)
    assert rust_time <= python_time * 1.5, f"Rust unexpectedly slow: {speedup:.2f}x"


def test_benchmark_repeated_calls():
    """Benchmark repeated calls to measure cache effectiveness."""
    if not RUST_AVAILABLE:
        pytest.skip("Rust module not available")

    n = 10000
    max_lag = 50
    n_calls = 100

    print(f"\nBenchmarking repeated calls: n={n}, max_lag={max_lag}, calls={n_calls}")

    # Generate data once
    data = np.random.randn(n)
    series = pd.Series(data)

    # Benchmark Python (repeated)
    start = time.perf_counter()
    for _ in range(n_calls):
        _ = compute_autocorrelation_python(series, max_lag)
    python_total = (time.perf_counter() - start) * 1000
    python_per_call = python_total / n_calls

    # Benchmark Rust (repeated)
    start = time.perf_counter()
    for _ in range(n_calls):
        _ = fft_autocorr.compute_autocorrelation(data, max_lag)
    rust_total = (time.perf_counter() - start) * 1000
    rust_per_call = rust_total / n_calls

    speedup = python_per_call / rust_per_call

    print(f"  Python total: {python_total:.1f} ms ({python_per_call:.3f} ms/call)")
    print(f"  Rust total:   {rust_total:.1f} ms ({rust_per_call:.3f} ms/call)")
    print(f"  Speedup: {speedup:.2f}x")

    # Verify that Rust is at least competitive (not slower)
    assert rust_total <= python_total * 1.5, f"Rust unexpectedly slow: {speedup:.2f}x"
