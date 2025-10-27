"""
Performance benchmark tests comparing Python (suboptimal) and Rust (optimized) implementations.

This module measures execution time and speedup across various input sizes and max_lag values.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "suboptimal"))

from processing import compute_autocorrelation as compute_autocorrelation_python

try:
    import fft_autocorr
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("WARNING: Rust module not available. Run 'maturin develop --release' in optimized/")


def benchmark_sizes(sizes=[100, 1000, 10000, 50000], max_lag=50, n_iterations=10):
    """Benchmark different array sizes with fixed max_lag."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Different Sizes (max_lag=50)")
    print("=" * 70)

    if not RUST_AVAILABLE:
        print("\nSkipping benchmark: Rust module not available")
        return None

    print(f"\nSizes: {sizes}")
    print(f"Max lag: {max_lag}")
    print(f"Iterations: {n_iterations}\n")

    print(f"{'Size':<10} {'Python (ms)':<15} {'Rust (ms)':<15} {'Speedup':<10} {'Method':<10}")
    print("-" * 65)

    results = []

    for size in sizes:
        # Warmup
        data = np.random.randn(size)
        series = pd.Series(data)
        _ = compute_autocorrelation_python(series, max_lag)
        if RUST_AVAILABLE:
            _ = fft_autocorr.compute_autocorrelation(data, max_lag)

        # Benchmark Python
        times_python = []
        for _ in range(n_iterations):
            data = np.random.randn(size)
            series = pd.Series(data)
            start = time.perf_counter()
            _ = compute_autocorrelation_python(series, max_lag)
            times_python.append(time.perf_counter() - start)

        python_time = np.median(times_python) * 1000

        # Benchmark Rust
        times_rust = []
        for _ in range(n_iterations):
            data = np.random.randn(size)
            start = time.perf_counter()
            _ = fft_autocorr.compute_autocorrelation(data, max_lag)
            times_rust.append(time.perf_counter() - start)

        rust_time = np.median(times_rust) * 1000

        speedup = python_time / rust_time

        # Detect method used (heuristic based on size and max_lag)
        method = "Direct" if max_lag < 100 and size < 10000 else "FFT"

        print(f"{size:<10} {python_time:<15.3f} {rust_time:<15.3f} {speedup:<10.2f}x {method:<10}")

        results.append({
            'size': size,
            'python_ms': python_time,
            'rust_ms': rust_time,
            'speedup': speedup,
            'method': method
        })

    return results


def benchmark_max_lags(n=10000, max_lags=[10, 50, 100, 200, 500], n_iterations=10):
    """Benchmark different max_lag values with fixed array size."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Different max_lag (n=10,000)")
    print("=" * 70)

    if not RUST_AVAILABLE:
        print("\nSkipping benchmark: Rust module not available")
        return None

    print(f"\nArray size: {n}")
    print(f"Max lags: {max_lags}")
    print(f"Iterations: {n_iterations}\n")

    print(f"{'Max Lag':<10} {'Python (ms)':<15} {'Rust (ms)':<15} {'Speedup':<10} {'Method':<10}")
    print("-" * 65)

    results = []

    for max_lag in max_lags:
        # Warmup
        data = np.random.randn(n)
        series = pd.Series(data)
        _ = compute_autocorrelation_python(series, max_lag)
        if RUST_AVAILABLE:
            _ = fft_autocorr.compute_autocorrelation(data, max_lag)

        # Benchmark Python
        times_python = []
        for _ in range(n_iterations):
            data = np.random.randn(n)
            series = pd.Series(data)
            start = time.perf_counter()
            _ = compute_autocorrelation_python(series, max_lag)
            times_python.append(time.perf_counter() - start)

        python_time = np.median(times_python) * 1000

        # Benchmark Rust
        times_rust = []
        for _ in range(n_iterations):
            data = np.random.randn(n)
            start = time.perf_counter()
            _ = fft_autocorr.compute_autocorrelation(data, max_lag)
            times_rust.append(time.perf_counter() - start)

        rust_time = np.median(times_rust) * 1000

        speedup = python_time / rust_time

        # Detect method used
        # Heuristic: direct if max_lag < ~150 for n=10000
        method = "Direct" if max_lag < 150 else "FFT"

        print(f"{max_lag:<10} {python_time:<15.3f} {rust_time:<15.3f} {speedup:<10.2f}x {method:<10}")

        results.append({
            'max_lag': max_lag,
            'python_ms': python_time,
            'rust_ms': rust_time,
            'speedup': speedup,
            'method': method
        })

    return results


def benchmark_repeated_calls(n=10000, max_lag=50, n_calls=100):
    """Benchmark repeated calls to measure cache effectiveness."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Repeated Calls (cache effectiveness)")
    print("=" * 70)

    if not RUST_AVAILABLE:
        print("\nSkipping benchmark: Rust module not available")
        return None

    print(f"\nArray size: {n}")
    print(f"Max lag: {max_lag}")
    print(f"Number of calls: {n_calls}\n")

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

    print(f"Python total: {python_total:.1f} ms ({python_per_call:.3f} ms/call)")
    print(f"Rust total:   {rust_total:.1f} ms ({rust_per_call:.3f} ms/call)")
    print(f"\nSpeedup: {speedup:.2f}x")

    return {
        'python_total_ms': python_total,
        'rust_total_ms': rust_total,
        'python_per_call_ms': python_per_call,
        'rust_per_call_ms': rust_per_call,
        'speedup': speedup
    }


def run_all_benchmarks():
    """Run all benchmark tests."""
    print("\n")
    print("=" * 70)
    print(" " * 18 + "BENCHMARK TEST SUITE")
    print("=" * 70)

    if not RUST_AVAILABLE:
        print("\nERROR: Rust module not available!")
        print("Build it with: cd optimized && maturin develop --release")
        return None

    results = {}

    # Run benchmarks
    results['sizes'] = benchmark_sizes()
    results['max_lags'] = benchmark_max_lags()
    results['repeated'] = benchmark_repeated_calls()

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    if results['sizes']:
        avg_speedup_sizes = np.mean([r['speedup'] for r in results['sizes']])
        print(f"\nAverage speedup across sizes: {avg_speedup_sizes:.2f}x")
        print(f"Range: {min(r['speedup'] for r in results['sizes']):.2f}x - {max(r['speedup'] for r in results['sizes']):.2f}x")

    if results['max_lags']:
        avg_speedup_lags = np.mean([r['speedup'] for r in results['max_lags']])
        print(f"\nAverage speedup across max_lags: {avg_speedup_lags:.2f}x")
        print(f"Range: {min(r['speedup'] for r in results['max_lags']):.2f}x - {max(r['speedup'] for r in results['max_lags']):.2f}x")

    if results['repeated']:
        print(f"\nRepeated calls speedup: {results['repeated']['speedup']:.2f}x")

    print("\n" + "=" * 70)
    print("BENCHMARKS COMPLETE")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
    if results is None:
        sys.exit(1)
    sys.exit(0)
