"""
Quick benchmark script to determine optimal FFT setting for ACF computation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
import time

def quick_benchmark():
    print("=" * 70)
    print("QUICK ACF FFT BENCHMARK")
    print("=" * 70)
    print()

    series_sizes = [100, 1000, 10000, 50000]
    max_lag = 50
    n_iterations = 20

    results = {"fft_false": {}, "fft_true": {}}

    print(f"Series sizes: {series_sizes}")
    print(f"Max lag: {max_lag}")
    print(f"Iterations per test: {n_iterations}\n")

    for size in series_sizes:
        # Generate random time series
        np.random.seed(42)
        test_series = pd.Series(np.random.randn(size))

        # Benchmark fft=False
        start = time.perf_counter()
        for _ in range(n_iterations):
            acf(test_series, nlags=max_lag, fft=False)
        time_false = (time.perf_counter() - start) / n_iterations

        # Benchmark fft=True
        start = time.perf_counter()
        for _ in range(n_iterations):
            acf(test_series, nlags=max_lag, fft=True)
        time_true = (time.perf_counter() - start) / n_iterations

        results["fft_false"][size] = time_false
        results["fft_true"][size] = time_true

        speedup = time_false / time_true
        faster = "FFT" if speedup > 1 else "Direct"
        print(f"Size {size:>6}: fft=False: {time_false*1000:>8.4f}ms | "
              f"fft=True: {time_true*1000:>8.4f}ms | "
              f"Speedup: {speedup:>6.2f}x ({faster})")

    # Analyze results for larger series (>= 1000)
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    large_sizes = [s for s in series_sizes if s >= 1000]
    speedups = [results["fft_false"][size] / results["fft_true"][size] for size in large_sizes]
    avg_speedup = np.mean(speedups)

    use_fft = avg_speedup > 1.2  # Use FFT if at least 20% faster on average

    print(f"\nAverage speedup with FFT for series >= 1000: {avg_speedup:.2f}x")
    print(f"Decision: Use fft={use_fft}")

    # Update the processing.py file
    print("\n" + "=" * 70)
    print("UPDATING processing.py")
    print("=" * 70)

    processing_file = Path(__file__).parent / "suboptimal" / "processing.py"
    content = processing_file.read_text()

    # Replace the fft parameter
    if "fft=False)" in content:
        content = content.replace("fft=False)", f"fft={use_fft})")
        processing_file.write_text(content)
        print(f"\n✓ Updated compute_autocorrelation to use fft={use_fft}")
    elif "fft=True)" in content:
        content = content.replace("fft=True)", f"fft={use_fft})")
        processing_file.write_text(content)
        print(f"\n✓ Updated compute_autocorrelation to use fft={use_fft}")
    else:
        print("\n✗ Could not find the line to update")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return use_fft


if __name__ == "__main__":
    optimal_setting = quick_benchmark()
    print(f"\nFinal recommendation: fft={optimal_setting}")
