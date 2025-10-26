import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
import time
from typing import Dict, Tuple


def benchmark_acf_methods(
    series_sizes: list = [100, 1000, 10000, 100000],
    max_lag: int = 50,
    n_iterations: int = 100
) -> Dict[str, Dict[int, float]]:
    """
    Benchmark ACF computation with fft=False vs fft=True.

    Parameters
    ----------
    series_sizes : list
        List of series sizes to test
    max_lag : int
        Maximum lag to compute
    n_iterations : int
        Number of iterations for each benchmark

    Returns
    -------
    dict
        Dictionary with timing results for each method and series size
    """
    results = {"fft_false": {}, "fft_true": {}}

    print("Running ACF benchmarks...")
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
            acf(test_series, nlags=max_lag, fft=True)
        time_false = (time.perf_counter() - start) / n_iterations

        # Benchmark fft=True
        start = time.perf_counter()
        for _ in range(n_iterations):
            acf(test_series, nlags=max_lag, fft=True)
        time_true = (time.perf_counter() - start) / n_iterations

        results["fft_false"][size] = time_false
        results["fft_true"][size] = time_true

        speedup = time_false / time_true
        print(f"Size {size:>6}: fft=False: {time_false*1000:>8.4f}ms | "
              f"fft=True: {time_true*1000:>8.4f}ms | "
              f"Speedup: {speedup:>6.2f}x")

    return results


def determine_optimal_fft_setting(threshold_size: int = 1000) -> Tuple[bool, str]:
    """
    Run benchmarks and determine the optimal FFT setting.

    Parameters
    ----------
    threshold_size : int
        Series size to use for decision making

    Returns
    -------
    tuple
        (use_fft: bool, reasoning: str)
    """
    # Run quick benchmark
    results = benchmark_acf_methods(
        series_sizes=[100, 1000, 10000],
        max_lag=50,
        n_iterations=50
    )

    # Calculate average speedup for larger series
    large_sizes = [s for s in results["fft_false"].keys() if s >= threshold_size]
    speedups = [
        results["fft_false"][size] / results["fft_true"][size]
        for size in large_sizes
    ]
    avg_speedup = np.mean(speedups)

    use_fft = avg_speedup > 1.2  # Use FFT if at least 20% faster

    reasoning = (
        f"Average speedup with FFT for series >= {threshold_size}: {avg_speedup:.2f}x\n"
        f"Decision: Use fft={'True' if use_fft else 'False'}"
    )

    print(f"\n{reasoning}\n")

    return use_fft, reasoning


def compute_autocorrelation(series: pd.Series, max_lag: int = 1) -> pd.Series:
    """
    Compute autocorrelation for a pandas Series using statsmodels.

    Parameters
    ----------
    series : pd.Series
        Input time series data
    max_lag : int, default=1
        Maximum lag to compute autocorrelation for (from lag 1 to lag max_lag)

    Returns
    -------
    pd.Series
        Series with lag as index and corresponding autocorrelation values
        Index: lag values from 1 to max_lag
        Values: autocorrelation coefficients

    Examples
    --------
    >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> result = compute_autocorrelation(data, max_lag=3)
    >>> print(result)
    1    0.700000
    2    0.411765
    3    0.152941
    dtype: float64
    """
    # Calculate autocorrelation from lag 0 to max_lag
    # nlags parameter is max_lag (it will compute lag 0 to max_lag inclusive)
    autocorr_values = acf(series, nlags=max_lag, fft=True)

    # Create result series starting from lag 1 (skip lag 0 which is always 1.0)
    lags = range(1, max_lag + 1)
    result = pd.Series(autocorr_values[1:], index=lags)
    result.index.name = 'lag'
    result.name = 'autocorrelation'

    return result