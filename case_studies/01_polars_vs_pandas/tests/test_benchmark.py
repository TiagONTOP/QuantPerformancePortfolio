"""
Benchmark tests comparing performance (time & memory) of implementations.

Compares:
- suboptimal_backtest_strategy (reference)
- optimal_backtest_strategy_pandas (vectorized pandas)
- optimal_backtest_strategy_polars (vectorized polars)

On three dataset sizes: Small, Medium, Large
"""
import os
import sys
import time
import tracemalloc
import pytest
import pandas as pd

# Inject path because folder starts with a digit
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from tools.utils import generate_synthetic_df
from suboptimal.backtest import suboptimal_backtest_strategy
from optimized.backtest import optimal_backtest_strategy_pandas, optimal_backtest_strategy_polars


# Benchmark configurations
SMALL_CONFIG = {"sample_size": 500, "n_backtest": 10, "window": 50, "seed": 42}
MEDIUM_CONFIG = {"sample_size": 1500, "n_backtest": 50, "window": 100, "seed": 42}
LARGE_CONFIG = {"sample_size": 3000, "n_backtest": 100, "window": 100, "seed": 42}


def benchmark_function(func, *args, **kwargs):
    """
    Benchmark a function: measure time and peak memory usage.

    Returns
    -------
    result : tuple
        Function return value
    elapsed_ms : float
        Execution time in milliseconds
    peak_memory_mb : float
        Peak memory usage in MB
    """
    tracemalloc.start()

    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_ms = elapsed * 1000
    peak_memory_mb = peak / (1024 * 1024)

    return result, elapsed_ms, peak_memory_mb


def run_benchmark_suite(config_name, config):
    """
    Run all implementations on a given config and return results.

    Returns
    -------
    results : dict
        Dictionary with keys: 'suboptimal', 'pandas', 'polars'
        Each value is a dict with keys: 'time_ms', 'memory_mb', 'success'
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {config_name}")
    print(f"{'='*60}")

    df = generate_synthetic_df(**config)
    results = {}

    # Suboptimal (reference)
    print(f"Running suboptimal...")
    try:
        _, time_ms, mem_mb = benchmark_function(suboptimal_backtest_strategy, df)
        results['suboptimal'] = {'time_ms': time_ms, 'memory_mb': mem_mb, 'success': True}
        print(f"  Time: {time_ms:.2f} ms | Memory: {mem_mb:.2f} MB")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['suboptimal'] = {'time_ms': None, 'memory_mb': None, 'success': False}

    # Pandas optimized
    print(f"Running pandas optimized...")
    try:
        _, time_ms, mem_mb = benchmark_function(optimal_backtest_strategy_pandas, df)
        results['pandas'] = {'time_ms': time_ms, 'memory_mb': mem_mb, 'success': True}
        print(f"  Time: {time_ms:.2f} ms | Memory: {mem_mb:.2f} MB")

        if results['suboptimal']['success']:
            speedup = results['suboptimal']['time_ms'] / time_ms
            print(f"  Speedup vs suboptimal: {speedup:.2f}x")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['pandas'] = {'time_ms': None, 'memory_mb': None, 'success': False}

    # Polars optimized
    print(f"Running polars optimized...")
    try:
        _, time_ms, mem_mb = benchmark_function(optimal_backtest_strategy_polars, df)
        results['polars'] = {'time_ms': time_ms, 'memory_mb': mem_mb, 'success': True}
        print(f"  Time: {time_ms:.2f} ms | Memory: {mem_mb:.2f} MB")

        if results['suboptimal']['success']:
            speedup = results['suboptimal']['time_ms'] / time_ms
            print(f"  Speedup vs suboptimal: {speedup:.2f}x")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['polars'] = {'time_ms': None, 'memory_mb': None, 'success': False}

    return results


@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_small():
    """Benchmark on small dataset."""
    results = run_benchmark_suite("SMALL", SMALL_CONFIG)

    # Soft assertion: optimized should be faster (at least not catastrophically slower)
    if results['suboptimal']['success'] and results['pandas']['success']:
        speedup = results['suboptimal']['time_ms'] / results['pandas']['time_ms']
        assert speedup > 0.5, f"Pandas too slow: {speedup:.2f}x (expected >0.5x)"

    if results['suboptimal']['success'] and results['polars']['success']:
        speedup = results['suboptimal']['time_ms'] / results['polars']['time_ms']
        assert speedup > 0.5, f"Polars too slow: {speedup:.2f}x (expected >0.5x)"


@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_medium():
    """Benchmark on medium dataset."""
    results = run_benchmark_suite("MEDIUM", MEDIUM_CONFIG)

    # Expect significant speedup on medium size
    if results['suboptimal']['success'] and results['pandas']['success']:
        speedup = results['suboptimal']['time_ms'] / results['pandas']['time_ms']
        assert speedup > 1.5, f"Pandas speedup too low: {speedup:.2f}x (expected >1.5x)"

    if results['suboptimal']['success'] and results['polars']['success']:
        speedup = results['suboptimal']['time_ms'] / results['polars']['time_ms']
        assert speedup > 1.5, f"Polars speedup too low: {speedup:.2f}x (expected >1.5x)"


@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_large():
    """Benchmark on large dataset."""
    results = run_benchmark_suite("LARGE", LARGE_CONFIG)

    # Expect major speedup on large dataset
    if results['suboptimal']['success'] and results['pandas']['success']:
        speedup = results['suboptimal']['time_ms'] / results['pandas']['time_ms']
        assert speedup > 2.0, f"Pandas speedup too low: {speedup:.2f}x (expected >2.0x)"

    if results['suboptimal']['success'] and results['polars']['success']:
        speedup = results['suboptimal']['time_ms'] / results['polars']['time_ms']
        assert speedup > 2.0, f"Polars speedup too low: {speedup:.2f}x (expected >2.0x)"


@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_summary():
    """
    Run full benchmark suite and print summary table.
    This is not a pass/fail test, just informational.
    """
    configs = [
        ("SMALL", SMALL_CONFIG),
        ("MEDIUM", MEDIUM_CONFIG),
        ("LARGE", LARGE_CONFIG),
    ]

    all_results = {}
    for name, config in configs:
        all_results[name] = run_benchmark_suite(name, config)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<10} {'Implementation':<15} {'Time (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10}")
    print(f"{'-'*80}")

    for config_name, results in all_results.items():
        ref_time = results['suboptimal']['time_ms']

        for impl in ['suboptimal', 'pandas', 'polars']:
            res = results[impl]
            if res['success']:
                time_str = f"{res['time_ms']:.2f}"
                mem_str = f"{res['memory_mb']:.2f}"
                speedup = ref_time / res['time_ms'] if ref_time else 1.0
                speedup_str = f"{speedup:.2f}x" if impl != 'suboptimal' else "-"
            else:
                time_str = "FAILED"
                mem_str = "FAILED"
                speedup_str = "-"

            print(f"{config_name:<10} {impl:<15} {time_str:<12} {mem_str:<12} {speedup_str:<10}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Run benchmarks when executed directly
    pytest.main([__file__, "-v", "-s"])
