"""Performance benchmarks for GPU-accelerated GBM Monte Carlo simulation.

Benchmarks compare:
- Different backends (CPU, CuPy, Numba)
- Different dtypes (float32, float64)
- Different problem sizes
- Speedup calculations
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

from optimized.pricing import simulate_gbm_paths as simulate_gbm_gpu
from suboptimal.pricing import simulate_gbm_paths as simulate_gbm_cpu

CUPY_AVAILABLE = True


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(
        self,
        backend: str,
        dtype: str,
        n_paths: int,
        n_steps: int,
        elapsed_time: float,
        final_mean: float,
        final_std: float,
    ):
        self.backend = backend
        self.dtype = dtype
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.elapsed_time = elapsed_time
        self.final_mean = final_mean
        self.final_std = final_std

    def __repr__(self):
        return (
            f"BenchmarkResult(backend={self.backend}, dtype={self.dtype}, "
            f"n_paths={self.n_paths:,}, n_steps={self.n_steps}, "
            f"time={self.elapsed_time:.4f}s, final_mean={self.final_mean:.4f})"
        )


def run_benchmark(
    backend: str,
    n_paths: int,
    n_steps: int,
    dtype: np.dtype,
    seed: int = 42,
    warmup: bool = True,
) -> BenchmarkResult:
    """Run a single benchmark configuration.

    Parameters
    ----------
    backend : str
        Backend to use ('cpu', 'cupy', 'numba')
    n_paths : int
        Number of paths to simulate
    n_steps : int
        Number of time steps
    dtype : np.dtype
        Data type (np.float32 or np.float64)
    seed : int
        Random seed for reproducibility
    warmup : bool
        Whether to run a warmup iteration (important for GPU)

    Returns
    -------
    BenchmarkResult
        Benchmark results
    """
    # Select the appropriate function based on backend
    if backend == "cpu":
        simulate_fn = simulate_gbm_cpu
        params = {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": n_steps,
            "n_paths": n_paths,
            "dtype": dtype,
            "rng": np.random.default_rng(seed),
        }
    elif backend in ("cupy", "gpu"):
        simulate_fn = simulate_gbm_gpu
        params = {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": n_steps,
            "n_paths": n_paths,
            "dtype": dtype,
            "seed": seed,
        }
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Warmup run for GPU kernels
    if warmup and backend in ("cupy", "numba", "gpu"):
        _ = simulate_fn(**params)

    # Timed run
    start = time.perf_counter()
    t_grid, paths = simulate_fn(**params)
    elapsed = time.perf_counter() - start

    # Compute statistics
    final_mean = float(np.mean(paths[-1, :]))
    final_std = float(np.std(paths[-1, :]))

    return BenchmarkResult(
        backend=backend,
        dtype=str(dtype),
        n_paths=n_paths,
        n_steps=n_steps,
        elapsed_time=elapsed,
        final_mean=final_mean,
        final_std=final_std,
    )


class TestBenchmarkSmall:
    """Benchmarks for small problem sizes."""

    def test_small_problem_cpu(self):
        """Benchmark CPU on small problem."""
        result = run_benchmark("cpu", n_paths=10_000, n_steps=252, dtype=np.float64)
        print(f"\nSmall CPU: {result}")
        assert result.elapsed_time > 0

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_small_problem_cupy(self):
        """Benchmark CuPy on small problem."""
        result = run_benchmark("cupy", n_paths=10_000, n_steps=252, dtype=np.float32)
        print(f"\nSmall CuPy: {result}")
        assert result.elapsed_time > 0



class TestBenchmarkMedium:
    """Benchmarks for medium problem sizes."""

    def test_medium_problem_cpu(self):
        """Benchmark CPU on medium problem."""
        result = run_benchmark("cpu", n_paths=100_000, n_steps=252, dtype=np.float64)
        print(f"\nMedium CPU: {result}")
        assert result.elapsed_time > 0

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_medium_problem_cupy(self):
        """Benchmark CuPy on medium problem."""
        result = run_benchmark("cupy", n_paths=100_000, n_steps=252, dtype=np.float32)
        print(f"\nMedium CuPy: {result}")
        assert result.elapsed_time > 0



class TestBenchmarkLarge:
    """Benchmarks for large problem sizes."""

    def test_large_problem_cpu(self):
        """Benchmark CPU on large problem."""
        result = run_benchmark("cpu", n_paths=500_000, n_steps=252, dtype=np.float32)  # Reduced from 1M
        print(f"\nLarge CPU: {result}")
        assert result.elapsed_time > 0

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_large_problem_cupy(self):
        """Benchmark CuPy on large problem."""
        result = run_benchmark("cupy", n_paths=500_000, n_steps=252, dtype=np.float32)  # Reduced from 1M
        print(f"\nLarge CuPy: {result}")
        assert result.elapsed_time > 0



class TestSpeedupComparison:
    """Compare speedups between backends."""

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_cupy_vs_cpu_speedup(self):
        """Measure CuPy speedup vs CPU."""
        n_paths = 500_000
        n_steps = 252

        # CPU baseline
        cpu_result = run_benchmark("cpu", n_paths, n_steps, np.float32)

        # CuPy
        cupy_result = run_benchmark("cupy", n_paths, n_steps, np.float32)

        speedup = cpu_result.elapsed_time / cupy_result.elapsed_time
        print(f"\nCuPy vs CPU speedup: {speedup:.2f}x")
        print(f"  CPU time: {cpu_result.elapsed_time:.4f}s")
        print(f"  CuPy time: {cupy_result.elapsed_time:.4f}s")

        # Expect at least some speedup (even modest GPU should be faster)
        assert speedup > 1.0



class TestDtypePerformance:
    """Compare performance of float32 vs float64."""

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_cupy_dtype_comparison(self):
        """Compare float32 vs float64 on CuPy."""
        n_paths = 500_000  # Reduced from 1M to avoid Windows TDR timeout
        n_steps = 252

        # float32
        result_f32 = run_benchmark("cupy", n_paths, n_steps, np.float32)

        # float64
        result_f64 = run_benchmark("cupy", n_paths, n_steps, np.float64)

        ratio = result_f64.elapsed_time / result_f32.elapsed_time
        print(f"\nCuPy float64/float32 time ratio: {ratio:.2f}x")
        print(f"  float32 time: {result_f32.elapsed_time:.4f}s")
        print(f"  float64 time: {result_f64.elapsed_time:.4f}s")

        # float32 should be faster or comparable
        assert ratio >= 1.0


def comprehensive_benchmark_suite(
    backends: List[str] = None,
    problem_sizes: List[Tuple[int, int]] = None,
    dtypes: List[np.dtype] = None,
) -> Dict[str, List[BenchmarkResult]]:
    """Run a comprehensive benchmark suite.

    Parameters
    ----------
    backends : list of str, optional
        Backends to test. Default: all available.
    problem_sizes : list of (n_paths, n_steps), optional
        Problem sizes to test. Default: [(10k, 252), (100k, 252), (1M, 252)]
    dtypes : list of np.dtype, optional
        Data types to test. Default: [np.float32, np.float64]

    Returns
    -------
    dict
        Results organized by backend
    """
    if backends is None:
        backends = ["cpu"]
        if CUPY_AVAILABLE:
            backends.append("cupy")

    if problem_sizes is None:
        problem_sizes = [
            (10_000, 252),
            (100_000, 252),
            (500_000, 252),  # Reduced from 1M to avoid Windows TDR timeout
        ]

    if dtypes is None:
        dtypes = [np.float32, np.float64]

    results = {backend: [] for backend in backends}

    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)

    for backend in backends:
        print(f"\n{backend.upper()} Backend:")
        print("-" * 80)

        for n_paths, n_steps in problem_sizes:
            for dtype in dtypes:
                try:
                    result = run_benchmark(backend, n_paths, n_steps, dtype)
                    results[backend].append(result)

                    paths_m = n_paths / 1_000_000
                    print(
                        f"  {paths_m:.2f}M paths × {n_steps} steps, {dtype.__name__:7s}: "
                        f"{result.elapsed_time:7.4f}s  "
                        f"(mean={result.final_mean:.2f}, std={result.final_std:.2f})"
                    )
                except Exception as e:
                    print(f"  {n_paths:>10,} paths × {n_steps} steps, {dtype.__name__}: FAILED ({e})")

    # Summary table
    print("\n" + "=" * 80)
    print("SPEEDUP SUMMARY (vs CPU float64 baseline)")
    print("=" * 80)

    # Find CPU float64 baseline for each problem size
    cpu_results = {
        (r.n_paths, r.n_steps, r.dtype): r.elapsed_time
        for r in results.get("cpu", [])
    }

    for n_paths, n_steps in problem_sizes:
        print(f"\n{n_paths:,} paths × {n_steps} steps:")

        baseline_key = (n_paths, n_steps, "float64")
        if baseline_key in cpu_results:
            baseline_time = cpu_results[baseline_key]

            for backend in backends:
                for dtype in dtypes:
                    matching = [
                        r for r in results[backend]
                        if r.n_paths == n_paths and r.n_steps == n_steps and r.dtype == str(dtype)
                    ]
                    if matching:
                        r = matching[0]
                        speedup = baseline_time / r.elapsed_time
                        print(f"  {backend:6s} {dtype.__name__:7s}: {speedup:6.2f}x  ({r.elapsed_time:.4f}s)")

    return results


if __name__ == "__main__":
    # Run comprehensive benchmark
    results = comprehensive_benchmark_suite()

    # Additional analysis
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if CUPY_AVAILABLE:
        print("\n[CuPy Backend]")
        print("  - Best for: Large-scale vectorized operations")
        print("  - Recommended dtype: float32 (unless high precision needed)")
        print("  - Typical speedup: 10-100x vs CPU (hardware dependent)")

    print("\n[CPU Backend]")
    print("  - Best for: Small problems, no GPU available, validation")
    print("  - Use as baseline for performance comparisons")

    print("\n[General Tips]")
    print("  - Use float32 for production (2x faster on GPU, sufficient precision)")
    print("  - Use float64 for validation and high-precision requirements")
    print("  - Consider chunking (max_paths_per_chunk) for memory-constrained GPUs")
    print("  - Warmup runs important for accurate GPU benchmarking")
