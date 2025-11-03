"""Performance benchmarks for GPU-accelerated GBM Monte Carlo simulation (CuPy).

Benchmarks compare:
- GPU (optimized/pricing.py) vs CPU (suboptimal/pricing.py)
- Different dtypes (float32, float64) on GPU
- Different problem sizes
- Speedup calculations
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

from optimized.pricing import simulate_gbm_paths as simulate_gbm_gpu
CUPY_AVAILABLE = True

# Import CPU version for comparison
from suboptimal.pricing import simulate_gbm_paths as simulate_gbm_cpu


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


def run_benchmark_gpu(
    n_paths: int,
    n_steps: int,
    dtype: np.dtype,
    seed: int = 42,
    warmup: bool = True,
) -> BenchmarkResult:
    """Run a GPU benchmark."""
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

    # Warmup run for GPU kernels
    if warmup:
        _ = simulate_gbm_gpu(**params)

    # Timed run
    start = time.perf_counter()
    t_grid, paths = simulate_gbm_gpu(**params)
    elapsed = time.perf_counter() - start

    # Compute statistics
    final_mean = float(np.mean(paths[-1, :]))
    final_std = float(np.std(paths[-1, :]))

    return BenchmarkResult(
        backend="gpu",
        dtype=str(dtype),
        n_paths=n_paths,
        n_steps=n_steps,
        elapsed_time=elapsed,
        final_mean=final_mean,
        final_std=final_std,
    )


def run_benchmark_cpu(
    n_paths: int,
    n_steps: int,
    dtype: np.dtype,
    seed: int = 42,
) -> BenchmarkResult:
    """Run a CPU benchmark."""
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

    # Timed run
    start = time.perf_counter()
    t_grid, paths = simulate_gbm_cpu(**params)
    elapsed = time.perf_counter() - start

    # Compute statistics
    final_mean = float(np.mean(paths[-1, :]))
    final_std = float(np.std(paths[-1, :]))

    return BenchmarkResult(
        backend="cpu",
        dtype=str(dtype),
        n_paths=n_paths,
        n_steps=n_steps,
        elapsed_time=elapsed,
        final_mean=final_mean,
        final_std=final_std,
    )


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestBenchmarkSmall:
    """Benchmarks for small problem sizes."""

    def test_small_problem_gpu(self):
        """Benchmark GPU on small problem."""
        result = run_benchmark_gpu(n_paths=10_000, n_steps=252, dtype=np.float32)
        print(f"\nSmall GPU: {result}")
        assert result.elapsed_time > 0

    def test_small_problem_cpu(self):
        """Benchmark CPU on small problem."""
        result = run_benchmark_cpu(n_paths=10_000, n_steps=252, dtype=np.float64)
        print(f"\nSmall CPU: {result}")
        assert result.elapsed_time > 0


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestBenchmarkMedium:
    """Benchmarks for medium problem sizes."""

    def test_medium_problem_gpu(self):
        """Benchmark GPU on medium problem."""
        result = run_benchmark_gpu(n_paths=100_000, n_steps=252, dtype=np.float32)
        print(f"\nMedium GPU: {result}")
        assert result.elapsed_time > 0

    def test_medium_problem_cpu(self):
        """Benchmark CPU on medium problem."""
        result = run_benchmark_cpu(n_paths=100_000, n_steps=252, dtype=np.float64)
        print(f"\nMedium CPU: {result}")
        assert result.elapsed_time > 0


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestBenchmarkLarge:
    """Benchmarks for large problem sizes."""

    def test_large_problem_gpu(self):
        """Benchmark GPU on large problem."""
        result = run_benchmark_gpu(n_paths=500_000, n_steps=252, dtype=np.float32)  # Reduced from 1M
        print(f"\nLarge GPU: {result}")
        assert result.elapsed_time > 0

    def test_large_problem_cpu(self):
        """Benchmark CPU on large problem."""
        result = run_benchmark_cpu(n_paths=500_000, n_steps=252, dtype=np.float32)  # Reduced from 1M
        print(f"\nLarge CPU: {result}")
        assert result.elapsed_time > 0


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestSpeedupComparison:
    """Compare speedups between GPU and CPU."""

    def test_gpu_vs_cpu_speedup(self):
        """Measure GPU speedup vs CPU."""
        n_paths = 500_000
        n_steps = 252

        # CPU baseline
        cpu_result = run_benchmark_cpu(n_paths, n_steps, np.float32)

        # GPU
        gpu_result = run_benchmark_gpu(n_paths, n_steps, np.float32)

        speedup = cpu_result.elapsed_time / gpu_result.elapsed_time
        print(f"\nGPU vs CPU speedup: {speedup:.2f}x")
        print(f"  CPU time: {cpu_result.elapsed_time:.4f}s")
        print(f"  GPU time: {gpu_result.elapsed_time:.4f}s")

        # Expect significant speedup
        assert speedup > 1.0


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestDtypePerformance:
    """Compare performance of float32 vs float64."""

    def test_gpu_dtype_comparison(self):
        """Compare float32 vs float64 on GPU."""
        n_paths = 500_000  # Reduced from 1M to avoid Windows TDR timeout
        n_steps = 252

        # float32
        result_f32 = run_benchmark_gpu(n_paths, n_steps, np.float32)

        # float64
        result_f64 = run_benchmark_gpu(n_paths, n_steps, np.float64)

        ratio = result_f64.elapsed_time / result_f32.elapsed_time
        print(f"\nGPU float64/float32 time ratio: {ratio:.2f}x")
        print(f"  float32 time: {result_f32.elapsed_time:.4f}s")
        print(f"  float64 time: {result_f64.elapsed_time:.4f}s")

        # float32 should be faster
        assert ratio >= 1.0


def comprehensive_benchmark_suite(
    problem_sizes: List[Tuple[int, int]] = None,
    dtypes: List[np.dtype] = None,
) -> Dict[str, List[BenchmarkResult]]:
    """Run a comprehensive benchmark suite.

    Parameters
    ----------
    problem_sizes : list of (n_paths, n_steps), optional
        Problem sizes to test. Default: [(10k, 252), (100k, 252), (1M, 252)]
    dtypes : list of np.dtype, optional
        Data types to test. Default: [np.float32, np.float64]

    Returns
    -------
    dict
        Results organized by backend
    """
    if not CUPY_AVAILABLE:
        print("CuPy not available. Skipping GPU benchmarks.")
        return {}

    if problem_sizes is None:
        problem_sizes = [
            (10_000, 252),
            (100_000, 252),
            (500_000, 252),  # Reduced from 1M to avoid Windows TDR timeout
        ]

    if dtypes is None:
        dtypes = [np.float32, np.float64]

    results = {"gpu": [], "cpu": []}

    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK SUITE (GPU vs CPU)")
    print("=" * 80)

    for n_paths, n_steps in problem_sizes:
        print(f"\n{n_paths:,} paths × {n_steps} steps:")
        print("-" * 80)

        for dtype in dtypes:
            # GPU benchmark
            try:
                gpu_result = run_benchmark_gpu(n_paths, n_steps, dtype)
                results["gpu"].append(gpu_result)

                paths_m = n_paths / 1_000_000
                print(
                    f"  GPU {dtype.__name__:7s}: {gpu_result.elapsed_time:7.4f}s  "
                    f"(mean={gpu_result.final_mean:.2f}, std={gpu_result.final_std:.2f})"
                )
            except Exception as e:
                print(f"  GPU {dtype.__name__:7s}: FAILED ({e})")

        # CPU benchmark (float64 only for baseline)
        try:
            cpu_result = run_benchmark_cpu(n_paths, n_steps, np.float64)
            results["cpu"].append(cpu_result)
            print(
                f"  CPU float64: {cpu_result.elapsed_time:7.4f}s  "
                f"(mean={cpu_result.final_mean:.2f}, std={cpu_result.final_std:.2f})"
            )
        except Exception as e:
            print(f"  CPU float64: FAILED ({e})")

    # Summary table
    print("\n" + "=" * 80)
    print("SPEEDUP SUMMARY (vs CPU float64 baseline)")
    print("=" * 80)

    # Find CPU float64 baseline for each problem size
    cpu_results_map = {
        (r.n_paths, r.n_steps): r.elapsed_time
        for r in results.get("cpu", [])
    }

    for n_paths, n_steps in problem_sizes:
        print(f"\n{n_paths:,} paths × {n_steps} steps:")

        baseline_key = (n_paths, n_steps)
        if baseline_key in cpu_results_map:
            baseline_time = cpu_results_map[baseline_key]

            for dtype in dtypes:
                matching = [
                    r for r in results["gpu"]
                    if r.n_paths == n_paths and r.n_steps == n_steps and r.dtype == str(dtype)
                ]
                if matching:
                    r = matching[0]
                    speedup = baseline_time / r.elapsed_time
                    print(f"  GPU {dtype.__name__:7s}: {speedup:6.2f}x  ({r.elapsed_time:.4f}s)")

    return results


if __name__ == "__main__":
    if not CUPY_AVAILABLE:
        print("CuPy not available. Please install with: pip install cupy-cuda12x")
        exit(1)

    # Run comprehensive benchmark
    results = comprehensive_benchmark_suite()

    # Additional analysis
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("\n[GPU Backend (CuPy)]")
    print("  - Optimized for: Large-scale Monte Carlo simulations")
    print("  - Recommended dtype: float32 (2x faster, sufficient precision)")
    print("  - Typical speedup: 10-100x vs CPU (hardware dependent)")

    print("\n[CPU Backend (NumPy)]")
    print("  - Use for: Small problems (<10k paths), validation, no GPU")
    print("  - Located in: suboptimal/pricing.py")

    print("\n[General Tips]")
    print("  - Use float32 for production (2x faster on GPU)")
    print("  - Use float64 for validation and high-precision requirements")
    print("  - Consider chunking (max_paths_per_chunk) for memory-constrained GPUs")
    print("  - Warmup runs important for accurate GPU benchmarking")
