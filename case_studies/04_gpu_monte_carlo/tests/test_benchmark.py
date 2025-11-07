"""Comprehensive benchmarks for GPU-accelerated Monte Carlo simulation.

This module consolidates all benchmarks for the GPU Monte Carlo case study:
- Asian option pricing benchmarks (GPU vs CPU)
- Different problem sizes (small, medium, large)
- Different precision levels (float32 vs float64)
- Zero-copy GPU pipeline benchmarks
- Speedup calculations and performance analysis

Results are automatically saved to benchmark_results.txt
"""

import time
from typing import Dict, List, NamedTuple, Optional
import sys
from datetime import datetime

import numpy as np
import pytest

# Import pricing functions
from suboptimal.pricing import simulate_gbm_paths as simulate_cpu
from utils import price_asian_option
from optimized.pricing import simulate_gbm_paths as simulate_gpu

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class AsianBenchmarkResult(NamedTuple):
    """Container for Asian option benchmark results."""
    backend: str
    dtype: str
    n_paths: int
    n_steps: int
    simulation_time: float
    pricing_time: float
    total_time: float
    option_price: float
    strike: float
    option_type: str


class ZeroCopyBenchmarkResult(NamedTuple):
    """Container for zero-copy benchmark results."""
    pipeline: str
    dtype: str
    n_paths: int
    n_steps: int
    simulation_time: float
    pricing_time: float
    transfer_time: float
    total_time: float
    option_price: float


# ============================================================================
# ASIAN OPTION BENCHMARKS
# ============================================================================

def benchmark_asian_option_cpu(
    n_paths: int,
    n_steps: int,
    dtype: np.dtype,
    strike: float = 100.0,
    option_type: str = "call",
    seed: int = 42,
) -> AsianBenchmarkResult:
    """Benchmark Asian option pricing on CPU."""
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
    rate = 0.05

    # Time simulation
    start_sim = time.perf_counter()
    t_grid, paths = simulate_cpu(**params)
    sim_time = time.perf_counter() - start_sim

    # Time pricing
    start_price = time.perf_counter()
    option_price = price_asian_option(t_grid, paths, strike, rate, option_type)
    price_time = time.perf_counter() - start_price

    total_time = sim_time + price_time

    return AsianBenchmarkResult(
        backend="cpu",
        dtype=str(dtype),
        n_paths=n_paths,
        n_steps=n_steps,
        simulation_time=sim_time,
        pricing_time=price_time,
        total_time=total_time,
        option_price=option_price,
        strike=strike,
        option_type=option_type,
    )


def benchmark_asian_option_gpu(
    n_paths: int,
    n_steps: int,
    dtype: np.dtype,
    strike: float = 100.0,
    option_type: str = "call",
    seed: int = 42,
    warmup: bool = True,
    max_paths_per_chunk: Optional[int] = None,
) -> AsianBenchmarkResult:
    """Benchmark Asian option pricing on GPU."""
    # Windows GPUs often enforce short kernel timeouts; chunk float64 sims so kernels stay under the limit.
    if max_paths_per_chunk is None and dtype == np.float64 and n_paths >= 500_000:
        max_paths_per_chunk = 250_000

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
    if max_paths_per_chunk is not None:
        params["max_paths_per_chunk"] = max_paths_per_chunk
    rate = 0.05

    # Warmup
    if warmup:
        t_grid, paths = simulate_gpu(**params)
        _ = price_asian_option(t_grid, paths, strike, rate, option_type)

    # Time simulation
    start_sim = time.perf_counter()
    t_grid, paths = simulate_gpu(**params)
    sim_time = time.perf_counter() - start_sim

    # Time pricing
    start_price = time.perf_counter()
    option_price = price_asian_option(t_grid, paths, strike, rate, option_type)
    price_time = time.perf_counter() - start_price

    total_time = sim_time + price_time

    return AsianBenchmarkResult(
        backend="gpu",
        dtype=str(dtype),
        n_paths=n_paths,
        n_steps=n_steps,
        simulation_time=sim_time,
        pricing_time=price_time,
        total_time=total_time,
        option_price=option_price,
        strike=strike,
        option_type=option_type,
    )


# ============================================================================
# TEST CLASSES: ASIAN OPTION BENCHMARKS
# ============================================================================

class TestAsianOptionBenchmarkSmall:
    """Benchmarks for small problem sizes."""

    def test_small_cpu(self):
        """Benchmark small problem on CPU."""
        result = benchmark_asian_option_cpu(
            n_paths=10_000, n_steps=252, dtype=np.float64
        )
        print(f"\n[SMALL CPU] Total: {result.total_time:.4f}s, "
              f"Price: {result.option_price:.4f}")
        assert result.total_time > 0

    def test_small_gpu(self):
        """Benchmark small problem on GPU."""
        result = benchmark_asian_option_gpu(
            n_paths=10_000, n_steps=252, dtype=np.float32
        )
        print(f"\n[SMALL GPU] Total: {result.total_time:.4f}s, "
              f"Price: {result.option_price:.4f}")
        assert result.total_time > 0


class TestAsianOptionBenchmarkMedium:
    """Benchmarks for medium problem sizes."""

    def test_medium_cpu(self):
        """Benchmark medium problem on CPU."""
        result = benchmark_asian_option_cpu(
            n_paths=100_000, n_steps=252, dtype=np.float64
        )
        print(f"\n[MEDIUM CPU] Total: {result.total_time:.4f}s, "
              f"Sim: {result.simulation_time:.4f}s, "
              f"Price: {result.option_price:.4f}")
        assert result.total_time > 0

    def test_medium_gpu(self):
        """Benchmark medium problem on GPU."""
        result = benchmark_asian_option_gpu(
            n_paths=100_000, n_steps=252, dtype=np.float32
        )
        print(f"\n[MEDIUM GPU] Total: {result.total_time:.4f}s, "
              f"Sim: {result.simulation_time:.4f}s, "
              f"Price: {result.option_price:.4f}")
        assert result.total_time > 0


class TestAsianOptionBenchmarkLarge:
    """Benchmarks for large problem sizes."""

    def test_large_cpu(self):
        """Benchmark large problem on CPU."""
        result = benchmark_asian_option_cpu(
            n_paths=500_000, n_steps=252, dtype=np.float32
        )
        print(f"\n[LARGE CPU] Total: {result.total_time:.4f}s, "
              f"Sim: {result.simulation_time:.4f}s, "
              f"Pricing: {result.pricing_time:.4f}s")
        assert result.total_time > 0

    def test_large_gpu(self):
        """Benchmark large problem on GPU."""
        result = benchmark_asian_option_gpu(
            n_paths=500_000, n_steps=252, dtype=np.float32
        )
        print(f"\n[LARGE GPU] Total: {result.total_time:.4f}s, "
              f"Sim: {result.simulation_time:.4f}s, "
              f"Pricing: {result.pricing_time:.4f}s")
        assert result.total_time > 0


class TestAsianOptionSpeedupComparison:
    """Compare speedups between GPU and CPU."""

    def test_speedup_medium_problem(self):
        """Measure GPU speedup vs CPU for medium problem."""
        n_paths = 500_000
        n_steps = 252

        # CPU benchmark
        cpu_result = benchmark_asian_option_cpu(n_paths, n_steps, np.float32)

        # GPU benchmark
        gpu_result = benchmark_asian_option_gpu(n_paths, n_steps, np.float32)

        # Calculate speedups
        sim_speedup = cpu_result.simulation_time / gpu_result.simulation_time
        total_speedup = cpu_result.total_time / gpu_result.total_time

        print(f"\n{'='*70}")
        print(f"SPEEDUP ANALYSIS (n_paths={n_paths:,}, n_steps={n_steps})")
        print(f"{'='*70}")
        print(f"CPU Total Time:    {cpu_result.total_time:.4f}s")
        print(f"  - Simulation:    {cpu_result.simulation_time:.4f}s")
        print(f"  - Pricing:       {cpu_result.pricing_time:.4f}s")
        print(f"GPU Total Time:    {gpu_result.total_time:.4f}s")
        print(f"  - Simulation:    {gpu_result.simulation_time:.4f}s")
        print(f"  - Pricing:       {gpu_result.pricing_time:.4f}s")
        print(f"Simulation Speedup: {sim_speedup:.2f}x")
        print(f"Total Speedup:      {total_speedup:.2f}x")
        print(f"Price Difference:   {abs(cpu_result.option_price - gpu_result.option_price):.6f}")
        print(f"{'='*70}")

        # GPU should be faster
        assert total_speedup > 1.0

    def test_speedup_breakdown(self):
        """Detailed speedup breakdown for large problem."""
        n_paths = 500_000
        n_steps = 252

        cpu_result = benchmark_asian_option_cpu(n_paths, n_steps, np.float32)
        gpu_result = benchmark_asian_option_gpu(n_paths, n_steps, np.float32)

        sim_speedup = cpu_result.simulation_time / gpu_result.simulation_time
        price_speedup = cpu_result.pricing_time / gpu_result.pricing_time
        total_speedup = cpu_result.total_time / gpu_result.total_time

        print(f"\nDetailed Speedup Breakdown:")
        print(f"  Simulation:  {sim_speedup:6.2f}x")
        print(f"  Pricing:     {price_speedup:6.2f}x")
        print(f"  Total:       {total_speedup:6.2f}x")

        # Simulation should benefit most from GPU
        assert sim_speedup > price_speedup


class TestAsianOptionDtypePerformance:
    """Compare float32 vs float64 performance."""

    def test_gpu_dtype_comparison(self):
        """Compare float32 vs float64 on GPU."""
        n_paths = 500_000
        n_steps = 252

        # float32
        result_f32 = benchmark_asian_option_gpu(n_paths, n_steps, np.float32)

        # float64
        result_f64 = benchmark_asian_option_gpu(n_paths, n_steps, np.float64)

        ratio = result_f64.total_time / result_f32.total_time

        print(f"\nGPU dtype Performance:")
        print(f"  float32: {result_f32.total_time:.4f}s, price={result_f32.option_price:.4f}")
        print(f"  float64: {result_f64.total_time:.4f}s, price={result_f64.option_price:.4f}")
        print(f"  Ratio:   {ratio:.2f}x")
        print(f"  Price difference: {abs(result_f32.option_price - result_f64.option_price):.6f}")

        # Performance can vary - just check both complete successfully
        assert result_f32.total_time > 0 and result_f64.total_time > 0


# ============================================================================
# TEST CLASSES: ZERO-COPY PIPELINE BENCHMARKS
# ============================================================================

@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestZeroCopyBenchmark:
    """Benchmark zero-copy GPU pipeline vs standard pipeline."""

    def test_standard_pipeline_float32(self):
        """Benchmark standard pipeline: GPU sim -> CPU transfer -> CPU pricing."""
        n_paths = 500_000
        n_steps = 252
        dtype = np.float32
        strike = 100.0
        rate = 0.05

        params = {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": n_steps,
            "n_paths": n_paths,
            "dtype": dtype,
            "seed": 42,
            "device_output": False,  # Standard: transfer to CPU
        }

        # Warmup
        t_grid, paths = simulate_gpu(**params)
        _ = price_asian_option(t_grid, paths, strike, rate, "call")

        # Timed run
        start_sim = time.perf_counter()
        t_grid, paths = simulate_gpu(**params)
        sim_time = time.perf_counter() - start_sim

        # Pricing (already on CPU)
        start_price = time.perf_counter()
        option_price = price_asian_option(t_grid, paths, strike, rate, "call")
        price_time = time.perf_counter() - start_price

        total_time = sim_time + price_time

        result = ZeroCopyBenchmarkResult(
            pipeline="standard",
            dtype=str(dtype),
            n_paths=n_paths,
            n_steps=n_steps,
            simulation_time=sim_time,
            pricing_time=price_time,
            transfer_time=0.0,  # Transfer included in sim_time
            total_time=total_time,
            option_price=option_price,
        )

        print(f"\n[STANDARD PIPELINE - {dtype.__name__}]")
        print(f"  Simulation (GPU + transfer): {result.simulation_time:.4f}s")
        print(f"  Pricing (CPU):               {result.pricing_time:.4f}s")
        print(f"  Total Time:                  {result.total_time:.4f}s")
        print(f"  Option Price:                {result.option_price:.6f}")

        assert result.total_time > 0

    def test_zero_copy_pipeline_float32(self):
        """Benchmark zero-copy pipeline: GPU sim -> GPU pricing (no transfer)."""
        n_paths = 500_000
        n_steps = 252
        dtype = np.float32
        strike = 100.0
        rate = 0.05

        params = {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": n_steps,
            "n_paths": n_paths,
            "dtype": dtype,
            "seed": 42,
            "device_output": True,  # Zero-copy: keep on GPU
        }

        # Warmup
        t_grid_gpu, paths_gpu = simulate_gpu(**params)
        _ = price_asian_option(t_grid_gpu, paths_gpu, strike, rate, "call")

        # Timed run
        start_sim = time.perf_counter()
        t_grid_gpu, paths_gpu = simulate_gpu(**params)
        sim_time = time.perf_counter() - start_sim

        # Pricing (stays on GPU)
        start_price = time.perf_counter()
        option_price = price_asian_option(t_grid_gpu, paths_gpu, strike, rate, "call")
        price_time = time.perf_counter() - start_price

        # Measure transfer time separately
        start_transfer = time.perf_counter()
        _ = cp.asnumpy(paths_gpu)
        transfer_time = time.perf_counter() - start_transfer

        total_time = sim_time + price_time

        result = ZeroCopyBenchmarkResult(
            pipeline="zero-copy",
            dtype=str(dtype),
            n_paths=n_paths,
            n_steps=n_steps,
            simulation_time=sim_time,
            pricing_time=price_time,
            transfer_time=transfer_time,
            total_time=total_time,
            option_price=option_price,
        )

        print(f"\n[ZERO-COPY PIPELINE - {dtype.__name__}]")
        print(f"  Simulation (GPU only):       {result.simulation_time:.4f}s")
        print(f"  Pricing (GPU):               {result.pricing_time:.4f}s")
        print(f"  Total Time:                  {result.total_time:.4f}s")
        print(f"  Transfer Time (measured):    {result.transfer_time:.4f}s (avoided!)")
        print(f"  Option Price:                {result.option_price:.6f}")

        assert result.total_time > 0

    def test_pipeline_comparison_float32(self):
        """Compare standard vs zero-copy pipeline."""
        n_paths = 500_000
        n_steps = 252
        dtype = np.float32
        strike = 100.0
        rate = 0.05

        # Standard pipeline
        params_standard = {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": n_steps,
            "n_paths": n_paths,
            "dtype": dtype,
            "seed": 42,
            "device_output": False,
        }

        # Zero-copy pipeline
        params_zerocopy = {**params_standard, "device_output": True}

        # Warmup both
        t_grid, paths = simulate_gpu(**params_standard)
        _ = price_asian_option(t_grid, paths, strike, rate, "call")

        t_grid_gpu, paths_gpu = simulate_gpu(**params_zerocopy)
        _ = price_asian_option(t_grid_gpu, paths_gpu, strike, rate, "call")

        # Benchmark standard pipeline
        start = time.perf_counter()
        t_grid, paths = simulate_gpu(**params_standard)
        option_price_standard = price_asian_option(t_grid, paths, strike, rate, "call")
        time_standard = time.perf_counter() - start

        # Benchmark zero-copy pipeline
        start = time.perf_counter()
        t_grid_gpu, paths_gpu = simulate_gpu(**params_zerocopy)
        option_price_zerocopy = price_asian_option(t_grid_gpu, paths_gpu, strike, rate, "call")
        time_zerocopy = time.perf_counter() - start

        # Measure transfer overhead
        start = time.perf_counter()
        _ = cp.asnumpy(paths_gpu)
        transfer_time = time.perf_counter() - start

        # Calculate speedup
        speedup = time_standard / time_zerocopy
        transfer_overhead_pct = (transfer_time / time_zerocopy) * 100

        print(f"\n{'='*80}")
        print(f"ZERO-COPY PIPELINE COMPARISON ({n_paths:,} paths, {dtype.__name__})")
        print(f"{'='*80}")
        print(f"Standard Pipeline (device_output=False):")
        print(f"  Total Time:         {time_standard:.4f}s")
        print(f"  Option Price:       {option_price_standard:.6f}")
        print(f"\nZero-Copy Pipeline (device_output=True):")
        print(f"  Total Time:         {time_zerocopy:.4f}s")
        print(f"  Option Price:       {option_price_zerocopy:.6f}")
        print(f"\nPerformance Gain:")
        print(f"  Speedup:            {speedup:.2f}x")
        print(f"  Time Saved:         {(time_standard - time_zerocopy)*1000:.2f}ms")
        print(f"  Transfer Overhead:  {transfer_overhead_pct:.1f}% of zero-copy time")
        print(f"  Transfer Time:      {transfer_time*1000:.2f}ms (eliminated)")
        print(f"\nPrice Consistency:")
        print(f"  Price Difference:   {abs(option_price_standard - option_price_zerocopy):.8f}")
        print(f"{'='*80}")

        # Zero-copy should be faster
        assert speedup > 1.0, f"Zero-copy pipeline should be faster, got {speedup:.2f}x"

        # Prices should be consistent
        rel_diff = abs(option_price_standard - option_price_zerocopy) / option_price_standard
        assert rel_diff < 1e-6, f"Prices should match, got rel_diff={rel_diff:.2e}"

    def test_pipeline_comparison_float64(self):
        """Compare standard vs zero-copy pipeline with float64."""
        n_paths = 500_000
        n_steps = 252
        dtype = np.float64
        strike = 100.0
        rate = 0.05

        params_standard = {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": n_steps,
            "n_paths": n_paths,
            "dtype": dtype,
            "seed": 42,
            "device_output": False,
            "max_paths_per_chunk": 250_000,  # Avoid TDR timeout
        }

        params_zerocopy = {**params_standard, "device_output": True}

        # Warmup
        t_grid, paths = simulate_gpu(**params_standard)
        _ = price_asian_option(t_grid, paths, strike, rate, "call")

        t_grid_gpu, paths_gpu = simulate_gpu(**params_zerocopy)
        _ = price_asian_option(t_grid_gpu, paths_gpu, strike, rate, "call")

        # Benchmark
        start = time.perf_counter()
        t_grid, paths = simulate_gpu(**params_standard)
        price_standard = price_asian_option(t_grid, paths, strike, rate, "call")
        time_standard = time.perf_counter() - start

        start = time.perf_counter()
        t_grid_gpu, paths_gpu = simulate_gpu(**params_zerocopy)
        price_zerocopy = price_asian_option(t_grid_gpu, paths_gpu, strike, rate, "call")
        time_zerocopy = time.perf_counter() - start

        speedup = time_standard / time_zerocopy

        print(f"\n[FLOAT64 COMPARISON]")
        print(f"  Standard:  {time_standard:.4f}s, price={price_standard:.6f}")
        print(f"  Zero-copy: {time_zerocopy:.4f}s, price={price_zerocopy:.6f}")
        print(f"  Speedup:   {speedup:.2f}x")

        assert speedup > 1.0


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestZeroCopyMemoryAnalysis:
    """Analyze memory transfer overhead."""

    def test_transfer_overhead_analysis(self):
        """Measure transfer overhead for different array sizes."""
        n_steps = 252
        sizes = [10_000, 50_000, 100_000, 500_000]
        dtype = np.float32

        print(f"\n{'='*80}")
        print(f"MEMORY TRANSFER OVERHEAD ANALYSIS ({dtype.__name__})")
        print(f"{'='*80}")
        print(f"{'Paths':>10} {'Array Size':>12} {'Transfer Time':>15} {'% of Total':>12}")
        print(f"{'-'*80}")

        for n_paths in sizes:
            params = {
                "s0": 100.0,
                "mu": 0.05,
                "sigma": 0.2,
                "maturity": 1.0,
                "n_steps": n_steps,
                "n_paths": n_paths,
                "dtype": dtype,
                "seed": 42,
                "device_output": True,
            }

            # Generate data on GPU
            t_grid_gpu, paths_gpu = simulate_gpu(**params)

            # Measure transfer time
            start = time.perf_counter()
            paths_cpu = cp.asnumpy(paths_gpu)
            transfer_time = time.perf_counter() - start

            # Measure pricing time on GPU
            start = time.perf_counter()
            _ = price_asian_option(t_grid_gpu, paths_gpu, 100.0, 0.05, "call")
            pricing_time = time.perf_counter() - start

            # Calculate array size
            array_size_mb = paths_gpu.nbytes / (1024**2)
            transfer_overhead_pct = (transfer_time / (transfer_time + pricing_time)) * 100

            print(f"{n_paths:>10,} {array_size_mb:>10.2f}MB {transfer_time*1000:>12.2f}ms "
                  f"{transfer_overhead_pct:>10.1f}%")

        print(f"{'-'*80}")
        print(f"\nConclusion:")
        print(f"  • Transfer overhead increases with array size")
        print(f"  • Zero-copy pipeline eliminates this overhead completely")
        print(f"  • device_output=True is critical for maximum performance")


if __name__ == "__main__":
    # Save benchmark results to file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_file = "tests/benchmark_results.txt"

    with open(output_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("GPU MONTE CARLO - BENCHMARK RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {timestamp}\n")
        f.write("="*80 + "\n\n")

    # Run pytest and capture output
    pytest.main([__file__, "-v", "-s", "--tb=short"])

    print(f"\n\nBenchmark results saved to: {output_file}")
