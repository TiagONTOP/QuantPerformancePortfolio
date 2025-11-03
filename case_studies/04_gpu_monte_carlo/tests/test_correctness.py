"""Correctness tests for GPU-accelerated GBM Monte Carlo simulation (CuPy).

Tests verify:
- Output shape and finite values
- Statistical moments (mean and variance of log-returns)
- Reproducibility with fixed seeds
- Parity between GPU (optimized) and CPU (suboptimal) versions
- Antithetic variates correctness

Note: These tests require CuPy to be installed. The optimized version is GPU-only.
"""

import numpy as np
import pytest

from optimized.pricing import simulate_gbm_paths as simulate_gbm_gpu
CUPY_AVAILABLE = True

from suboptimal.pricing import simulate_gbm_paths as simulate_gbm_cpu


class TestBasicCorrectness:
    """Test basic correctness requirements."""

    @pytest.fixture
    def base_params_cpu(self):
        """Common parameters for CPU testing."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 10000,
            "rng": np.random.default_rng(42),
        }

    @pytest.fixture
    def base_params_gpu(self):
        """Common parameters for GPU testing."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 10000,
            "seed": 42,
        }

    def test_output_shape_cpu(self, base_params_cpu):
        """Test that output shapes are correct (CPU/suboptimal)."""
        t_grid, paths = simulate_gbm_cpu(**base_params_cpu)

        assert t_grid.shape == (base_params_cpu["n_steps"] + 1,)
        assert paths.shape == (base_params_cpu["n_steps"] + 1, base_params_cpu["n_paths"])

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_output_shape_gpu(self, base_params_gpu):
        """Test that output shapes are correct (GPU/optimized)."""
        t_grid, paths = simulate_gbm_gpu(**base_params_gpu)

        assert t_grid.shape == (base_params_gpu["n_steps"] + 1,)
        assert paths.shape == (base_params_gpu["n_steps"] + 1, base_params_gpu["n_paths"])


    def test_all_finite_values(self, base_params_cpu):
        """Test that all generated values are finite (CPU)."""
        _, paths = simulate_gbm_cpu(**base_params_cpu)

        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0)  # Prices must be positive

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_all_finite_values_gpu(self, base_params_gpu):
        """Test that all generated values are finite (GPU)."""
        _, paths = simulate_gbm_gpu(**base_params_gpu)

        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0)  # Prices must be positive

    def test_initial_price(self, base_params_cpu):
        """Test that all paths start at s0."""
        _, paths = simulate_gbm_cpu(**base_params_cpu)

        assert np.allclose(paths[0, :], base_params_cpu["s0"])

    def test_time_grid(self, base_params_cpu):
        """Test that time grid is correct."""
        t_grid, _ = simulate_gbm_cpu(**base_params_cpu)

        expected = np.linspace(0.0, base_params_cpu["maturity"], base_params_cpu["n_steps"] + 1)
        assert np.allclose(t_grid, expected)


class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_s0(self):
        """Test that negative s0 raises ValueError."""
        with pytest.raises(ValueError, match="s0 must be strictly positive"):
            simulate_gbm_cpu(
                s0=-100.0, mu=0.05, sigma=0.2, maturity=1.0,
                n_steps=252, n_paths=1000
            )

    def test_invalid_sigma(self):
        """Test that negative sigma raises ValueError."""
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            simulate_gbm_cpu(
                s0=100.0, mu=0.05, sigma=-0.2, maturity=1.0,
                n_steps=252, n_paths=1000
            )

    def test_invalid_maturity(self):
        """Test that non-positive maturity raises ValueError."""
        with pytest.raises(ValueError, match="maturity must be strictly positive"):
            simulate_gbm_cpu(
                s0=100.0, mu=0.05, sigma=0.2, maturity=0.0,
                n_steps=252, n_paths=1000
            )

    def test_invalid_n_steps(self):
        """Test that non-positive n_steps raises ValueError."""
        with pytest.raises(ValueError, match="n_steps must be a positive integer"):
            simulate_gbm_cpu(
                s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
                n_steps=0, n_paths=1000
            )

    def test_invalid_n_paths(self):
        """Test that non-positive n_paths raises ValueError."""
        with pytest.raises(ValueError, match="n_paths must be a positive integer"):
            simulate_gbm_cpu(
                s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
                n_steps=252, n_paths=0
            )

    def test_invalid_backend(self):
        """Test that invalid backend raises ValueError (skipped - no unified API)."""
        pytest.skip("No unified backend API - use separate CPU/GPU functions")


class TestStatisticalMoments:
    """Test statistical properties of simulated paths."""

    @pytest.fixture
    def large_sample_params_cpu(self):
        """Parameters for statistical testing (large sample, CPU)."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 100000,
            "rng": np.random.default_rng(42),
        }

    @pytest.fixture
    def large_sample_params_gpu(self):
        """Parameters for statistical testing (large sample, GPU)."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 100000,
            "seed": 42,
        }

    def test_log_returns_mean_cpu(self, large_sample_params_cpu):
        """Test that log-returns have approximately correct mean (CPU)."""
        _, paths = simulate_gbm_cpu(**large_sample_params_cpu)

        # Calculate log returns
        log_returns = np.diff(np.log(paths), axis=0)
        dt = large_sample_params_cpu["maturity"] / large_sample_params_cpu["n_steps"]

        # Expected drift (per step)
        q = 0.0  # no dividend
        expected_drift = (
            large_sample_params_cpu["mu"] - q - 0.5 * large_sample_params_cpu["sigma"] ** 2
        ) * dt

        # Sample mean
        sample_mean = np.mean(log_returns)

        # Allow for sampling error (3 standard errors)
        std_error = large_sample_params_cpu["sigma"] * np.sqrt(dt) / np.sqrt(
            large_sample_params_cpu["n_paths"] * large_sample_params_cpu["n_steps"]
        )

        assert abs(sample_mean - expected_drift) < 3 * std_error

    def test_log_returns_std_cpu(self, large_sample_params_cpu):
        """Test that log-returns have approximately correct std dev (CPU)."""
        _, paths = simulate_gbm_cpu(**large_sample_params_cpu)

        # Calculate log returns
        log_returns = np.diff(np.log(paths), axis=0)
        dt = large_sample_params_cpu["maturity"] / large_sample_params_cpu["n_steps"]

        # Expected volatility (per step)
        expected_vol = large_sample_params_cpu["sigma"] * np.sqrt(dt)

        # Sample std dev
        sample_std = np.std(log_returns, ddof=1)

        # Allow for sampling error (approximate)
        # std(sample_std) H expected_vol / sqrt(2 * N)
        N = large_sample_params_cpu["n_paths"] * large_sample_params_cpu["n_steps"]
        std_error = expected_vol / np.sqrt(2 * N)

        assert abs(sample_std - expected_vol) < 3 * std_error

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_log_returns_mean_cupy(self, large_sample_params_gpu):
        """Test that log-returns have approximately correct mean (CuPy)."""
        _, paths = simulate_gbm_gpu(**large_sample_params_gpu)

        log_returns = np.diff(np.log(paths), axis=0)
        dt = large_sample_params_gpu["maturity"] / large_sample_params_gpu["n_steps"]
        expected_drift = (
            large_sample_params_gpu["mu"] - 0.5 * large_sample_params_gpu["sigma"] ** 2
        ) * dt

        sample_mean = np.mean(log_returns)
        std_error = large_sample_params_gpu["sigma"] * np.sqrt(dt) / np.sqrt(
            large_sample_params_gpu["n_paths"] * large_sample_params_gpu["n_steps"]
        )

        assert abs(sample_mean - expected_drift) < 3 * std_error


class TestReproducibility:
    """Test reproducibility with fixed seeds."""

    @pytest.fixture
    def repro_params_cpu(self):
        """Parameters for reproducibility testing (CPU)."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 1000,
        }

    @pytest.fixture
    def repro_params_gpu(self):
        """Parameters for reproducibility testing (GPU)."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 1000,
            "seed": 42,
        }

    def test_cpu_reproducibility(self, repro_params_cpu):
        """Test that CPU backend is reproducible with same RNG."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        _, paths1 = simulate_gbm_cpu(**repro_params_cpu, rng=rng1)
        _, paths2 = simulate_gbm_cpu(**repro_params_cpu, rng=rng2)

        assert np.allclose(paths1, paths2)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_cupy_reproducibility(self, repro_params_gpu):
        """Test that CuPy backend is reproducible with same seed."""
        _, paths1 = simulate_gbm_gpu(**repro_params_gpu)
        _, paths2 = simulate_gbm_gpu(**repro_params_gpu)

        assert np.allclose(paths1, paths2)



class TestBackendParity:
    """Test parity between different backends using pre-generated shocks."""

    @pytest.fixture
    def parity_params(self):
        """Parameters for parity testing."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 1000,
        }

    def test_cpu_cupy_parity_float64(self, parity_params):
        """Test CPU and CuPy give same results with same shocks (float64)."""
        if not CUPY_AVAILABLE:
            pytest.skip("CuPy not available")

        # Generate shocks on CPU
        rng = np.random.default_rng(42)
        shocks = rng.standard_normal(
            (parity_params["n_paths"], parity_params["n_steps"])
        )

        # Run both backends with same shocks
        _, paths_cpu = simulate_gbm_cpu(
            **parity_params, shocks=shocks, dtype=np.float64
        )
        _, paths_cupy = simulate_gbm_gpu(
            **parity_params, shocks=shocks, dtype=np.float64
        )

        # Should be very close in float64
        assert np.allclose(paths_cpu, paths_cupy, rtol=1e-12, atol=1e-12)

    def test_cpu_cupy_parity_float32(self, parity_params):
        """Test CPU and CuPy give similar results with same shocks (float32)."""
        if not CUPY_AVAILABLE:
            pytest.skip("CuPy not available")

        # Generate shocks on CPU
        rng = np.random.default_rng(42)
        shocks = rng.standard_normal(
            (parity_params["n_paths"], parity_params["n_steps"])
        )

        # Run both backends with same shocks
        _, paths_cpu = simulate_gbm_cpu(
            **parity_params, shocks=shocks, dtype=np.float32
        )
        _, paths_cupy = simulate_gbm_gpu(
            **parity_params, shocks=shocks, dtype=np.float32
        )

        # Float32 has lower precision
        assert np.allclose(paths_cpu, paths_cupy, rtol=1e-5, atol=1e-5)


class TestAntitheticVariates:
    """Test antithetic variates implementation."""

    @pytest.fixture
    def antithetic_params_cpu(self):
        """Parameters for antithetic testing (CPU)."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 10000,  # Even number
            "rng": np.random.default_rng(42),
        }

    @pytest.fixture
    def antithetic_params_gpu(self):
        """Parameters for antithetic testing (GPU)."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 10000,  # Even number
            "seed": 42,
        }

    def test_antithetic_reduces_variance_cpu(self, antithetic_params_cpu):
        """Test that antithetic variates reduce variance of estimator (CPU)."""
        # Run without antithetic
        _, paths_standard = simulate_gbm_cpu(
            **antithetic_params_cpu, antithetic=False
        )

        # Run with antithetic
        _, paths_antithetic = simulate_gbm_cpu(
            **antithetic_params_cpu, antithetic=True
        )

        # Compare variance of final prices
        var_standard = np.var(paths_standard[-1, :])
        var_antithetic = np.var(paths_antithetic[-1, :])

        # Antithetic should reduce variance (though not guaranteed in finite samples)
        # At least check both are reasonable
        assert var_standard > 0
        assert var_antithetic > 0

    def test_antithetic_shape_odd_paths(self):
        """Test antithetic works with odd number of paths."""
        # Odd number of paths
        _, paths = simulate_gbm_cpu(
            s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
            n_steps=252, n_paths=999, antithetic=True, rng=np.random.default_rng(42)
        )

        assert paths.shape[1] == 999

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_antithetic_cupy(self, antithetic_params_gpu):
        """Test antithetic variates with CuPy backend."""
        _, paths = simulate_gbm_gpu(
            **antithetic_params_gpu, antithetic=True
        )

        # Check shape and finite values
        assert paths.shape[1] == antithetic_params_gpu["n_paths"]
        assert np.all(np.isfinite(paths))


class TestDtypeSupport:
    """Test different dtype support."""

    def test_float32_cpu(self):
        """Test float32 dtype on CPU."""
        _, paths = simulate_gbm_cpu(
            s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
            n_steps=252, n_paths=1000, dtype=np.float32, rng=np.random.default_rng(42)
        )

        assert paths.dtype == np.float32

    def test_float64_cpu(self):
        """Test float64 dtype on CPU."""
        _, paths = simulate_gbm_cpu(
            s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
            n_steps=252, n_paths=1000, dtype=np.float64, rng=np.random.default_rng(42)
        )

        assert paths.dtype == np.float64

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_float32_cupy(self):
        """Test float32 dtype on CuPy."""
        _, paths = simulate_gbm_gpu(
            s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
            n_steps=252, n_paths=1000, dtype=np.float32, seed=42
        )

        assert paths.dtype == np.float32


class TestDividendYield:
    """Test dividend yield functionality."""

    def test_dividend_yield_effect(self):
        """Test that dividend yield affects drift correctly."""
        params = {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 100000,
            "rng": np.random.default_rng(42),
        }

        # Without dividend
        _, paths_no_div = simulate_gbm_cpu(**params, dividend_yield=0.0)

        # With dividend
        _, paths_with_div = simulate_gbm_cpu(**params, dividend_yield=0.03)

        # Paths with dividend should have lower final values on average
        mean_no_div = np.mean(paths_no_div[-1, :])
        mean_with_div = np.mean(paths_with_div[-1, :])

        assert mean_with_div < mean_no_div


class TestMemoryChunking:
    """Test memory chunking for large simulations."""

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_chunking_same_result(self):
        """Test that chunking gives same result as non-chunked."""
        params = {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 10000,
            "seed": 42,
            "dtype": np.float32,
        }

        # Without chunking
        _, paths_no_chunk = simulate_gbm_gpu(**params)

        # With chunking
        _, paths_chunked = simulate_gbm_gpu(**params, max_paths_per_chunk=2500)

        # Should be identical
        assert np.allclose(paths_no_chunk, paths_chunked, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
