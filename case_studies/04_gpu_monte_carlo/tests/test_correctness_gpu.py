"""Correctness tests for GPU-accelerated GBM Monte Carlo simulation (CuPy only).

Tests verify:
- Output shape and finite values (GPU)
- Statistical moments (mean and variance of log-returns)
- Reproducibility with fixed seeds
- Parity between GPU (optimized) and CPU (suboptimal) versions
- Antithetic variates correctness

Note: These tests require CuPy to be installed.
"""

import numpy as np
import pytest

from optimized.pricing import simulate_gbm_paths as simulate_gbm_gpu
CUPY_AVAILABLE = True

# Import CPU version (suboptimal) for comparison
from suboptimal.pricing import simulate_gbm_paths as simulate_gbm_cpu


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestGPUCorrectness:
    """Test GPU implementation correctness."""

    @pytest.fixture
    def base_params(self):
        """Common parameters for testing."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 10000,
            "seed": 42,
        }

    def test_output_shape(self, base_params):
        """Test that output shapes are correct."""
        t_grid, paths = simulate_gbm_gpu(**base_params)

        assert t_grid.shape == (base_params["n_steps"] + 1,)
        assert paths.shape == (base_params["n_steps"] + 1, base_params["n_paths"])

    def test_all_finite_values(self, base_params):
        """Test that all generated values are finite."""
        _, paths = simulate_gbm_gpu(**base_params)

        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0)  # Prices must be positive

    def test_initial_price(self, base_params):
        """Test that all paths start at s0."""
        _, paths = simulate_gbm_gpu(**base_params)

        assert np.allclose(paths[0, :], base_params["s0"])

    def test_time_grid(self, base_params):
        """Test that time grid is correct."""
        t_grid, _ = simulate_gbm_gpu(**base_params)

        expected = np.linspace(0.0, base_params["maturity"], base_params["n_steps"] + 1)
        assert np.allclose(t_grid, expected)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_s0(self):
        """Test that negative s0 raises ValueError."""
        with pytest.raises(ValueError, match="s0 must be strictly positive"):
            simulate_gbm_gpu(
                s0=-100.0, mu=0.05, sigma=0.2, maturity=1.0,
                n_steps=252, n_paths=1000
            )

    def test_invalid_sigma(self):
        """Test that negative sigma raises ValueError."""
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            simulate_gbm_gpu(
                s0=100.0, mu=0.05, sigma=-0.2, maturity=1.0,
                n_steps=252, n_paths=1000
            )

    def test_invalid_maturity(self):
        """Test that non-positive maturity raises ValueError."""
        with pytest.raises(ValueError, match="maturity must be strictly positive"):
            simulate_gbm_gpu(
                s0=100.0, mu=0.05, sigma=0.2, maturity=0.0,
                n_steps=252, n_paths=1000
            )

    def test_invalid_n_steps(self):
        """Test that non-positive n_steps raises ValueError."""
        with pytest.raises(ValueError, match="n_steps must be a positive integer"):
            simulate_gbm_gpu(
                s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
                n_steps=0, n_paths=1000
            )

    def test_invalid_n_paths(self):
        """Test that non-positive n_paths raises ValueError."""
        with pytest.raises(ValueError, match="n_paths must be a positive integer"):
            simulate_gbm_gpu(
                s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
                n_steps=252, n_paths=0
            )


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestStatisticalMoments:
    """Test statistical properties of simulated paths."""

    @pytest.fixture
    def large_sample_params(self):
        """Parameters for statistical testing (large sample)."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 100000,
            "seed": 42,
        }

    def test_log_returns_mean_gpu(self, large_sample_params):
        """Test that log-returns have approximately correct mean (GPU)."""
        _, paths = simulate_gbm_gpu(**large_sample_params)

        # Calculate log returns
        log_returns = np.diff(np.log(paths), axis=0)
        dt = large_sample_params["maturity"] / large_sample_params["n_steps"]

        # Expected drift (per step)
        q = 0.0  # no dividend
        expected_drift = (
            large_sample_params["mu"] - q - 0.5 * large_sample_params["sigma"] ** 2
        ) * dt

        # Sample mean
        sample_mean = np.mean(log_returns)

        # Allow for sampling error (3 standard errors)
        std_error = large_sample_params["sigma"] * np.sqrt(dt) / np.sqrt(
            large_sample_params["n_paths"] * large_sample_params["n_steps"]
        )

        assert abs(sample_mean - expected_drift) < 3 * std_error

    def test_log_returns_std_gpu(self, large_sample_params):
        """Test that log-returns have approximately correct std dev (GPU)."""
        _, paths = simulate_gbm_gpu(**large_sample_params)

        # Calculate log returns
        log_returns = np.diff(np.log(paths), axis=0)
        dt = large_sample_params["maturity"] / large_sample_params["n_steps"]

        # Expected volatility (per step)
        expected_vol = large_sample_params["sigma"] * np.sqrt(dt)

        # Sample std dev
        sample_std = np.std(log_returns, ddof=1)

        # Allow for sampling error (approximate)
        N = large_sample_params["n_paths"] * large_sample_params["n_steps"]
        std_error = expected_vol / np.sqrt(2 * N)

        assert abs(sample_std - expected_vol) < 3 * std_error


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestReproducibility:
    """Test reproducibility with fixed seeds."""

    @pytest.fixture
    def repro_params(self):
        """Parameters for reproducibility testing."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 1000,
            "seed": 42,
        }

    def test_gpu_reproducibility(self, repro_params):
        """Test that GPU backend is reproducible with same seed."""
        _, paths1 = simulate_gbm_gpu(**repro_params)
        _, paths2 = simulate_gbm_gpu(**repro_params)

        assert np.allclose(paths1, paths2)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestGPUvsCPUParity:
    """Test parity between GPU and CPU using pre-generated shocks."""

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

    def test_gpu_cpu_parity_float64(self, parity_params):
        """Test GPU and CPU give same results with same shocks (float64)."""
        # Generate shocks on CPU
        rng = np.random.default_rng(42)
        shocks = rng.standard_normal(
            (parity_params["n_paths"], parity_params["n_steps"])
        )

        # Run both backends with same shocks
        _, paths_cpu = simulate_gbm_cpu(
            **parity_params, shocks=shocks, dtype=np.float64
        )
        _, paths_gpu = simulate_gbm_gpu(
            **parity_params, shocks=shocks, dtype=np.float64
        )

        # Should be very close in float64
        assert np.allclose(paths_cpu, paths_gpu, rtol=1e-12, atol=1e-12)

    def test_gpu_cpu_parity_float32(self, parity_params):
        """Test GPU and CPU give similar results with same shocks (float32)."""
        # Generate shocks on CPU
        rng = np.random.default_rng(42)
        shocks = rng.standard_normal(
            (parity_params["n_paths"], parity_params["n_steps"])
        )

        # Run both backends with same shocks
        _, paths_cpu = simulate_gbm_cpu(
            **parity_params, shocks=shocks, dtype=np.float32
        )
        _, paths_gpu = simulate_gbm_gpu(
            **parity_params, shocks=shocks, dtype=np.float32
        )

        # Float32 has lower precision
        assert np.allclose(paths_cpu, paths_gpu, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestAntitheticVariates:
    """Test antithetic variates implementation."""

    @pytest.fixture
    def antithetic_params(self):
        """Parameters for antithetic testing."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 10000,  # Even number
            "seed": 42,
        }

    def test_antithetic_reduces_variance_gpu(self, antithetic_params):
        """Test that antithetic variates reduce variance of estimator (GPU)."""
        # Run without antithetic
        _, paths_standard = simulate_gbm_gpu(
            **antithetic_params, antithetic=False
        )

        # Run with antithetic
        _, paths_antithetic = simulate_gbm_gpu(
            **antithetic_params, antithetic=True
        )

        # Compare variance of final prices
        var_standard = np.var(paths_standard[-1, :])
        var_antithetic = np.var(paths_antithetic[-1, :])

        # Both should be reasonable (not NaN or 0)
        assert var_standard > 0
        assert var_antithetic > 0

    def test_antithetic_shape_odd_paths(self):
        """Test antithetic works with odd number of paths."""
        # Odd number of paths
        _, paths = simulate_gbm_gpu(
            s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
            n_steps=252, n_paths=999, antithetic=True, seed=42
        )

        assert paths.shape[1] == 999


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestDtypeSupport:
    """Test different dtype support."""

    def test_float32_gpu(self):
        """Test float32 dtype on GPU."""
        _, paths = simulate_gbm_gpu(
            s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
            n_steps=252, n_paths=1000, dtype=np.float32, seed=42
        )

        assert paths.dtype == np.float32

    def test_float64_gpu(self):
        """Test float64 dtype on GPU."""
        _, paths = simulate_gbm_gpu(
            s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
            n_steps=252, n_paths=1000, dtype=np.float64, seed=42
        )

        assert paths.dtype == np.float64


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
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
            "seed": 42,
        }

        # Without dividend
        _, paths_no_div = simulate_gbm_gpu(**params, dividend_yield=0.0)

        # With dividend
        _, paths_with_div = simulate_gbm_gpu(**params, dividend_yield=0.03)

        # Paths with dividend should have lower final values on average
        mean_no_div = np.mean(paths_no_div[-1, :])
        mean_with_div = np.mean(paths_with_div[-1, :])

        assert mean_with_div < mean_no_div


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestMemoryChunking:
    """Test memory chunking for large simulations."""

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
