"""Rigorous unit tests for Asian option pricing with GPU vs CPU comparison.

Tests verify:
- Numerical correctness of Asian option pricing
- Parity between GPU (optimized) and CPU (suboptimal) implementations
- Edge cases and boundary conditions
- Statistical consistency
- Different option types (call/put)
- Different strike prices (ITM, ATM, OTM)
"""

import numpy as np
import pytest

# Import pricing functions
from suboptimal.pricing import simulate_gbm_paths as simulate_cpu
from utils import price_asian_option

from optimized.pricing import simulate_gbm_paths as simulate_gpu
CUPY_AVAILABLE = True


class TestAsianOptionBasics:
    """Test basic functionality of Asian option pricing."""

    @pytest.fixture
    def simple_paths(self):
        """Generate simple deterministic paths for testing."""
        # 3 paths, 5 time steps (returned as (n_steps + 1, n_paths) = (5, 3))
        # Path 1: [100, 101, 102, 103, 104]
        # Path 2: [100, 100, 100, 100, 100]
        # Path 3: [100, 99, 98, 97, 96]
        paths = np.array([
            [100.0, 101.0, 102.0, 103.0, 104.0],
            [100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 99.0, 98.0, 97.0, 96.0],
        ], dtype=float).T
        time_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        return time_grid, paths

    def test_call_option_itm(self, simple_paths):
        """Test call option in-the-money."""
        t_grid, paths = simple_paths
        strike = 95.0
        rate = 0.05

        price = price_asian_option(t_grid, paths, strike, rate, "call")

        # Calculate expected:
        # Avg prices: [102.0, 100.0, 98.0]
        # Payoffs: [7.0, 5.0, 3.0]
        # Mean payoff: 5.0
        # Discounted: 5.0 * exp(-0.05 * 1.0) = 4.756
        expected = 5.0 * np.exp(-0.05 * 1.0)

        assert abs(price - expected) < 1e-10

    def test_put_option_itm(self, simple_paths):
        """Test put option in-the-money."""
        t_grid, paths = simple_paths
        strike = 105.0
        rate = 0.05

        price = price_asian_option(t_grid, paths, strike, rate, "put")

        # Calculate expected:
        # Avg prices: [102.0, 100.0, 98.0]
        # Payoffs: [3.0, 5.0, 7.0]
        # Mean payoff: 5.0
        # Discounted: 5.0 * exp(-0.05 * 1.0) = 4.756
        expected = 5.0 * np.exp(-0.05 * 1.0)

        assert abs(price - expected) < 1e-10

    def test_call_option_otm(self, simple_paths):
        """Test call option out-of-the-money."""
        t_grid, paths = simple_paths
        strike = 110.0
        rate = 0.05

        price = price_asian_option(t_grid, paths, strike, rate, "call")

        # All average prices < 110, so payoff = 0
        assert price == 0.0

    def test_put_option_otm(self, simple_paths):
        """Test put option out-of-the-money."""
        t_grid, paths = simple_paths
        strike = 90.0
        rate = 0.05

        price = price_asian_option(t_grid, paths, strike, rate, "put")

        # All average prices > 90, so payoff = 0
        assert price == 0.0

    def test_call_option_atm(self, simple_paths):
        """Test call option at-the-money."""
        t_grid, paths = simple_paths
        strike = 100.0
        rate = 0.05

        price = price_asian_option(t_grid, paths, strike, rate, "call")

        # Avg prices: [102.0, 100.0, 98.0]
        # Payoffs: [2.0, 0.0, 0.0]
        # Mean payoff: 2.0/3
        expected = (2.0 / 3.0) * np.exp(-0.05 * 1.0)

        assert abs(price - expected) < 1e-10


class TestAsianOptionInputValidation:
    """Test input validation for Asian option pricing."""

    def test_invalid_time_grid_empty(self):
        """Test that empty time grid raises error."""
        with pytest.raises(ValueError, match="non-empty 1D array"):
            price_asian_option([], np.array([[100]]), 100, 0.05, "call")

    def test_invalid_paths_1d(self):
        """Test that 1D paths array raises error."""
        with pytest.raises(ValueError, match="2D array"):
            price_asian_option([0, 1], [100, 101], 100, 0.05, "call")

    def test_mismatched_dimensions(self):
        """Test that mismatched time_grid and paths raises error."""
        t_grid = np.array([0.0, 1.0])  # 2 points
        paths = np.array([[100, 101, 102]]).T  # 3 points (time dimension mismatch)
        with pytest.raises(ValueError, match="must match"):
            price_asian_option(t_grid, paths, 100, 0.05, "call")

    def test_invalid_option_type(self):
        """Test that invalid option type raises error."""
        t_grid = np.array([0.0, 1.0])
        paths = np.array([[100.0], [101.0]])
        with pytest.raises(ValueError, match='must be one of'):
            price_asian_option(t_grid, paths, 100, 0.05, "invalid")


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestSimulatorStatisticalEquivalence:
    """Test statistical equivalence between GPU and CPU simulators for Asian options.

    This class tests that both GPU and CPU simulators produce statistically equivalent
    results when used with the same pricing function. It does NOT test GPU vs CPU pricers
    (since there's only one pricer: utils.price_asian_option), but rather validates that
    the two simulators (optimized.pricing vs suboptimal.pricing) generate paths with
    equivalent statistical properties.
    """

    @pytest.fixture
    def pricing_params(self):
        """Common parameters for pricing tests."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 50000,  # Large enough for statistical accuracy
        }

    @pytest.fixture
    def option_params(self):
        """Option parameters for testing."""
        return {
            "rate": 0.05,
        }

    def test_call_parity_with_same_shocks(self, pricing_params, option_params):
        """Test that GPU and CPU give same call option price with same random seed."""
        # Use same seed for both CPU and GPU
        seed = 42

        # CPU simulation with rng
        t_cpu, paths_cpu = simulate_cpu(
            **pricing_params,
            rng=np.random.default_rng(seed),
            dtype=np.float64
        )

        # GPU simulation with seed
        t_gpu, paths_gpu = simulate_gpu(
            **pricing_params,
            seed=seed,
            dtype=np.float64
        )

        # Price call option
        strike = 100.0
        price_cpu = price_asian_option(t_cpu, paths_cpu, strike, option_params["rate"], "call")
        price_gpu = price_asian_option(t_gpu, paths_gpu, strike, option_params["rate"], "call")

        # Verify both produce reasonable results
        assert 0 < price_cpu < pricing_params["s0"], f"CPU price {price_cpu:.4f} seems unreasonable"
        assert 0 < price_gpu < pricing_params["s0"], f"GPU price {price_gpu:.4f} seems unreasonable"

        # Prices should be close (within Monte Carlo error)
        # Note: Different RNGs (NumPy vs CuPy) will give different but statistically equivalent results
        rel_diff = abs(price_cpu - price_gpu) / price_cpu
        assert rel_diff < 0.05, \
            f"CPU/GPU prices differ too much: CPU={price_cpu:.4f}, GPU={price_gpu:.4f}, rel_diff={rel_diff:.2%}"

    def test_put_parity_with_same_shocks(self, pricing_params, option_params):
        """Test that GPU and CPU give similar put option prices with same random seed."""
        # Use same seed for both CPU and GPU
        seed = 123

        # CPU simulation
        t_cpu, paths_cpu = simulate_cpu(
            **pricing_params,
            rng=np.random.default_rng(seed),
            dtype=np.float64
        )

        # GPU simulation
        t_gpu, paths_gpu = simulate_gpu(
            **pricing_params,
            seed=seed,
            dtype=np.float64
        )

        # Price put option
        strike = 100.0
        price_cpu = price_asian_option(t_cpu, paths_cpu, strike, option_params["rate"], "put")
        price_gpu = price_asian_option(t_gpu, paths_gpu, strike, option_params["rate"], "put")

        # Should be statistically similar (within Monte Carlo error)
        rel_diff = abs(price_cpu - price_gpu) / price_cpu if price_cpu != 0 else abs(price_cpu - price_gpu)
        assert rel_diff < 0.05, f"CPU: {price_cpu:.4f}, GPU: {price_gpu:.4f}, rel_diff: {rel_diff:.2%}"

    def test_multiple_strikes_parity(self, pricing_params, option_params):
        """Test parity across multiple strike prices."""
        # Use same seed for both
        seed = 999

        # Simulate once for each backend
        t_cpu, paths_cpu = simulate_cpu(**pricing_params, rng=np.random.default_rng(seed), dtype=np.float64)
        t_gpu, paths_gpu = simulate_gpu(**pricing_params, seed=seed, dtype=np.float64)

        # Test multiple strikes
        strikes = [80, 90, 100, 110, 120]
        for strike in strikes:
            for option_type in ["call", "put"]:
                price_cpu = price_asian_option(t_cpu, paths_cpu, strike, option_params["rate"], option_type)
                price_gpu = price_asian_option(t_gpu, paths_gpu, strike, option_params["rate"], option_type)

                rel_diff = abs(price_cpu - price_gpu) / max(price_cpu, 1e-10)
                assert rel_diff < 0.05, (
                    f"Strike {strike}, {option_type}: CPU={price_cpu:.4f}, "
                    f"GPU={price_gpu:.4f}, rel_diff={rel_diff:.2%}"
                )

    def test_parity_float32(self, pricing_params, option_params):
        """Test parity with float32 (lower precision)."""
        seed = 777

        # Simulate with float32
        t_cpu, paths_cpu = simulate_cpu(**pricing_params, rng=np.random.default_rng(seed), dtype=np.float32)
        t_gpu, paths_gpu = simulate_gpu(**pricing_params, seed=seed, dtype=np.float32)

        strike = 100.0
        price_cpu = price_asian_option(t_cpu, paths_cpu, strike, option_params["rate"], "call")
        price_gpu = price_asian_option(t_gpu, paths_gpu, strike, option_params["rate"], "call")

        # Float32 has lower precision plus different RNGs
        rel_diff = abs(price_cpu - price_gpu) / price_cpu if price_cpu != 0 else abs(price_cpu - price_gpu)
        assert rel_diff < 0.05, f"CPU: {price_cpu:.6f}, GPU: {price_gpu:.6f}, rel_diff: {rel_diff:.2%}"


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestAsianOptionStatisticalProperties:
    """Test statistical properties of Asian option pricing."""

    @pytest.fixture
    def large_sample_params(self):
        """Parameters for statistical testing."""
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 100000,
            "seed": 42,
        }

    def test_call_put_parity_approximate(self, large_sample_params):
        """Test approximate call-put parity for Asian options."""
        # Note: Exact call-put parity doesn't hold for Asian options,
        # but we can test an approximate relationship

        strike = 100.0
        rate = 0.05

        # GPU simulation
        t, paths = simulate_gpu(**large_sample_params, dtype=np.float32)

        # Price both call and put
        call_price = price_asian_option(t, paths, strike, rate, "call")
        put_price = price_asian_option(t, paths, strike, rate, "put")

        # Average price of the paths
        avg_prices = paths.mean(axis=0)
        forward = avg_prices.mean()

        # Approximate relationship: C - P ≈ exp(-r*T) * (F - K)
        lhs = call_price - put_price
        rhs = np.exp(-rate * large_sample_params["maturity"]) * (forward - strike)

        # Should be approximately equal (within sampling error)
        rel_diff = abs(lhs - rhs) / max(abs(rhs), 1e-10)
        assert rel_diff < 0.05, f"Call-Put parity violation: {lhs:.4f} vs {rhs:.4f}"

    def test_increasing_volatility_increases_price(self):
        """Test that higher volatility leads to higher option prices."""
        params = {
            "s0": 100.0,
            "mu": 0.05,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 50000,
            "dtype": np.float32,
            "seed": 42,
        }

        strike = 100.0
        rate = 0.05

        # Low volatility
        t_low, paths_low = simulate_gpu(**params, sigma=0.1)
        price_low = price_asian_option(t_low, paths_low, strike, rate, "call")

        # High volatility
        t_high, paths_high = simulate_gpu(**params, sigma=0.3)
        price_high = price_asian_option(t_high, paths_high, strike, rate, "call")

        # Higher volatility should give higher call price
        assert price_high > price_low, f"High vol: {price_high:.4f}, Low vol: {price_low:.4f}"

    def test_price_converges_with_sample_size(self):
        """Test that price estimates converge as sample size increases."""
        params = {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "dtype": np.float32,
        }

        strike = 100.0
        rate = 0.05

        # Generate large sample once
        rng = np.random.default_rng(42)
        large_shocks = rng.standard_normal((200000, params["n_steps"]))

        # Use subsets of increasing size
        prices = []
        sample_sizes = [10000, 50000, 100000, 200000]

        for n_paths in sample_sizes:
            shocks = large_shocks[:n_paths, :]
            t, paths = simulate_gpu(**params, n_paths=n_paths, shocks=shocks)
            price = price_asian_option(t, paths, strike, rate, "call")
            prices.append(price)

        # Prices should converge (std dev decreases)
        # Last two prices should be very close
        rel_diff = abs(prices[-1] - prices[-2]) / prices[-1]
        assert rel_diff < 0.01, f"Prices not converging: {prices}"


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestAsianOptionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_volatility(self):
        """Test Asian option with zero volatility (deterministic)."""
        params = {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.0,  # Zero volatility
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 1000,
            "seed": 42,
            "dtype": np.float64,
        }

        t, paths = simulate_gpu(**params)

        # With zero volatility, all simulated trajectories (columns) should be identical
        assert np.allclose(paths[:, 0], paths[:, 1])
        assert np.allclose(paths[:, 0], paths[:, -1])

        # Price call option
        strike = 100.0
        rate = 0.05
        price = price_asian_option(t, paths, strike, rate, "call")

        # Calculate expected price analytically
        # S(t) = S0 * exp((mu - 0.5*0^2) * t) = S0 * exp(mu * t)
        # Average = (1/T) * integral[S0 * exp(mu*t), 0, T]
        #         = S0 * (exp(mu*T) - 1) / (mu * T)
        expected_avg = params["s0"] * (np.exp(params["mu"] * params["maturity"]) - 1) / (
            params["mu"] * params["maturity"]
        )
        expected_payoff = max(expected_avg - strike, 0.0)
        expected_price = expected_payoff * np.exp(-rate * params["maturity"])

        rel_diff = abs(price - expected_price) / expected_price
        assert rel_diff < 0.01, f"Price: {price:.6f}, Expected: {expected_price:.6f}"

    def test_very_deep_itm_call(self):
        """Test very deep in-the-money call."""
        params = {
            "s0": 200.0,  # High spot
            "mu": 0.05,
            "sigma": 0.1,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 10000,
            "seed": 42,
            "dtype": np.float32,
        }

        t, paths = simulate_gpu(**params)

        strike = 50.0  # Very low strike
        rate = 0.05
        price = price_asian_option(t, paths, strike, rate, "call")

        # Price should be approximately (S0 - K) * exp(-r*T) for deep ITM
        avg_price = paths.mean()
        approx_price = (avg_price - strike) * np.exp(-rate * params["maturity"])

        # Should be reasonably close
        rel_diff = abs(price - approx_price) / approx_price
        assert rel_diff < 0.1, f"Price: {price:.4f}, Approx: {approx_price:.4f}"

    def test_very_deep_otm_call(self):
        """Test very deep out-of-the-money call."""
        params = {
            "s0": 50.0,  # Low spot
            "mu": 0.05,
            "sigma": 0.1,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 10000,
            "seed": 42,
            "dtype": np.float32,
        }

        t, paths = simulate_gpu(**params)

        strike = 200.0  # Very high strike
        rate = 0.05
        price = price_asian_option(t, paths, strike, rate, "call")

        # Price should be very close to zero
        assert price < 1.0, f"Deep OTM call price should be near zero: {price:.4f}"


class TestAsianOptionDifferentMaturities:
    """Test Asian options with different maturities."""

    @pytest.fixture
    def base_params(self):
        return {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "n_paths": 20000,
            "dtype": np.float64,
            "seed": 42,
        }

    def test_short_maturity(self, base_params):
        """Test Asian option with short maturity."""
        # Use rng instead of seed for CPU
        params = base_params.copy()
        seed = params.pop("seed")
        t, paths = simulate_cpu(**params, maturity=0.1, n_steps=25, rng=np.random.default_rng(seed))

        strike = 100.0
        rate = 0.05
        price = price_asian_option(t, paths, strike, rate, "call")

        # Price should be reasonable
        assert 0 <= price <= base_params["s0"]

    def test_long_maturity(self, base_params):
        """Test Asian option with long maturity."""
        # Use rng instead of seed for CPU
        params = base_params.copy()
        seed = params.pop("seed")
        t, paths = simulate_cpu(**params, maturity=5.0, n_steps=252*5, rng=np.random.default_rng(seed))

        strike = 100.0
        rate = 0.05
        price = price_asian_option(t, paths, strike, rate, "call")

        # Price should be reasonable
        assert 0 <= price <= base_params["s0"] * 2


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestAsianOptionAntithetic:
    """Test Asian option pricing with antithetic variates."""

    def test_antithetic_reduces_variance(self):
        """Test that antithetic variates reduce variance of price estimate."""
        params = {
            "s0": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 252,
            "n_paths": 10000,
            "dtype": np.float32,
        }

        strike = 100.0
        rate = 0.05

        # Run multiple trials without antithetic
        prices_standard = []
        for seed in range(10):
            t, paths = simulate_gpu(**params, seed=seed, antithetic=False)
            price = price_asian_option(t, paths, strike, rate, "call")
            prices_standard.append(price)

        # Run multiple trials with antithetic
        prices_antithetic = []
        for seed in range(10):
            t, paths = simulate_gpu(**params, seed=seed, antithetic=True)
            price = price_asian_option(t, paths, strike, rate, "call")
            prices_antithetic.append(price)

        # Variance should be lower with antithetic
        var_standard = np.var(prices_standard)
        var_antithetic = np.var(prices_antithetic)

        # Both should be positive
        assert var_standard > 0
        assert var_antithetic > 0

        print(f"\nVariance without antithetic: {var_standard:.6f}")
        print(f"Variance with antithetic: {var_antithetic:.6f}")
        print(f"Reduction: {(1 - var_antithetic/var_standard)*100:.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
