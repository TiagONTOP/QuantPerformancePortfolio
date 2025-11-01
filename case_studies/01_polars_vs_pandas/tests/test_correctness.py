"""
Test numerical parity between suboptimal and optimized implementations.

Ensures that:
1. optimal_backtest_strategy_pandas produces EXACTLY the same results as suboptimal
2. optimal_backtest_strategy_polars produces EXACTLY the same results as suboptimal

Tested across multiple seeds and data sizes.
"""
import os
import sys
import pytest
import numpy as np

# Inject path because folder starts with a digit
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from tools.utils import generate_synthetic_df, parity_assert
from suboptimal.backtest import suboptimal_backtest_strategy
from optimized.backtest import optimal_backtest_strategy_pandas, optimal_backtest_strategy_polars


# Test configurations
SMALL_CONFIG = {"sample_size": 500, "n_backtest": 10, "window": 50}
MEDIUM_CONFIG = {"sample_size": 1500, "n_backtest": 50, "window": 100}
LARGE_CONFIG = {"sample_size": 3000, "n_backtest": 100, "window": 100}

SEEDS = [42, 43, 100]


@pytest.mark.parametrize("seed", SEEDS)
def test_pandas_vs_suboptimal_small(seed):
    """Test pandas vectorization vs suboptimal on small dataset."""
    df = generate_synthetic_df(**SMALL_CONFIG, seed=seed)

    ref_returns, ref_equity = suboptimal_backtest_strategy(df)
    opt_returns, opt_equity = optimal_backtest_strategy_pandas(df)

    parity_assert(opt_returns, ref_returns, atol=1e-12, label=f"Pandas returns (seed={seed}, small)")
    parity_assert(opt_equity, ref_equity, atol=1e-8, label=f"Pandas equity (seed={seed}, small)")


@pytest.mark.parametrize("seed", SEEDS)
def test_pandas_vs_suboptimal_medium(seed):
    """Test pandas vectorization vs suboptimal on medium dataset."""
    df = generate_synthetic_df(**MEDIUM_CONFIG, seed=seed)

    ref_returns, ref_equity = suboptimal_backtest_strategy(df)
    opt_returns, opt_equity = optimal_backtest_strategy_pandas(df)

    parity_assert(opt_returns, ref_returns, atol=1e-12, label=f"Pandas returns (seed={seed}, medium)")
    parity_assert(opt_equity, ref_equity, atol=2e-8, label=f"Pandas equity (seed={seed}, medium)")


@pytest.mark.parametrize("seed", SEEDS)
def test_pandas_vs_suboptimal_large(seed):
    """Test pandas vectorization vs suboptimal on large dataset."""
    df = generate_synthetic_df(**LARGE_CONFIG, seed=seed)

    ref_returns, ref_equity = suboptimal_backtest_strategy(df)
    opt_returns, opt_equity = optimal_backtest_strategy_pandas(df)

    parity_assert(opt_returns, ref_returns, atol=1e-12, label=f"Pandas returns (seed={seed}, large)")
    parity_assert(opt_equity, ref_equity, atol=6e-8, label=f"Pandas equity (seed={seed}, large)")


@pytest.mark.parametrize("seed", SEEDS)
def test_polars_vs_suboptimal_small(seed):
    """Test polars vectorization vs suboptimal on small dataset."""
    pytest.importorskip("polars")  # Skip if polars not installed

    df = generate_synthetic_df(**SMALL_CONFIG, seed=seed)

    ref_returns, ref_equity = suboptimal_backtest_strategy(df)
    opt_returns, opt_equity = optimal_backtest_strategy_polars(df)

    parity_assert(opt_returns, ref_returns, atol=1e-12, label=f"Polars returns (seed={seed}, small)")
    parity_assert(opt_equity, ref_equity, atol=2e-8, label=f"Polars equity (seed={seed}, small)")


@pytest.mark.parametrize("seed", SEEDS)
def test_polars_vs_suboptimal_medium(seed):
    """Test polars vectorization vs suboptimal on medium dataset."""
    pytest.importorskip("polars")

    df = generate_synthetic_df(**MEDIUM_CONFIG, seed=seed)

    ref_returns, ref_equity = suboptimal_backtest_strategy(df)
    opt_returns, opt_equity = optimal_backtest_strategy_polars(df)

    parity_assert(opt_returns, ref_returns, atol=1e-12, label=f"Polars returns (seed={seed}, medium)")
    parity_assert(opt_equity, ref_equity, atol=2e-8, label=f"Polars equity (seed={seed}, medium)")


@pytest.mark.parametrize("seed", SEEDS)
def test_polars_vs_suboptimal_large(seed):
    """Test polars vectorization vs suboptimal on large dataset."""
    pytest.importorskip("polars")

    df = generate_synthetic_df(**LARGE_CONFIG, seed=seed)

    ref_returns, ref_equity = suboptimal_backtest_strategy(df)
    opt_returns, opt_equity = optimal_backtest_strategy_polars(df)

    parity_assert(opt_returns, ref_returns, atol=1e-12, label=f"Polars returns (seed={seed}, large)")
    parity_assert(opt_equity, ref_equity, atol=6e-8, label=f"Polars equity (seed={seed}, large)")


def test_pandas_polars_parity():
    """Test that pandas and polars implementations produce identical results."""
    pytest.importorskip("polars")

    df = generate_synthetic_df(**MEDIUM_CONFIG, seed=42)

    pandas_returns, pandas_equity = optimal_backtest_strategy_pandas(df)
    polars_returns, polars_equity = optimal_backtest_strategy_polars(df)

    parity_assert(polars_returns, pandas_returns, atol=1e-12, label="Polars vs Pandas returns")
    parity_assert(polars_equity, pandas_equity, atol=2e-8, label="Polars vs Pandas equity")


if __name__ == "__main__":
    # Run all tests when executed directly
    pytest.main([__file__, "-v", "-s"])
