"""
Test edge cases and robustness of backtest implementations.

Ensures all implementations handle corner cases correctly:
1. Extreme window sizes (window >= n_obs-1)
2. No-trade scenarios (very high thresholds)
3. Single asset portfolios (n_backtest=1)
4. Zero returns (all log_return_* = 0)
5. Column order invariance (permuted columns)
6. NaN handling in signals and returns
7. Invalid parameter validation
8. Output metadata (dtypes, series names, index)
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

# Inject path because folder starts with a digit
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from utils import generate_synthetic_df, parity_assert
from suboptimal.backtest import suboptimal_backtest_strategy
from optimized.backtest import optimal_backtest_strategy_pandas, optimal_backtest_strategy_polars


# ========== Test: Extreme window size ==========

def test_window_too_large_pandas():
    """Test pandas implementation with window >= len(df)-1 (should return empty series)."""
    df = generate_synthetic_df(sample_size=100, n_backtest=5, window=50, seed=42)

    # Window = len(df) or larger → no valid trades
    huge_window = len(df)

    ref_returns, ref_equity = suboptimal_backtest_strategy(df, signal_sigma_window_size=huge_window)
    opt_returns, opt_equity = optimal_backtest_strategy_pandas(df, signal_sigma_window_size=huge_window)

    # Both should return empty series with same index
    assert len(ref_returns) == 0, "Reference should return empty with window >= n_obs"
    assert len(opt_returns) == 0, "Pandas should return empty with window >= n_obs"
    assert ref_returns.index.equals(opt_returns.index), "Empty series indices should match"


def test_window_too_large_polars():
    """Test polars implementation with window >= len(df)-1."""
    pytest.importorskip("polars")

    df = generate_synthetic_df(sample_size=100, n_backtest=5, window=50, seed=42)
    huge_window = len(df)

    ref_returns, ref_equity = suboptimal_backtest_strategy(df, signal_sigma_window_size=huge_window)
    opt_returns, opt_equity = optimal_backtest_strategy_polars(df, signal_sigma_window_size=huge_window)

    assert len(ref_returns) == 0, "Reference should return empty"
    assert len(opt_returns) == 0, "Polars should return empty"
    assert ref_returns.index.equals(opt_returns.index), "Empty series indices should match"


# ========== Test: No-trade scenario ==========

def test_no_trade_high_thresholds_pandas():
    """Test with impossibly high thresholds → no positions taken, returns should be ~0."""
    df = generate_synthetic_df(sample_size=500, n_backtest=10, window=50, seed=42)

    # Set thresholds so high that no signals will trigger
    ref_returns, ref_equity = suboptimal_backtest_strategy(
        df, signal_sigma_thr_long=1e9, signal_sigma_thr_short=1e9
    )
    opt_returns, opt_equity = optimal_backtest_strategy_pandas(
        df, signal_sigma_thr_long=1e9, signal_sigma_thr_short=1e9
    )

    # All returns should be exactly 0 (no positions → no P&L)
    parity_assert(opt_returns, ref_returns, atol=1e-12, label="No-trade returns (pandas)")
    parity_assert(opt_equity, ref_equity, atol=1e-8, label="No-trade equity (pandas)")

    # Equity should be constant (start capital)
    assert np.allclose(ref_equity.values, ref_equity.iloc[0], atol=1e-6), \
        "Equity should be constant with no trades"


def test_no_trade_high_thresholds_polars():
    """Test polars with no-trade scenario."""
    pytest.importorskip("polars")

    df = generate_synthetic_df(sample_size=500, n_backtest=10, window=50, seed=42)

    ref_returns, ref_equity = suboptimal_backtest_strategy(
        df, signal_sigma_thr_long=1e9, signal_sigma_thr_short=1e9
    )
    opt_returns, opt_equity = optimal_backtest_strategy_polars(
        df, signal_sigma_thr_long=1e9, signal_sigma_thr_short=1e9
    )

    parity_assert(opt_returns, ref_returns, atol=1e-12, label="No-trade returns (polars)")
    parity_assert(opt_equity, ref_equity, atol=1e-8, label="No-trade equity (polars)")


# ========== Test: Single asset ==========

def test_single_asset_pandas():
    """Test with only one asset (n_backtest=1)."""
    df = generate_synthetic_df(sample_size=500, n_backtest=1, window=50, seed=42)

    ref_returns, ref_equity = suboptimal_backtest_strategy(df)
    opt_returns, opt_equity = optimal_backtest_strategy_pandas(df)

    parity_assert(opt_returns, ref_returns, atol=1e-12, label="Single asset returns (pandas)")
    parity_assert(opt_equity, ref_equity, atol=1e-8, label="Single asset equity (pandas)")

    # Check metadata
    assert opt_returns.name == "strategy_return", "Returns series should have correct name"
    assert opt_equity.name == "portfolio_equity", "Equity series should have correct name"


def test_single_asset_polars():
    """Test polars with single asset."""
    pytest.importorskip("polars")

    df = generate_synthetic_df(sample_size=500, n_backtest=1, window=50, seed=42)

    ref_returns, ref_equity = suboptimal_backtest_strategy(df)
    opt_returns, opt_equity = optimal_backtest_strategy_polars(df)

    parity_assert(opt_returns, ref_returns, atol=1e-12, label="Single asset returns (polars)")
    parity_assert(opt_equity, ref_equity, atol=1e-8, label="Single asset equity (polars)")


# ========== Test: Zero returns ==========

def test_zero_returns_pandas():
    """Test with all returns set to zero → equity should be constant."""
    df = generate_synthetic_df(sample_size=500, n_backtest=10, window=50, seed=42)

    # Set all log_return columns to 0
    for col in df.columns:
        if col.startswith("log_return_"):
            df[col] = 0.0

    ref_returns, ref_equity = suboptimal_backtest_strategy(df)
    opt_returns, opt_equity = optimal_backtest_strategy_pandas(df)

    parity_assert(opt_returns, ref_returns, atol=1e-12, label="Zero returns (pandas)")
    parity_assert(opt_equity, ref_equity, atol=1e-8, label="Zero returns equity (pandas)")

    # With zero returns and transaction costs, equity should decline slightly or stay flat
    # depending on whether positions are taken
    # Key check: both implementations should match exactly
    assert opt_equity.dtype == np.float64, "Equity dtype should be float64"


def test_zero_returns_polars():
    """Test polars with zero returns."""
    pytest.importorskip("polars")

    df = generate_synthetic_df(sample_size=500, n_backtest=10, window=50, seed=42)

    for col in df.columns:
        if col.startswith("log_return_"):
            df[col] = 0.0

    ref_returns, ref_equity = suboptimal_backtest_strategy(df)
    opt_returns, opt_equity = optimal_backtest_strategy_polars(df)

    parity_assert(opt_returns, ref_returns, atol=1e-12, label="Zero returns (polars)")
    parity_assert(opt_equity, ref_equity, atol=1e-8, label="Zero returns equity (polars)")


# ========== Test: Column permutation invariance ==========

def test_column_order_invariance_pandas():
    """Test that column order doesn't affect results (after numeric sorting)."""
    df = generate_synthetic_df(sample_size=500, n_backtest=10, window=50, seed=42)

    # Compute with original order
    ref_returns, ref_equity = optimal_backtest_strategy_pandas(df)

    # Shuffle columns randomly
    cols = df.columns.tolist()
    np.random.seed(123)
    np.random.shuffle(cols)
    df_shuffled = df[cols]

    # Compute with shuffled order
    shuf_returns, shuf_equity = optimal_backtest_strategy_pandas(df_shuffled)

    # Results should be identical (implementation sorts numerically)
    parity_assert(shuf_returns, ref_returns, atol=1e-12, label="Shuffled columns returns (pandas)")
    parity_assert(shuf_equity, ref_equity, atol=1e-12, label="Shuffled columns equity (pandas)")


def test_column_order_invariance_polars():
    """Test polars column order invariance."""
    pytest.importorskip("polars")

    df = generate_synthetic_df(sample_size=500, n_backtest=10, window=50, seed=42)

    ref_returns, ref_equity = optimal_backtest_strategy_polars(df)

    cols = df.columns.tolist()
    np.random.seed(123)
    np.random.shuffle(cols)
    df_shuffled = df[cols]

    shuf_returns, shuf_equity = optimal_backtest_strategy_polars(df_shuffled)

    parity_assert(shuf_returns, ref_returns, atol=1e-12, label="Shuffled columns returns (polars)")
    parity_assert(shuf_equity, ref_equity, atol=1e-12, label="Shuffled columns equity (polars)")


# ========== Test: NaN handling ==========

def test_nan_in_signals_pandas():
    """Test with NaN values in signal columns → should handle gracefully."""
    df = generate_synthetic_df(sample_size=500, n_backtest=10, window=50, seed=42)

    # Inject some NaN values into signals
    df.loc[df.index[100:110], "signal_1"] = np.nan
    df.loc[df.index[200:205], "signal_5"] = np.nan

    ref_returns, ref_equity = suboptimal_backtest_strategy(df)
    opt_returns, opt_equity = optimal_backtest_strategy_pandas(df)

    # Should still produce valid output with same shape
    assert len(opt_returns) == len(ref_returns), "NaN signals: length mismatch"
    assert opt_returns.index.equals(ref_returns.index), "NaN signals: index mismatch"

    # Numerical parity (NaN handling should be consistent)
    parity_assert(opt_returns, ref_returns, atol=1e-12, label="NaN signals returns (pandas)")
    parity_assert(opt_equity, ref_equity, atol=1e-8, label="NaN signals equity (pandas)")


def test_nan_in_signals_polars():
    """Test polars with NaN in signals."""
    pytest.importorskip("polars")

    df = generate_synthetic_df(sample_size=500, n_backtest=10, window=50, seed=42)

    df.loc[df.index[100:110], "signal_1"] = np.nan
    df.loc[df.index[200:205], "signal_5"] = np.nan

    ref_returns, ref_equity = suboptimal_backtest_strategy(df)
    opt_returns, opt_equity = optimal_backtest_strategy_polars(df)

    parity_assert(opt_returns, ref_returns, atol=1e-12, label="NaN signals returns (polars)")
    parity_assert(opt_equity, ref_equity, atol=1e-8, label="NaN signals equity (polars)")


# ========== Test: Invalid parameters ==========

def test_invalid_threshold_params_suboptimal():
    """Test that invalid threshold parameters raise assertion errors (suboptimal)."""
    df = generate_synthetic_df(sample_size=200, n_backtest=5, window=50, seed=42)

    # signal_sigma_thr_long < signal_sigma_thr_short should fail
    with pytest.raises((AssertionError, ValueError)):
        suboptimal_backtest_strategy(
            df,
            signal_sigma_thr_long=0.5,
            signal_sigma_thr_short=2.0
        )


def test_invalid_threshold_params_pandas():
    """Test pandas implementation rejects invalid threshold params."""
    df = generate_synthetic_df(sample_size=200, n_backtest=5, window=50, seed=42)

    with pytest.raises((AssertionError, ValueError)):
        optimal_backtest_strategy_pandas(
            df,
            signal_sigma_thr_long=0.5,
            signal_sigma_thr_short=2.0
        )


def test_invalid_threshold_params_polars():
    """Test polars implementation rejects invalid threshold params."""
    pytest.importorskip("polars")

    df = generate_synthetic_df(sample_size=200, n_backtest=5, window=50, seed=42)

    with pytest.raises((AssertionError, ValueError)):
        optimal_backtest_strategy_polars(
            df,
            signal_sigma_thr_long=0.5,
            signal_sigma_thr_short=2.0
        )


# ========== Test: Output metadata ==========

def test_output_metadata_pandas():
    """Verify output series have correct names, dtypes, and index properties."""
    df = generate_synthetic_df(sample_size=500, n_backtest=10, window=50, seed=42)

    returns, equity = optimal_backtest_strategy_pandas(df)

    # Check series names
    assert returns.name == "strategy_return", f"Returns name should be 'strategy_return', got '{returns.name}'"
    assert equity.name == "portfolio_equity", f"Equity name should be 'portfolio_equity', got '{equity.name}'"

    # Check dtypes
    assert returns.dtype == np.float64, f"Returns dtype should be float64, got {returns.dtype}"
    assert equity.dtype == np.float64, f"Equity dtype should be float64, got {equity.dtype}"

    # Check index is strictly increasing (monotonic)
    assert returns.index.is_monotonic_increasing, "Returns index should be monotonic increasing"
    assert equity.index.is_monotonic_increasing, "Equity index should be monotonic increasing"

    # Check index alignment
    assert returns.index.equals(equity.index), "Returns and equity should have same index"

    # Check index dtype (should be datetime64)
    assert pd.api.types.is_datetime64_any_dtype(returns.index), "Index should be datetime64"


def test_output_metadata_polars():
    """Verify polars output metadata."""
    pytest.importorskip("polars")

    df = generate_synthetic_df(sample_size=500, n_backtest=10, window=50, seed=42)

    returns, equity = optimal_backtest_strategy_polars(df)

    assert returns.name == "strategy_return", f"Returns name should be 'strategy_return', got '{returns.name}'"
    assert equity.name == "portfolio_equity", f"Equity name should be 'portfolio_equity', got '{equity.name}'"
    assert returns.dtype == np.float64, f"Returns dtype should be float64, got {returns.dtype}"
    assert equity.dtype == np.float64, f"Equity dtype should be float64, got {equity.dtype}"
    assert returns.index.is_monotonic_increasing, "Returns index should be monotonic increasing"
    assert equity.index.is_monotonic_increasing, "Equity index should be monotonic increasing"
    assert returns.index.equals(equity.index), "Returns and equity should have same index"
    assert pd.api.types.is_datetime64_any_dtype(returns.index), "Index should be datetime64"


def test_metadata_parity_all_implementations():
    """Verify all three implementations produce identical metadata."""
    df = generate_synthetic_df(sample_size=500, n_backtest=10, window=50, seed=42)

    ref_returns, ref_equity = suboptimal_backtest_strategy(df)
    pandas_returns, pandas_equity = optimal_backtest_strategy_pandas(df)

    # Names should match
    assert pandas_returns.name == ref_returns.name, "Returns name mismatch: pandas vs suboptimal"
    assert pandas_equity.name == ref_equity.name, "Equity name mismatch: pandas vs suboptimal"

    # Indices should match
    assert pandas_returns.index.equals(ref_returns.index), "Returns index mismatch: pandas vs suboptimal"
    assert pandas_equity.index.equals(ref_equity.index), "Equity index mismatch: pandas vs suboptimal"

    # Check polars if available
    try:
        pytest.importorskip("polars")
        polars_returns, polars_equity = optimal_backtest_strategy_polars(df)

        assert polars_returns.name == ref_returns.name, "Returns name mismatch: polars vs suboptimal"
        assert polars_equity.name == ref_equity.name, "Equity name mismatch: polars vs suboptimal"
        assert polars_returns.index.equals(ref_returns.index), "Returns index mismatch: polars vs suboptimal"
        assert polars_equity.index.equals(ref_equity.index), "Equity index mismatch: polars vs suboptimal"
    except pytest.skip.Exception:
        pass  # Skip polars check if not installed


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "-s"])
