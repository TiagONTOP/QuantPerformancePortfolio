"""
Optimized (vectorized) implementations of the backtest strategy.

Two versions:
1. optimal_backtest_strategy_pandas: Pure pandas vectorization
2. optimal_backtest_strategy_polars: Hybrid Polars/NumPy for rolling operations

Both produce EXACTLY the same results as suboptimal/backtest.py::suboptimal_backtest_strategy
with numerical parity (atol=1e-12).
"""
import pandas as pd
import numpy as np
import polars as pl

# Import custom exception from centralized module
import sys
import os
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.exceptions import InvalidParameterError



def optimal_backtest_strategy_pandas(
    df: pd.DataFrame,
    signal_sigma_window_size: int = 100,
    transaction_cost_rate: float = 0.0001,
    signal_sigma_thr_long: float = 1.0,
    signal_sigma_thr_short: float = 1.0,
    start_capital: float = 1_000_000.0
) -> tuple[pd.Series, pd.Series]:
    """
    Vectorized pandas implementation of the backtest strategy.

    SEMANTICS (matching suboptimal reference):
    - Capital per asset initialized as start_capital / n_assets
    - Position per asset ∈ {-1, 0, +1} decided at t using signal_t vs sigma_t
    - sigma_t = rolling std on [t-window, t) → .rolling(window).std().shift(1)
    - Position applied on return_{t+1}
    - Transaction costs:
        * 0 → ±1: 1 × fee
        * ±1 → 0: 1 × fee
        * +1 ↔ -1: 2 × fee (exit + entry)
    - Update: cap_{t+1,j} = cap_{t,j} * (1 + desired_t * r_{t+1} - cost_rate * cost_mult)
    - Output indexed from t = window_size+1 to end (same as reference)

    Parameters
    ----------
    df : pd.DataFrame
        Columns: signal_1, ..., signal_N, log_return_1, ..., log_return_N
        Index: datetime
    signal_sigma_window_size : int
        Rolling window for signal std
    transaction_cost_rate : float
        Transaction cost rate per unit trade
    signal_sigma_thr_long : float
        Long threshold multiplier
    signal_sigma_thr_short : float
        Short threshold multiplier
    start_capital : float
        Initial portfolio capital

    Returns
    -------
    strategy_returns : pd.Series
        Daily portfolio returns, indexed [window_size+1:]
    portfolio_equity : pd.Series
        Portfolio equity curve, indexed [window_size+1:]
    """
    if signal_sigma_thr_long < signal_sigma_thr_short:
        raise InvalidParameterError(
            f"The long threshold ({signal_sigma_thr_long}) must be >= "
            f"short threshold ({signal_sigma_thr_short})"
        )

    # Extract and sort columns numerically
    signal = df.filter(like="signal").sort_index(
        axis=1,
        key=lambda c: c.str.extract(r'(\d+)', expand=False).astype(float).fillna(-1)
    )
    log_return = df.filter(like="log_return").sort_index(
        axis=1,
        key=lambda c: c.str.extract(r'(\d+)', expand=False).astype(float).fillna(-1)
    )

    n_obs = len(df)
    n_assets = signal.shape[1]
    window = signal_sigma_window_size

    # Rolling sigma (excluding current row)
    sigma = signal.rolling(window=window, min_periods=1).std().shift(1)

    # Desired position at t (applied on r_{t+1})
    # If sigma <= 0 or NaN, desired = 0
    thr_long = signal_sigma_thr_long * sigma
    thr_short = -signal_sigma_thr_short * sigma

    # Vectorized position logic with explicit NaN handling (parity with reference loop)
    signal_vals = signal.to_numpy()
    sigma_vals = sigma.to_numpy()
    thr_long_vals = thr_long.to_numpy()
    thr_short_vals = thr_short.to_numpy()

    desired_vals = np.zeros_like(signal_vals, dtype=np.int8)
    valid_sigma = np.isfinite(sigma_vals) & (sigma_vals > 0)
    valid_signal = np.isfinite(signal_vals)

    long_mask = valid_sigma & valid_signal & (signal_vals > thr_long_vals)
    short_mask = valid_sigma & valid_signal & (signal_vals < thr_short_vals)

    desired_vals[long_mask] = 1
    desired_vals[short_mask] = -1

    # Mimic reference: no positions before first valid window or on last row
    desired_vals[:window, :] = 0
    desired_vals[n_obs - 1:, :] = 0

    desired = pd.DataFrame(desired_vals, index=signal.index, columns=log_return.columns, dtype=np.int8)

    # Previous position (for transaction cost calculation)
    pos_prev = desired.shift(1, fill_value=0).astype(np.int8)

    # Transaction cost multiplier
    # change: position changed (0/1)
    # flip: position flipped sign (0/1)
    change = (desired != pos_prev).astype(np.int8)
    flip = ((desired * pos_prev) == -1).astype(np.int8)
    cost_mult = (change + flip).astype(np.float64)  # 0, 1, or 2

    # Future return (applied on current decision)
    r_next = log_return.shift(-1).fillna(0.0)

    # Growth factor per asset per timestep
    # G_t = 1 + desired_t * r_{t+1} - transaction_cost_rate * cost_mult_t
    G = 1.0 + desired.astype(np.float64) * r_next - transaction_cost_rate * cost_mult

    # Valid zone: trades happen at t ∈ [window, n_obs-2], applied on r_{t+1}
    # So G[t] is valid for t ∈ [window, n_obs-2]
    # For t < window or t >= n_obs-1, set G = 1
    valid_start = window
    valid_end = n_obs - 1

    G_safe = G.copy()
    G_safe.iloc[:valid_start] = 1.0
    G_safe.iloc[valid_end:] = 1.0

    # Clip negative growth (defensive)
    G_safe = G_safe.clip(lower=0.0)

    # Cumulative capital per asset
    # At each timestep t, cap[t] = cap[t-1] * G[t]
    # Starting from cap0
    cap0 = start_capital / n_assets
    cap_path = cap0 * G_safe.cumprod(axis=0)

    # Handle floor: if cap becomes 0, it stays 0
    dead = (cap_path <= 0).cummax(axis=0)
    cap_path = cap_path.where(~dead, 0.0)

    # Portfolio equity at each timestep
    equity = cap_path.sum(axis=1)

    # Strategy returns: (equity[t] - equity[t-1]) / equity[t-1]
    equity_prev = equity.shift(1)
    strategy_returns = ((equity - equity_prev) / equity_prev).fillna(0.0)

    # Output: Reference semantics
    # Reference loop: for t in [window, n_obs-2]:
    #   - at iteration t, computes equity_after using r[t+1]
    #   - stores result with index df.index[t+1]
    # In vectorized form:
    #   - equity[window] uses G[window] which depends on r[window+1]
    #   - this should be output with label df.index[window+1]
    # Solution: take equity[window:n_obs-1] and reindex to [window+1:n_obs]
    equity_slice = equity.iloc[window:n_obs-1]
    returns_slice = strategy_returns.iloc[window:n_obs-1]

    out_index = df.index[window + 1:n_obs]
    equity_out = pd.Series(equity_slice.values, index=out_index, name="portfolio_equity")
    strategy_returns_out = pd.Series(returns_slice.values, index=out_index, name="strategy_return")

    return strategy_returns_out, equity_out


def optimal_backtest_strategy_polars(
    df: pd.DataFrame,
    signal_sigma_window_size: int = 100,
    transaction_cost_rate: float = 0.0001,
    signal_sigma_thr_long: float = 1.0,
    signal_sigma_thr_short: float = 1.0,
    start_capital: float = 1_000_000.0
) -> tuple[pd.Series, pd.Series]:
    """
    Hybrid Polars/NumPy implementation optimized for performance.

    Strategy:
    - Use Polars for rolling window operations (Rust-based, faster than Pandas)
    - Use NumPy for cross-asset matrix operations (vectorized)

    This hybrid approach leverages the strengths of each library:
    - Polars: Columnar operations, rolling windows (Rust implementation)
    - NumPy: Dense matrix operations, cumulative products (compiled C/Fortran)

    Pure Polars is actually SLOWER for this use case because:
    1. Too many intermediate allocations with .with_columns()
    2. Polars is designed for SQL-like operations, not matrix math
    3. Creating N columns for N assets is inefficient (better as matrix)

    Returns EXACTLY the same results as optimal_backtest_strategy_pandas.

    Parameters and semantics: see optimal_backtest_strategy_pandas docstring.

    Returns
    -------
    strategy_returns : pd.Series
        Daily portfolio returns (as pandas Series for consistency)
    portfolio_equity : pd.Series
        Portfolio equity curve (as pandas Series for consistency)
    """
    if signal_sigma_thr_long < signal_sigma_thr_short:
        raise InvalidParameterError(
            f"The long threshold ({signal_sigma_thr_long}) must be >= "
            f"short threshold ({signal_sigma_thr_short})"
        )

    # Extract and sort columns numerically
    signal_pd = df.filter(like="signal").sort_index(
        axis=1,
        key=lambda c: c.str.extract(r'(\d+)', expand=False).astype(float).fillna(-1)
    )
    log_return_pd = df.filter(like="log_return").sort_index(
        axis=1,
        key=lambda c: c.str.extract(r'(\d+)', expand=False).astype(float).fillna(-1)
    )

    window = signal_sigma_window_size
    n_obs = len(df)
    n_assets = signal_pd.shape[1]

    # === USE POLARS FOR ROLLING OPERATIONS (its strength) ===
    # Convert to Polars for fast rolling std (Rust implementation)
    signal_pl = pl.from_pandas(signal_pd)

    # Determine correct rolling_std keyword for current Polars version
    rolling_std_kwargs = {"min_samples": 1}
    if signal_pl.width > 0:
        try:
            _ = pl.col(signal_pl.columns[0]).rolling_std(
                window_size=window, **rolling_std_kwargs
            )
        except TypeError:
            rolling_std_kwargs = {"min_periods": 1}

    # Polars' rolling_std is 2-3x faster than Pandas (Rust vs Python/C)
    sigma_exprs = [
        pl.col(col).rolling_std(window_size=window, **rolling_std_kwargs).shift(1).fill_null(0)
        for col in signal_pl.columns
    ]
    sigma_pl = signal_pl.select(sigma_exprs)

    # === CONVERT TO NUMPY FOR MATRIX OPERATIONS (its strength) ===
    # NumPy is optimal for dense matrix operations (vectorized C/Fortran)
    signal_np = signal_pl.to_numpy()
    sigma_np = sigma_pl.to_numpy()
    log_return_np = log_return_pd.values

    # Vectorized position logic
    thr_long = signal_sigma_thr_long * sigma_np
    thr_short = -signal_sigma_thr_short * sigma_np

    desired = np.zeros_like(signal_np, dtype=np.int8)
    valid_sigma_np = np.isfinite(sigma_np) & (sigma_np > 0)
    valid_signal_np = np.isfinite(signal_np)
    long_mask_np = valid_sigma_np & valid_signal_np & (signal_np > thr_long)
    short_mask_np = valid_sigma_np & valid_signal_np & (signal_np < thr_short)
    desired[long_mask_np] = 1
    desired[short_mask_np] = -1

    # Zero positions outside trading horizon (matches reference loop)
    desired[:window, :] = 0
    desired[n_obs - 1:, :] = 0

    # Transaction costs
    pos_prev = np.roll(desired, 1, axis=0)
    pos_prev[0, :] = 0

    change = (desired != pos_prev).astype(np.int8)
    flip = ((desired * pos_prev) == -1).astype(np.int8)
    cost_mult = (change + flip).astype(np.float64)

    # Future returns
    r_next = np.roll(log_return_np, -1, axis=0)
    r_next[-1, :] = 0.0

    # Growth factor
    G = 1.0 + desired.astype(np.float64) * r_next - transaction_cost_rate * cost_mult

    # Valid zone masking
    G[:window, :] = 1.0
    G[n_obs-1:, :] = 1.0
    G = np.maximum(G, 0.0)

    # Cumulative capital (NumPy's cumprod is vectorized)
    cap0 = start_capital / n_assets
    cap_path = cap0 * np.cumprod(G, axis=0)

    # Handle floor
    dead = np.maximum.accumulate(cap_path <= 0, axis=0)
    cap_path[dead] = 0.0

    # Portfolio equity
    equity = cap_path.sum(axis=1)

    # Strategy returns
    equity_prev = np.roll(equity, 1)
    equity_prev[0] = equity[0]
    strategy_returns = np.where(
        equity_prev > 0,
        (equity - equity_prev) / equity_prev,
        0.0
    )

    # Slice output
    equity_slice = equity[window:n_obs-1]
    returns_slice = strategy_returns[window:n_obs-1]

    out_index = df.index[window + 1:n_obs]
    strategy_returns_out = pd.Series(returns_slice, index=out_index, name="strategy_return")
    equity_out = pd.Series(equity_slice, index=out_index, name="portfolio_equity")

    return strategy_returns_out, equity_out
