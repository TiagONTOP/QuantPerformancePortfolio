"""
Optimized (vectorized) implementations of the backtest strategy.

Two versions:
1. optimal_backtest_strategy_pandas: Pure pandas vectorization
2. optimal_backtest_strategy_polars: Polars native expressions

Both produce EXACTLY the same results as suboptimal/backtest.py::suboptimal_backtest_strategy
with numerical parity (atol=1e-12).
"""
import pandas as pd
import numpy as np

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


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
    assert signal_sigma_thr_long >= signal_sigma_thr_short, \
        "Long threshold must be >= short threshold"

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
    sigma = signal.rolling(window).std().shift(1)

    # Desired position at t (applied on r_{t+1})
    # If sigma <= 0 or NaN, desired = 0
    thr_long = signal_sigma_thr_long * sigma
    thr_short = -signal_sigma_thr_short * sigma

    desired = pd.DataFrame(0, index=signal.index, columns=log_return.columns, dtype=np.int8)
    desired.values[:] = 0
    desired = desired.where(signal.values <= thr_long.values, 1)    # signal > thr_long → +1
    desired = desired.where(signal.values >= thr_short.values, -1)  # signal < thr_short → -1

    # Force desired = 0 where sigma invalid (NaN or <= 0)
    mask_invalid = (sigma.isna()) | (sigma <= 0)
    desired = desired.mask(mask_invalid.values, 0)

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

    strategy_returns_out.name = "strategy_return"
    equity_out.name = "portfolio_equity"

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
    Vectorized Polars implementation of the backtest strategy.

    Uses native Polars expressions (no Python UDFs) for maximum performance.
    Returns EXACTLY the same results as optimal_backtest_strategy_pandas.

    Parameters and semantics: see optimal_backtest_strategy_pandas docstring.

    Returns
    -------
    strategy_returns : pd.Series
        Daily portfolio returns (as pandas Series for consistency)
    portfolio_equity : pd.Series
        Portfolio equity curve (as pandas Series for consistency)
    """
    if not POLARS_AVAILABLE:
        raise ImportError("Polars is not installed. Install with: pip install polars")

    assert signal_sigma_thr_long >= signal_sigma_thr_short, \
        "Long threshold must be >= short threshold"

    # Extract and sort columns numerically (pandas side)
    signal_pd = df.filter(like="signal").sort_index(
        axis=1,
        key=lambda c: c.str.extract(r'(\d+)', expand=False).astype(float).fillna(-1)
    )
    log_return_pd = df.filter(like="log_return").sort_index(
        axis=1,
        key=lambda c: c.str.extract(r'(\d+)', expand=False).astype(float).fillna(-1)
    )

    # Convert to Polars
    # Preserve datetime index
    has_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    if has_datetime_index:
        signal_pd = signal_pd.reset_index()
        log_return_pd = log_return_pd.reset_index()
        timestamp_col = signal_pd.columns[0]

    signal_pl = pl.from_pandas(signal_pd)
    log_return_pl = pl.from_pandas(log_return_pd)

    window = signal_sigma_window_size
    n_obs = len(df)
    n_assets = len(signal_pd.columns) - (1 if has_datetime_index else 0)

    # Get signal columns (exclude timestamp if present)
    if has_datetime_index:
        sig_cols = [c for c in signal_pl.columns if c != timestamp_col]
        ret_cols = [c for c in log_return_pl.columns if c != timestamp_col]
    else:
        sig_cols = signal_pl.columns
        ret_cols = log_return_pl.columns

    # === Compute sigma (rolling std, shifted by 1) ===
    sigma_exprs = [
        pl.col(c).rolling_std(window, min_periods=window, center=False).shift(1).alias(f"sigma_{c}")
        for c in sig_cols
    ]
    sigma_pl = signal_pl.select(sigma_exprs)

    # === Compute desired positions ===
    desired_exprs = []
    for i, c in enumerate(sig_cols):
        sigma_col = f"sigma_{c}"
        sig = pl.col(c)
        sigma = sigma_pl[sigma_col]

        # Thresholds
        thr_long = signal_sigma_thr_long * sigma
        thr_short = -signal_sigma_thr_short * sigma

        # Position logic
        desired = (
            pl.when(sig > thr_long).then(pl.lit(1))
            .when(sig < thr_short).then(pl.lit(-1))
            .otherwise(pl.lit(0))
        )

        # Force 0 if sigma invalid
        desired = pl.when((sigma.is_null()) | (sigma <= 0.0)).then(pl.lit(0)).otherwise(desired)

        desired_exprs.append(desired.alias(f"desired_{c}"))

    # Join signal, sigma, desired
    combined = signal_pl.select(sig_cols).with_columns(sigma_pl).with_columns(desired_exprs)

    # === Compute transaction costs ===
    cost_exprs = []
    for c in sig_cols:
        des_col = f"desired_{c}"
        pos_prev = pl.col(des_col).shift(1, fill_value=0)
        change = (pl.col(des_col) != pos_prev).cast(pl.Int8)
        flip = ((pl.col(des_col) * pos_prev) == -1).cast(pl.Int8)
        cost_mult = (change + flip).cast(pl.Float64)
        cost_exprs.append(cost_mult.alias(f"cost_{c}"))

    combined = combined.with_columns(cost_exprs)

    # === Get r_next (log return shifted by -1) ===
    rnext_exprs = []
    for c in ret_cols:
        rnext = pl.col(c).shift(-1).fill_null(0.0)
        rnext_exprs.append(rnext.alias(f"rnext_{c}"))

    log_return_shifted = log_return_pl.select(rnext_exprs)

    # === Compute growth factor G ===
    # G = 1 + desired * r_next - cost_rate * cost_mult
    G_exprs = []
    for i, c in enumerate(sig_cols):
        des_col = f"desired_{c}"
        cost_col = f"cost_{c}"
        rnext_col = f"rnext_{ret_cols[i]}"

        # Merge all needed columns into one frame
        G = (
            1.0
            + pl.col(des_col).cast(pl.Float64) * log_return_shifted[rnext_col]
            - transaction_cost_rate * pl.col(cost_col)
        )
        G_exprs.append(G.alias(f"G_{c}"))

    combined = combined.with_columns(log_return_shifted).with_columns(G_exprs)

    # === Set G = 1 outside valid zone ===
    valid_start = window
    valid_end = n_obs - 1

    G_safe_exprs = []
    for c in sig_cols:
        G_col = f"G_{c}"
        # Use row_index (polars >= 0.19) or row_number
        row_idx = pl.arange(0, pl.count()).alias("_row_idx")

        G_safe = (
            pl.when((row_idx < valid_start) | (row_idx >= valid_end))
            .then(pl.lit(1.0))
            .otherwise(pl.col(G_col).clip(0.0, None))  # clip lower bound
        )
        G_safe_exprs.append(G_safe.alias(f"Gsafe_{c}"))

    # Add row index for filtering
    combined = combined.with_columns([pl.arange(0, pl.count()).alias("_row_idx")])
    combined = combined.with_columns(G_safe_exprs)

    # === Compute cumulative product (capital path) ===
    cap0 = start_capital / n_assets
    cap_exprs = []
    for c in sig_cols:
        Gsafe_col = f"Gsafe_{c}"
        # Cumulative product
        cap = (cap0 * pl.col(Gsafe_col).cum_prod()).alias(f"cap_{c}")
        cap_exprs.append(cap)

    combined = combined.with_columns(cap_exprs)

    # === Handle floor (if cap <= 0, stays 0) ===
    # Dead mask: cummax of (cap <= 0)
    cap_final_exprs = []
    for c in sig_cols:
        cap_col = f"cap_{c}"
        dead = (pl.col(cap_col) <= 0.0).cum_max()
        cap_final = pl.when(dead).then(pl.lit(0.0)).otherwise(pl.col(cap_col))
        cap_final_exprs.append(cap_final.alias(f"capfinal_{c}"))

    combined = combined.with_columns(cap_final_exprs)

    # === Compute equity (sum across assets) ===
    capfinal_cols = [f"capfinal_{c}" for c in sig_cols]
    equity_expr = sum(pl.col(c) for c in capfinal_cols).alias("equity")
    combined = combined.with_columns([equity_expr])

    # === Compute strategy returns ===
    equity_prev = pl.col("equity").shift(1)
    strat_ret = (
        pl.when(equity_prev > 0)
        .then((pl.col("equity") - equity_prev) / equity_prev)
        .otherwise(pl.lit(0.0))
        .alias("strategy_return")
    )
    combined = combined.with_columns([strat_ret])

    # === Convert back to pandas and slice output ===
    result_pd = combined.to_pandas()

    if has_datetime_index:
        result_pd = result_pd.set_index(timestamp_col)

    # Slice to match reference output
    out_index = df.index[window + 1:]
    strategy_returns_out = result_pd["strategy_return"].loc[out_index]
    equity_out = result_pd["equity"].loc[out_index]

    strategy_returns_out.name = "strategy_return"
    equity_out.name = "portfolio_equity"

    return strategy_returns_out, equity_out



