"""
Utilities for data generation and metrics computation.
Reproduces EXACTLY the logic from suboptimal/backtest.py.
"""
import pandas as pd
import numpy as np
import exchange_calendars as ec


def numeric_sort_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Sort columns with given prefix numerically (extract numbers from column names).
    E.g., signal_1, signal_2, ..., signal_100 -> sorted properly by numeric ID.
    """
    cols = df.filter(like=prefix)
    sorted_cols = cols.sort_index(
        axis=1,
        key=lambda c: c.str.extract(r'(\d+)', expand=False).astype(float).fillna(-1)
    )
    return sorted_cols


def generate_synthetic_df(
    sample_size: int = 3000,
    n_backtest: int = 100,
    ann: int = 252,
    sigma: float = None,
    mu: float = None,
    ic: float = 0.05,
    window: int = 100,
    seed: int = 43
) -> pd.DataFrame:
    """
    Generate synthetic data EXACTLY as in suboptimal/backtest.py.

    Returns a DataFrame with:
      - columns: signal_1, signal_2, ..., signal_N, log_return_1, log_return_2, ..., log_return_N
      - index: NASDAQ calendar dates (tz-naive)

    The signal_t is constructed to be correlated with return_{t+1}, then shifted back.
    """
    if sigma is None:
        sigma = 0.3 / np.sqrt(ann)
    if mu is None:
        mu = 0.0 / ann

    assert 0 <= ic < 1, "Information Coefficient must be in [0, 1)"

    # Build calendar index
    cal = ec.get_calendar("XNAS")
    start_session = cal.sessions[cal.sessions >= pd.Timestamp("2000-01-01")][0]
    available_sessions = cal.sessions[cal.sessions >= start_session]

    # We need sample_size + window + 1 sessions for proper backtest
    sessions_needed = sample_size + window + 1
    num_sessions = min(sessions_needed, len(available_sessions))
    sessions = available_sessions[:num_sessions]

    # Convert to naive datetime
    dates = sessions.tz_localize(None) if sessions.tz is None else sessions.tz_convert("UTC").tz_localize(None)

    actual_sample_size = len(dates)

    # Generate synthetic log returns
    rng = np.random.default_rng(seed)
    log_returns = rng.standard_normal(size=(actual_sample_size, n_backtest)) * sigma + mu

    # Signal correlated with future returns
    IC = float(ic)
    eps = rng.standard_normal(size=(actual_sample_size, n_backtest))
    # signal_future[t] correlates with return[t]
    signal_future = IC * log_returns + np.sqrt(1 - IC**2) * eps * sigma

    # Shift so that signal_t is used to decide position applied on return_{t+1}
    # signal[t] = signal_future[t+1], and signal[-1] = 0
    signal = np.vstack([signal_future[1:], np.zeros((1, n_backtest))])

    # Build DataFrames
    signal_df = pd.DataFrame(
        signal,
        columns=[f"signal_{i+1}" for i in range(n_backtest)],
        index=dates
    )
    log_returns_df = pd.DataFrame(
        log_returns,
        columns=[f"log_return_{i+1}" for i in range(n_backtest)],
        index=dates
    )

    final_df = pd.concat([signal_df, log_returns_df], axis=1)
    return final_df


def safe_rolling_sigma(signal: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling std on signal columns, excluding current row (shift by 1).
    Returns sigma with same shape as signal.
    """
    return signal.rolling(window).std().shift(1)


def sharpe_ratio(r: pd.Series, periods_per_year: int = 252) -> tuple[float, float]:
    """
    Compute annualized Sharpe ratio and its naive t-statistic.
    Returns (sharpe_ratio, t_stat).
    """
    r = r.dropna()
    if len(r) == 0:
        return np.nan, np.nan

    mu = r.mean()
    sd = r.std(ddof=1)

    if sd == 0 or not np.isfinite(sd):
        return np.nan, np.nan

    sr = (mu / sd) * np.sqrt(periods_per_year)
    # Naive t-stat (ignoring autocorrelation)
    t_sr = sr * np.sqrt(len(r) / periods_per_year)

    return sr, t_sr


def capm_alpha_beta_tstats(rp: pd.Series, rm: pd.Series) -> tuple[float, float, float, float]:
    """
    Compute CAPM alpha, beta and their t-statistics via OLS.
    rp = portfolio returns, rm = market returns.

    Returns (alpha, beta, t_alpha, t_beta).
    """
    df_ = pd.DataFrame({"rp": rp, "rm": rm}).dropna()

    if len(df_) < 3:
        return np.nan, np.nan, np.nan, np.nan

    y = df_["rp"].values.reshape(-1, 1)
    x = np.column_stack([np.ones(len(df_)), df_["rm"].values])  # [const, market]

    xtx = x.T @ x
    xtx_inv = np.linalg.inv(xtx)
    beta_hat = xtx_inv @ (x.T @ y)

    alpha = float(beta_hat[0, 0])
    beta = float(beta_hat[1, 0])

    # Residuals and variance
    y_hat = x @ beta_hat
    resid = y - y_hat
    n = len(df_)
    k = 2
    sigma2 = float((resid.T @ resid).item() / (n - k))

    # Covariance matrix of beta_hat
    cov_beta = xtx_inv * sigma2
    se_alpha = float(np.sqrt(cov_beta[0, 0]))
    se_beta = float(np.sqrt(cov_beta[1, 1]))

    t_alpha = alpha / se_alpha if se_alpha != 0 else np.nan
    t_beta = beta / se_beta if se_beta != 0 else np.nan

    return alpha, beta, t_alpha, t_beta


def parity_assert(a: pd.Series, b: pd.Series, atol: float = 1e-12, label: str = "", rtol: float = 0.0):
    """
    Assert that two Series are numerically equal within tolerance.
    Raises AssertionError if not.

    Parameters
    ----------
    a, b : pd.Series
        Series to compare
    atol : float
        Absolute tolerance
    rtol : float
        Relative tolerance (not used if 0)
    label : str
        Label for error message
    """
    import numpy as np

    # Check index equality
    assert a.index.equals(b.index), f"{label}: Index mismatch"

    # Check shape
    assert a.shape == b.shape, f"{label}: Shape mismatch {a.shape} vs {b.shape}"

    # Check values
    if rtol > 0:
        # Use both absolute and relative tolerance
        assert np.allclose(a.values, b.values, atol=atol, rtol=rtol), \
            f"{label}: Values differ beyond tolerance (atol={atol}, rtol={rtol})"
    else:
        # Use only absolute tolerance
        diff = np.abs(a.values - b.values)
        max_diff = np.max(diff)
        assert max_diff <= atol, \
            f"{label}: Max difference {max_diff:.2e} exceeds tolerance {atol:.2e}"
