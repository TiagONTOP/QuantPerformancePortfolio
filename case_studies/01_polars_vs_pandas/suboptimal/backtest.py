import logging

import exchange_calendars as ec
import numpy as np
import pandas as pd

# Import custom exception and utilities from centralized modules
import sys
import os
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.exceptions import InvalidParameterError
from tools.utils import sharpe_ratio, capm_alpha_beta_tstats

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =========================
# Parameters
# =========================
SAMPLE_SIZE = 3000
N_BACKTEST = 100
ANN = 252
SIGMA = 0.3 / np.sqrt(ANN)
MU = 0.0 / ANN
INFORMATION_COEFFICIENT = 0.05

if not (0 <= INFORMATION_COEFFICIENT < 1):
    raise InvalidParameterError("Information Coefficient must be in [0, 1) interval")
SIGNAL_SIGMA_THR_LONG = 1
SIGNAL_SIGMA_THR_SHORT = 1
TRANSACTION_COST_RATE = 0.0001      # proportional cost (per entry/exit)
SIGNAL_SIGMA_WINDOW_SIZE = 100
START_CAPITAL = 1_000_000.0         # initial portfolio capital
SEED = 43


# ==========================================================
# Backtest (suboptimal for performance)
# - initial capital split 1/N per asset
# - position per asset ∈ {-1,0,+1} (short/cash/long)
# - decision at t, P&L applied on r_{t+1}
# - proportional costs to asset capital, debited at each trade
# - no cross-asset rebalancing (independent)
# ==========================================================
def suboptimal_backtest_strategy(
    df: pd.DataFrame,
    signal_sigma_window_size: int = SIGNAL_SIGMA_WINDOW_SIZE,
    transaction_cost_rate: float = TRANSACTION_COST_RATE,
    signal_sigma_thr_long: float = SIGNAL_SIGMA_THR_LONG,
    signal_sigma_thr_short: float = SIGNAL_SIGMA_THR_SHORT,
    start_capital: float = START_CAPITAL,
):
    if signal_sigma_thr_long < signal_sigma_thr_short:
        raise InvalidParameterError(
            f"The long threshold ({signal_sigma_thr_long}) must be >= "
            f"short threshold ({signal_sigma_thr_short})"
        )

    # Numeric-lexical column sorting (intentionally verbose)
    sig = df.filter(like="signal").sort_index(
        axis=1,
        key=lambda c: c.str.extract(r'(\d+)', expand=False).astype(float).fillna(-1)
    )
    rets = df.filter(like="log_return").sort_index(
        axis=1,
        key=lambda c: c.str.extract(r'(\d+)', expand=False).astype(float).fillna(-1)
    )

    n_obs = len(df)
    n_assets = sig.shape[1]

    # Capital per asset (equal-weighted) and positions per asset
    cap = np.full(n_assets, start_capital / n_assets, dtype=float)
    pos = np.zeros(n_assets, dtype=int)   # -1, 0, +1

    # History tracking
    daily_ret_list = []
    equity_list = []

    # Start when we have a complete window AND there's a t+1 remaining to realize P&L
    for t in range(signal_sigma_window_size, n_obs - 1):
        total_before = float(cap.sum())

        # Standard deviations on [t-window, t) (rolling std "past")
        sigmas = sig.iloc[t - signal_sigma_window_size:t].std()

        # Signal at t (decision for r_{t+1})
        s_t = sig.iloc[t]
        r_next = rets.iloc[t + 1]

        # Suboptimal loop per asset
        for j in range(n_assets):
            unit_sigma = float(sigmas.iloc[j]) if np.isfinite(sigmas.iloc[j]) else 0.0
            if unit_sigma <= 0:
                desired = 0  # if no usable historical volatility → cash
            else:
                s_val = float(s_t.iloc[j])
                if s_val < -signal_sigma_thr_short * unit_sigma:
                    desired = -1
                elif s_val >  signal_sigma_thr_long  * unit_sigma:
                    desired = +1
                else:
                    desired = 0

            # Transaction costs (proportional to current asset capital)
            # 0 -> ±1 : 1× cost ; ±1 -> 0 : 1× ; +1 <-> -1 : 2× (exit + entry)
            change = (pos[j] != desired)
            flip = (pos[j] == 1 and desired == -1) or (pos[j] == -1 and desired == 1)
            cost_mult = (2 if flip else (1 if change else 0))
            cost_amt = transaction_cost_rate * cap[j] * cost_mult

            # P&L on r_{t+1} with all-in exposure (beginning-of-day capital)
            rj = float(r_next.iloc[j])
            pnl_j = cap[j] * (desired * rj)

            # Update capital + costs
            cap[j] = cap[j] + pnl_j - cost_amt

            # Update position
            pos[j] = desired

            # (Optional) floor stop if cap becomes nearly zero
            if not np.isfinite(cap[j]) or cap[j] < 0:
                cap[j] = 0.0

        total_after = float(cap.sum())
        equity_list.append(total_after)

        # Daily portfolio return (on capital before)
        if total_before > 0:
            daily_ret = (total_after - total_before) / total_before
        else:
            daily_ret = 0.0
        daily_ret_list.append(daily_ret)

    # Time series aligned on r_{t+1} (so index starting from t=window+1..)
    out_index = df.index[signal_sigma_window_size + 1:]
    strategy_returns = pd.Series(daily_ret_list, index=out_index, name="strategy_return")
    portfolio_equity = pd.Series(equity_list, index=out_index, name="portfolio_equity")
    return strategy_returns, portfolio_equity


if __name__ == "__main__":
    # =========================
    # DataFrames + time index
    # =========================
    cal = ec.get_calendar("XNAS")  # NASDAQ
    # Find first valid session on or after 2020-01-01
    start_session = cal.sessions[cal.sessions >= pd.Timestamp("2000-01-01")][0]
    # Get available sessions, limited by what the calendar has
    available_sessions = cal.sessions[cal.sessions >= start_session]

    # We need extra sessions for the signal window + 1 for the future return
    # So if we want SAMPLE_SIZE final observations, we need SAMPLE_SIZE + SIGNAL_SIGMA_WINDOW_SIZE + 1 sessions
    sessions_needed = SAMPLE_SIZE + SIGNAL_SIGMA_WINDOW_SIZE + 1
    num_sessions = min(sessions_needed, len(available_sessions))
    sessions = available_sessions[:num_sessions]
    # sessions is already UTC-aware, convert to naive
    dates = sessions.tz_localize(None) if sessions.tz is None else sessions.tz_convert("UTC").tz_localize(None)

    # Use actual number of sessions for data generation
    actual_sample_size = len(dates)

    # =========================
    # Synthetic data
    # =========================
    rng = np.random.default_rng(SEED)
    log_returns = rng.standard_normal(size=(actual_sample_size, N_BACKTEST)) * SIGMA + MU

    # Signal correlated to future raw returns
    IC = float(INFORMATION_COEFFICIENT)
    eps = rng.standard_normal(size=(actual_sample_size, N_BACKTEST))
    # signal_t = IC * return_{t} + sqrt(1-IC²) * epsilon * sigma
    signal_future = IC * log_returns + np.sqrt(1 - IC**2) * eps * SIGMA
    # Shift so that signal_t is used to decide position applied on return_{t+1}
    signal = np.vstack([signal_future[1:], np.zeros((1, N_BACKTEST))])

    def col_corr(a, b):
        a = a - a.mean(axis=0, keepdims=True)
        b = b - b.mean(axis=0, keepdims=True)
        num = (a * b).sum(axis=0)
        den = np.sqrt((a*a).sum(axis=0) * (b*b).sum(axis=0))
        den = np.where(den == 0, np.nan, den)
        return num / den

    # Realized IC between signal_t and return_{t+1}
    ic_realized = np.nanmean(col_corr(signal[:-1], log_returns[1:]))
    logger.info(f"Target IC={IC:.3f} | Realized IC~{ic_realized:.3f}")
    signal_df = pd.DataFrame(
        signal, columns=[f"signal_{i+1}" for i in range(N_BACKTEST)], index=dates
    )
    log_returns_df = pd.DataFrame(
        log_returns, columns=[f"log_return_{i+1}" for i in range(N_BACKTEST)], index=dates
    )
    final_df = pd.concat([signal_df, log_returns_df], axis=1)
    logger.info(f"Generated data shape: {final_df.shape}")
    logger.debug(f"Data sample:\n{final_df.head()}")

    # Only run demo when executed directly, not on import
    strategy_returns, portfolio_equity = suboptimal_backtest_strategy(final_df)

    # =========================
    # Basic stats
    # =========================
    # Benchmark "market" = equal-weighted average of initial returns
    market_returns = log_returns_df.mean(axis=1).loc[strategy_returns.index]

    # Use centralized metric functions from tools/utils.py
    sr, t_sr = sharpe_ratio(strategy_returns, periods_per_year=ANN)
    alpha, beta, t_alpha, t_beta = capm_alpha_beta_tstats(strategy_returns, market_returns)

    # =========================
    # Summary
    # =========================
    logger.info("\n==== SUMMARY STATS ====")
    logger.info(f"Observations: {len(strategy_returns)}")
    logger.info(f"Sharpe (annualized, rf=0): {sr:.4f} | t(Sharpe): {t_sr:.4f}")
    logger.info(f"CAPM alpha (rf=0): {alpha:.6f} | t(alpha): {t_alpha:.4f}")
    logger.info(f"CAPM beta : {beta:.6f} | t(beta): {t_beta:.4f}")
    logger.info("\nLast few rows:")
    logger.info(f"\n{pd.concat([strategy_returns.tail(3), portfolio_equity.tail(3)], axis=1)}")
