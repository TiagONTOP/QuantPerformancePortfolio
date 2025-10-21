import pandas as pd
import numpy as np
import exchange_calendars as ec

# =========================
# Paramètres (garde esprit)
# =========================
SAMPLE_SIZE = 3000
N_BACKTEST = 100
SIGMA = 0.5/np.sqrt(252)
MU = 0.01/252
INFORMATION_COEFICIENT = 0.05
assert 0 <= INFORMATION_COEFICIENT < 1, ValueError("Information Coeficient must be in [0, 1[ interval")
SIGNAL_SIGMA_THR_LONG = 1
SIGNAL_SIGMA_THR_SHORT = 1
TRANSACTION_COST_RATE = 0.0001      # coût proportionnel (par entrée/sortie)
SIGNAL_SIGMA_WINDOW_SIZE = 100
START_CAPITAL = 1_000_000.0         # capital initial du portefeuille
ANN = 252                            # jours ouvrés/an (Sharpe)

# =========================
# Données synthétiques
# =========================
rng = np.random.default_rng(42)
log_returns = rng.standard_normal(size=(SAMPLE_SIZE, N_BACKTEST)) * SIGMA + MU

# Signal corrélé aux rendements futurs bruts (style de ton brouillon)
IC = float(INFORMATION_COEFICIENT)
eps = rng.standard_normal(size=(SAMPLE_SIZE, N_BACKTEST))
# signal_t = IC * return_{t} + sqrt(1-IC²) * epsilon * sigma
signal_future = IC * log_returns + np.sqrt(1 - IC**2) * eps * SIGMA
# Shift pour que signal_t serve à décider la position appliquée sur return_{t+1}
signal = np.vstack([signal_future[1:], np.zeros((1, N_BACKTEST))])

def col_corr(a, b):
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a*a).sum(axis=0) * (b*b).sum(axis=0))
    den = np.where(den == 0, np.nan, den)
    return num / den

# IC réalisé entre signal_t et return_{t+1}
ic_realized = np.nanmean(col_corr(signal[:-1], log_returns[1:]))
print(f"Target IC={IC:.3f} | Realized IC~{ic_realized:.3f}")

# =========================
# DataFrames + index temps
# =========================
cal = ec.get_calendar("XNAS")  # NASDAQ
# Find first valid session on or after 2020-01-01
start_session = cal.sessions[cal.sessions >= pd.Timestamp("2020-01-01")][0]
# Get available sessions, limited by what the calendar has
available_sessions = cal.sessions[cal.sessions >= start_session]
num_sessions = min(SAMPLE_SIZE, len(available_sessions))
sessions = available_sessions[:num_sessions]
dates = pd.DatetimeIndex(sessions.tz_convert("UTC")).tz_localize(None)
signal_df = pd.DataFrame(signal, columns=[f"signal_{i+1}" for i in range(N_BACKTEST)], index=dates)
log_returns_df = pd.DataFrame(log_returns, columns=[f"log_return_{i+1}" for i in range(N_BACKTEST)], index=dates)
final_df = pd.concat([signal_df, log_returns_df], axis=1)
print(final_df.head())

# ==========================================================
# Backtest (sous-optimal en perf)
# - capital initial réparti 1/N par actif
# - position par actif ∈ {-1,0,+1} (short/cash/long)
# - décision à t, P&L appliqué sur r_{t+1}
# - coûts proportionnels au capital de l'actif, débités à chaque trade
# - pas de rebalancing cross-actifs (indépendant)
# ==========================================================
def suboptimal_backtest_strategy(df: pd.DataFrame, 
                                 signal_sigma_window_size: int = SIGNAL_SIGMA_WINDOW_SIZE,
                                 transaction_cost_rate: float = TRANSACTION_COST_RATE, 
                                 signal_sigma_thr_long: float = SIGNAL_SIGMA_THR_LONG,
                                 signal_sigma_thr_short: float = SIGNAL_SIGMA_THR_SHORT,
                                 start_capital: float = START_CAPITAL):
    assert signal_sigma_thr_long >= signal_sigma_thr_short, ValueError("The long threshold must be >= short threshold")

    # Tri numérico-lexical des colonnes (volontairement verbeux)
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

    # Capital par actif (équipondéré) et positions par actif
    cap = np.full(n_assets, start_capital / n_assets, dtype=float)
    pos = np.zeros(n_assets, dtype=int)   # -1, 0, +1

    # Historique
    daily_ret_list = []
    equity_list = []

    # On démarre quand on a une fenêtre complète ET qu’il reste un t+1 pour réaliser le P&L
    for t in range(signal_sigma_window_size, n_obs - 1):
        total_before = float(cap.sum())

        # Écarts-types sur [t-window, t) (rolling std "passé")
        sigmas = sig.iloc[t - signal_sigma_window_size:t].std()

        # Signal à t (décision pour r_{t+1})
        s_t = sig.iloc[t]
        r_next = rets.iloc[t + 1]

        # Boucle sous-optimale par actif
        for j in range(n_assets):
            unit_sigma = float(sigmas.iloc[j]) if np.isfinite(sigmas.iloc[j]) else 0.0
            if unit_sigma <= 0:
                desired = 0  # si pas de volatilité historique exploitable → cash
            else:
                s_val = float(s_t.iloc[j])
                if s_val < -signal_sigma_thr_short * unit_sigma:
                    desired = -1
                elif s_val >  signal_sigma_thr_long  * unit_sigma:
                    desired = +1
                else:
                    desired = 0

            # Coûts de transaction (proportionnels au capital courant de l'actif)
            # 0 -> ±1 : 1× coût ; ±1 -> 0 : 1× ; +1 <-> -1 : 2× (sortie + entrée)
            change = (pos[j] != desired)
            flip = (pos[j] == 1 and desired == -1) or (pos[j] == -1 and desired == 1)
            cost_mult = (2 if flip else (1 if change else 0))
            cost_amt = transaction_cost_rate * cap[j] * cost_mult

            # P&L sur r_{t+1} avec exposition all-in (capital de début de jour)
            rj = float(r_next.iloc[j])
            pnl_j = cap[j] * (desired * rj)

            # Mise à jour capital + coûts
            cap[j] = cap[j] + pnl_j - cost_amt

            # Mise à jour position
            pos[j] = desired

            # (Optionnel) stop plancher si cap devient quasi nul
            if not np.isfinite(cap[j]) or cap[j] < 0:
                cap[j] = 0.0

        total_after = float(cap.sum())
        equity_list.append(total_after)

        # Rendement quotidien du portefeuille (sur capital avant)
        if total_before > 0:
            daily_ret = (total_after - total_before) / total_before
        else:
            daily_ret = 0.0
        daily_ret_list.append(daily_ret)

    # Séries temporelles alignées sur r_{t+1} (donc index à partir de t=window+1..)
    out_index = df.index[signal_sigma_window_size + 1:]
    strategy_returns = pd.Series(daily_ret_list, index=out_index, name="strategy_return")
    portfolio_equity = pd.Series(equity_list, index=out_index, name="portfolio_equity")
    return strategy_returns, portfolio_equity

strategy_returns, portfolio_equity = suboptimal_backtest_strategy(final_df)

# =========================
# Stats de base
# =========================
# Benchmark "market" = moyenne égal-pondérée des returns initiaux
market_returns = log_returns_df.mean(axis=1).loc[strategy_returns.index]

def sharpe_ratio(r: pd.Series, periods_per_year: int = ANN):
    r = r.dropna()
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan, np.nan
    sr = (mu / sd) * np.sqrt(periods_per_year)
    # t-stat naïf du Sharpe (ign. auto-corr) : Lo (2002) ferait mieux, mais on reste simple ici
    t_sr = sr * np.sqrt(len(r) / periods_per_year)
    return sr, t_sr

sr, t_sr = sharpe_ratio(strategy_returns)

def capm_alpha_beta_tstats(rp: pd.Series, rm: pd.Series):
    df_ = pd.DataFrame({"rp": rp, "rm": rm}).dropna()
    if len(df_) < 3:
        return np.nan, np.nan, np.nan, np.nan
    y = df_["rp"].values.reshape(-1, 1)
    x = np.column_stack([np.ones(len(df_)), df_["rm"].values])  # [const, marché]
    xtx = x.T @ x
    xtx_inv = np.linalg.inv(xtx)
    beta_hat = xtx_inv @ (x.T @ y)
    alpha = float(beta_hat[0, 0])
    beta = float(beta_hat[1, 0])
    y_hat = x @ beta_hat
    resid = y - y_hat
    n = len(df_); k = 2
    sigma2 = float((resid.T @ resid).item() / (n - k))
    cov_beta = xtx_inv * sigma2
    se_alpha = float(np.sqrt(cov_beta[0, 0]))
    se_beta  = float(np.sqrt(cov_beta[1, 1]))
    t_alpha = alpha / se_alpha if se_alpha != 0 else np.nan
    t_beta  = beta  / se_beta  if se_beta  != 0 else np.nan
    return alpha, beta, t_alpha, t_beta

alpha, beta, t_alpha, t_beta = capm_alpha_beta_tstats(strategy_returns, market_returns)

# =========================
# Résumé
# =========================
print("\n==== SUMMARY STATS ====")
print(f"Observations: {len(strategy_returns)}")
print(f"Sharpe (annualisé, rf=0): {sr:.4f} | t(Sharpe): {t_sr:.4f}")
print(f"CAPM alpha (rf=0): {alpha:.6f} | t(alpha): {t_alpha:.4f}")
print(f"CAPM beta : {beta:.6f} | t(beta): {t_beta:.4f}")
print("\nDernières lignes :")
print(pd.concat([strategy_returns.tail(3), portfolio_equity.tail(3)], axis=1))
