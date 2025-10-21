import pandas as pd
import numpy as np

SAMPLE_SIZE = 3000
N_BACKTEST = 100
SIGMA = 0.5
MU = 0.01
INFORMATION_COEFICIENT = 0.8
assert INFORMATION_COEFICIENT >= 0 and INFORMATION_COEFICIENT < 1, ValueError("Information Coeficient must be in [0, 1[ interval")
SIGNAL_SIGMA_THR_LONG = 1
SIGNAL_SIGMA_THR_SHORT = 1
TRANSACTION_COST = 0.0001
SIGNAL_SIGMA_WINDOW_SIZE = 100

rng = np.random.default_rng(42)
log_returns = rng.standard_normal(size=(SAMPLE_SIZE, N_BACKTEST)) * SIGMA + MU

# Génération du signal corrélé aux rendements futurs bruts
IC = float(INFORMATION_COEFICIENT)
eps = rng.standard_normal(size=(SAMPLE_SIZE, N_BACKTEST))
# signal_t = IC * return_{t+1} + sqrt(1-IC²) * epsilon * sigma
# Shift pour avoir signal_t aligné avec return_t (et prédire return_{t+1})
signal_future = IC * log_returns + np.sqrt(1 - IC**2) * eps * SIGMA
signal = np.vstack([signal_future[1:], np.zeros((1, N_BACKTEST))])  # Shift: signal_t prédit return_{t+1}

def col_corr(a, b):
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a*a).sum(axis=0) * (b*b).sum(axis=0))
    return num / np.where(den == 0, np.nan, den)

# Corrélation entre signal_t et return_{t+1}
ic_realized = np.nanmean(col_corr(signal[:-1], log_returns[1:]))
print(f"Target IC={IC:.3f} | Realized IC~{ic_realized:.3f}")

signal = pd.DataFrame(signal, columns=["signal_{}".format(c) for c in range(1, len(signal.columns)+1)])
log_returns = pd.DataFrame(log_returns, columns=["log_return_{}".format(c) for c in range(1, len(log_returns.columns)+1)])
final_df = pd.DataFrame(np.concatenate([signal, log_returns], axis=1))

print(final_df.head())

def suboptimal_backtest_strategy(df: pd.DataFrame, 
                                 signal_sigma_window_size: int = SIGNAL_SIGMA_WINDOW_SIZE,
                                 transaction_cost: float = TRANSACTION_COST, 
                                 signal_sigma_thr_long: float = SIGNAL_SIGMA_THR_LONG,
                                 signal_sigma_thr_short: float = SIGNAL_SIGMA_THR_SHORT,
                                 ):
    assert signal_sigma_thr_long > signal_sigma_thr_short, ValueError("The long threshold must always be greater than the short in case of ")
    backtest_returns = pd.Series([0], index=df.index)
    signal = df.filter(like="signal")
    signal = signal.sort_index(
        axis=1,
        key=lambda c: c.str.extract(r'(\d+)', expand=False).astype(float).fillna(-1)
    )
    log_returns = df.filter(like="log_return")
    log_returns = log_returns.sort_index(
        axis=1,
        key=lambda c: c.str.extract(r'(\d+)', expand=False).astype(float).fillna(-1)
    )
    pos_flag = None
    for index in range(signal_sigma_window_size, len(df)+1):
        sigmas = signal.iloc[index-signal_sigma_window_size:signal_sigma_window_size].std()
        before_signal = signal.iloc[index-1]
        actual_returns = log_returns.iloc[index]
        for second_index in range(len(sigmas)):
            unit_actual_returns = actual_returns.iloc[second_index]
            unit_actual_sigma = sigmas.iloc[second_index]
            if before_signal < -signal_sigma_thr_short * unit_actual_sigma:
                if pos_flag == None:
                    backtest_returns.append(-unit_actual_returns - transaction_cost)
                    pos_flag = "short"
                elif pos_flag == "long":
                    backtest_returns.append(-unit_actual_returns - transaction_cost*2)
                    pos_flag = "short"
                else:
                    backtest_returns.append(-unit_actual_returns)
                
            elif before_signal > signal_sigma_thr_long * unit_actual_sigma:
                if pos_flag == None:
                    backtest_returns.append(unit_actual_returns - transaction_cost)
                    pos_flag = "long"
                elif pos_flag == "short":
                    backtest_returns.append(unit_actual_returns - transaction_cost*2)
                    pos_flag = "long"
                else:
                    backtest_returns.append(unit_actual_returns)
            else:
                if pos_flag in {"long", "short"}:
                    pos_flag = None
                    backtest_returns.append(transaction_cost)
        
        
                
            
