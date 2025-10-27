# gbm_ascii.py
from __future__ import annotations
import numpy as np
from math import log, exp, sqrt, erf
from typing import Callable, Optional, Tuple
import matplotlib.pyplot as plt

Array = np.ndarray

def _ensure_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()

def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    q: float = 0.0,
    *,
    antithetic: bool = False,
    dtype=np.float64,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Array, Array]:
    """
    Exact GBM:
        S_t = S0 * exp( (mu - q - 0.5*sigma^2) * t + sigma * W_t )

    Returns (t, paths) with paths.shape = (n_paths, n_steps+1).
    Fully vectorized over paths and steps.
    """
    rng = _ensure_rng(rng)
    dt = T / n_steps
    drift = (mu - q - 0.5 * sigma * sigma) * dt
    vol = sigma * sqrt(dt)

    base = n_paths if not antithetic else (n_paths + 1) // 2
    Z = rng.standard_normal(size=(base, n_steps)).astype(dtype, copy=False)
    if antithetic:
        Z = np.vstack([Z, -Z])[:n_paths]
    else:
        if Z.shape[0] != n_paths:
            Z = np.resize(Z, (n_paths, n_steps))

    log_returns = drift + vol * Z                                 
    cums = np.cumsum(log_returns, axis=1, dtype=dtype)            
    logS0 = np.array(log(S0), dtype=dtype)
    logS = np.empty((n_paths, n_steps + 1), dtype=dtype)
    logS[:, 0] = logS0
    logS[:, 1:] = logS0 + cums
    S = np.exp(logS, dtype=dtype)

    t = np.linspace(0.0, T, n_steps + 1, dtype=dtype)
    return t, S


if __name__ == "__main__":
    t, S = simulate_gbm_paths(
        100,
        0.01,
        0.5,
        2,
        2*252,
        100,
        0,
        antithetic=True,
        rng=42
    )

    print(np.mean(S.ravel())*252)
    print(np.std(S.ravel()) * np.sqrt(252))

    plt.plot(t, S)
    plt.show()

    