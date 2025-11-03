from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray, DTypeLike

Array = NDArray[np.float64]
RngLike = Union[np.random.Generator, np.random.RandomState]


def _ensure_rng(rng: Optional[RngLike]) -> RngLike:
    """Return a usable random number generator, defaulting to NumPy's Generator."""
    return rng if rng is not None else np.random.default_rng()


def simulate_gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    dividend_yield: float = 0.0,
    *,
    antithetic: bool = False,
    dtype: DTypeLike = np.float64,
    rng: Optional[RngLike] = None,
    shocks: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """
    Simulate geometric Brownian motion paths using the model's closed-form solution.

    The process under the risk-neutral measure is
        S_t = s0 * exp((mu - q - 0.5 * sigma^2) * t + sigma * W_t)

    Parameters
    ----------
    s0 : float
        Initial asset price. Must be strictly positive.
    mu : float
        Drift of the process under the chosen measure (annualised).
    sigma : float
        Volatility parameter. Must be non-negative.
    maturity : float
        Time horizon in years. Must be strictly positive.
    n_steps : int
        Number of time steps (excluding the initial time).
    n_paths : int
        Number of Monte Carlo paths to generate.
    dividend_yield : float, optional
        Continuous dividend yield, default is 0.0.
    antithetic : bool, optional
        If True, use antithetic variates to reduce variance. Default is False.
    dtype : numpy dtype, optional
        Target floating-point precision for the generated paths. Default is float64.
    rng : numpy random Generator or RandomState, optional
        Random number generator to use. Defaults to `np.random.default_rng()`.
    shocks : ndarray, optional
        Pre-generated standard normal shocks of shape (n_paths, n_steps).
        Used for testing and exact reproducibility.
        If provided, antithetic and rng parameters are ignored.

    Returns
    -------
    tuple[ndarray, ndarray]
        A tuple containing the time grid of shape (n_steps + 1,) and the simulated
        paths of shape (n_steps + 1, n_paths).

    Raises
    ------
    ValueError
        If any of the input parameters are invalid.
    """
    if s0 <= 0.0:
        raise ValueError("s0 must be strictly positive to take its logarithm.")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")
    if maturity <= 0.0:
        raise ValueError("maturity must be strictly positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer.")

    target_dtype = np.dtype(dtype)

    dt = maturity / float(n_steps)
    drift = (mu - dividend_yield - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)

    if shocks is None:
        # Generate shocks using RNG
        rng = _ensure_rng(rng)
        base_paths = n_paths if not antithetic else (n_paths + 1) // 2
        shocks = rng.standard_normal(size=(base_paths, n_steps))
        shocks = np.asarray(shocks, dtype=target_dtype)
        if antithetic:
            shocks = np.concatenate((shocks, -shocks), axis=0)[:n_paths]
    else:
        # Use pre-generated shocks
        shocks = np.asarray(shocks, dtype=target_dtype)

    log_returns = drift + vol * shocks
    cumulative_returns = np.cumsum(log_returns, axis=1, dtype=target_dtype)

    log_paths = np.empty((n_steps + 1, n_paths), dtype=target_dtype)
    log_s0 = np.array(np.log(s0), dtype=target_dtype)
    log_paths[0, :] = log_s0
    log_paths[1:, :] = (log_s0 + cumulative_returns).T

    paths = np.empty_like(log_paths)
    np.exp(log_paths, out=paths)

    time_grid = np.linspace(0.0, maturity, n_steps + 1, dtype=target_dtype)
    return time_grid, paths


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t_grid, spot_paths = simulate_gbm_paths(
        s0=100.0,
        mu=0.01,
        sigma=0.5,
        maturity=2.0,
        n_steps=2 * 252,
        n_paths=100,
        dividend_yield=0.0,
        antithetic=True,
        rng=np.random.default_rng(42),
    )

    log_returns = np.diff(np.log(spot_paths), axis=0)
    sigma_hat = np.std(log_returns, ddof=1) * np.sqrt(252)
    drift_hat = np.mean(log_returns) * 252
    mu_hat = drift_hat + 0.5 * sigma_hat ** 2
    print("actual drift :", drift_hat, "expected :", -0.115)
    print("actual sigma :", sigma_hat, "expected :", 0.5)
    print("actual mu :", mu_hat, "expected :", 0.01)

    for idx in range(spot_paths.shape[1]):
        plt.plot(t_grid, spot_paths[:, idx])
    plt.show()
