from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

_OptionLiteral = Literal["C", "Call", "CALL", "c", "call", "P", "Put", "PUT", "p", "put"]


def _normalize_option_type(o_type: _OptionLiteral) -> Literal["call", "put"]:
    """Map user-provided option labels to lowercase canonical values."""
    normalized = str(o_type).strip().lower()
    mapping = {"c": "call", "call": "call", "p": "put", "put": "put"}
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError('o_type must be one of "call", "put", "c", or "p"') from exc


def price_asian_option(
    time_grid: ArrayLike,
    paths: ArrayLike,
    strike: float,
    rate: float,
    o_type: _OptionLiteral,
) -> float:
    """
    Price an arithmetic-average Asian option from pre-simulated paths.

    Parameters
    ----------
    time_grid : array-like of shape (n_steps + 1,)
        Increasing time instants (in years) associated with the simulated paths.
    paths : array-like of shape (n_steps + 1, n_paths)
        Simulated asset paths laid out as columns. Values can include the initial spot.
    strike : float
        Strike price of the option.
    rate : float
        Continuously compounded risk-free rate.
    o_type : {"call", "put", "c", "p"}
        Option payoff type. Inputs are case-insensitive.

    Returns
    -------
    float
        Discounted option value estimated via Monte Carlo sampling.
    """
    time_grid_arr = np.asarray(time_grid, dtype=np.float64)
    if time_grid_arr.ndim != 1 or time_grid_arr.size == 0:
        raise ValueError("time_grid must be a non-empty 1D array-like sequence")

    paths_arr: NDArray[np.float64] = np.asarray(paths, dtype=np.float64)
    if paths_arr.ndim != 2:
        raise ValueError("paths must be a 2D array-like object of shape (n_steps + 1, n_paths)")
    if paths_arr.shape[0] != time_grid_arr.size:
        raise ValueError(
            f"paths first dimension ({paths_arr.shape[0]}) must match len(time_grid) ({time_grid_arr.size})"
        )

    option_type = _normalize_option_type(o_type)
    avg_prices = paths_arr.mean(axis=0)

    if option_type == "call":
        payoff = np.maximum(avg_prices - strike, 0.0)
    else:
        payoff = np.maximum(strike - avg_prices, 0.0)

    discount_factor = float(np.exp(-rate * time_grid_arr[-1]))
    return float(discount_factor * payoff.mean())
