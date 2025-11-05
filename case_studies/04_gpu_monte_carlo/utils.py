from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

_OptionLiteral = Literal["C", "Call", "CALL", "c", "call", "P", "Put", "PUT", "p", "put"]


def get_array_module(arr):
    """
    Detect if the array is NumPy or CuPy and return the corresponding module.

    Parameters
    ----------
    arr : array-like
        Input array (NumPy or CuPy).

    Returns
    -------
    module
        numpy or cupy module depending on input array type.
    """
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp
    return np


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

    This function is backend-agnostic and works with both NumPy (CPU) and CuPy (GPU) arrays.
    When CuPy arrays are provided, all computations remain on the GPU until the final result.

    Parameters
    ----------
    time_grid : array-like of shape (n_steps + 1,)
        Increasing time instants (in years) associated with the simulated paths.
        Can be a NumPy ndarray (CPU) or CuPy ndarray (GPU).
    paths : array-like of shape (n_steps + 1, n_paths)
        Simulated asset paths laid out as columns. Values can include the initial spot.
        Can be a NumPy ndarray (CPU) or CuPy ndarray (GPU).
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

    Notes
    -----
    Performance optimization:
        - For GPU arrays (CuPy), all computations stay on GPU until final transfer
        - For CPU arrays (NumPy), computations use standard NumPy operations
        - Preserves the dtype of the input arrays (no forced conversion to float64)
    """
    # Detect backend (NumPy or CuPy)
    xp = get_array_module(paths)

    # Convert to arrays using the appropriate backend, preserving dtype
    time_grid_arr = xp.asarray(time_grid)
    if time_grid_arr.ndim != 1 or time_grid_arr.size == 0:
        raise ValueError("time_grid must be a non-empty 1D array-like sequence")

    paths_arr = xp.asarray(paths)
    if paths_arr.ndim != 2:
        raise ValueError("paths must be a 2D array-like object of shape (n_steps + 1, n_paths)")
    if paths_arr.shape[0] != time_grid_arr.size:
        raise ValueError(
            f"paths first dimension ({paths_arr.shape[0]}) must match len(time_grid) ({time_grid_arr.size})"
        )

    option_type = _normalize_option_type(o_type)

    # All computations use the detected backend (xp)
    avg_prices = paths_arr.mean(axis=0)

    if option_type == "call":
        payoff = xp.maximum(avg_prices - strike, 0.0)
    else:
        payoff = xp.maximum(strike - avg_prices, 0.0)

    discount_factor = xp.exp(-rate * time_grid_arr[-1])
    price = discount_factor * payoff.mean()

    # Return a Python scalar (convert from GPU if needed)
    return float(price)
