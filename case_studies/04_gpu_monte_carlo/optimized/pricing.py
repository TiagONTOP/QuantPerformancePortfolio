"""GPU-accelerated Monte Carlo simulation for geometric Brownian motion paths.

This module provides optimized implementation using CuPy (vectorized GPU).
CuPy is required - this is the GPU-only optimized version.

For CPU-based simulation, use suboptimal/pricing.py instead.

Installation
------------
Required:
    pip install cupy-cuda12x  # or cupy-cuda11x depending on your CUDA version

For CUDA Toolkit on Windows:
    Download from https://developer.nvidia.com/cuda-downloads

Example
-------
    from optimized.pricing import simulate_gbm_paths
    import numpy as np

    # GPU-accelerated simulation (10-100x faster than CPU)
    t, paths = simulate_gbm_paths(
        s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
        n_steps=252, n_paths=1_000_000,
        dtype=np.float32, seed=42
    )
"""

from __future__ import annotations

import os
import platform
import warnings
from typing import Optional, Tuple

import cupy as cp
import numpy as np
from numpy.typing import DTypeLike, NDArray

Array = NDArray[np.float64]


def _estimate_memory_gb(n_paths: int, n_steps: int, dtype: np.dtype) -> float:
    """Estimate GPU memory usage in GB for the simulation."""
    itemsize = dtype.itemsize
    # Memory for: shocks (n_paths × n_steps), paths ((n_steps + 1) × n_paths), intermediates
    memory_bytes = (2 * n_paths * n_steps + n_paths) * itemsize
    return memory_bytes / (1024**3)


def _get_available_gpu_memory_gb() -> Optional[float]:
    """Return available GPU memory in GB for the current device."""
    try:
        device = cp.cuda.Device()
        free_bytes, _ = device.mem_info
        return free_bytes / (1024**3)
    except cp.cuda.runtime.CUDARuntimeError:
        return None
    except Exception:
        return None


def _get_available_host_memory_gb() -> Optional[float]:
    """Return available system RAM in GB."""
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        pass

    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            return (page_size * avail_pages) / (1024**3)
        except (OSError, ValueError, AttributeError):
            pass

    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys / (1024**3)
        except Exception:
            pass

    return None


def _validate_inputs(
    s0: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
) -> None:
    """Validate input parameters for GBM simulation."""
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
    dtype: DTypeLike = np.float32,
    device_output: bool = False,
    max_paths_per_chunk: Optional[int] = None,
    seed: Optional[int] = None,
    shocks: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """
    Simulate geometric Brownian motion paths using GPU acceleration (CuPy).

    This is the optimized GPU version. For CPU computation, use suboptimal/pricing.py.

    The process under the risk-neutral measure is:
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
        Target floating-point precision. Default is float32.
        Use float32 for best GPU performance (2x faster than float64).
        Use float64 for higher precision validation.
    device_output : bool, optional
        If True, return GPU arrays (CuPy ndarray) to avoid CPU transfer.
        If False, transfer results to CPU (NumPy ndarray). Default is False.
    max_paths_per_chunk : int, optional
        Process paths in chunks to limit GPU memory usage.
        Useful for very large simulations. Default is None (no chunking).
    seed : int, optional
        Random seed for reproducibility on GPU.
    shocks : ndarray, optional
        Pre-generated standard normal shocks of shape (n_paths, n_steps).
        Used for testing and exact reproducibility.
        If provided, antithetic and seed parameters are ignored.

    Returns
    -------
    tuple[ndarray, ndarray]
        A tuple containing:
        - time_grid: array of shape (n_steps + 1,)
        - paths: array of shape (n_steps + 1, n_paths)

        By default, returns NumPy ndarrays (CPU).
        If device_output=True, returns CuPy ndarrays (GPU).

    Raises
    ------
    ValueError
        If any of the input parameters are invalid.

    Notes
    -----
    GPU Memory estimation:
        memory ≈ (2 × n_paths × n_steps + n_paths) × sizeof(dtype)

    For example, 1M paths × 252 steps × float32 ≈ 2 GB GPU memory.

    Performance recommendations:
        - Use float32 for production (2x faster, sufficient precision)
        - Use float64 for validation (higher precision)
        - Use max_paths_per_chunk for memory-constrained GPUs
        - Typical speedup: 10-100x vs CPU (hardware dependent)

    Examples
    --------
    Basic usage:

    >>> import numpy as np
    >>> t, paths = simulate_gbm_paths(
    ...     s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
    ...     n_steps=252, n_paths=100000, seed=42
    ... )

    Maximum performance with float32:

    >>> t, paths = simulate_gbm_paths(
    ...     s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
    ...     n_steps=252, n_paths=1000000,
    ...     dtype=np.float32, seed=42
    ... )

    Chunking for very large simulations:

    >>> t, paths = simulate_gbm_paths(
    ...     s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
    ...     n_steps=252, n_paths=10000000,
    ...     max_paths_per_chunk=1000000, seed=42
    ... )

    Keep results on GPU for further processing:

    >>> import cupy as cp
    >>> t_gpu, paths_gpu = simulate_gbm_paths(
    ...     s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
    ...     n_steps=252, n_paths=1000000,
    ...     device_output=True, seed=42
    ... )
    >>> # Continue processing on GPU
    >>> payoff = cp.maximum(paths_gpu[-1, :] - 105.0, 0.0)
    >>> option_price = float(cp.mean(payoff) * cp.exp(-0.05 * 1.0))
    """
    # Validate inputs
    _validate_inputs(s0, sigma, maturity, n_steps, n_paths)
    target_dtype = np.dtype(dtype)

    # Estimate memory and warn if large
    mem_gb = _estimate_memory_gb(n_paths, n_steps, target_dtype)
    if max_paths_per_chunk is None:
        gpu_free_gb = _get_available_gpu_memory_gb()
        host_free_gb = _get_available_host_memory_gb()
        warning_reasons = []

        if gpu_free_gb is not None:
            gpu_threshold = gpu_free_gb * 0.8
            if mem_gb > gpu_threshold:
                warning_reasons.append(
                    f"GPU usage estimate {mem_gb:.2f} GB exceeds 80% of free GPU memory ({gpu_free_gb:.2f} GB)"
                )

        if host_free_gb is not None:
            host_threshold = host_free_gb * 0.8
            if mem_gb > host_threshold:
                warning_reasons.append(
                    f"GPU result transfer requires {mem_gb:.2f} GB which exceeds 80% of available system RAM ({host_free_gb:.2f} GB)"
                )

        if not warning_reasons and gpu_free_gb is None and host_free_gb is None and mem_gb > 4.0:
            warning_reasons.append(
                f"estimated GPU memory {mem_gb:.2f} GB exceeds the default safety threshold (4.00 GB)"
            )

        if warning_reasons:
            warnings.warn(
                "High memory pressure detected: "
                + "; ".join(warning_reasons)
                + ". Consider using max_paths_per_chunk to reduce memory usage.",
                ResourceWarning,
            )

    # Compute constants
    dt = maturity / float(n_steps)
    drift_scalar = float((mu - dividend_yield - 0.5 * sigma * sigma) * dt)
    vol_scalar = float(sigma * np.sqrt(dt))
    log_s0_scalar = float(np.log(s0))

    # Set seed if provided
    if seed is not None:
        cp.random.seed(seed)

    def _simulate_chunk(chunk_paths: int, chunk_shocks: Optional[Array] = None) -> cp.ndarray:
        """Simulate a chunk of paths on GPU."""
        if chunk_shocks is None:
            # Generate shocks on GPU
            base_paths = chunk_paths if not antithetic else (chunk_paths + 1) // 2
            gpu_shocks = cp.random.standard_normal(
                size=(base_paths, n_steps), dtype=target_dtype
            )
            if antithetic:
                # Apply antithetic variates
                gpu_shocks = cp.concatenate(
                    (gpu_shocks, -gpu_shocks), axis=0
                )[:chunk_paths]
        else:
            # Use pre-generated shocks
            gpu_shocks = cp.asarray(chunk_shocks, dtype=target_dtype)

        # Vectorized computation on GPU
        log_returns = drift_scalar + vol_scalar * gpu_shocks
        cumulative_returns = cp.cumsum(log_returns, axis=1, dtype=target_dtype)

        # Build log_paths with time dimension first
        log_paths = cp.empty((n_steps + 1, chunk_paths), dtype=target_dtype)
        log_paths[0, :] = log_s0_scalar
        log_paths[1:, :] = (log_s0_scalar + cumulative_returns).T

        # Exponentiate to get prices
        chunk_paths_result = cp.exp(log_paths)
        return chunk_paths_result

    # Handle chunking if requested
    if max_paths_per_chunk is not None and max_paths_per_chunk < n_paths and shocks is None:
        all_paths = []
        for start_idx in range(0, n_paths, max_paths_per_chunk):
            end_idx = min(start_idx + max_paths_per_chunk, n_paths)
            chunk_size = end_idx - start_idx
            chunk_result = _simulate_chunk(chunk_size)
            all_paths.append(chunk_result)
        paths_gpu = cp.concatenate(all_paths, axis=1)
    else:
        # Single chunk (with or without pre-generated shocks)
        paths_gpu = _simulate_chunk(n_paths, shocks)

    # Time grid
    time_grid = cp.linspace(0.0, maturity, n_steps + 1, dtype=target_dtype)

    # Transfer to CPU if needed
    if device_output:
        # Return both arrays on GPU for zero-copy pipeline
        return time_grid, paths_gpu
    else:
        # Transfer both arrays to CPU
        return cp.asnumpy(time_grid), cp.asnumpy(paths_gpu)


if __name__ == "__main__":
    import time

    print("GPU-Accelerated GBM Monte Carlo Simulation (CuPy)")
    print("=" * 60)
    print("CuPy backend only - optimized for maximum performance")
    print()

    # Configuration
    s0 = 100.0
    mu = 0.05
    sigma = 0.2
    maturity = 1.0
    n_steps = 252
    n_paths = 1_000_000
    seed = 42

    print(f"Simulating {n_paths:,} paths with {n_steps} steps")
    print()

    # Benchmark float32 (recommended)
    print("Testing float32 (recommended for production)...")
    start = time.perf_counter()
    t_grid, paths = simulate_gbm_paths(
        s0=s0, mu=mu, sigma=sigma, maturity=maturity,
        n_steps=n_steps, n_paths=n_paths,
        dtype=np.float32, seed=seed
    )
    elapsed_f32 = time.perf_counter() - start
    print(f"  Time: {elapsed_f32:.4f}s")
    print(f"  Final price (mean): {paths[-1, :].mean():.4f}")
    print(f"  Final price (std): {paths[-1, :].std():.4f}")
    print()

    # Benchmark float64 (validation)
    print("Testing float64 (higher precision)...")
    start = time.perf_counter()
    t_grid, paths = simulate_gbm_paths(
        s0=s0, mu=mu, sigma=sigma, maturity=maturity,
        n_steps=n_steps, n_paths=n_paths,
        dtype=np.float64, seed=seed
    )
    elapsed_f64 = time.perf_counter() - start
    print(f"  Time: {elapsed_f64:.4f}s")
    print(f"  Final price (mean): {paths[-1, :].mean():.4f}")
    print(f"  Final price (std): {paths[-1, :].std():.4f}")
    print()

    ratio = elapsed_f64 / elapsed_f32
    print(f"Performance: float32 is {ratio:.2f}x faster than float64")
    print()

    print("Performance Tips:")
    print("  - Use dtype=np.float32 for production (more than 2x faster)")
    print("  - Use dtype=np.float64 for validation")
    print("  - Use max_paths_per_chunk for memory-constrained GPUs")
    print("  - Use device_output=True to keep results on GPU")
    print()
    print("For CPU computation, use suboptimal/pricing.py")
