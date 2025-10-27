"""GPU-accelerated Monte Carlo simulation for geometric Brownian motion paths.

This module provides optimized implementations using CuPy (vectorized GPU) and
Numba CUDA (custom kernels) with automatic fallback to CPU when GPU is unavailable.

Installation
------------
For GPU support, install:
    pip install cupy-cuda12x  # or cupy-cuda11x depending on your CUDA version
    pip install numba

For CUDA Toolkit on Windows:
    Download from https://developer.nvidia.com/cuda-downloads
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray

Array = NDArray[np.float_]
RngLike = Union[np.random.Generator, np.random.RandomState]
Backend = Literal["auto", "cupy", "numba", "cpu"]


# Check for GPU backend availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    from numba import cuda
    NUMBA_AVAILABLE = cuda.is_available()
except (ImportError, Exception):
    NUMBA_AVAILABLE = False
    cuda = None


def _ensure_rng(rng: Optional[RngLike]) -> RngLike:
    """Return a usable random number generator, defaulting to NumPy's Generator."""
    return rng if rng is not None else np.random.default_rng()


def _estimate_memory_gb(n_paths: int, n_steps: int, dtype: np.dtype) -> float:
    """Estimate GPU memory usage in GB for the simulation."""
    itemsize = dtype.itemsize
    # Memory for: shocks (n_paths × n_steps), paths (n_paths × n_steps+1), intermediates
    memory_bytes = (2 * n_paths * n_steps + n_paths) * itemsize
    return memory_bytes / (1024**3)


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


def _simulate_gbm_cpu(
    s0: float,
    mu: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    dividend_yield: float,
    antithetic: bool,
    dtype: np.dtype,
    rng: RngLike,
    shocks: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """CPU implementation of GBM simulation (fallback)."""
    dt = maturity / float(n_steps)
    drift = (mu - dividend_yield - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)

    if shocks is None:
        base_paths = n_paths if not antithetic else (n_paths + 1) // 2
        shocks = rng.standard_normal(size=(base_paths, n_steps))
        shocks = np.asarray(shocks, dtype=dtype)
        if antithetic:
            shocks = np.concatenate((shocks, -shocks), axis=0)[:n_paths]
    else:
        shocks = np.asarray(shocks, dtype=dtype)

    log_returns = drift + vol * shocks
    cumulative_returns = np.cumsum(log_returns, axis=1, dtype=dtype)

    log_paths = np.empty((n_paths, n_steps + 1), dtype=dtype)
    log_s0 = np.array(np.log(s0), dtype=dtype)
    log_paths[:, 0] = log_s0
    log_paths[:, 1:] = log_s0 + cumulative_returns

    paths = np.empty_like(log_paths)
    np.exp(log_paths, out=paths)

    time_grid = np.linspace(0.0, maturity, n_steps + 1, dtype=dtype)
    return time_grid, paths


def _simulate_gbm_cupy(
    s0: float,
    mu: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    dividend_yield: float,
    antithetic: bool,
    dtype: np.dtype,
    seed: Optional[int],
    device_output: bool,
    max_paths_per_chunk: Optional[int],
    shocks: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """CuPy vectorized GPU implementation."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available. Install with: pip install cupy-cuda12x")

    dt = maturity / float(n_steps)
    drift_scalar = float((mu - dividend_yield - 0.5 * sigma * sigma) * dt)
    vol_scalar = float(sigma * np.sqrt(dt))
    log_s0_scalar = float(np.log(s0))

    # Set seed if provided
    if seed is not None:
        cp.random.seed(seed)

    def _simulate_chunk(chunk_paths: int) -> cp.ndarray:
        """Simulate a chunk of paths on GPU."""
        if shocks is None:
            base_paths = chunk_paths if not antithetic else (chunk_paths + 1) // 2
            chunk_shocks = cp.random.standard_normal(
                size=(base_paths, n_steps), dtype=dtype
            )
            if antithetic:
                chunk_shocks = cp.concatenate(
                    (chunk_shocks, -chunk_shocks), axis=0
                )[:chunk_paths]
        else:
            chunk_shocks = cp.asarray(shocks, dtype=dtype)

        # Vectorized computation
        log_returns = drift_scalar + vol_scalar * chunk_shocks
        cumulative_returns = cp.cumsum(log_returns, axis=1, dtype=dtype)

        # Build log_paths
        log_paths = cp.empty((chunk_paths, n_steps + 1), dtype=dtype)
        log_paths[:, 0] = log_s0_scalar
        log_paths[:, 1:] = log_s0_scalar + cumulative_returns

        # Exponentiate
        chunk_paths_result = cp.exp(log_paths)
        return chunk_paths_result

    # Handle chunking if requested
    if max_paths_per_chunk is not None and max_paths_per_chunk < n_paths:
        all_paths = []
        for start_idx in range(0, n_paths, max_paths_per_chunk):
            end_idx = min(start_idx + max_paths_per_chunk, n_paths)
            chunk_size = end_idx - start_idx
            chunk_result = _simulate_chunk(chunk_size)
            all_paths.append(chunk_result)
        paths_gpu = cp.concatenate(all_paths, axis=0)
    else:
        paths_gpu = _simulate_chunk(n_paths)

    # Time grid
    time_grid = cp.linspace(0.0, maturity, n_steps + 1, dtype=dtype)

    # Transfer to CPU if needed
    if device_output:
        return cp.asnumpy(time_grid), paths_gpu
    else:
        return cp.asnumpy(time_grid), cp.asnumpy(paths_gpu)


def _simulate_gbm_numba(
    s0: float,
    mu: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    dividend_yield: float,
    antithetic: bool,
    dtype: np.dtype,
    seed: Optional[int],
    device_output: bool,
) -> Tuple[Array, Array]:
    """Numba CUDA kernel implementation (1 thread per path)."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError(
            "Numba CUDA is not available. Install with: pip install numba\n"
            "And ensure CUDA Toolkit is installed."
        )

    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64, xoroshiro128p_normal_float32

    dt = maturity / float(n_steps)
    drift = (mu - dividend_yield - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    log_s0 = np.log(s0)

    # Determine base paths for antithetic
    base_paths = n_paths if not antithetic else (n_paths + 1) // 2
    actual_paths = n_paths

    # Create output array
    paths = np.empty((actual_paths, n_steps + 1), dtype=dtype)
    paths[:, 0] = s0

    # Initialize RNG states
    rng_seed = seed if seed is not None else np.random.randint(0, 2**31)
    rng_states = create_xoroshiro128p_states(base_paths, seed=rng_seed)

    # Define kernel based on dtype
    if dtype == np.float32:
        @cuda.jit(fastmath=True)
        def gbm_kernel(paths_out, rng_states, drift, vol, log_s0, n_steps, antithetic, base_paths):
            idx = cuda.grid(1)
            if idx < base_paths:
                log_price = log_s0
                for t in range(n_steps):
                    z = xoroshiro128p_normal_float32(rng_states, idx)
                    log_price += drift + vol * z
                    paths_out[idx, t + 1] = np.exp(log_price)

                # Handle antithetic path if needed
                if antithetic and idx + base_paths < paths_out.shape[0]:
                    log_price_anti = log_s0
                    cuda.syncthreads()
                    for t in range(n_steps):
                        z = xoroshiro128p_normal_float32(rng_states, idx)
                        log_price_anti += drift - vol * z  # Negated shock
                        paths_out[idx + base_paths, t + 1] = np.exp(log_price_anti)
    else:  # float64
        @cuda.jit(fastmath=True)
        def gbm_kernel(paths_out, rng_states, drift, vol, log_s0, n_steps, antithetic, base_paths):
            idx = cuda.grid(1)
            if idx < base_paths:
                log_price = log_s0
                for t in range(n_steps):
                    z = xoroshiro128p_normal_float64(rng_states, idx)
                    log_price += drift + vol * z
                    paths_out[idx, t + 1] = np.exp(log_price)

                # Handle antithetic path if needed
                if antithetic and idx + base_paths < paths_out.shape[0]:
                    log_price_anti = log_s0
                    cuda.syncthreads()
                    for t in range(n_steps):
                        z = xoroshiro128p_normal_float64(rng_states, idx)
                        log_price_anti += drift - vol * z  # Negated shock
                        paths_out[idx + base_paths, t + 1] = np.exp(log_price_anti)

    # Launch kernel
    threads_per_block = 256
    blocks = (base_paths + threads_per_block - 1) // threads_per_block

    d_paths = cuda.to_device(paths)
    gbm_kernel[blocks, threads_per_block](
        d_paths, rng_states, drift, vol, log_s0, n_steps, antithetic, base_paths
    )

    if device_output:
        paths_result = d_paths
    else:
        paths_result = d_paths.copy_to_host()

    time_grid = np.linspace(0.0, maturity, n_steps + 1, dtype=dtype)
    return time_grid, paths_result


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
    backend: Backend = "auto",
    device_output: bool = False,
    max_paths_per_chunk: Optional[int] = None,
    seed: Optional[int] = None,
    shocks: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """
    Simulate geometric Brownian motion paths using GPU or CPU acceleration.

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
        Target floating-point precision. Default is float64.
        Use float32 for better GPU performance with acceptable precision.
    rng : numpy random Generator or RandomState, optional
        Random number generator for CPU backend. Defaults to `np.random.default_rng()`.
    backend : {"auto", "cupy", "numba", "cpu"}, optional
        Backend to use for computation:
        - "auto": Automatically select best available (CuPy > Numba > CPU)
        - "cupy": Use CuPy vectorized implementation (fastest for large simulations)
        - "numba": Use Numba CUDA kernel (good for custom control)
        - "cpu": Use NumPy CPU implementation
        Default is "auto".
    device_output : bool, optional
        If True and using GPU backend, return GPU arrays (CuPy ndarray or Numba device array).
        If False, transfer results to CPU (NumPy ndarray). Default is False.
    max_paths_per_chunk : int, optional
        For CuPy backend, process paths in chunks to limit memory usage.
        Useful for very large simulations. Default is None (no chunking).
    seed : int, optional
        Random seed for reproducibility (used by GPU backends).
        For CPU backend, use the `rng` parameter instead.
    shocks : ndarray, optional
        Pre-generated standard normal shocks of shape (n_paths, n_steps).
        Used for testing and exact reproducibility across backends.
        If provided, antithetic and seed parameters are ignored.

    Returns
    -------
    tuple[ndarray, ndarray]
        A tuple containing:
        - time_grid: array of shape (n_steps + 1,)
        - paths: array of shape (n_paths, n_steps + 1)

        Arrays are NumPy ndarrays by default, or GPU arrays if device_output=True.

    Raises
    ------
    ValueError
        If any of the input parameters are invalid.
    RuntimeError
        If requested backend is not available.

    Notes
    -----
    Memory estimation (GPU):
        memory H (2 × n_paths × n_steps + n_paths) × sizeof(dtype)

    For example, 1M paths × 252 steps × float32 H 2 GB GPU memory.

    Performance recommendations:
        - Use float32 for better GPU throughput (usually sufficient for pricing)
        - Use float64 for higher precision validation
        - Use max_paths_per_chunk for memory-constrained GPUs
        - CuPy backend is typically fastest for large vectorized operations
        - Numba backend offers more control and is good for custom modifications

    Examples
    --------
    Basic usage with automatic backend selection:

    >>> t, paths = simulate_gbm_paths(
    ...     s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
    ...     n_steps=252, n_paths=100000, seed=42
    ... )

    Force GPU backend with float32 for maximum performance:

    >>> t, paths = simulate_gbm_paths(
    ...     s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
    ...     n_steps=252, n_paths=1000000,
    ...     backend="cupy", dtype=np.float32, seed=42
    ... )

    Use chunking for very large simulations:

    >>> t, paths = simulate_gbm_paths(
    ...     s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
    ...     n_steps=252, n_paths=10000000,
    ...     backend="cupy", max_paths_per_chunk=1000000, seed=42
    ... )

    Test reproducibility with pre-generated shocks:

    >>> rng = np.random.default_rng(42)
    >>> shocks = rng.standard_normal((10000, 252))
    >>> t_cpu, paths_cpu = simulate_gbm_paths(
    ...     s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
    ...     n_steps=252, n_paths=10000, backend="cpu", shocks=shocks
    ... )
    >>> t_gpu, paths_gpu = simulate_gbm_paths(
    ...     s0=100.0, mu=0.05, sigma=0.2, maturity=1.0,
    ...     n_steps=252, n_paths=10000, backend="cupy", shocks=shocks
    ... )
    >>> np.allclose(paths_cpu, paths_gpu, rtol=1e-6)
    True
    """
    # Validate inputs
    _validate_inputs(s0, sigma, maturity, n_steps, n_paths)
    target_dtype = np.dtype(dtype)

    # Estimate memory and warn if large
    mem_gb = _estimate_memory_gb(n_paths, n_steps, target_dtype)
    if mem_gb > 4.0 and backend in ("auto", "cupy") and max_paths_per_chunk is None:
        warnings.warn(
            f"Estimated GPU memory: {mem_gb:.2f} GB. "
            f"Consider using max_paths_per_chunk to reduce memory usage.",
            ResourceWarning,
        )

    # Backend selection logic
    if backend == "auto":
        if CUPY_AVAILABLE:
            selected_backend = "cupy"
        elif NUMBA_AVAILABLE:
            selected_backend = "numba"
        else:
            selected_backend = "cpu"
    else:
        selected_backend = backend

    # Dispatch to appropriate backend
    if selected_backend == "cpu":
        rng = _ensure_rng(rng)
        return _simulate_gbm_cpu(
            s0, mu, sigma, maturity, n_steps, n_paths,
            dividend_yield, antithetic, target_dtype, rng, shocks
        )

    elif selected_backend == "cupy":
        if not CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy backend requested but not available. "
                "Install with: pip install cupy-cuda12x"
            )
        return _simulate_gbm_cupy(
            s0, mu, sigma, maturity, n_steps, n_paths,
            dividend_yield, antithetic, target_dtype, seed,
            device_output, max_paths_per_chunk, shocks
        )

    elif selected_backend == "numba":
        if not NUMBA_AVAILABLE:
            raise RuntimeError(
                "Numba backend requested but not available. "
                "Install with: pip install numba and ensure CUDA Toolkit is installed."
            )
        if shocks is not None:
            warnings.warn(
                "Pre-generated shocks are not supported with Numba backend. "
                "Falling back to CPU.",
                UserWarning,
            )
            rng = _ensure_rng(rng)
            return _simulate_gbm_cpu(
                s0, mu, sigma, maturity, n_steps, n_paths,
                dividend_yield, antithetic, target_dtype, rng, shocks
            )
        return _simulate_gbm_numba(
            s0, mu, sigma, maturity, n_steps, n_paths,
            dividend_yield, antithetic, target_dtype, seed, device_output
        )

    else:
        raise ValueError(
            f"Unknown backend: {selected_backend}. "
            f"Must be one of: 'auto', 'cupy', 'numba', 'cpu'"
        )


if __name__ == "__main__":
    import time

    print("GPU-Accelerated GBM Monte Carlo Simulation")
    print("=" * 50)
    print(f"CuPy available: {CUPY_AVAILABLE}")
    print(f"Numba CUDA available: {NUMBA_AVAILABLE}")
    print()

    # Configuration
    s0 = 100.0
    mu = 0.05
    sigma = 0.2
    maturity = 1.0
    n_steps = 252
    n_paths = 1_000_000
    seed = 42

    # Benchmark different backends
    backends = []
    if CUPY_AVAILABLE:
        backends.append("cupy")
    if NUMBA_AVAILABLE:
        backends.append("numba")
    backends.append("cpu")

    print(f"Simulating {n_paths:,} paths with {n_steps} steps\n")

    results = {}
    for backend_name in backends:
        print(f"Testing {backend_name.upper()} backend...")
        start = time.perf_counter()

        t_grid, paths = simulate_gbm_paths(
            s0=s0, mu=mu, sigma=sigma, maturity=maturity,
            n_steps=n_steps, n_paths=n_paths,
            backend=backend_name, dtype=np.float32, seed=seed
        )

        elapsed = time.perf_counter() - start
        results[backend_name] = (elapsed, paths)

        print(f"  Time: {elapsed:.4f}s")
        print(f"  Final price (mean): {paths[:, -1].mean():.4f}")
        print(f"  Final price (std): {paths[:, -1].std():.4f}")
        print()

    # Compute speedups
    if "cpu" in results:
        cpu_time = results["cpu"][0]
        print("Speedup vs CPU:")
        for backend_name in backends:
            if backend_name != "cpu":
                speedup = cpu_time / results[backend_name][0]
                print(f"  {backend_name.upper()}: {speedup:.2f}x")
