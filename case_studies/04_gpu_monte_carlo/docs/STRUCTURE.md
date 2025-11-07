# STRUCTURE.md — Technical Analysis of GPU Optimization

This document details the architecture, implementation choices, and technical principles underlying the GPU (CuPy) optimization compared to the CPU (NumPy) version.

---

## 🔬 1. The Core Problem — Simulating Geometric Brownian Motion (GBM)

The computational bottleneck is the simulation of **N** asset-price paths following a **Geometric Brownian Motion** (GBM).

Discretized analytical form for a time step $\Delta t$:

$$
S_{t_i} = S_{t_{i-1}} \exp\left((\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t}\, Z_i\right)
$$

where $Z_i \sim \mathcal{N}(0,1)$ is a random shock.

Expressed in log-space:

$$
\ln(S_{t_i}) = \ln(S_{t_{i-1}}) + (\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t}\, Z_i
$$

Simulating all trajectories from $t_0$ to $T$ requires:

1. Generating a random-shock matrix $Z$ of shape $(N_{paths} \times N_{steps})$  
2. Computing log-returns $R_{i,j} = \text{drift} + \text{vol} \times Z_{i,j}$  
3. Applying a cumulative sum (`cumsum`) along the time axis (`axis=1`)  
4. Adding $\ln(S_0)$ and exponentiating (`exp`) to recover price paths

The computational load is dominated by steps 1 (RNG) and 3 (`cumsum`), both large-scale matrix operations.

**Key insight:**  
This is an **embarrassingly parallel** problem — each path $i$ is fully independent of every other path $j \neq i$.  
It is thus ideal for hardware-level parallelization.

---

## 🧠 2. Suboptimal (CPU) Implementation — `suboptimal/pricing.py`

This version is "suboptimal" not because the NumPy code is poor, but because the **hardware (CPU)** is inherently inefficient for this class of problem.

### Design Philosophy

- Full vectorization with **NumPy**
- Avoids Python `for` loops (which are slow) by using array-wise operations (`np.cumsum`, `np.exp`), which run in compiled C/Fortran (BLAS/LAPACK)

### Example: `simulate_gbm_paths` (CPU)

```python
# 1. Shock generation on CPU
shocks = rng.standard_normal(size=(base_paths, n_steps))

# 2. Element-wise vectorization
log_returns = drift + vol * shocks

# 3. Axis-wise operation (costly step)
cumulative_returns = np.cumsum(log_returns, axis=1, dtype=target_dtype)

# 4. Assembly and exponentiation
log_paths[1:, :] = (log_s0 + cumulative_returns).T
np.exp(log_paths, out=paths)
````

### CPU Bottleneck

Even though NumPy is vectorized, it uses only a few CPU cores (e.g. 4 cores / 8 threads).
Parallelism relies on **SIMD** (Single Instruction, Multiple Data) instructions such as AVX2, operating on tiny vectors (e.g. 4 × `float64` or 8 × `float32`).

For $1{,}000{,}000 \times 252$ operations, the CPU remains mostly sequential compared to a GPU.
All data reside in system RAM (DDR3), which has far lower bandwidth than GPU VRAM (GDDR5/HBM).

---

## ⚡ 3. Optimized (GPU) Implementation — `optimized/pricing.py`

This version leverages the GPU’s massive parallelism by replacing `numpy` with `cupy`.

### Design Philosophy

* **CuPy** provides a nearly drop-in API replacement for NumPy
* Each CuPy call (e.g. `cp.random.standard_normal`, `cp.cumsum`) compiles and launches a **CUDA kernel** that runs across thousands of GPU cores
* The GPU uses **SIMT** (Single Instruction, Multiple Threads) — thousands of threads execute the same instruction (e.g. “generate random number”) on distinct data simultaneously

### Example: `simulate_gbm_paths` (GPU)

The code is almost identical to the CPU version, but every operation executes on the GPU:

```python
# 1. Random shocks (CUDA kernel)
gpu_shocks = cp.random.standard_normal(
    size=(base_paths, n_steps), dtype=target_dtype
)

# 2. Element-wise operation (CUDA kernel)
log_returns = drift_scalar + vol_scalar * gpu_shocks

# 3. Axis-wise cumulative sum (optimized CUDA kernel)
cumulative_returns = cp.cumsum(log_returns, axis=1, dtype=target_dtype)

# 4. Assembly and exponentiation (CUDA kernel)
log_paths[1:, :] = (log_s0 + cumulative_returns).T
chunk_paths_result = cp.exp(log_paths)
```

### Critical Difference — Zero Data Transfer

The performance gain comes not only from raw compute speed but from **avoiding host↔device memory transfers**.

1. `gpu_shocks` is allocated directly in **VRAM**
2. `log_returns` computed in **VRAM**
3. `cumulative_returns` in **VRAM**
4. `log_paths` and final results remain in **VRAM**

Data never leave high-bandwidth GPU memory (GDDR5).
Only at the end — via `cp.asnumpy(paths_gpu)` — are results copied back to CPU RAM.
If the pricing logic (e.g. Asian payoff) also uses CuPy, this transfer can be avoided entirely.

---

## 🛠️ 4. Technical Optimizations in Detail

While swapping `np` → `cp` is the foundation, deeper understanding of hardware constraints is essential for real performance.

### 4.1. `float32` vs `float64` — Single vs Double Precision

This is the **most impactful** optimization on consumer GPUs.

* Gaming-class GPUs (e.g. GTX 980 Ti, Maxwell) have **severely limited FP64** throughput — typically $1/32$ of FP32 performance
* Data-center GPUs (Tesla V100, A100) offer far better FP64 ratios ($1/2$ or $1/3$ of FP32)
* In Monte Carlo simulations, **statistical noise** ($O(1/\sqrt{N})$) dominates machine precision error

**Conclusion:**

Our benchmarks confirm this. On the GTX 980 Ti, using `float32` (0.367s) is **1.81× faster** than using `float64` (0.666s).

While the theoretical FP64 *compute* limit for this architecture is 1/32 of FP32, our workload is bound by memory bandwidth and `cumsum` operations, not just raw computation. The **1.81× speedup** (and 50% VRAM saving) from using `float32` is a massive gain for a negligible loss in precision relative to Monte Carlo error

---

### 4.2. VRAM Management — `max_paths_per_chunk`

GPU memory is **fixed and limited** (6 GB on the 980 Ti).
Simulating $10^7 \times 252$ steps in `float64` would require tens of gigabytes — impossible.

* **Problem:** Over-allocation raises
  `cupy.cuda.runtime.CUDARuntimeError: out of memory`
* **Solution:** **Chunking** the workload
* The parameter `max_paths_per_chunk` splits the simulation into manageable batches
* A Python `for` loop calls `_simulate_chunk` for each batch, then concatenates results via `cp.concatenate`

**Trade-off:** Slight Python overhead (multiple kernel launches) in exchange for fitting within available VRAM — a standard compute/throughput trade-off.

---

### 4.3. Random Number Generation (RNG)

* CPU (NumPy) uses **PCG64** via `np.random.default_rng(seed)`
* GPU (CuPy) uses **Philox-4x32-10** via `cp.random.seed(seed)`
* These are **different algorithms**, so identical seeds yield **different sequences**

Thus:

* Option prices differ numerically but remain statistically equivalent
* Dedicated tests (in `test_correctness.py`) inject **pre-generated identical shock matrices** to verify mathematical equivalence:
  when fed the same inputs, both CPU and GPU functions produce **bit-level identical outputs** (within floating-point tolerance)

---

## ✅ Summary

| Aspect                   | CPU (NumPy)                   | GPU (CuPy)                      |
| :----------------------- | :---------------------------- | :------------------------------ |
| **Parallelism model**    | SIMD (few cores)              | SIMT (thousands of threads)     |
| **Memory bandwidth**     | DDR3 (~25 GB/s)               | GDDR5 (~250 GB/s)               |
| **Precision sweet spot** | FP64                          | FP32                            |
| **Bottleneck**           | RNG + cumsum (RAM-bound)      | Kernel launch overhead (minor)  |
| **Ideal use case**       | Small or sequential workloads | Massive Monte Carlo simulations |

> **In essence:** the GPU implementation achieves its acceleration not by algorithmic change,
> but by aligning the same mathematical process with hardware that matches its intrinsic parallel structure.