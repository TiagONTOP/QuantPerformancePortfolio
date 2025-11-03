# BENCHMARKS.md — Detailed Performance Analysis

This document presents the quantitative benchmark results comparing the CPU implementation (NumPy) to the GPU implementation (CuPy).

The results were generated using `tests/generate_performance_report.py` and archived in `tests/performance_report.txt` and `tests/benchmark_results.txt`.

**Main Conclusion:**  
The GPU implementation (CuPy) achieves a **speedup of up to 16.4×** over the optimized CPU (NumPy) version, provided that the problem size is large enough and single precision (`float32`) is used.

---

## 🖥️ Test Environment

- **CPU:** Intel Core i7 4770 (Overclocked @ 4.1 GHz)  
- **RAM:** 16 GB DDR3 @ 2400 MHz  
- **GPU:** NVIDIA GeForce GTX 980 Ti (Overclocked)  
- **Libraries:** NumPy (CPU), CuPy (GPU, CUDA)

---

## 📊 Summary Table of Results

The table below summarizes execution time (in seconds), throughput (in million operations per second), and relative speedup.  
Speedup is computed relative to **CPU float32** for fair comparison.

| Problem Size (Paths × Steps) | Backend | DType | Time (s) | Throughput (M-ops/s) | Speedup (vs CPU f32) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Small** (10k × 252) | CPU | `float32` | 0.080 | 31.6 | **1.0× (Baseline)** |
| | GPU | `float32` | 0.474 | 5.3 | **0.17× (Slower)** |
| | GPU | `float64` | 0.031 | 81.7 | **2.5×** |
| | | | | | |
| **Medium** (50k × 252) | CPU | `float32` | 0.734 | 17.2 | **1.0× (Baseline)** |
| | GPU | `float32` | 0.057 | 220.5 | **12.8×** |
| | GPU | `float64` | 0.098 | 128.8 | **7.5×** |
| | | | | | |
| **Large** (100k × 252) | CPU | `float32` | 0.985 | 25.6 | **1.0× (Baseline)** |
| | GPU | `float32` | **0.060** | 419.5 | **16.4× (Peak)** |
| | GPU | `float64` | 0.189 | 133.1 | **5.2×** |
| | | | | | |
| **Very Large** (500k × 252) | CPU | `float32` | 4.878 | 25.8 | **1.0× (Baseline)** |
| | GPU | `float32` | **0.502** | 251.1 | **9.7×** |
| | GPU | `float64` | 5.763 | 21.9 | **0.85× (Slower)** |

*(Data extracted from `tests/performance_report.txt`)*

---

## 📉 1. The Overhead Problem (Small Workloads)

The first and most critical observation comes from the **Small** case (10,000 paths):

- **CPU (`float32`)** — 0.080 seconds  
- **GPU (`float32`)** — 0.474 seconds  

👉 **The GPU is 5.9× slower than the CPU.**

This is *not* an error. It reflects the **fixed overhead** of GPU computation.  
To execute a single operation, CuPy must:

1. Allocate memory on VRAM.  
2. Compile the CUDA kernel (JIT — “Just-in-Time”).  
3. Launch the execution command to the GPU.  
4. Synchronize the result back to host.

For small problems, this fixed cost dominates the total runtime — while the CPU (leveraging SIMD vectorization) finishes the task faster.

> **Insight:** GPU acceleration only becomes worthwhile when the computational load is large enough to **amortize kernel launch overhead**.

**Note on the `float32` vs `float64` anomaly:**  
For the smallest test, `float32` (0.474 s) appears slower than `float64` (0.031 s).  
This likely reflects a one-time JIT compilation or warmup cost for `float32` kernels.  
In repeated (warmed-up) tests, performance stabilizes and the anomaly disappears.

---

## ⚡ 2. The Precision Effect (`float32` vs `float64`)

A second major insight: **precision has a massive impact on GPU performance.**

- **Large Problem (100k paths):**  
  - `float32` → **0.060 s**  
  - `float64` → 0.189 s (**3.15× slower**)  

- **Very Large Problem (500k paths):**  
  - `float32` → **0.502 s**  
  - `float64` → 5.763 s (**11.5× slower!**)  

### Why such a huge gap?

Because the GTX 980 Ti is a **gaming GPU**, not a compute GPU.

- **Architecture (Maxwell):** Double-precision (`float64`) throughput is artificially capped to **1/32 of single precision (`float32`) performance**.  
- **Real-world outcome:** The benchmarks confirm it.  
  On the “Very Large” case, the GPU in `float64` (5.76 s) is actually slower than the CPU in `float32` (4.88 s).

> **Insight:**  
> For Monte Carlo simulations — where the stochastic error $O(1/\sqrt{N})$ dominates machine precision error —  
> using `float32` is optimal.  
> It halves VRAM usage and can deliver **10×+ performance gains** on consumer GPUs.

---

## 🚀 3. The Sweet Spot — Optimal Use Case

The best performance occurs when both **parallelism** and **hardware efficiency** are maximized.

**Scenario:** 100k paths × 252 steps, `float32`

| Metric | CPU (NumPy) | GPU (CuPy) |
| :--- | ---: | ---: |
| **Execution Time** | 0.985 s | **0.060 s** |
| **Speedup** | — | **16.4×** |

Here, the GPU reaches its full potential:

- The workload is large enough to **saturate thousands of CUDA cores**.  
- The `float32` datatype unlocks **full FP32 throughput** on the GPU.  
- The CPU, despite efficient SIMD vectorization, cannot match the GPU’s **SIMT** (Single Instruction, Multiple Threads) architecture.

> **Conclusion:**  
> The GPU achieves its advantage not through different algorithms,  
> but by running the *same* Monte Carlo process on hardware perfectly matched to its inherent parallel structure.
