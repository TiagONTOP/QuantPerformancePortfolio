# 04\_gpu\_monte\_carlo — Monte Carlo Pricing Optimization with CuPy

This project is a quantitative-finance case study on performance optimization, demonstrating the **massive acceleration** achieved by porting a Monte Carlo (MC) pricing simulation from **CPU (NumPy)** to **GPU (CuPy)**.

The example focuses on pricing an **Asian option**, whose value depends on the simulated average of a **Geometric Brownian Motion (GBM)** process.

-----

## 🎯 Project Objective

The goal is to quantitatively compare two implementations of GBM path generation:

1.  **Baseline (`suboptimal/pricing.py`)** — a standard vectorized implementation using **NumPy**.
    Efficient on CPU, but limited by the inherently sequential (or lightly parallel) nature of CPU execution.

2.  **Optimized (`optimized/pricing.py`)** — a GPU-based implementation using **CuPy**, a drop-in replacement for NumPy that runs computations on NVIDIA GPUs.
    The two codes are almost identical semantically, illustrating a *drop-in optimization* approach that exploits massive GPU parallelism without rewriting core logic.

-----

## 🚀 Key Results — The "Speedup"

The acceleration from NumPy → CuPy is **massive**, especially for large simulation sizes and single-precision arithmetic.

For the "Large" problem (500,000 paths × 252 timesteps), based on verified benchmark results:

| Backend | Precision | Execution Time | **Speedup (vs CPU)** |
| :--- | :--- | :--- | :--- |
| **CPU (NumPy)** | `float32` | **4.785 s** | 1.0× (baseline) |
| **GPU (CuPy) — Standard** | `float32` | **0.349 s** | **13.7×** |
| **GPU (CuPy) — Zero-Copy** | `float32` | **0.114 s** | **42.0×** ⚡ |

### Precision Trade-Off — FP32 vs FP64

A crucial finding is the impact of numerical precision (`dtype`) on GPU performance:

  - **Single precision (`float32`)** — delivers **maximum performance**.
    On the test GPU, `float32` is **1.81× faster** than `float64` for the same problem.
    Ideal for Monte Carlo simulations where statistical noise dominates machine precision.

  - **Double precision (`float64`)** — still provides strong speedup (**7.2× vs CPU**).
    Use when high precision is required for validation or sensitivity analysis.

For Monte Carlo pricing, where statistical noise usually outweighs machine precision, **`float32` is the optimal choice**.

Full benchmark data are available in `tests/benchmark_results.txt`.

### ⚡ Zero-Copy GPU Pipeline — The Game Changer

**The killer feature**: Our **zero-copy GPU pipeline** keeps both simulation and pricing entirely on GPU, eliminating CPU-GPU memory transfers.

Using `device_output=True` with the backend-agnostic `price_asian_option()` function provides:

  - **3.06× additional speedup** over standard GPU pipeline (0.349s → 0.114s)
  - **Total speedup of 42.0× vs CPU baseline** (4.785s → 0.114s)
  - **156ms of transfer time completely eliminated**

This is the **true power** of GPU optimization: not just faster compute, but **zero-copy architecture**.

-----

## 📂 Project Structure

```
04_gpu_monte_carlo/
├── docs/
│   ├── BENCHMARKS.md                  # Detailed performance analysis
│   ├── README.md                      # Documentation landing page (this file)
│   ├── STRUCTURE.md                   # Technical implementation details
│   └── TESTS.md                       # Test suite documentation
├── optimized/
│   ├── __init__.py
│   └── pricing.py                     # Optimized GPU implementation (CuPy)
├── suboptimal/
│   ├── __init__.py
│   └── pricing.py                     # Baseline CPU implementation (NumPy)
├── tests/
│   ├── test_correctness.py            # ✅ ALL correctness tests (44 tests)
│   │                                  #    - GBM simulation tests (CPU + GPU)
│   │                                  #    - Asian option pricing tests
│   │                                  #    - Input validation tests
│   │                                  #    - Statistical parity tests
│   ├── test_benchmark.py              # ⚡ ALL performance benchmarks (14 benchmarks)
│   │                                  #    - Small/Medium/Large problem sizes
│   │                                  #    - CPU vs GPU comparisons
│   │                                  #    - Zero-copy pipeline benchmarks
│   │                                  #    - Memory transfer analysis
│   └── benchmark_results.txt          # 📈 Generated: performance benchmark results
├── poetry.lock
├── pyproject.toml                     # Poetry configuration and dependencies
└── utils.py                           # Shared utility functions (Asian option pricer)
```

-----

## 🛠️ Installation & Usage

This project uses [**Poetry**](https://python-poetry.org/) for dependency and environment management.

```bash
# 1. Navigate to the project directory
cd case_studies/04_gpu_monte_carlo

# 2. Install Poetry (if not already installed)
pip install poetry

# 3. Configure Poetry to create a local .venv
poetry config virtualenvs.in-project true

# 4. Install dependencies
poetry install --no-root

# 5. Activate the virtual environment
.venv\Scripts\activate
```

### Installing CuPy (for CUDA 11/12)

Choose the build matching your CUDA Toolkit:

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x (or if CUDA 12 fails)
pip install cupy-cuda11x
```

-----

## ▶️ Running Tests and Benchmarks

```bash
# Run ALL Correctness Tests (44 tests)
python -m pytest tests/test_correctness.py -v

# Run ALL Performance Benchmarks (14 benchmarks)
python -m pytest tests/test_benchmark.py -v -s
```

## 🖥️ Benchmark Environment

All benchmarks were run on the following hardware:

| Component | Specification |
| :--- | :--- |
| **CPU** | Intel Core i7-4770 (Haswell) OC @ 4.1 GHz |
| **Motherboard** | ASUS Z87 |
| **RAM** | 16 GB DDR3 @ 2400 MHz |
| **GPU** | NVIDIA GeForce GTX 980 Ti (OC) |
| | *Architecture:* Maxwell (2nd Gen) |
| | *Compute Capability:* 5.2 |

-----

> ⚡ **Summary:** By porting the pipeline to CuPy, the standard end-to-end simulation achieves a **13.7× speedup**. By further implementing a **zero-copy architecture** (`device_output=True`), the total acceleration reaches **42.0×** — a massive, quantifiable gain from strategic GPU optimization on mid-range hardware.