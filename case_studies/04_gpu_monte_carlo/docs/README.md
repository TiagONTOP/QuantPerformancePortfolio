# 04_gpu_monte_carlo — Monte Carlo Pricing Optimization with CuPy

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

## 🚀 Key Results — The “Speedup”

The acceleration from NumPy → CuPy is significant, especially for large simulation sizes and single-precision arithmetic.

For the “Large” problem (100,000 paths × 252 timesteps), based on the methodologically-sound benchmark results:

| Backend | Precision | Execution Time | **Speedup (vs CPU float32)** |
| :--- | :--- | :--- | :--- |
| **CPU (NumPy)** | `float32` | **0.482 s** | 1.0× |
| **GPU (CuPy)** | `float32` | **0.078 s** | **6.20×** |

### Precision Trade-Off — FP32 vs FP64

A crucial finding is the impact of numerical precision (`dtype`) on performance:

  - **Single precision (`float32`)** — delivers the **maximum speedup (6.20×)**.
    Ideal for most consumer GPUs where FP32 units dominate.

  - **Double precision (`float64`)** — provides smaller gains.
    On the test GPU (NVIDIA GTX 980 Ti, Maxwell), the speedup for the same "Large" problem was only **3.30x**, and for the "Very Large" problem, the GPU was **0.83x** (slower than the CPU). This confirms that FP64 throughput is a significant bottleneck on this hardware.

For Monte Carlo pricing, where statistical noise usually outweighs machine precision, **`float32` is almost always optimal**.

Full benchmark data are available in [`docs/BENCHMARKS.md`](https://www.google.com/search?q=./docs/BENCHMARKS.md) and `tests/performance_report.txt`.

### ✨ Zero-Copy GPU Pipeline

**New capability**: The project now supports a **zero-copy GPU pipeline** where both simulation and pricing run entirely on GPU, eliminating CPU-GPU memory transfers.

Using `device_output=True` with the backend-agnostic `price_asian_option()` function enables an additional **1.2-2.0× speedup** over the standard pipeline, bringing total acceleration to **~7.5-12.5× vs CPU**.

See [`docs/CORRECTIONS_APPLIED.md`](https://www.google.com/search?q=./docs/CORRECTIONS_APPLIED.md) for technical details.

-----

## 📂 Project Structure

```

04_gpu_monte_carlo/                   
├── docs/
│   ├── BENCHMARKS.md                  # Detailed performance analysis
│   ├── README.md                      # Documentation landing page
│   ├── STRUCTURE.md                   # Technical implementation details
│   └── TESTS.md                       # Test documentation
├── optimized/
│   ├── __pycache__/
│   ├── __init__.py
│   └── pricing.py                     # Optimized GPU implementation (CuPy)
├── suboptimal/
│   ├── __pycache__/
│   ├── __init__.py
│   └── pricing.py                     # Baseline CPU implementation (NumPy)
├── tests/
│   ├── __pycache__/
│   ├── benchmark_results.txt          # Aggregated benchmark results
│   ├── generate_performance_report.py # Generates detailed performance report
│   ├── performance_report.txt         # Exported performance summary
│   ├── run_all_tests_and_benchmarks.py# Runs all tests and benchmarks
│   ├── test_asian_option_benchmark_zero_copy.py # Zero-copy pipeline benchmark
│   ├── test_asian_option_benchmark.py # Asian option pricing benchmarks
│   ├── test_asian_option_correctness.py # Asian option pricing correctness
│   ├── test_benchmark_gpu.py          # GPU benchmark suite
│   ├── test_correctness_gpu.py        # GPU vs CPU numerical parity tests
│   ├── test_correctness.py            # Generic correctness tests
│   └── test_results.txt               # Consolidated test output logs
├── poetry.lock
├── pyproject.toml                     # Poetry configuration and dependencies
└── utils.py                           # Shared utility functions
````

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
````

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

### Quick Validation

Validate that all corrections are working correctly:

```bash
# Validate backend-agnostic pricing and zero-copy pipeline
python scripts/validate_fixes.py
```

This script tests:

  - Backend detection (NumPy/CuPy)
  - CPU pricer with NumPy arrays
  - GPU pricer with CuPy arrays (zero-copy)
  - Pipeline consistency
  - Call/Put option types

### Comprehensive Testing

Two main scripts are provided in the `tests/` directory:

```bash
# 1. Generate a detailed performance report with CPU/GPU comparisons
#    Output: tests/performance_report.txt
python tests/generate_performance_report.py

# 2. Run ALL unit tests and benchmarks via pytest
#    Output: tests/test_results.txt (correctness tests)
#            tests/benchmark_results.txt (performance benchmarks)
python tests/run_all_tests_and_benchmarks.py
```

**Note:** These scripts now generate **separate reports**:

  - `performance_report.txt` — Detailed performance metrics with CPU/GPU comparisons using identical random seeds
  - `test_results.txt` — Correctness test results (GPU correctness, general correctness, Asian option correctness)
  - `benchmark_results.txt` — Benchmark test results (GPU benchmarks, Asian benchmarks, zero-copy benchmarks)

### Zero-Copy Pipeline Benchmark

To measure the performance gain of the zero-copy GPU pipeline:

```bash
# Run comprehensive zero-copy benchmark suite
python tests/test_asian_option_benchmark_zero_copy.py
```

This demonstrates the additional speedup achieved by keeping all data on GPU.

-----

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

> ⚡ **Summary:** By swapping NumPy for CuPy with almost no code changes, Monte Carlo pricing achieves a **6.2× GPU acceleration** on mid-range hardware — a striking illustration of how quantitative-finance simulations can benefit from GPU parallelism.

```