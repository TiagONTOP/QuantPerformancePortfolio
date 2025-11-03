# 04_gpu_monte_carlo — Monte Carlo Pricing Optimization with CuPy

This project is a quantitative-finance case study on performance optimization, demonstrating the **massive acceleration** achieved by porting a Monte Carlo (MC) pricing simulation from **CPU (NumPy)** to **GPU (CuPy)**.

The example focuses on pricing an **Asian option**, whose value depends on the simulated average of a **Geometric Brownian Motion (GBM)** process.

---

## 🎯 Project Objective

The goal is to quantitatively compare two implementations of GBM path generation:

1. **Baseline (`suboptimal/pricing.py`)** — a standard vectorized implementation using **NumPy**.  
   Efficient on CPU, but limited by the inherently sequential (or lightly parallel) nature of CPU execution.

2. **Optimized (`optimized/pricing.py`)** — a GPU-based implementation using **CuPy**, a drop-in replacement for NumPy that runs computations on NVIDIA GPUs.  
   The two codes are almost identical semantically, illustrating a *drop-in optimization* approach that exploits massive GPU parallelism without rewriting core logic.

---

## 🚀 Key Results — The “Speedup”

The acceleration from NumPy → CuPy is dramatic, especially for large simulation sizes and single-precision arithmetic.

For the “Large” problem (100 000 paths × 252 timesteps):

| Backend | Precision | Execution Time | **Speedup (vs CPU float32)** |
| :--- | :--- | :--- | :--- |
| **CPU (NumPy)** | `float32` | ≈ 0.985 s | 1.0× |
| **GPU (CuPy)** | `float32` | **≈ 0.060 s** | **≈ 16.4×** |

### Precision Trade-Off — FP32 vs FP64

A crucial finding is the impact of numerical precision (`dtype`) on performance:

- **Single precision (`float32`)** — delivers the **maximum speedup (~16.4×)**.  
  Ideal for most consumer GPUs where FP32 units dominate.

- **Double precision (`float64`)** — provides smaller gains (~1×–6×).  
  On the test GPU (NVIDIA GTX 980 Ti, Maxwell), FP64 throughput is only 1⁄32 of FP32.

For Monte Carlo pricing, where statistical noise usually outweighs machine precision, **`float32` is almost always optimal**.

Full benchmark data are available in [`BENCHMARKS.md`](./BENCHMARKS.md) and `tests/performance_report.txt`.

---

## 📂 Project Structure

```

/
├── optimized/
│   └── pricing.py          # Optimized GPU implementation (CuPy)
├── suboptimal/
│   └── pricing.py          # Baseline CPU implementation (NumPy)
├── tests/
│   ├── test_correctness.py          # Numerical-parity unit tests
│   ├── test_asian_option_*.py       # Asian-option-specific tests
│   ├── test_benchmark_*.py          # Pytest-benchmark scripts
│   ├── generate_performance_report.py  # Produces detailed timing report
│   └── run_all_tests_and_benchmarks.py # Runs all tests + benchmarks
├── pyproject.toml          # Poetry + dependency configuration
├── README.md               # High-level overview (this file)
├── STRUCTURE.md            # Technical implementation details
├── TESTS.md                # Unit-test documentation
└── BENCHMARKS.md           # Detailed performance analysis

````

---

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

---

## ▶️ Running Tests and Benchmarks

Two main scripts are provided in the `tests/` directory:

```bash
# 1. Generate a detailed performance report (tests/performance_report.txt)
python tests/generate_performance_report.py

# 2. Run ALL unit tests and benchmarks via pytest
# Results are shown in-console and saved to tests/benchmark_results.txt
python tests/run_all_tests_and_benchmarks.py
```

---

## 🖥️ Benchmark Environment

All benchmarks were run on the following hardware:

| Component       | Specification                             |
| :-------------- | :---------------------------------------- |
| **CPU**         | Intel Core i7-4770 (Haswell) OC @ 4.1 GHz |
| **Motherboard** | ASUS Z87                                  |
| **RAM**         | 16 GB DDR3 @ 2400 MHz                     |
| **GPU**         | NVIDIA GeForce GTX 980 Ti (OC)            |
|                 | *Architecture:* Maxwell (2nd Gen)         |
|                 | *Compute Capability:* 5.2                 |

---

> ⚡ **Summary:** By swapping NumPy for CuPy with almost no code changes, Monte Carlo pricing achieves a **16× GPU acceleration** on mid-range hardware — a striking illustration of how quantitative-finance simulations can benefit from GPU parallelism.