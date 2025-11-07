# Quant Performance Portfolio (Python / Rust / Polars)

> **"Your Python backtests are slow. Your trading system is slower. I make both fast."**

This repository is a collection of high-impact case studies demonstrating performance optimization in quantitative finance systems.  
It serves as a **technical proof-of-work** for my performance engineering services.

---

## 1. The Problem — “Slow Code Kills Your Alpha.”

In quantitative finance, slow, unoptimized Python code is not a technical inconvenience — it’s a **direct commercial cost**.

- **High Iteration Latency:** When a backtest takes **8 hours** instead of 5 minutes, your quants can’t iterate fast enough to innovate.  
- **Cloud Cost Explosion:** Inefficient computations on AWS/GCP waste budget that should fund research.  
- **Production Failures:** Naive prototypes (`pandas`, `numpy` scripts) pushed into production collapse under real market data loads.

---

## 2. The Solution — A Productized Optimization Service

I provide a structured, two-phase optimization service specifically designed for quant research and trading systems.

- **Phase 1 — The Audit:**  
  A scoped engagement (up to 5 days).  
  I profile the codebase (you give me), identify the top three bottlenecks, and deliver a **technical roadmap** for fixing them.

- **Phase 2 — The Optimization:**  
  Targeted rewriting of critical “hot paths.”  
  I migrate `pandas` → `polars`, `numpy` → `cupy`, or rewrite performance-critical Python loops in **Rust** (via PyO3 bindings).

---

## 3. The Proof — Case Studies

Each directory below is a self-contained, reproducible project.  
Every case study includes **correctness tests** (`test_correctness.py` to ensure numerical parity) and **performance benchmarks** (`test_benchmark.py`).

---

### 📈 [Case Study 01: Pandas vs. Polars](./case_studies/01_polars_vs_pandas/README.md)

- **Problem:** A naive `pandas` backtesting loop with $O(T \times W \times N)$ complexity (Time × Window × Assets).  
- **Solution:** Refactored into a vectorized $O(T \times N)$ implementation using a **Polars + NumPy hybrid** — Polars for Rust-based rolling operations, NumPy for matrix algebra.  
- **Result:** **~615× acceleration** on large datasets, with a **19% reduction in Python memory footprint** and strict numerical parity (`atol < 6e-8`).

---

### 🦀 [Case Study 02: HFT Orderbook in Rust](./case_studies/02_hft_orderbook_rust/hft_optimization/README.md)

- **Problem:** A `HashMap`-based L2 order book suffers from cache misses, heap allocations, and ~150 ns read latency.  
- **Solution:** Solution: Replaced with a Ring Buffer + Bitset architecture in Rust, using the integer price tick as a relative index (via a moving anchor) within the buffer, designed for full L1 cache residency (~34 KB).
- **Result:** **~5.35× faster updates**, **~177–546× faster reads** (sub-nanosecond latency), and **zero heap allocation** in the hot path.

---

### 🔬 [Case Study 03: FFT Autocorrelation](./case_studies/03_fft_autocorrelation/README.md)

- **Problem:** Problem: Beating SciPy's $O(n \log n)$ autocorrelation baseline by using adaptive algorithms and low-level memory management in Rust.
- **Solution:** Implemented an adaptive Rust algorithm that systematically outperforms the $O(n \log n)$ FFT baseline by selecting a faster direct $O(n \cdot k)$ method for small lags and eliminating memory overhead via zero-allocation buffer pools.  
- **Result:** The adaptive Rust implementation achieves consistent 2.6×–70× speedups over SciPy's FFT baseline while maintaining near bit-level numerical accuracy.

---

### 💻 [Case Study 04: GPU Monte Carlo](./case_studies/04_gpu_monte_carlo/README.md)

- **Problem:** Monte Carlo simulation for Asian option pricing is CPU-bound under NumPy.  
- **Solution:** Drop-in replacement of `numpy` with `cupy`, offloading all parallel computations to an NVIDIA GPU.  
- **Result:** By porting the pipeline to CuPy, the standard end-to-end simulation achieves a **13.7× speedup**. By further implementing a **zero-copy architecture** (`device_output=True`), the total acceleration reaches **42.0×**

---

## 4. How to Verify These Results

This portfolio is composed of **independent, isolated projects**.  
Each case study (`case_studies/`) contains its own environment, dependencies (`pyproject.toml` or `Cargo.toml`), and test suite.

To validate any result, you must navigate to its directory and follow its local instructions.

---

### Example: Verifying the “01_polars_vs_pandas” Study

```bash
cd case_studies/01_polars_vs_pandas
# pip install poetry
poetry config virtualenvs.in-project true
poetry install

# Activate environment
.venv/Scripts/activate    # Windows
source .venv/bin/activate # Linux / macOS
### Commands
# Fast tests (recommended)
python -m pytest -v -m "not benchmark"

# Only correctness
python -m pytest tests/test_correctness.py -v

# Only edge cases
python -m pytest tests/test_edge_cases.py -v

# Full benchmark (~2 min)
python -m pytest tests/test_benchmark.py -v -s
```
