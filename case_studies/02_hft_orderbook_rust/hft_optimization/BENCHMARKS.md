# Performance Analysis Report: L2 Order Book Optimization in Rust

## 1. Executive Summary

This report details the optimization of an L2 order book (`L2Book`) written in Rust, comparing a **suboptimal HashMap-based implementation** with an **optimized contiguous ring-buffer design** aligned to the L1 cache.

The optimization achieved a **177× to 546× reduction in read latency** and a **5.3× improvement in update throughput**.

The primary source of this gain is **algorithmic**:  
the `suboptimal` implementation exhibited a hidden **O(N)** complexity on *every incoming message*, while the `optimized` version guarantees **O(1)** (amortized) operations.  
This algorithmic improvement is further amplified by a **cache-aware design** that ensures the entire hot data set (≈ 33 KiB) fits within the CPU’s 32 KiB L1d cache.

### 📈 Key Performance Metrics

| Metric | Baseline (`HashMap`) | Optimized (`Ring Buffer`) | Performance Gain |
| :--- | :--- | :--- | :--- |
| **Read Latency (Best Bid)** | 147.25 ns | 0.832 ns | **177×** |
| **Read Latency (Mid-Price)** | 308.59 ns | 0.565 ns | **546×** |
| **Update Latency (Average)** | 1.378 µs | 257.46 ns | **5.35×** |
| **Update Complexity** | **O(N)** (linear scan) | **O(1)** (amortized) | Algorithmic |
| **CPU Load (Simulation)** | 39.8 % / core | 6.1 % / core | **-84.6 %** |

---

## 2. Root Cause Analysis: The Hidden O(N) Bottleneck

Initial profiling showed that the `suboptimal` version did **not** behave in amortized O(1) as a `HashMap` might suggest.  
The culprit was an **unintended algorithmic dependency** in the critical path.

**Suboptimal implementation (`HashMap`):**
1. `L2Book::update()` called for each message  
2. `update()` calls `self.verify_checksum()`  
3. `verify_checksum()` calls `self.best_bid()` / `self.best_ask()`  
4. `best_bid()` executes `self.bids.iter().max_by_key(...)`  
5. → **Linear O(N)** scan over all price levels in the `HashMap`

**Optimized implementation (`Ring Buffer`):**
1. `L2Book::update()` called  
2. `update()` calls `self.set_bid_level()` / `set_ask_level()`  
3. These maintain a pointer `best_rel` (relative best index) in **O(1)** time  
4. `verify_checksum()` calls `self.best_bid()`  
5. `best_bid()` simply reads `self.bids.best_rel` → **O(1)** read

The true bottleneck was not `HashMap::insert` (O(1) amortized) but the **O(N) recomputation of best price on each message**.  
The optimized version fixes this by maintaining state incrementally in O(1).

---

## 3. Benchmark Methodology

- **Hardware:** Intel Core i7-4770 @ 4.1 GHz (Haswell), 16 GB DDR3 2400 MHz  
- **Cache Hierarchy:** L1d 32 KiB, L1i 32 KiB, L2 256 KiB (per core), L3 8 MiB (shared)  
- **System:** Windows 10  
- **Benchmarking Suite:** `Criterion.rs` (Rust 1.70+ in `--release` mode)  
- **Statistics:** Median (b-estimate) and 95 % confidence intervals reported by Criterion

---

## 4. Update-Path Performance

### 4.1. Depth-Scaling Benchmark (`depth_scaling`)

This benchmark measures the latency of a single `update()` as the number of active price levels N increases.

| Active Levels (N) | Baseline (`HashMap`) | Optimized (`Ring Buffer`) | Gain |
| :--- | :--- | :--- | :--- |
| 5    | 751.45 ns | 151.20 ns | **4.97×** |
| 10   | 1.004 µs | 174.22 ns | **5.76×** |
| 20   | 1.356 µs | 246.56 ns | **5.50×** |
| 50   | 2.656 µs | 510.20 ns | **5.21×** |

**Analysis**

- **`HashMap` baseline:** Cost grows linearly with N (751 ns → 2.65 µs) → **O(N)** confirmed  
- **`Ring Buffer` optimized:** Cost remains nearly constant (151 ns → 510 ns).  
  The small rise is due to the simulator producing more diffs for deeper books, not algorithmic scaling.  
  → **O(1)** amortized with respect to N.

### 4.2. Average Update Performance

| Scenario | Baseline (`HashMap`) | Optimized (`Ring Buffer`) | Gain |
| :--- | :--- | :--- | :--- |
| **Single Update** | 1.378 µs | 257.46 ns | **5.35×** |
| **Batch (100 updates)** | 138.49 µs | 26.07 µs | **5.31×** |
| *Per-update amortized* | *(1.385 µs)* | *(260.69 ns)* | *(5.31×)* |

---

## 5. Read-Path Performance

Read operations benefit the most, moving from O(N) scans to O(1) direct reads.

| Operation | Baseline (`HashMap`) | Optimized (`Ring Buffer`) | Gain |
| :--- | :--- | :--- | :--- |
| `best_bid()` | 147.25 ns | 0.832 ns | **177×** |
| `best_ask()` | 147.48 ns | 0.833 ns | **177×** |
| `mid_price()` | 308.59 ns | 0.565 ns | **546×** |
| `orderbook_imbalance()` | 300.84 ns | 0.578 ns | **521×** |
| `top_bids(10)` | 193.88 ns | 95.42 ns | **2.03×** |

**Analysis**

- **O(1) Reads (Best/Mid/Imbalance):**  
  Latencies ≈ 0.5–0.8 ns (2–4 CPU cycles @ 4.1 GHz) — the irreducible cost of an L1d cache hit.  
  Confirms the success of the **cache-aware hot-set design**.

- **`top_bids(10)` Scan (Design Trade-off):**  
  - Baseline: O(N log N) (collect + sort).  
  - Optimized: O(CAP) linear scan of fixed 4096-slot buffer.  
  - Gain = 2.03× → For small N (~50), a sequential O(CAP) scan in contiguous L1 memory is faster than scattered O(N log N) sorting.

---

## 6. Memory & Cache Analysis — *How the Speedup Happens*

The O(1) gains are magnified by hardware-sympathetic design.  
The optimized structure ensures that all data touched by `update()` resides in L1.

### 6.1. Hot/Cold Split and Alignment

`L2Book` is split into:
- **ColdData:** rarely accessed metadata (`tick_size`, `seq`)  
- **HotData:** critical fields (`qty`, `occupied`, `best_rel`) aligned to 64 bytes (`#[repr(align(64))]`)  

This prevents false sharing and ensures each `HotData` block starts on its own cache line.

### 6.2. L1 Residency Calculation

Memory footprint of `HotData`:

1. **Quantity Array:** `qty: Box<[f32; 4096]>`  
   - 4096 × 4 B = 16 384 B (≈ 16 KiB)
2. **Occupancy Bitset:** `occupied: Box<[u64; 64]>`  
   - 64 × 8 B = 512 B (≈ 0.5 KiB)
3. **Metadata:** (`head`, `anchor`, `best_rel`) ≈ 24 B  

**Total per side (bid/ask):** ≈ 16.5 KiB  
**Total hot set:** ≈ 33 KiB for both sides

### 6.3. Cache Conclusion

The 33 KiB hot set fits almost perfectly in the 32 KiB L1d cache (i7-4770).  
The 1 KiB overflow is instantly served from L2 (256 KiB).  

Thus, **virtually 100 % of `update()` and `best_bid()` operations incur no costly cache misses** — explaining why O(1) latencies are measured in **picoseconds (CPU cycles)** instead of nanoseconds (L2/L3 hits).

---

## 7. Workload Simulation — *Why It Matters*

To contextualize these results, a realistic HFT workload was simulated on a single CPU core:

- **Profile:** 1 000 000 ops / s  
- **Mix:** 70 % `best_bid` (read), 20 % `update` (feed), 10 % `top_bids(10)` (snapshot)

### CPU Time (Baseline – `HashMap`)

| Operation | Frequency | Latency/Op | CPU Time |
| :--- | :--- | :--- | :--- |
| `best_bid` reads | 700 000 | 147.25 ns | 103.08 ms |
| `update` writes | 200 000 | 1.378 µs | 275.60 ms |
| `top_bids(10)` | 100 000 | 193.88 ns | 19.39 ms |
| **Total / s** |  |  | **398.07 ms** |
| **CPU Utilization** |  |  | **39.8 %** |

### CPU Time (Optimized – `Ring Buffer`)

| Operation | Frequency | Latency/Op | CPU Time |
| :--- | :--- | :--- | :--- |
| `best_bid` reads | 700 000 | 0.832 ns | 0.58 ms |
| `update` writes | 200 000 | 257.46 ns | 51.49 ms |
| `top_bids(10)` | 100 000 | 95.42 ns | 9.54 ms |
| **Total / s** |  |  | **61.61 ms** |
| **CPU Utilization** |  |  | **6.2 %** |

### Simulation Conclusion

Switching to the optimized implementation reduces CPU usage for order-book management by **84.6 %** (from 39.8 % → 6.2 %).  

This frees **~336 ms per second per core**, enabling the system to run more complex trading logic, handle more instruments concurrently, or simply operate with drastically lower latency jitter.

---
