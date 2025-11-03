# Implementation Analysis: HFT L2 Order Book Optimization

## 1. Introduction

This document presents a structural analysis of two implementations of an L2 order book (`L2Book`):

1. **Baseline (`suboptimal`)** — a naïve implementation using Rust’s standard `HashMap`.
2. **Optimized (`optimized`)** — a cache-aware implementation designed for L1 residency, using a *ring buffer* and *bitsets*.

The goal is to dissect the architecture, design decisions, and trade-offs that explain the **177×–546× read latency reduction** and **5.3× update speedup** measured in [BENCHMARKS.md](./BENCHMARKS.md).

The key insight is that the performance gap stems not only from cache locality, but from eliminating a **hidden O(N) algorithmic flaw** in the baseline.

---

## 2. Baseline Implementation (`suboptimal`)

**File:** `src/suboptimal/book.rs`

### 2.1. Architecture

The baseline implementation maps price levels (ticks) to quantities via a `HashMap`:

```rust
pub struct L2Book {
    pub seq: u64,
    pub tick_size: f64,
    pub lot_size: f64,
    pub bids: HashMap<Price, Qty>, // price_tick -> size
    pub asks: HashMap<Price, Qty>,
}
````

### 2.2. Main Bottlenecks

At first glance:

* **Memory Allocations:** Every `insert` can trigger unpredictable heap reallocations.
* **Cache Misses:** `HashMap` data is scattered across memory, forcing frequent cache misses.

However, profiling revealed a **much deeper algorithmic flaw**.

### 2.3. Hidden O(N) Bottleneck

The critical update path is not amortized O(1) as expected.

1. `L2Book::update()` is called per message.
2. It calls `self.verify_checksum()`.
3. That in turn calls `self.best_bid()` and `self.best_ask()`.
4. `best_bid()` performs a **full linear scan**:

```rust
#[inline]
pub fn best_bid(&self) -> Option<(Price, Qty)> {
    self.bids.iter().max_by_key(|(p, _)| *p).map(|(p, q)| (*p, *q)) // <-- O(N)
}
```

**Result:** Every incoming message triggers an O(N) scan of both maps.
Benchmarks confirm perfect linear scaling: 751 ns at N = 5 → 2.65 µs at N = 50.

### 2.4. Real Complexity

| Function      | Complexity     | Explanation                                     |
| :------------ | :------------- | :---------------------------------------------- |
| `update()`    | **O(N)**       | due to `verify_checksum()` calling `best_bid()` |
| `best_bid()`  | **O(N)**       | full iteration                                  |
| `top_bids(n)` | **O(N log N)** | collect + sort                                  |
| `bid_depth()` | **O(1)**       | length query only                               |

---

## 3. Optimized Implementation (`optimized`)

**File:** `src/optimized/book.rs`

A full redesign delivering O(1) operations and cache-level determinism.

### 3.1. Architecture: Hot/Cold Separation and Alignment

```rust
#[repr(align(64))] // 64B cache-line alignment
#[derive(Clone)]
struct HotData {
    qty: Box<[f32; CAP]>,              // CAP = 4096
    occupied: Box<[u64; BITSET_SIZE]>, // BITSET_SIZE = 64
    head: usize,
    anchor: i64,
    best_rel: usize, // key to O(1) performance
}

#[derive(Clone, Serialize, Deserialize)]
struct ColdData {
    seq: u64,
    tick_size: f64,
    lot_size: f64,
    initialized: bool,
}

pub struct L2Book {
    #[serde(skip)] bids: HotData,
    #[serde(skip)] asks: HotData,
    #[serde(flatten)] cold: ColdData,
}
```

The 64-byte alignment avoids false sharing and ensures `bids` and `asks` begin on separate cache lines.

### 3.2. Mechanism 1: Moving-Anchor Ring Buffer

The book uses a **fixed-size circular buffer** (`CAP = 4096`) representing a sliding window on the price axis.

* `anchor` → reference tick where `rel = 0`
* `rel` → relative tick offset from anchor
* `head` → physical index of anchor within `qty`
* **O(1) Mapping:**

```rust
#[inline(always)]
fn rel_to_phys(&self, rel: usize) -> usize {
    (self.head + rel) & CAP_MASK // fast modulo, CAP = power of two
}
```

### 3.3. Mechanism 2: Constant-Time Best-Price Tracking

The `best_rel` field caches the relative index of the best price.

* Updated in O(1) at each `set_qty()` call.
* Read instantly by `best_bid()`:

```rust
#[inline(always)]
pub fn best_bid(&self) -> Option<(Price, Qty)> {
    if self.bids.best_rel == usize::MAX { None } else {
        let price = self.bid_rel_to_price(self.bids.best_rel);
        let qty = self.bids.get_qty(self.bids.best_rel);
        Some((price, qty as Qty))
    }
}
```

Result: **831 ps latency** — equivalent to a single L1 cache load.

Fallback via bitset scanning (64×u64 = 4096 bits) is still constant O(1).

### 3.4. Mechanism 3: L1 Cache Budgeting

* **`qty` arrays:** 2 × (4096 × 4 B) = **32 KiB**
* **`occupied` bitsets:** 2 × 512 B = **1 KiB**
* **Metadata:** ≈ 48 B

**Total hot set:** ≈ 33 KiB → perfectly fits within the 32 KiB L1d cache (slight spill absorbed by L2).
The choice of `f32` over `f64` was **crucial** for this fit.

### 3.5. Mechanism 4: Recentering Logic

`recenter()` (marked `#[cold]`) slides the buffer window when price moves beyond 4096 ticks:

1. Compute new anchor
2. Shift head pointer
3. Wipe newly exposed segment

Small shifts amortize to O(1); large displacements (> 2048 ticks) trigger a full reset.

---

## 4. Architectural Comparison

### 4.1. Complexity Summary

| Operation     | Baseline (`HashMap`) | Optimized (`Ring Buffer`) | Explanation                            |
| :------------ | :------------------- | :------------------------ | :------------------------------------- |
| `update()`    | **O(N)**             | **O(1)** (amortized)      | Replaces O(N) scan with cached pointer |
| `best_bid()`  | **O(N)**             | **O(1)**                  | Reads `best_rel` directly              |
| `top_bids(n)` | **O(N log N)**       | **O(CAP)**                | Linear scan over fixed 4096 slots      |
| `bid_depth()` | **O(1)**             | **O(CAP/64)**             | Popcount on fixed bitset               |

### 4.2. Design Trade-offs

| Aspect             | Baseline (`HashMap`) | Optimized (`Ring Buffer`)   |
| :----------------- | :------------------- | :-------------------------- |
| Price range        | Unlimited            | Fixed (4096 ticks)          |
| Quantity precision | `f64` (64-bit)       | `f32` (32-bit, L1-friendly) |
| Memory usage       | Variable (∝ N)       | Fixed (~33 KiB)             |
| Code size          | ≈ 185 LOC            | ≈ 740 LOC                   |

---

## 5. Code Quality and Safety

* **Unit Tests:** 13 tests in `optimized/book.rs` (recentering, overflow, NaN sanitization) vs 3 in baseline.
* **Input Validation:** `set_bid_level()` and `set_ask_level()` handle `NaN`, `Inf`, and negative/zero quantities safely.
* **Compile-time Assertions:** Constants verify `CAP` is power-of-two and divisible by 64, with zero runtime cost.

---

## 6. Hardware Context

All benchmarks were executed on the following configuration:

| Component       | Specification                                       |
| :-------------- | :-------------------------------------------------- |
| **CPU**         | Intel Core i7-4770 (Haswell) overclocked to 4.1 GHz |
| **Motherboard** | ASUS Z87 series                                     |
| **RAM**         | 16 GB DDR3-2400 MHz                                 |
| **GPU**         | NVIDIA GTX 980 Ti (OC)                              |
| **OS**          | Windows 10 (64-bit)                                 |

This setup provides a realistic single-core latency profile for mid-2010s hardware — ideal for demonstrating micro-architectural gains without GPU or NUMA effects.

---

## 7. Conclusion

The `suboptimal` implementation is not merely inefficient — it is **algorithmically unsuitable** for HFT, with an O(N) update cost.

The `optimized` design introduces O(1) state tracking (`best_rel`) and maximizes **mechanical sympathy**:
a 33 KiB `HotData` structure intentionally sized to fit within a 32 KiB L1 cache.

The result is a system where **read latencies are measured in picoseconds, not nanoseconds**, showcasing the synergy between **algorithmic minimalism and hardware-level optimization**.

```
