# Case Study 02: HFT L2 Orderbook Optimization

## Executive Summary

This case study demonstrates extreme performance optimization of a Level 2 (L2) orderbook for High-Frequency Trading (HFT) systems. By replacing standard `HashMap` data structures with cache-optimized ring buffers and bitsets, we achieve **sub-nanosecond read latencies** and **5.5x faster updates**.

**Key Results:**

  - **5.5x faster updates**: 242 ns vs. 1.338 µs (`HashMap` baseline)
  - **175-560x faster reads**: 0.53-0.90 ns vs. 147-310 ns for best bid/ask queries
  - **L1 Cache-Budgeted**: Entire \~34 KiB hot data set fits within L1D cache (32-64 KiB)
  - **Zero-Allocation Hot Path**: Predictable latency with no allocator jitter

## Problem Statement

In High-Frequency Trading, orderbook operations are executed millions of times per second. Market makers, arbitrageurs, and HFT firms require:

  - **Sub-nanosecond** best bid/ask (BBO) queries (critical for decision latency)
  - **Sub-microsecond** price level updates (to maintain orderbook state)
  - **Predictable latency** (no allocator calls, no hash collisions, no cache misses)
  - **100% correctness** (any data corruption can lead to millions in losses)

Standard data structures (`HashMap`, `BTreeMap`) fail to meet these requirements due to:

  - Heap allocations during updates
  - Cache misses from scattered memory access
  - Hash computation overhead
  - Unpredictable performance (hash collisions, tree rebalancing)

## Solution Approach

Replace the `HashMap`-based orderbook with an ultra-optimized Rust implementation:

1.  **Hot/Cold Data Split**: Isolate data accessed every *tick* (quantities, bitset) from "cold" data (metadata, `tick_size`).
2.  **Dynamic Ring Buffer**: A fixed-capacity array (`CAP=4096`) indexed by a price relative to a movable **anchor** (`anchor`). The anchor is re-centered (`recenter`) when the price drifts, keeping writes O(1) in the vast majority of cases.
3.  **Bitset Tracking**: A `[u64; 64]` bitset tracks occupied price levels to find the best price.
4.  **Cached BBO**: The *best price* (`best_rel`) is stored and updated, enabling O(1) BBO queries (a simple memory read).
5.  **Zero-Allocation**: All memory is pre-allocated; no allocations occur in the critical hot path (`update`, `best_bid`).
6.  **Memory Alignment**: The `HotData` struct is 64-byte aligned to prevent cache-line false sharing.

## Technologies Used

### Core Implementation

  - **Rust (1.70+)**: Systems programming language with zero-cost abstractions
  - **Ring Buffer**: Manually implemented with `Box<[f32; CAP]>`
  - **Bitset**: `Box<[u64; BITSET_SIZE]>` for fast bitwise operations

### Testing & Benchmarking

  - **Rust Unit Tests**: `cargo test`
  - **Criterion**: Statistical benchmarking
  - **`adler`**: For (optimized) checksum verification

### Development Tools

  - **Cargo**: Rust package manager and build system
  - **rustc**: Rust compiler

## Repository Structure

```
02_hft_orderbook_rust/
  hft_optimization/
    benches/
      optimized_vs_suboptimal.rs
      orderbook_update.rs
    scripts/
    src/
      bin/
      common/
        messages.rs
        mod.rs
        types.rs
      optimized/
        book.rs
        mod.rs
      suboptimal/
        book.rs
        messages.rs
        mod.rs
        simulator.rs
        types.rs
      lib.rs
      main.rs
```

## Quick Start

### Prerequisites

Install the Rust toolchain (1.70 or higher):

```bash
# Install rustup (Rust installer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
rustc --version
cargo --version
```

### Build & Test

```bash
# Navigate to project directory
cd case_studies/02_hft_orderbook_rust/hft_optimization

# Build in release mode (required for accurate performance)
cargo build --release

# Run all unit tests
cargo test
```

### Run Benchmarks

```bash
# Run all benchmarks (may take several minutes)
cargo bench

# Run a specific benchmark suite
cargo bench --bench optimized_vs_suboptimal

# View benchmark results
# Open in browser: target/criterion/report/index.html
```

## Usage Examples

### Basic Orderbook Operations

```rust
use hft_optimisation::optimized::book::L2Book;
use hft_optimisation::common::types::{Price, Qty, Side};
use hft_optimisation::common::messages::{L2UpdateMsg, L2Diff, MsgType};

// Create an orderbook (tick_size, lot_size)
// Capacity is fixed at 4096 levels per side (L1-optimized)
let mut book = L2Book::new(0.1, 0.001);

// Apply an update message (snapshot or diff)
let msg = L2UpdateMsg {
    msg_type: MsgType::L2Update,
    symbol: "BTC-USDT".to_string(),
    ts: 123456789,
    seq: 1,
    diffs: vec![
        L2Diff {
            side: Side::Bid,
            price_tick: 100_000, // Price in ticks
            size: 10.0,         // Quantity
        },
        L2Diff {
            side: Side::Ask,
            price_tick: 100_010,
            size: 5.0,
        },
    ],
    checksum: 0, // Note: checksum is verified inside update()
};

// Update the book
// Checksum verification happens here (if not disabled)
book.update(&msg, "BTC-USDT");

// Query best bid/ask (O(1) read, sub-nanosecond)
// Returns Option<(Price, Qty)>
if let Some((price, qty)) = book.best_bid() {
    println!("Best bid: {} @ {}", price, qty);
}

if let Some((price, qty)) = book.best_ask() {
    println!("Best ask: {} @ {}", price, qty);
}

// Query mid price and spread
if let Some(mid) = book.mid_price() {
    println!("Mid Price: {}", mid);
}
if let Some(spread) = book.spread_ticks() {
    println!("Spread: {} ticks", spread);
}
```

### Top-of-Book Depth Queries

```rust
// Get best 5 levels on each side
// Note: This scans the bitset locally, O(N) where N=depth
let top_bids = book.top_bids(5);
let top_asks = book.top_asks(5);

println!("--- Top 5 Bids ---");
for (price, qty) in top_bids {
    println!("Bid: {} @ {}", price, qty);
}
```

### Orderbook Metrics

```rust
// Compute orderbook imbalance over top 5 levels
if let Some(imbalance) = book.orderbook_imbalance_depth(5) {
    println!("Imbalance (5 levels): {:.2}%", imbalance * 100.0);
}

// Get total number of price levels (O(1) via bitset popcount)
let bid_levels = book.bid_depth();
let ask_levels = book.ask_depth();
println!("Total depth: {} bids / {} asks", bid_levels, ask_levels);
```

## Performance Highlights

### Latency Comparison

| Operation | `HashMap` (Baseline) | `Ring Buffer` (Optimized) | Speedup |
|---|---|---|---|
| Best bid query | 148 ns | **0.845 ns** | **≈175x** |
| Best ask query | 151 ns | **0.831 ns** | **≈181x** |
| Mid price query | 310 ns | **0.585 ns** | **≈530x** |
| Update (1 level) | 1.338 µs | **242 ns** | **≈5.5x** |
| Update (100 levels)| 151 µs | **26.3 µs** | **≈5.7x** |
| Top 10 bids | 196 ns | **90.3 ns** | **≈2.2x** |

### Memory Footprint (Hot Data)

| Component | Size (per side) | Total (Bids + Asks) | Location |
|---|---|---|---|
| Quantities buffer (`[f32; 4096]`) | 16 KiB | 32 KiB | L1 Cache |
| Bitset (`[u64; 64]`) | 512 Bytes | 1 KiB | L1 Cache |
| Metadata (anchor, head, best\_rel) | \~24 Bytes | \~48 Bytes | Registers/L1 |
| **Total (Hot Data)** | **\~16.5 KiB** | **\~33-34 KiB** | **Fits in L1D** |

*Reference:*

  - L1D Cache (Modern Intel/AMD): 32-64 KiB per core
  - L1 Cache Latency: \~1 ns
  - RAM Latency: \~60-100 ns

## Key Optimization Techniques

### 1\. Hot/Cold Data Split

**Problem**: Infrequently used metadata (like `tick_size` or `seq`) pollutes cache lines that hold critical data (quantities).

**Solution**: The `L2Book` contains two `HotData` structs (bids/asks) and one `ColdData` struct.

  - `HotData`: Marked `#[repr(align(64))]`. Contains `qty`, `occupied`, `head`, `anchor`, `best_rel`. This data is read/written on every *tick*.
  - `ColdData`: Contains `seq`, `tick_size`, `lot_size`, `initialized`. Accessed rarely.

**Benefit**: Ensures that a 64-byte cache line contains *only* critical data, maximizing spatial and temporal locality.

### 2\. Anchor-Based Dynamic Ring Buffer

**Problem**: A static `Vec[price]` is unusable if the price is (e.g.) 50,000. A `HashMap` is slow.

**Solution**: A ring buffer of `CAP=4096` levels.

  - `anchor`: A reference price (in ticks) that corresponds to logical index `rel=0`.
  - `head`: The *physical* index in the `qty` array that corresponds to `rel=0`.
  - `rel_to_phys(rel)`: `(self.head + rel) & 4095` (fast modulo via bitmask).
  - `price_to_rel(price)`: `price - self.asks.anchor` (for asks).

If an `update` arrives for a price outside the 4096-tick window, a `#[cold]` (rarely called) function `recenter` moves the anchor and head, copying relevant data.

**Benefit**: O(1) access for all price updates *within a 4096-tick window* (the vast majority). Uses `f32` (4B) instead of `f64` (8B) to double cache density.

### 3\. Bitset and Cached Best Price

**Problem**: Finding the best price in a `HashMap` is O(N). Scanning a 4096-element array is O(CAP), which is too slow (though O(1) in time).

**Solution**:

1.  **Bitset**: An `occupied: [u64; 64]` (512 bytes) tracks which relative levels have a quantity \> EPS. Bit `i` is set to 1 if `qty[i] > 0`.
2.  **Cached Best Price**: `best_rel: usize` stores the *current* relative index of the best price.
3.  **BBO Query**: `best_bid()` simply reads `self.bids.best_rel`. This is an O(1) memory read, hence the \< 1 ns latency.
4.  **Update**: If an `update` *removes* the `best_rel`, we call `find_first_from_head()`, which scans the 64 `u64`s of the bitset (O(64) `trailing_zeros` ops, \~O(1) in practice) to find the new best.

**Benefit**: True O(1) BBO queries. The cost of searching is only paid when the BBO level is removed, not on every query.

### 4\. Zero-Allocation Hot Path

**Problem**: Memory allocation (e.g., `Vec::push`, `HashMap::insert`) is non-deterministic and slow (100-1000 ns).

**Solution**:

  - `HotData` uses `Box::new([0.0; CAP])` at initialization (one-time).
  - The `update` function only performs writes into this pre-allocated array.
  - Even the `verify_checksum` (marked `#[cold]`) avoids `format!` and uses `itoa::Buffer` to write integers to a stack buffer, preventing any heap allocation.

**Benefit**: Predictable and ultra-low latency for *every* update.

## Testing & Validation

### Unit Tests

The project includes comprehensive unit tests (`#[cfg(test)]` in `optimized/book.rs`) covering:

  - Basic operations (insert, remove)
  - BBO (Best Bid/Ask) logic
  - `EPS` threshold logic (tiny quantities)
  - `recenter` logic (large price jumps, margin tests)
  - Massive price jumps (re-seeding)
  - Input sanitization (`NaN`, `Inf`)
  - Correctness of `top_bids` / `orderbook_imbalance_depth`

<!-- end list -->

```bash
# Run tests
cargo test
```

## Tradeoffs & Limitations

**1. Price Window**

  - **Optimized**: Fixed 4096 price level capacity. If the price moves more than 4096 ticks at once (e.g., a massive flash crash), the `recenter` logic will perform a full "re-seed" (clear the book).
  - **Baseline**: Unlimited price range (bound by `i64`).

**2. Memory Usage**

  - **Optimized**: Fixed, \~34 KiB per book (whether 10 or 4000 levels are active).
  - **Baseline**: Variable, proportional to the number of *active* levels.
  - **Impact**: The optimized version is less memory-efficient for very sparse books but more efficient if the book is dense.

**3. Precision**

  - **Optimized**: Uses `f32` for quantities (7 decimal digits of precision).
  - **Baseline**: Uses `f64` (15 decimal digits).
  - **Impact**: `f32` is sufficient for most HFT use cases (e.g., max 16M lots with 0.001 precision) and doubles cache density.

**4. Code Complexity**

  - **Optimized**: \~500 lines of complex logic (anchor management, `recenter`, bit-twiddling).
  - **Baseline**: \~150 lines of simple `HashMap` logic.
  - **Impact**: Maintaining the optimized version requires significant expertise.

## Lessons Learned

1.  **Cache is King**: The difference between 1 ns (L1) and 100 ns (RAM) is 100x. Design *must* start with cache locality (Hot/Cold split, `f32` vs `f64`, alignment).
2.  **Pay the Cost "Offline"**: The cost of finding the BBO (O(N) bitset scan) is paid only when the BBO is *removed*, making the *read* (the most frequent operation) O(1).
3.  **Zero-Allocation is Non-Negotiable**: Latency predictability is often more important than average speed. Allocations are the enemy of predictability.
4.  **Measure, Don't Guess**: `Criterion` and source analysis (`HashMap::iter().max_by_key()`) revealed BBO reads were the O(N) bottleneck in the baseline, not updates.