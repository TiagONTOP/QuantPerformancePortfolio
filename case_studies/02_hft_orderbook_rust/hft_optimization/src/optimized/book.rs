use serde::{Deserialize, Serialize};
use adler::Adler32;

use crate::common::types::{Price, Qty, Side};
use crate::common::messages::L2UpdateMsg;

/// Capacity optimized for L1 cache (4096 levels = 16 KB qty + 512 B bitset per side)
/// Total hot set: ~34 KB (fits in 32 KB L1 + small overflow to L2)
const CAP: usize = 1 << 12; // 4096 levels per side
const CAP_MASK: usize = CAP - 1;

/// Bitset for tracking occupied levels (64 u64s = 4096 bits)
const BITSET_SIZE: usize = CAP / 64;

/// Compile-time assertions for critical invariants using array length trick
/// Ensures CAP is power of 2 (for fast modulo via bitmasking) and multiple of 64 (for word-aligned bitset clearing)
/// Also ensures BITSET_SIZE is power of 2 (for fast word index modulo)
/// If assertion fails, compilation will fail with array length mismatch
#[allow(dead_code)]
const fn bool_to_usize(b: bool) -> usize { b as usize }
const _ASSERT_CAP_POW2: [(); 1] = [(); bool_to_usize(CAP.is_power_of_two())];
const _ASSERT_CAP_DIV64: [(); 1] = [(); bool_to_usize(CAP % 64 == 0)];
const _ASSERT_BITSET_POW2: [(); 1] = [(); bool_to_usize((BITSET_SIZE & (BITSET_SIZE - 1)) == 0)];

/// Epsilon for quantity comparison (avoid denormal flapping)
const EPS: f32 = 1e-9;

/// Recenter hysteresis margins (prevents oscillation near boundaries)
/// Hard boundary: MUST recenter if rel outside [0, CAP) - prevents out-of-bounds access
/// Soft boundary: recenter if within [0, CAP) but close to edges - improves locality
const RECENTER_LOW_MARGIN: usize = 64;
const RECENTER_HIGH_MARGIN: usize = CAP - RECENTER_LOW_MARGIN; // 4032

/// Hot data: accessed on every update - cache-line aligned
/// WARNING: Clone derives present for L2Book compatibility, but copying ~16KB per side
/// If cloning frequently in hot path, consider removing Clone and using Arc or manual impl
#[repr(align(64))]
#[derive(Clone)]
struct HotData {
    /// Quantities per level (f32 for cache density: 4B vs 8B)
    /// Ring buffer: physical index = (head + rel) & CAP_MASK
    /// rel ∈ [0, CAP) unsigned, rel=0 maps to anchor price
    qty: Box<[f32; CAP]>,

    /// Bitset for O(1) occupied level tracking
    /// Bit i set => level i has qty > EPS
    occupied: Box<[u64; BITSET_SIZE]>,

    /// Logical head offset in the ring buffer
    /// head always points to the physical index of rel=0 (anchor)
    head: usize,

    /// Anchor price (in ticks) - virtual reference price for rel=0
    /// Not necessarily a real level; recentering places current price ~CAP/2 from anchor for locality
    /// For bids: anchor = price + CAP/2 (prices below anchor have positive rel)
    /// For asks: anchor = price - CAP/2 (prices above anchor have positive rel)
    anchor: i64,

    /// Best price relative to anchor (usize::MAX if no levels)
    /// Smallest occupied rel (highest price for bids, lowest for asks)
    best_rel: usize,

    /// Instrumentation: count recenter events (for latency correlation)
    #[cfg(debug_assertions)]
    recenter_count: u64,
}

impl HotData {
    #[inline(always)]
    fn new() -> Self {
        Self {
            qty: Box::new([0.0; CAP]),
            occupied: Box::new([0u64; BITSET_SIZE]),
            head: 0,
            anchor: 0,
            best_rel: usize::MAX, // Sentinel: no levels
            #[cfg(debug_assertions)]
            recenter_count: 0,
        }
    }

    /// Convert relative index to physical index in ring buffer
    /// rel ∈ [0, CAP) unsigned
    #[inline(always)]
    fn rel_to_phys(&self, rel: usize) -> usize {
        (self.head + rel) & CAP_MASK
    }

    /// Get quantity at relative index
    #[inline(always)]
    fn get_qty(&self, rel: usize) -> f32 {
        let phys = self.rel_to_phys(rel);
        self.qty[phys]
    }

    /// Set quantity at relative index and update bitset
    #[inline(always)]
    fn set_qty(&mut self, rel: usize, qty: f32) {
        let phys = self.rel_to_phys(rel);
        self.qty[phys] = qty;

        // Update bitset using EPS threshold
        let word_idx = phys / 64;
        let bit_pos = phys % 64;

        if qty > EPS {
            self.occupied[word_idx] |= 1u64 << bit_pos;
        } else {
            self.occupied[word_idx] &= !(1u64 << bit_pos);
        }
    }

    /// Check if a relative index is occupied
    #[inline(always)]
    fn is_occupied(&self, rel: usize) -> bool {
        let phys = self.rel_to_phys(rel);
        let word_idx = phys / 64;
        let bit_pos = phys % 64;
        (self.occupied[word_idx] & (1u64 << bit_pos)) != 0
    }

    /// Find first occupied level from head (smallest rel)
    /// Returns usize::MAX if empty
    /// This works for both bid and ask since we want smallest rel
    #[inline(always)]
    fn find_first_from_head(&self) -> usize {
        let head_word = self.head / 64;
        let head_bit = self.head % 64;

        // Check first word (may be partial)
        let mask = !((1u64 << head_bit).wrapping_sub(1));
        let word = self.occupied[head_word] & mask;
        if word != 0 {
            let bit_pos = word.trailing_zeros() as usize;
            let phys = head_word * 64 + bit_pos;
            // Calculate rel with unsigned wrap
            return (phys.wrapping_sub(self.head)) & CAP_MASK;
        }

        // Check remaining words
        for i in 1..BITSET_SIZE {
            let word_idx = (head_word + i) & (BITSET_SIZE - 1);
            let word = self.occupied[word_idx];
            if word != 0 {
                let bit_pos = word.trailing_zeros() as usize;
                let phys = word_idx * 64 + bit_pos;
                return (phys.wrapping_sub(self.head)) & CAP_MASK;
            }
        }

        usize::MAX
    }

    /// Clear a band of the ring buffer starting at physical index
    /// Used to clear slots that exit the window during smart shift recenter
    #[cold]
    fn clear_band_phys(&mut self, phys_start: usize, count: usize) {
        if count == 0 {
            return;
        }

        // Clear qty and bitset for each slot in the band
        for i in 0..count {
            let phys = (phys_start + i) & CAP_MASK;
            self.qty[phys] = 0.0;

            let word_idx = phys / 64;
            let bit_pos = phys % 64;
            self.occupied[word_idx] &= !(1u64 << bit_pos);
        }
    }

    /// Recenter the ring buffer when price moves out of acceptable range
    ///
    /// CORRECT SMART SHIFT IMPLEMENTATION:
    /// - For shift > 0 (anchor rises): prices "descend" in rel space
    ///   → head_new = head_old - shift (to keep slots stable)
    ///   → clear band at top: [CAP - shift, CAP)
    ///
    /// - For shift < 0 (anchor descends): prices "rise" in rel space
    ///   → head_new = head_old + |shift| (to keep slots stable)
    ///   → clear band at bottom: [0, |shift|)
    ///
    /// This ensures:
    /// 1. Prices in the overlap keep their physical slot (cache-friendly)
    /// 2. Only O(|shift|) writes instead of O(CAP)
    /// 3. NO ghost liquidity (cleared slots = exactly those exiting window)
    ///
    /// Fallback to full reseed if |shift| >= CAP (huge jump)
    #[cold]
    fn recenter(&mut self, new_anchor: i64, shift_amount: i64) {
        #[cfg(debug_assertions)]
        {
            self.recenter_count += 1;
        }

        if shift_amount == 0 {
            return; // No-op
        }

        let abs_shift = if shift_amount >= 0 {
            shift_amount as usize
        } else {
            shift_amount.wrapping_neg() as usize
        };

        // Huge jump: full reseed (fallback safety)
        if abs_shift >= CAP {
            self.qty.fill(0.0);
            self.occupied.fill(0);
            self.anchor = new_anchor;
            self.head = 0;
            self.best_rel = usize::MAX;
            return;
        }

        // Smart shift: O(|shift|) clearing
        if shift_amount > 0 {
            // Anchor rises: clear top band [CAP - shift, CAP)
            let clear_rel_start = CAP - abs_shift;
            let phys_start = (self.head + clear_rel_start) & CAP_MASK;
            self.clear_band_phys(phys_start, abs_shift);

            // Adjust head: head_new = head_old - shift (to keep slots stable)
            self.head = self.head.wrapping_sub(abs_shift) & CAP_MASK;

        } else {
            // Anchor descends: clear bottom band [0, |shift|)
            let phys_start = self.head;
            self.clear_band_phys(phys_start, abs_shift);

            // Adjust head: head_new = head_old + |shift| (to keep slots stable)
            self.head = (self.head + abs_shift) & CAP_MASK;
        }

        self.anchor = new_anchor;

        // Recalculate best (may have moved out or stayed)
        self.best_rel = self.find_first_from_head();
    }

    /// Check if HARD recenter is needed (rel outside valid range [0, CAP))
    /// CRITICAL: Must recenter to prevent out-of-bounds access
    #[inline(always)]
    fn needs_recenter_hard(&self, rel_signed: i64) -> bool {
        rel_signed < 0 || rel_signed >= CAP as i64
    }

    /// Check if SOFT recenter is needed (rel in valid range but near boundaries)
    /// OPTIMIZATION: Recenter proactively to maintain good locality
    #[inline(always)]
    fn needs_recenter_soft(&self, rel_signed: i64) -> bool {
        if rel_signed < 0 || rel_signed >= CAP as i64 {
            return false; // Hard recenter handles this
        }
        let rel = rel_signed as usize;
        rel < RECENTER_LOW_MARGIN || rel >= RECENTER_HIGH_MARGIN
    }
}

impl Default for HotData {
    fn default() -> Self {
        Self::new()
    }
}

/// Cold data: rarely accessed - keep separate to avoid cache pollution
#[derive(Clone, Serialize, Deserialize)]
struct ColdData {
    /// The one "hot" field in `ColdData`.
    /// Checked on every `update()` for O(1) sequence continuity.
    /// It lives here as it's metadata, distinct from the large,
    /// 64-byte-aligned `HotData` arrays.
    seq: u64,
    tick_size: f64,
    lot_size: f64,
    /// Explicit initialization flag (avoids relying on anchor==0 sentinel)
    #[serde(skip)]
    initialized: bool,
}

/// L2 Book ultra-optimized for HFT with strict L1 cache discipline
///
/// Key optimizations:
/// - CAP=4096 for L1 fit (~34 KB total hot data)
/// - Ring buffer with fixed capacity (power of 2)
/// - f32 quantities for 2x cache density vs f64
/// - Bitset for O(1) best tracking via CPU intrinsics
/// - Hot/cold split with 64-byte alignment
/// - Zero allocations in hot path
/// - EPS threshold (1e-9) to avoid denormal flapping
/// - Hard/soft boundary checks for safety + locality
/// - Correct negative shift recenter (backward anchor movement)
/// - Block-level band clearing for large recenters
///
/// Single-writer design - for multi-reader, consider double-buffer + atomic swap
#[derive(Clone, Serialize, Deserialize)]
pub struct L2Book {
    #[serde(skip)]
    bids: HotData,

    #[serde(skip)]
    asks: HotData,

    #[serde(flatten)]
    cold: ColdData,
}

impl L2Book {
    /// Create a new L2Book with fixed L1-optimized capacity
    pub fn new(tick_size: f64, lot_size: f64) -> Self {
        Self {
            bids: HotData::new(),
            asks: HotData::new(),
            cold: ColdData {
                seq: 0,
                tick_size,
                lot_size,
                initialized: false,
            },
        }
    }

    /// For compatibility with old API (ignores capacity)
    pub fn with_capacity(tick_size: f64, lot_size: f64, _capacity: usize) -> Self {
        Self::new(tick_size, lot_size)
    }

    /// Initialize anchors on first update
    /// Uses explicit initialized flag to avoid relying on anchor==0 sentinel
    /// Centers the anchor immediately to avoid triggering recenter on the second update
    #[cold]
    fn initialize_anchors(&mut self, msg: &L2UpdateMsg) {
        if self.cold.initialized {
            return;
        }

        let mut first_bid = None;
        let mut first_ask = None;

        for diff in &msg.diffs {
            match diff.side {
                Side::Bid if first_bid.is_none() => first_bid = Some(diff.price_tick),
                Side::Ask if first_ask.is_none() => first_ask = Some(diff.price_tick),
                _ => {}
            }
        }

        // Center the anchor immediately to provide maximum room for price movement
        // This avoids triggering a recenter on the very first subsequent update
        //
        // For bids: anchor = price + CAP/2 (prices below anchor get positive rel)
        //   - first_bid at price P gets rel = anchor - P = (P + CAP/2) - P = CAP/2
        //   - This places the first bid in the CENTER of the window [0, CAP)
        //   - Higher bids (P+1, P+2, ...) get smaller rel (CAP/2 - 1, CAP/2 - 2, ...)
        //   - Lower bids (P-1, P-2, ...) get larger rel (CAP/2 + 1, CAP/2 + 2, ...)
        //
        // For asks: anchor = price - CAP/2 (prices above anchor get positive rel)
        //   - first_ask at price P gets rel = P - anchor = P - (P - CAP/2) = CAP/2
        //   - This places the first ask in the CENTER of the window [0, CAP)
        //   - Lower asks (P-1, P-2, ...) get smaller rel (CAP/2 - 1, CAP/2 - 2, ...)
        //   - Higher asks (P+1, P+2, ...) get larger rel (CAP/2 + 1, CAP/2 + 2, ...)
        //
        // With CAP=4096, the first price sits at rel=2048, giving 2048 slots in each
        // direction before hitting the soft recenter margins at 64 and 4032.

        if let Some(bid) = first_bid {
            self.bids.anchor = bid + (CAP / 2) as i64;
            // head points to physical index of rel=0, which should map to anchor
            // Since we want rel=0 to map to anchor, and rel=CAP/2 to map to first_bid,
            // we need: (head + CAP/2) & CAP_MASK = physical index for first_bid
            // Starting with head=0 would put rel=0 at index 0, rel=CAP/2 at index CAP/2
            // But we want rel=CAP/2 to be at the "natural" position after centering.
            // Actually, with the ring buffer, we can just use head=0 and the math works out:
            // rel = anchor - price, so price = anchor - rel
            // first_bid will have rel = (anchor - first_bid) = CAP/2
            // Physical index = (head + rel) & CAP_MASK = (0 + CAP/2) & CAP_MASK = CAP/2
            self.bids.head = 0;
        }
        if let Some(ask) = first_ask {
            self.asks.anchor = ask - (CAP / 2) as i64;
            // Same reasoning: rel = price - anchor
            // first_ask will have rel = (first_ask - anchor) = CAP/2
            // Physical index = (head + rel) & CAP_MASK = (0 + CAP/2) & CAP_MASK = CAP/2
            self.asks.head = 0;
        }

        self.cold.initialized = true;
    }

    /// Convert bid price to relative index
    /// rel = anchor - price (higher price = smaller rel = better bid)
    #[inline(always)]
    fn bid_price_to_rel(&self, price: Price) -> i64 {
        self.bids.anchor - price
    }

    /// Convert bid relative index to price
    #[inline(always)]
    fn bid_rel_to_price(&self, rel: usize) -> Price {
        self.bids.anchor - rel as i64
    }

    /// Convert ask price to relative index
    /// rel = price - anchor (lower price = smaller rel = better ask)
    #[inline(always)]
    fn ask_price_to_rel(&self, price: Price) -> i64 {
        price - self.asks.anchor
    }

    /// Convert ask relative index to price
    #[inline(always)]
    fn ask_rel_to_price(&self, rel: usize) -> Price {
        self.asks.anchor + rel as i64
    }

    /// Set bid level with hard/soft recentering and branchless best tracking
    #[inline(always)]
    fn set_bid_level(&mut self, price: Price, qty: f32) {
        // Sanitize: reject NaN/inf, treat tiny/negative as zero
        let sanitized_qty = if qty.is_finite() && qty > EPS { qty } else { 0.0 };

        let mut rel_signed = self.bid_price_to_rel(price);

        // Check if recentering needed (hard or soft)
        if self.bids.needs_recenter_hard(rel_signed) || self.bids.needs_recenter_soft(rel_signed) {
            // Recenter: move anchor to keep price in acceptable range
            let new_anchor = price + (CAP / 2) as i64;
            let shift = new_anchor - self.bids.anchor;
            self.bids.recenter(new_anchor, shift);

            // Recalculate rel after recenter (no recursion!)
            rel_signed = self.bid_price_to_rel(price);
        }

        // CRITICAL: Hard boundary check BEFORE cast to prevent out-of-bounds
        if rel_signed < 0 || rel_signed >= CAP as i64 {
            // Still out of range after recenter (shouldn't happen) - skip to prevent corruption
            return;
        }

        let rel = rel_signed as usize;
        self.bids.set_qty(rel, sanitized_qty);

        // Update best: branchless with sentinels
        // Best bid = smallest rel (highest price)
        if sanitized_qty > EPS {
            if self.bids.best_rel == usize::MAX || rel < self.bids.best_rel {
                self.bids.best_rel = rel;
            }
        } else if rel == self.bids.best_rel {
            // Best was removed, find new best
            self.bids.best_rel = self.bids.find_first_from_head();
        }
    }

    /// Set ask level with hard/soft recentering and branchless best tracking
    #[inline(always)]
    fn set_ask_level(&mut self, price: Price, qty: f32) {
        // Sanitize: reject NaN/inf, treat tiny/negative as zero
        let sanitized_qty = if qty.is_finite() && qty > EPS { qty } else { 0.0 };

        let mut rel_signed = self.ask_price_to_rel(price);

        // Check if recentering needed (hard or soft)
        if self.asks.needs_recenter_hard(rel_signed) || self.asks.needs_recenter_soft(rel_signed) {
            // Recenter: move anchor to keep price in acceptable range
            let new_anchor = price - (CAP / 2) as i64;
            let shift = new_anchor - self.asks.anchor; // CRITICAL: shift = new - old (was reversed)
            self.asks.recenter(new_anchor, shift);

            // Recalculate rel after recenter (no recursion!)
            rel_signed = self.ask_price_to_rel(price);
        }

        // CRITICAL: Hard boundary check BEFORE cast to prevent out-of-bounds
        if rel_signed < 0 || rel_signed >= CAP as i64 {
            // Still out of range after recenter (shouldn't happen) - skip to prevent corruption
            return;
        }

        let rel = rel_signed as usize;
        self.asks.set_qty(rel, sanitized_qty);

        // Update best: branchless with sentinels
        // Best ask = smallest rel (lowest price)
        if sanitized_qty > EPS {
            if self.asks.best_rel == usize::MAX || rel < self.asks.best_rel {
                self.asks.best_rel = rel;
            }
        } else if rel == self.asks.best_rel {
            // Best was removed, find new best
            self.asks.best_rel = self.asks.find_first_from_head();
        }
    }

    /// Get best bid - O(1)
    #[inline(always)]
    pub fn best_bid(&self) -> Option<(Price, Qty)> {
        if self.bids.best_rel == usize::MAX {
            None
        } else {
            let price = self.bid_rel_to_price(self.bids.best_rel);
            let qty = self.bids.get_qty(self.bids.best_rel);
            Some((price, qty as Qty))
        }
    }

    /// Get best ask - O(1)
    #[inline(always)]
    pub fn best_ask(&self) -> Option<(Price, Qty)> {
        if self.asks.best_rel == usize::MAX {
            None
        } else {
            let price = self.ask_rel_to_price(self.asks.best_rel);
            let qty = self.asks.get_qty(self.asks.best_rel);
            Some((price, qty as Qty))
        }
    }

    /// Update the book with a message.
    /// This is the primary hot path for ingestion.
    pub fn update(&mut self, msg: &L2UpdateMsg, symbol: &str) -> bool {
        
        // --- 1. Common Hot Path Logic ---
        
        // Initialize anchors on the first message.
        // This is a `#[cold]` function, so it won't pollute the i-cache.
        self.initialize_anchors(msg);

        // Apply all diffs in the message.
        // This loop is the "hot path" proper. Every operation inside
        // `set_bid_level` / `set_ask_level` is O(1) and L1-cache resident.
        for diff in &msg.diffs {
            let qty = diff.size as f32;
            match diff.side {
                Side::Bid => self.set_bid_level(diff.price_tick, qty),
                Side::Ask => self.set_ask_level(diff.price_tick, qty),
            }
        }

        // --- 2. Dual-Mode Validation (A/B Switch) ---
        //
        // This `#[cfg]` block implements a clean A/B switch for validation,
        // which is the core of this case study.
        //
        // By default, we compile in "Benchmark Mode" to create an
        // apples-to-apples comparison.
        //
        // With the feature flag, we compile in "Production Mode" to show
        // the *actual* O(1) HFT check.
        //
        
        #[cfg(not(feature = "no_checksum"))]
        {
            // ----- "A" Build (Default / Benchmark Mode) -----
            //
            // We call the full state hash (our pedagogical proxy).
            // This is the *exact same workload* as the `suboptimal` version,
            // allowing the benchmark to fairly compare:
            //   - `suboptimal`: O(N) BBO read
            //   - `optimized`:  O(1) BBO read
            //
            self.cold.seq = msg.seq;
            self.verify_checksum(symbol, msg.seq, msg.checksum)
        }

        #[cfg(feature = "no_checksum")]
        {
            // ----- "B" Build (Production / HFT Mode) -----
            //
            // This is the *real* production logic. We replace the heavy,
            // pedagogical hash with the true HFT check: a simple, O(1)
            // strict integer continuity check.
            //
            // A production feed handler MUST ensure the sequence is *exactly*
            // one greater than the last known sequence.
            //
            
            if self.cold.initialized {
                // We are initialized and have a sequence number.
                
                if msg.seq <= self.cold.seq {
                    // This is a stale or duplicate message. We ignore it
                    // but do not consider it a fatal error.
                    return false; 
                }

                if msg.seq > self.cold.seq + 1 {
                    // --- GAP DETECTED ---
                    // We are at seq 100, but msg 102 arrived. We missed 101.
                    // The book is now corrupt. A real system must resync.
                    // We fail the update immediately.
                    // 
                    return false;
                }
                
                // If we are here, msg.seq == self.cold.seq + 1.
                // This is the "golden path".

            } 
            // else: This is the first message (self.cold.initialized == false),
            // so we accept it as the baseline.

            // Commit the new, valid sequence number.
            self.cold.seq = msg.seq;
            true
        }
    }

    /// Verifies a full state checksum in an HFT-optimized manner.
    ///
    /// This function is the "A" part of our "A/B" validation switch
    /// (see comments in `update()`). It is compiled *only* for the default
    /// benchmark build.
    ///
    /// Its purpose is to provide an apples-to-apples comparison with the
    /// `suboptimal` version by performing the *exact same logical workload*
    /// (a full BBO-based hash), but with all HFT optimizations applied.
    ///
    /// # Key Optimizations vs. `suboptimal`:
    ///
    /// 1.  **O(1) BBO Reads:**
    ///     Calls `best_bid()`/`best_ask()` in O(1) (reading `best_rel`),
    ///     which is the core fix for the O(N) bug.
    ///
    /// 2.  **`#[cold]` / `#[inline(never)]`:**
    ///     Protects the L1i (Instruction Cache) from pollution. We tell the
    ///     compiler this is an auxiliary path, not part of the core `update` logic.
    ///
    /// 3.  **Zero Heap Allocation:**
    ///     Uses `itoa::Buffer` (stack) instead of `format!` (heap) to
    ///     eliminate allocator latency, a mandatory HFT practice.
    ///
    #[cold]
    #[inline(never)]
    fn verify_checksum(&self, symbol: &str, seq: u64, expected_checksum: u32) -> bool {
        // O(1) read - Now safe and fast (L1d cache hit).
        let (bb, _) = self.best_bid().unwrap_or((0, 0.0));
        // O(1) read - Also safe and fast.
        let (aa, _) = self.best_ask().unwrap_or((0, 0.0));

        // Build the hash payload without ANY heap allocation.
        let mut hasher = Adler32::new();
        hasher.write_slice(symbol.as_bytes());
        hasher.write_slice(b"|");

        // Use `itoa` to format the u64 `seq` onto a stack buffer.
        let mut seq_buf = itoa::Buffer::new();
        let seq_str = seq_buf.format(seq);
        hasher.write_slice(seq_str.as_bytes());
        hasher.write_slice(b"|");

        // Use `itoa` to format the i64 `bb` onto a stack buffer.
        let mut bb_buf = itoa::Buffer::new();
        let bb_str = bb_buf.format(bb);
        hasher.write_slice(bb_str.as_bytes());
        hasher.write_slice(b"|");

        // Use `itoa` to format the i64 `aa` onto a stack buffer.
        let mut aa_buf = itoa::Buffer::new();
        let aa_str = aa_buf.format(aa);
        hasher.write_slice(aa_str.as_bytes());

        let computed = hasher.checksum();
        computed == expected_checksum
    }

    /// Mid-price in ticks - O(1)
    #[inline(always)]
    pub fn mid_price_ticks(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some((bid as f64 + ask as f64) / 2.0),
            _ => None,
        }
    }

    /// Mid-price in real price - O(1)
    #[inline(always)]
    pub fn mid_price(&self) -> Option<f64> {
        self.mid_price_ticks().map(|mid| mid * self.cold.tick_size)
    }

    /// Orderbook imbalance at best level - O(1)
    #[inline(always)]
    pub fn orderbook_imbalance(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((_, bid_size)), Some((_, ask_size))) => {
                let total = bid_size + ask_size;
                if total > EPS as f64 {
                    Some((bid_size - ask_size) / total)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Orderbook imbalance over depth levels - LOCAL around best
    /// Collects exactly 'depth' PRESENT levels (skips empty slots)
    /// This is O(depth) and only scans around the best
    #[inline]
    pub fn orderbook_imbalance_depth(&self, depth: usize) -> Option<f64> {
        if self.bids.best_rel == usize::MAX || self.asks.best_rel == usize::MAX {
            return None;
        }

        let mut total_bid_size = 0.0f32;
        let mut total_ask_size = 0.0f32;
        let mut bid_count = 0;
        let mut ask_count = 0;

        // Scan bids: collect exactly 'depth' present levels
        let mut rel = self.bids.best_rel;
        for _ in 0..CAP {
            // Safety: wrap modulo to prevent infinite loop
            if bid_count >= depth {
                break;
            }
            if self.bids.is_occupied(rel) {
                total_bid_size += self.bids.get_qty(rel);
                bid_count += 1;
            }
            rel = (rel + 1) & CAP_MASK;
            if rel == self.bids.best_rel {
                break; // Wrapped around
            }
        }

        // Scan asks: collect exactly 'depth' present levels
        let mut rel = self.asks.best_rel;
        for _ in 0..CAP {
            if ask_count >= depth {
                break;
            }
            if self.asks.is_occupied(rel) {
                total_ask_size += self.asks.get_qty(rel);
                ask_count += 1;
            }
            rel = (rel + 1) & CAP_MASK;
            if rel == self.asks.best_rel {
                break; // Wrapped around
            }
        }

        if bid_count == 0 || ask_count == 0 {
            return None;
        }

        let total = total_bid_size + total_ask_size;
        if total > EPS {
            Some(((total_bid_size - total_ask_size) / total) as f64)
        } else {
            None
        }
    }

    /// Spread in ticks - O(1)
    #[inline(always)]
    pub fn spread_ticks(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some(ask - bid),
            _ => None,
        }
    }

    /// Spread in real price - O(1)
    #[inline(always)]
    pub fn spread(&self) -> Option<f64> {
        self.spread_ticks().map(|s| s as f64 * self.cold.tick_size)
    }

    /// Bid depth count - O(1) via bitset popcount
    pub fn bid_depth(&self) -> usize {
        self.bids.occupied.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Ask depth count - O(1) via bitset popcount
    pub fn ask_depth(&self) -> usize {
        self.asks.occupied.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Top N bids - collects exactly N present levels (or less if book shallow)
    pub fn top_bids(&self, n: usize) -> Vec<(Price, Qty)> {
        let mut result = Vec::with_capacity(n);
        if self.bids.best_rel == usize::MAX {
            return result;
        }

        let mut rel = self.bids.best_rel;
        for _ in 0..CAP {
            if result.len() >= n {
                break;
            }
            if self.bids.is_occupied(rel) {
                let price = self.bid_rel_to_price(rel);
                let qty = self.bids.get_qty(rel);
                result.push((price, qty as Qty));
            }
            rel = (rel + 1) & CAP_MASK;
            if rel == self.bids.best_rel {
                break; // Wrapped around
            }
        }
        result
    }

    /// Top N asks - collects exactly N present levels (or less if book shallow)
    pub fn top_asks(&self, n: usize) -> Vec<(Price, Qty)> {
        let mut result = Vec::with_capacity(n);
        if self.asks.best_rel == usize::MAX {
            return result;
        }

        let mut rel = self.asks.best_rel;
        for _ in 0..CAP {
            if result.len() >= n {
                break;
            }
            if self.asks.is_occupied(rel) {
                let price = self.ask_rel_to_price(rel);
                let qty = self.asks.get_qty(rel);
                result.push((price, qty as Qty));
            }
            rel = (rel + 1) & CAP_MASK;
            if rel == self.asks.best_rel {
                break; // Wrapped around
            }
        }
        result
    }

    /// Accessor for seq (for compatibility)
    #[inline(always)]
    pub fn seq(&self) -> u64 {
        self.cold.seq
    }

    /// Get recenter count (debug builds only)
    #[cfg(debug_assertions)]
    pub fn recenter_count(&self) -> (u64, u64) {
        (self.bids.recenter_count, self.asks.recenter_count)
    }
}

impl Default for L2Book {
    fn default() -> Self {
        Self::new(0.01, 0.001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::messages::{L2Diff, MsgType};

    #[test]
    fn test_l1_optimized_basic() {
        let mut book = L2Book::new(0.1, 0.001);

        let msg = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 0,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 1000, size: 10.0 },
                L2Diff { side: Side::Ask, price_tick: 1002, size: 8.0 },
            ],
            checksum: 0,
        };

        book.update(&msg, "BTC-USDT");

        assert_eq!(book.best_bid(), Some((1000, 10.0)));
        assert_eq!(book.best_ask(), Some((1002, 8.0)));
        assert_eq!(book.mid_price_ticks(), Some(1001.0));
    }

    #[test]
    fn test_eps_threshold() {
        let mut book = L2Book::new(0.1, 0.001);

        // Bootstrap
        let bootstrap = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 0,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 1000, size: 10.0 },
            ],
            checksum: 0,
        };
        book.update(&bootstrap, "BTC-USDT");

        // Very small quantity (below EPS) should not be tracked
        let tiny_update = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 1,
            seq: 2,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 999, size: 1e-12 },
            ],
            checksum: 0,
        };
        book.update(&tiny_update, "BTC-USDT");

        // Best should still be 1000, not 999 (because 999's qty is below EPS)
        assert_eq!(book.best_bid(), Some((1000, 10.0)));
    }

    #[test]
    fn test_recenter_threshold() {
        let mut book = L2Book::new(0.1, 0.001);

        // Initialize
        let init = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 0,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 50000, size: 10.0 },
            ],
            checksum: 0,
        };
        book.update(&init, "BTC-USDT");

        #[cfg(debug_assertions)]
        let initial_recenters = book.recenter_count().0;

        // Add prices within margin (should not trigger recenter)
        for i in 1..RECENTER_LOW_MARGIN as i64 {
            let msg = L2UpdateMsg {
                msg_type: MsgType::L2Update,
                symbol: "BTC-USDT".to_string(),
                ts: i,
                seq: (i + 1) as u64,
                diffs: vec![
                    L2Diff { side: Side::Bid, price_tick: 50000 - i, size: 1.0 },
                ],
                checksum: 0,
            };
            book.update(&msg, "BTC-USDT");
        }

        #[cfg(debug_assertions)]
        {
            let recenters_after = book.recenter_count().0;
            assert_eq!(recenters_after, initial_recenters, "Should not recenter within margin");
        }
    }

    #[test]
    fn test_massive_wraparound() {
        let mut book = L2Book::new(0.1, 0.001);

        // Initialize
        let init = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 0,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 50000, size: 100.0 },
            ],
            checksum: 0,
        };
        book.update(&init, "BTC-USDT");

        // Add many levels around best (within CAP=4096 capacity)
        // Note: With CAP=4096 and recentering, we can't maintain all 800 levels
        // Recentering will happen and clear parts of the old range
        for i in 1..400u64 {
            let msg = L2UpdateMsg {
                msg_type: MsgType::L2Update,
                symbol: "BTC-USDT".to_string(),
                ts: i as i64,
                seq: i + 1,
                diffs: vec![
                    L2Diff { side: Side::Bid, price_tick: 50000 - i as i64, size: (i % 100 + 1) as f64 },
                ],
                checksum: 0,
            };
            book.update(&msg, "BTC-USDT");
        }

        // Best should remain near original price (may trigger recentering)
        let best = book.best_bid().unwrap();
        assert!(best.0 >= 49000 && best.0 <= 51000, "Best price should be in reasonable range");
        // With CAP=4096 and recentering thresholds, expect at least 200 levels maintained
        assert!(book.bid_depth() >= 200, "Should have many levels after wraparound, got {}", book.bid_depth());
    }

    #[test]
    fn test_large_price_jump_reseed() {
        let mut book = L2Book::new(0.1, 0.001);

        // Initialize at 50000
        let init = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 0,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 50000, size: 10.0 },
            ],
            checksum: 0,
        };
        book.update(&init, "BTC-USDT");

        // Jump to 60000 (> CAP/2 = 2048 ticks away) - should trigger reseed
        let jump = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 1,
            seq: 2,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 60000, size: 20.0 },
            ],
            checksum: 0,
        };
        book.update(&jump, "BTC-USDT");

        // Should work without panic - price may be adjusted by recenter logic
        let best = book.best_bid().unwrap();
        assert!(best.1 > 19.0 && best.1 < 21.0, "Quantity should be ~20.0");

        // After large jump and reseed, depth should be small
        assert!(book.bid_depth() <= 10, "Old levels should be cleared after reseed");
    }

    #[test]
    fn test_depth_collection_exact() {
        let mut book = L2Book::new(0.1, 0.001);

        // Create book with gaps (empty levels between present ones)
        let msg = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 0,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 1000, size: 100.0 },
                L2Diff { side: Side::Bid, price_tick: 998, size: 80.0 },  // Gap at 999
                L2Diff { side: Side::Bid, price_tick: 996, size: 60.0 },  // Gap at 997
                L2Diff { side: Side::Ask, price_tick: 1001, size: 50.0 },
                L2Diff { side: Side::Ask, price_tick: 1003, size: 40.0 }, // Gap at 1002
                L2Diff { side: Side::Ask, price_tick: 1005, size: 30.0 }, // Gap at 1004
            ],
            checksum: 0,
        };
        book.update(&msg, "BTC-USDT");

        // Request depth=3 should return exactly 3 present levels per side
        let top_bids = book.top_bids(3);
        assert_eq!(top_bids.len(), 3);
        assert_eq!(top_bids[0], (1000, 100.0));
        assert_eq!(top_bids[1], (998, 80.0));
        assert_eq!(top_bids[2], (996, 60.0));

        let top_asks = book.top_asks(3);
        assert_eq!(top_asks.len(), 3);
        assert_eq!(top_asks[0], (1001, 50.0));
        assert_eq!(top_asks[1], (1003, 40.0));
        assert_eq!(top_asks[2], (1005, 30.0));

        // Imbalance depth should use exactly 3 present levels
        let imb = book.orderbook_imbalance_depth(3).unwrap();
        let expected = (240.0 - 120.0) / 360.0; // (100+80+60 - 50+40+30) / total
        assert!((imb - expected).abs() < 1e-6);
    }

    #[test]
    fn test_band_clearing_after_recenter() {
        let mut book = L2Book::new(0.1, 0.001);

        // Initialize
        let init = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 0,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 50000, size: 10.0 },
                L2Diff { side: Side::Bid, price_tick: 49999, size: 9.0 },
                L2Diff { side: Side::Bid, price_tick: 49998, size: 8.0 },
            ],
            checksum: 0,
        };
        book.update(&init, "BTC-USDT");

        // Force recenter by jumping beyond high margin
        let jump = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 1,
            seq: 2,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 50000 + RECENTER_HIGH_MARGIN as i64, size: 1.0 },
            ],
            checksum: 0,
        };
        book.update(&jump, "BTC-USDT");

        // Check that old levels are cleared (band should be zeroed)
        // In a proper implementation, levels outside the new window should not appear
        let depth = book.bid_depth();
        assert!(depth <= 10, "Old levels should be cleared after recenter, got depth={}", depth);
    }

    #[test]
    fn test_no_infinite_recursion() {
        let mut book = L2Book::new(0.1, 0.001);

        // Initialize
        let init = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 0,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 50000, size: 10.0 },
            ],
            checksum: 0,
        };
        book.update(&init, "BTC-USDT");

        // Try to trigger recenter, then immediately update same price
        // Should not cause infinite recursion
        let recenter_and_update = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 1,
            seq: 2,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 50000 + RECENTER_HIGH_MARGIN as i64 + 100, size: 5.0 },
                L2Diff { side: Side::Bid, price_tick: 50000 + RECENTER_HIGH_MARGIN as i64 + 100, size: 6.0 }, // Update same level
            ],
            checksum: 0,
        };
        book.update(&recenter_and_update, "BTC-USDT");

        // Should complete without stack overflow
        assert!(book.best_bid().is_some());
    }

    #[test]
    fn test_negative_shift_recenter() {
        let mut book = L2Book::new(0.1, 0.001);

        // Initialize with bid at 50000
        let init = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 0,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 50000, size: 10.0 },
            ],
            checksum: 0,
        };
        book.update(&init, "BTC-USDT");

        // Add many levels BELOW to force anchor down (negative shift)
        for i in 1..200 {
            let msg = L2UpdateMsg {
                msg_type: MsgType::L2Update,
                symbol: "BTC-USDT".to_string(),
                ts: i as i64,
                seq: i + 1,
                diffs: vec![
                    L2Diff { side: Side::Bid, price_tick: 50000 - i as i64, size: (i % 10 + 1) as f64 },
                ],
                checksum: 0,
            };
            book.update(&msg, "BTC-USDT");
        }

        // Now add levels ABOVE original anchor (should trigger negative shift recenter)
        for i in 1..200 {
            let msg = L2UpdateMsg {
                msg_type: MsgType::L2Update,
                symbol: "BTC-USDT".to_string(),
                ts: (200 + i) as i64,
                seq: 200 + i + 1,
                diffs: vec![
                    L2Diff { side: Side::Bid, price_tick: 50000 + i as i64, size: (i % 10 + 1) as f64 },
                ],
                checksum: 0,
            };
            book.update(&msg, "BTC-USDT");
        }

        // Best bid should be the highest price
        let best = book.best_bid().unwrap();
        assert!(best.0 >= 50100, "Best bid should be high price, got {}", best.0);
        assert!(best.1 > 0.0, "Best bid should have quantity");

        // Should have multiple levels
        assert!(book.bid_depth() >= 50, "Should have many levels after negative shift, got {}", book.bid_depth());
    }

    #[test]
    fn test_nan_inf_sanitization() {
        let mut book = L2Book::new(0.1, 0.001);

        // Initialize with valid levels
        let init = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 0,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 1000, size: 10.0 },
                L2Diff { side: Side::Ask, price_tick: 1002, size: 8.0 },
            ],
            checksum: 0,
        };
        book.update(&init, "BTC-USDT");

        // Try to update with NaN quantities (should be rejected/sanitized to 0)
        let nan_update = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 1,
            seq: 2,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 999, size: f64::NAN },
                L2Diff { side: Side::Ask, price_tick: 1003, size: f64::NAN },
            ],
            checksum: 0,
        };
        book.update(&nan_update, "BTC-USDT");

        // Best bid/ask should remain at original valid levels (NaN was sanitized to 0)
        assert_eq!(book.best_bid(), Some((1000, 10.0)));
        assert_eq!(book.best_ask(), Some((1002, 8.0)));

        // Try to update with infinity quantities (should be rejected/sanitized to 0)
        let inf_update = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 2,
            seq: 3,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 998, size: f64::INFINITY },
                L2Diff { side: Side::Ask, price_tick: 1004, size: f64::NEG_INFINITY },
            ],
            checksum: 0,
        };
        book.update(&inf_update, "BTC-USDT");

        // Best bid/ask should still remain at original valid levels
        assert_eq!(book.best_bid(), Some((1000, 10.0)));
        assert_eq!(book.best_ask(), Some((1002, 8.0)));

        // Update with very small quantities below EPS (should also be treated as 0)
        let tiny_update = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 3,
            seq: 4,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 997, size: 1e-12 },
                L2Diff { side: Side::Ask, price_tick: 1005, size: 1e-12 },
            ],
            checksum: 0,
        };
        book.update(&tiny_update, "BTC-USDT");

        // Best should still be original levels (tiny quantities below EPS ignored)
        assert_eq!(book.best_bid(), Some((1000, 10.0)));
        assert_eq!(book.best_ask(), Some((1002, 8.0)));

        // Now update original best levels with NaN (should remove them and find new best)
        let remove_best = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 4,
            seq: 5,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 1000, size: f64::NAN },
                L2Diff { side: Side::Ask, price_tick: 1002, size: f64::NAN },
            ],
            checksum: 0,
        };
        book.update(&remove_best, "BTC-USDT");

        // Best levels should now be empty (no other valid levels)
        assert_eq!(book.best_bid(), None);
        assert_eq!(book.best_ask(), None); //
    }

    #[test]
    fn test_small_shift_recenter_no_ghost_liquidity() {
        // This test verifies that the "small shift" recenter logic correctly
        // clears bands and doesn't leave ghost liquidity in the bitset.
        //
        // Scenario:
        // 1. Create multiple bid levels
        // 2. Trigger a small shift recenter (not a full reseed)
        // 3. Remove the current best bid
        // 4. Verify that best_bid() returns the correct next level, NOT a ghost

        let mut book = L2Book::new(0.1, 0.001);

        // T=1: Initialize with bid at 50000
        let msg1 = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 1,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 50000, size: 10.0 },
            ],
            checksum: 0,
        };
        book.update(&msg1, "BTC-USDT");
        assert_eq!(book.best_bid(), Some((50000, 10.0)));

        // T=2-10: Add multiple bid levels (both above and below 50000)
        for i in 1..=5u64 {
            let msg = L2UpdateMsg {
                msg_type: MsgType::L2Update,
                symbol: "BTC-USDT".to_string(),
                ts: i as i64 + 1,
                seq: i + 1,
                diffs: vec![
                    L2Diff { side: Side::Bid, price_tick: 50000 + i as i64, size: (10 + i) as f64 },
                    L2Diff { side: Side::Bid, price_tick: 50000 - i as i64, size: (10 - i) as f64 },
                ],
                checksum: 0,
            };
            book.update(&msg, "BTC-USDT");
        }

        // Verify we have multiple levels (1 initial + 5*2 = 11 levels)
        let depth_initial = book.bid_depth();
        assert!(depth_initial >= 10, "Should have at least 10 bid levels, got {}", depth_initial);
        assert_eq!(book.best_bid().unwrap().0, 50005); // Highest bid

        // T=11: Trigger a SMALL SHIFT recenter (within CAP/2, so not a full reseed)
        // Move price up by RECENTER_LOW_MARGIN to trigger soft recenter
        let shift_price = 50000 + RECENTER_LOW_MARGIN as i64;
        let msg_shift = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 11,
            seq: 12,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: shift_price, size: 100.0 },
            ],
            checksum: 0,
        };
        book.update(&msg_shift, "BTC-USDT");

        // After recenter, the new best should be the higher price
        assert_eq!(book.best_bid(), Some((shift_price, 100.0)));

        // Verify depth - after small shift, some old levels should be preserved
        let depth_after_shift = book.bid_depth();
        assert!(depth_after_shift >= 1, "Should have at least the new level");

        // T=12: Remove the current best bid
        let msg_remove = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 12,
            seq: 13,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: shift_price, size: 0.0 },
            ],
            checksum: 0,
        };
        book.update(&msg_remove, "BTC-USDT");

        // CRITICAL ASSERTION: After removing the best, the next best should be
        // either a valid preserved level OR None (if all were cleared).
        // It should NEVER be a "ghost" price that doesn't actually exist.
        let new_best = book.best_bid();

        if let Some((price, qty)) = new_best {
            // If there's a new best, verify it's actually valid
            assert!(qty > 0.0, "Best bid quantity should be > 0, got {}", qty);
            assert!(price < shift_price, "New best should be below removed price, got {} vs {}", price, shift_price);

            // Verify this price actually has quantity in the book
            // by checking that bid_depth is consistent
            assert!(book.bid_depth() >= 1, "If there's a best bid, depth should be >= 1");

            // CRITICAL: Verify this is NOT a ghost - check it's in the valid range
            // It should be one of the levels we added earlier or preserved during recenter
            let is_valid = (price >= 49995 && price <= 50005) || (price > shift_price - 100 && price < shift_price);
            assert!(is_valid, "Price {} appears to be a ghost (outside expected ranges)", price);
        } else {
            // If best is None, depth should be 0
            assert_eq!(book.bid_depth(), 0,
                "If best_bid() is None, bid_depth() should be 0, got {}", book.bid_depth());
        }
    }

    #[test]
    fn test_negative_shift_recenter_no_ghost_liquidity() {
        // Test the negative shift path (anchor moves backward) for ghost liquidity
        let mut book = L2Book::new(0.1, 0.001);

        // Initialize
        let msg1 = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 1,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 50000, size: 10.0 },
            ],
            checksum: 0,
        };
        book.update(&msg1, "BTC-USDT");

        // Add levels ABOVE the initial price
        for i in 1..=20u64 {
            let msg = L2UpdateMsg {
                msg_type: MsgType::L2Update,
                symbol: "BTC-USDT".to_string(),
                ts: i as i64 + 1,
                seq: i + 1,
                diffs: vec![
                    L2Diff { side: Side::Bid, price_tick: 50000 + i as i64, size: i as f64 },
                ],
                checksum: 0,
            };
            book.update(&msg, "BTC-USDT");
        }

        let initial_best = book.best_bid().unwrap();
        assert_eq!(initial_best.0, 50020); // Highest bid

        // Trigger a NEGATIVE shift by adding a very low bid
        // This forces anchor to move backward (negative shift_amount)
        let low_price = 50000 - RECENTER_LOW_MARGIN as i64 - 10;
        let msg_shift = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 22,
            seq: 23,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: low_price, size: 5.0 },
            ],
            checksum: 0,
        };
        book.update(&msg_shift, "BTC-USDT");

        // After negative shift recenter, verify book state is consistent
        let depth_after_shift = book.bid_depth();
        assert!(depth_after_shift >= 1, "Should have at least one level after shift");

        // Get all bids and verify they're all valid (no ghosts)
        let top_bids = book.top_bids(100);
        for (price, qty) in &top_bids {
            assert!(*qty > 0.0, "All bid quantities should be > 0, found ghost at price {} with qty {}", price, qty);
        }

        // Verify depth is consistent with top_bids
        assert_eq!(top_bids.len(), depth_after_shift,
            "top_bids length should match bid_depth, got {} vs {}", top_bids.len(), depth_after_shift);
    }

    #[test]
    fn test_wraparound_shift_no_corruption() {
        // Test the wraparound case in negative shift (when new_head > old_head)
        let mut book = L2Book::new(0.1, 0.001);

        // Create a scenario that will cause wraparound
        // 1. Initialize near the end of the capacity range
        let msg1 = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 1,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 60000, size: 100.0 },
            ],
            checksum: 0,
        };
        book.update(&msg1, "BTC-USDT");

        // 2. Add many levels to fill up part of the book
        for i in 1..=50u64 {
            let msg = L2UpdateMsg {
                msg_type: MsgType::L2Update,
                symbol: "BTC-USDT".to_string(),
                ts: i as i64 + 1,
                seq: i + 1,
                diffs: vec![
                    L2Diff { side: Side::Bid, price_tick: 60000 + i as i64, size: i as f64 },
                    L2Diff { side: Side::Bid, price_tick: 60000 - i as i64, size: i as f64 },
                ],
                checksum: 0,
            };
            book.update(&msg, "BTC-USDT");
        }

        let initial_depth = book.bid_depth();
        assert!(initial_depth >= 50, "Should have many levels initially");

        // 3. Force a recenter that might cause wraparound
        let shift_price = 60000 + 200;
        let msg_shift = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 52,
            seq: 53,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: shift_price, size: 10.0 },
            ],
            checksum: 0,
        };
        book.update(&msg_shift, "BTC-USDT");

        // 4. Verify no corruption: all reported levels should have valid quantities
        let all_bids = book.top_bids(1000);
        for (price, qty) in &all_bids {
            assert!(*qty > 1e-9,
                "Found corrupted/ghost level at price {} with qty {} (should be > EPS)",
                price, qty);
        }

        // 5. Verify consistency: depth should match actual count of valid levels
        assert_eq!(all_bids.len(), book.bid_depth(),
            "Depth mismatch: top_bids returned {} levels but bid_depth reports {}",
            all_bids.len(), book.bid_depth());

        // 6. Remove all levels one by one and verify no ghosts appear
        for (price, _) in all_bids.clone() {
            let msg_remove = L2UpdateMsg {
                msg_type: MsgType::L2Update,
                symbol: "BTC-USDT".to_string(),
                ts: 100 + price as i64,
                seq: 100 + price as u64,
                diffs: vec![
                    L2Diff { side: Side::Bid, price_tick: price, size: 0.0 },
                ],
                checksum: 0,
            };
            book.update(&msg_remove, "BTC-USDT");
        }

        // After removing all, book should be empty
        assert_eq!(book.best_bid(), None, "Book should be empty after removing all levels");
        assert_eq!(book.bid_depth(), 0, "Depth should be 0 after removing all levels");
    }

    #[test]
    fn test_no_ghost_liquidity_after_soft_recenter() {
        // Bug scenario (with CAP=4096):
        // 1. Add bid at price 50000 with qty 10.0
        // 2. Trigger soft recenter by adding a bid far away
        // 3. After recenter, the qty 10.0 should either:
        //    - Still be at price 50000 (if 50000 is in the new window)
        //    - Be cleared (if 50000 is outside the new window)
        //    - NEVER be re-labeled to a different price like 49994 or 50006

        let mut book = L2Book::new(0.1, 0.001);

        // Step 1: Add initial bid at 50000 with qty 10.0
        let msg1 = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 1,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 50000, size: 10.0 },
            ],
            checksum: 0,
        };
        book.update(&msg1, "BTC-USDT");

        // Verify initial state
        assert_eq!(book.best_bid(), Some((50000, 10.0)), "Initial bid should be at 50000");

        // Get all bids before recenter - should be exactly one at 50000
        let bids_before = book.top_bids(100);
        assert_eq!(bids_before.len(), 1, "Should have exactly 1 bid before recenter");
        assert_eq!(bids_before[0], (50000, 10.0), "Should be 50000@10.0");

        // Step 2: Trigger soft recenter by adding a bid that causes negative shift
        // Add a bid far below to trigger anchor moving down
        let trigger_price = 50000 - RECENTER_LOW_MARGIN as i64 - 100;
        let msg2 = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 2,
            seq: 2,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: trigger_price, size: 5.0 },
            ],
            checksum: 0,
        };
        book.update(&msg2, "BTC-USDT");

        // Step 3: Get all bids after recenter
        let bids_after = book.top_bids(100);

        // CRITICAL ASSERTION: Check that 50000@10.0 is either:
        // A) Still at exactly 50000 with qty 10.0 (if preserved)
        // B) Not present at all (if cleared)
        // C) NEVER at a different price with qty 10.0 (NO GHOST)

        let mut found_original = false;

        for (price, qty) in &bids_after {
            // Check if we found the original level
            if *price == 50000 && (*qty - 10.0).abs() < 0.001 {
                found_original = true;
            }

            // Check for ghost: qty 10.0 at a DIFFERENT price
            if *price != 50000 && (*qty - 10.0).abs() < 0.001 {
                panic!(
                    "GHOST LIQUIDITY DETECTED! Original bid 50000@10.0 has been re-labeled to {}@{:.1}. \
                     This is the exact bug from the audit: soft recenter shifted quantities by ±2*shift ticks.",
                    price, qty
                );
            }
        }

        // Additional check: if the trigger price is in the book, verify its quantity
        let has_trigger = bids_after.iter().any(|(p, q)| *p == trigger_price && (*q - 5.0).abs() < 0.001);
        assert!(has_trigger, "The trigger price {}@5.0 should be in the book", trigger_price);

        // If we found the original at 50000, great! The window included it.
        // If we didn't find it, that's also acceptable - it was cleared.
        // But we should NEVER find it at a different price (panic would have triggered above).

        println!("Test passed: found_original={}, bids_after={:?}",
                 found_original, bids_after);
    }
}
