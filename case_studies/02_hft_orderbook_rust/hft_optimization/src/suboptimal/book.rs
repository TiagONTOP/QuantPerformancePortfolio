use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use adler::Adler32;

use crate::common::types::{Price, Qty, Side};
use crate::common::messages::L2UpdateMsg;

/// L2 Book: price-aggregated order book
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct L2Book {
    pub seq: u64,
    pub tick_size: f64,
    pub lot_size: f64,
    pub bids: HashMap<Price, Qty>, // price_tick -> size
    pub asks: HashMap<Price, Qty>,
}

impl L2Book {
    /// Creates a new empty L2Book
    pub fn new(tick_size: f64, lot_size: f64) -> Self {
        Self {
            seq: 0,
            tick_size,
            lot_size,
            bids: HashMap::new(),
            asks: HashMap::new(),
        }
    }

    /// Returns the best bid (highest price)
    #[inline]
    pub fn best_bid(&self) -> Option<(Price, Qty)> {
        self.bids.iter().max_by_key(|(p, _)| *p).map(|(p, q)| (*p, *q))
    }

    /// Returns the best ask (lowest price)
    #[inline]
    pub fn best_ask(&self) -> Option<(Price, Qty)> {
        self.asks.iter().min_by_key(|(p, _)| *p).map(|(p, q)| (*p, *q))
    }

    /// Calculates the mid-price in ticks
    /// Returns None if bid or ask is absent
    #[inline]
    pub fn mid_price_ticks(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some((bid as f64 + ask as f64) / 2.0),
            _ => None,
        }
    }

    /// Calculates the mid-price in real price (mid_price_ticks * tick_size)
    #[inline]
    pub fn mid_price(&self) -> Option<f64> {
        self.mid_price_ticks().map(|mid_ticks| mid_ticks * self.tick_size)
    }

    /// Calculates the orderbook imbalance at the best level
    /// Imbalance = (bid_size - ask_size) / (bid_size + ask_size)
    /// Returns a value between -1.0 and 1.0
    /// - Positive value: more liquidity on bid side (buy pressure)
    /// - Negative value: more liquidity on ask side (sell pressure)
    /// - Returns None if bid or ask is absent
    #[inline]
    pub fn orderbook_imbalance(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((_, bid_size)), Some((_, ask_size))) => {
                let total = bid_size + ask_size;
                if total > 1e-9 {
                    Some((bid_size - ask_size) / total)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Calculates the weighted orderbook imbalance over multiple levels
    /// depth: number of levels to consider on each side
    /// Returns None if the book is empty
    pub fn orderbook_imbalance_depth(&self, depth: usize) -> Option<f64> {
        if self.bids.is_empty() || self.asks.is_empty() {
            return None;
        }

        // Get the N best bids (sorted descending by price)
        let mut bid_levels: Vec<_> = self.bids.iter().collect();
        bid_levels.sort_by_key(|(p, _)| std::cmp::Reverse(*p));
        let bid_levels: Vec<_> = bid_levels.into_iter().take(depth).collect();

        // Get the N best asks (sorted ascending by price)
        let mut ask_levels: Vec<_> = self.asks.iter().collect();
        ask_levels.sort_by_key(|(p, _)| *p);
        let ask_levels: Vec<_> = ask_levels.into_iter().take(depth).collect();

        let total_bid_size: f64 = bid_levels.iter().map(|(_, &qty)| qty).sum();
        let total_ask_size: f64 = ask_levels.iter().map(|(_, &qty)| qty).sum();

        let total = total_bid_size + total_ask_size;
        if total > 1e-9 {
            Some((total_bid_size - total_ask_size) / total)
        } else {
            None
        }
    }

    /// Updates the L2Book with an L2UpdateMsg message
    /// Applies all message diffs and updates the sequence number
    /// Returns true if the checksum is valid, false otherwise
    pub fn update(&mut self, msg: &L2UpdateMsg, symbol: &str) -> bool {
        // Apply each diff
        for diff in &msg.diffs {
            match diff.side {
                Side::Bid => {
                    if diff.size == 0.0 {
                        // Remove the level
                        self.bids.remove(&diff.price_tick);
                    } else {
                        // Insert or update
                        self.bids.insert(diff.price_tick, diff.size);
                    }
                }
                Side::Ask => {
                    if diff.size == 0.0 {
                        // Remove the level
                        self.asks.remove(&diff.price_tick);
                    } else {
                        // Insert or update
                        self.asks.insert(diff.price_tick, diff.size);
                    }
                }
            }
        }

        // Update the sequence number
        self.seq = msg.seq;

        // Verify the checksum
        self.verify_checksum(symbol, msg.seq, msg.checksum)
    }

    /// Verifies the checksum of the L2Book
    /// Format: "symbol|seq|best_bid_price|best_ask_price"
    fn verify_checksum(&self, symbol: &str, seq: u64, expected_checksum: u32) -> bool {
        let (bb, _) = self.best_bid().unwrap_or((0, 0.0));
        let (aa, _) = self.best_ask().unwrap_or((0, 0.0));
        let payload = format!("{}|{}|{}|{}", symbol, seq, bb, aa);

        let mut hasher = Adler32::new();
        hasher.write_slice(payload.as_bytes());
        let computed = hasher.checksum();

        computed == expected_checksum
    }

    /// Calculates the spread in ticks
    #[inline]
    pub fn spread_ticks(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calculates the spread in real price
    #[inline]
    pub fn spread(&self) -> Option<f64> {
        self.spread_ticks().map(|s| s as f64 * self.tick_size)
    }

    /// Returns the number of levels on the bid side
    #[inline]
    pub fn bid_depth(&self) -> usize {
        self.bids.len()
    }

    /// Returns the number of levels on the ask side
    #[inline]
    pub fn ask_depth(&self) -> usize {
        self.asks.len()
    }

    /// Returns the N best bid levels (sorted by price descending)
    pub fn top_bids(&self, n: usize) -> Vec<(Price, Qty)> {
        let mut levels: Vec<_> = self.bids.iter().map(|(p, q)| (*p, *q)).collect();
        levels.sort_by_key(|(p, _)| std::cmp::Reverse(*p));
        levels.into_iter().take(n).collect()
    }

    /// Returns the N best ask levels (sorted by price ascending)
    pub fn top_asks(&self, n: usize) -> Vec<(Price, Qty)> {
        let mut levels: Vec<_> = self.asks.iter().map(|(p, q)| (*p, *q)).collect();
        levels.sort_by_key(|(p, _)| *p);
        levels.into_iter().take(n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mid_price() {
        let mut book = L2Book::new(0.1, 0.001);
        book.bids.insert(1000, 10.0);
        book.asks.insert(1002, 8.0);

        assert_eq!(book.mid_price_ticks(), Some(1001.0));
        let mid = book.mid_price().unwrap();
        assert!((mid - 100.1).abs() < 1e-9);
    }

    #[test]
    fn test_orderbook_imbalance() {
        let mut book = L2Book::new(0.1, 0.001);
        book.bids.insert(1000, 15.0);
        book.asks.insert(1002, 5.0);

        let imbalance = book.orderbook_imbalance().unwrap();
        // (15 - 5) / (15 + 5) = 10 / 20 = 0.5
        assert!((imbalance - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_spread() {
        let mut book = L2Book::new(0.1, 0.001);
        book.bids.insert(1000, 10.0);
        book.asks.insert(1003, 8.0);

        assert_eq!(book.spread_ticks(), Some(3));
        let spread = book.spread().unwrap();
        assert!((spread - 0.3).abs() < 1e-9);
    }
}
