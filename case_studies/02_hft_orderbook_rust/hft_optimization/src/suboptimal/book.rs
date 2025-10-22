use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use adler::Adler32;

use crate::suboptimal::types::{Price, Qty, Side};
use crate::suboptimal::messages::L2UpdateMsg;

/// L2 Book : carnet d'ordres agrégé par niveaux de prix
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct L2Book {
    pub seq: u64,
    pub tick_size: f64,
    pub lot_size: f64,
    pub bids: HashMap<Price, Qty>, // price_tick -> size
    pub asks: HashMap<Price, Qty>,
}

impl L2Book {
    /// Crée un nouveau L2Book vide
    pub fn new(tick_size: f64, lot_size: f64) -> Self {
        Self {
            seq: 0,
            tick_size,
            lot_size,
            bids: HashMap::new(),
            asks: HashMap::new(),
        }
    }

    /// Retourne le meilleur bid (prix le plus élevé)
    #[inline]
    pub fn best_bid(&self) -> Option<(Price, Qty)> {
        self.bids.iter().max_by_key(|(p, _)| *p).map(|(p, q)| (*p, *q))
    }

    /// Retourne le meilleur ask (prix le plus bas)
    #[inline]
    pub fn best_ask(&self) -> Option<(Price, Qty)> {
        self.asks.iter().min_by_key(|(p, _)| *p).map(|(p, q)| (*p, *q))
    }

    /// Calcule le mid-price en ticks
    /// Retourne None si le bid ou l'ask est absent
    #[inline]
    pub fn mid_price_ticks(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some((bid as f64 + ask as f64) / 2.0),
            _ => None,
        }
    }

    /// Calcule le mid-price en prix réel (mid_price_ticks * tick_size)
    #[inline]
    pub fn mid_price(&self) -> Option<f64> {
        self.mid_price_ticks().map(|mid_ticks| mid_ticks * self.tick_size)
    }

    /// Calcule l'orderbook imbalance au meilleur niveau
    /// Imbalance = (bid_size - ask_size) / (bid_size + ask_size)
    /// Retourne une valeur entre -1.0 et 1.0
    /// - Valeur positive : plus de liquidité côté bid (pression d'achat)
    /// - Valeur négative : plus de liquidité côté ask (pression de vente)
    /// - Retourne None si le bid ou l'ask est absent
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

    /// Calcule l'orderbook imbalance pondéré sur plusieurs niveaux
    /// depth: nombre de niveaux à considérer de chaque côté
    /// Retourne None si le livre est vide
    pub fn orderbook_imbalance_depth(&self, depth: usize) -> Option<f64> {
        if self.bids.is_empty() || self.asks.is_empty() {
            return None;
        }

        // Récupère les N meilleurs bids (triés décroissants par prix)
        let mut bid_levels: Vec<_> = self.bids.iter().collect();
        bid_levels.sort_by_key(|(p, _)| std::cmp::Reverse(*p));
        let bid_levels: Vec<_> = bid_levels.into_iter().take(depth).collect();

        // Récupère les N meilleurs asks (triés croissants par prix)
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

    /// Met à jour le L2Book avec un message L2UpdateMsg
    /// Applique toutes les différences du message et met à jour le numéro de séquence
    /// Retourne true si le checksum est valide, false sinon
    pub fn update(&mut self, msg: &L2UpdateMsg, symbol: &str) -> bool {
        // Applique chaque différence
        for diff in &msg.diffs {
            match diff.side {
                Side::Bid => {
                    if diff.size == 0.0 {
                        // Suppression du niveau
                        self.bids.remove(&diff.price_tick);
                    } else {
                        // Insertion ou mise à jour
                        self.bids.insert(diff.price_tick, diff.size);
                    }
                }
                Side::Ask => {
                    if diff.size == 0.0 {
                        // Suppression du niveau
                        self.asks.remove(&diff.price_tick);
                    } else {
                        // Insertion ou mise à jour
                        self.asks.insert(diff.price_tick, diff.size);
                    }
                }
            }
        }

        // Met à jour le numéro de séquence
        self.seq = msg.seq;

        // Vérifie le checksum
        self.verify_checksum(symbol, msg.seq, msg.checksum)
    }

    /// Vérifie le checksum du L2Book
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

    /// Calcule le spread en ticks
    #[inline]
    pub fn spread_ticks(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calcule le spread en prix réel
    #[inline]
    pub fn spread(&self) -> Option<f64> {
        self.spread_ticks().map(|s| s as f64 * self.tick_size)
    }

    /// Retourne le nombre de niveaux côté bid
    #[inline]
    pub fn bid_depth(&self) -> usize {
        self.bids.len()
    }

    /// Retourne le nombre de niveaux côté ask
    #[inline]
    pub fn ask_depth(&self) -> usize {
        self.asks.len()
    }

    /// Retourne les N meilleurs niveaux de bids (triés par prix décroissant)
    pub fn top_bids(&self, n: usize) -> Vec<(Price, Qty)> {
        let mut levels: Vec<_> = self.bids.iter().map(|(p, q)| (*p, *q)).collect();
        levels.sort_by_key(|(p, _)| std::cmp::Reverse(*p));
        levels.into_iter().take(n).collect()
    }

    /// Retourne les N meilleurs niveaux d'asks (triés par prix croissant)
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
