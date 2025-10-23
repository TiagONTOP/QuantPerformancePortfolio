use serde::{Deserialize, Serialize};
use adler::Adler32;

use crate::common::types::{Price, Qty, Side};
use crate::common::messages::L2UpdateMsg;

/// L2 Book optimisé : carnet d'ordres utilisant des Vec pour un accès O(1)
///
/// Stratégie d'optimisation :
/// - Utilise des Vec au lieu de HashMap pour stocker les niveaux de prix
/// - L'indice dans le Vec correspond au prix en ticks relatif à un prix de référence
/// - Si qty == 0.0, le niveau n'existe pas (évite les allocations)
/// - Accès O(1) au lieu de O(log n) avec HashMap
/// - Cache le best_bid et best_ask pour éviter les itérations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct L2Book {
    pub seq: u64,
    pub tick_size: f64,
    pub lot_size: f64,

    // Prix de référence (anchor) pour la conversion tick <-> index
    // Les bids sont stockés relativement à ce point
    // Les asks aussi
    bid_anchor: Price,  // Prix de référence pour les bids
    ask_anchor: Price,  // Prix de référence pour les asks

    // Vecteurs pour stocker les quantités
    // Index = (price_tick - anchor)
    // Valeur = quantité (0.0 si le niveau n'existe pas)
    bids: Vec<Qty>,     // Index 0 = bid_anchor, Index 1 = bid_anchor - 1, etc.
    asks: Vec<Qty>,     // Index 0 = ask_anchor, Index 1 = ask_anchor + 1, etc.

    // Cache des meilleurs prix (pour performance)
    cached_best_bid: Option<(Price, Qty)>,
    cached_best_ask: Option<(Price, Qty)>,
}

impl L2Book {
    /// Crée un nouveau L2Book vide avec une capacité initiale pour les Vec
    ///
    /// # Arguments
    /// * `tick_size` - Taille d'un tick (ex: 0.1 pour BTC)
    /// * `lot_size` - Taille minimum de lot (ex: 0.001)
    /// * `initial_capacity` - Capacité initiale des Vec (défaut: 1000 niveaux de chaque côté)
    pub fn new(tick_size: f64, lot_size: f64) -> Self {
        Self::with_capacity(tick_size, lot_size, 1000)
    }

    /// Crée un nouveau L2Book avec une capacité personnalisée
    pub fn with_capacity(tick_size: f64, lot_size: f64, capacity: usize) -> Self {
        Self {
            seq: 0,
            tick_size,
            lot_size,
            bid_anchor: 0,
            ask_anchor: 0,
            bids: vec![0.0; capacity],
            asks: vec![0.0; capacity],
            cached_best_bid: None,
            cached_best_ask: None,
        }
    }

    /// Initialise les anchors lors du premier update (bootstrap)
    /// Doit être appelé avec le premier message qui contient des prix
    fn initialize_anchors(&mut self, msg: &L2UpdateMsg) {
        if self.bid_anchor == 0 && self.ask_anchor == 0 {
            // Trouve le premier bid et ask dans le message
            let mut first_bid = None;
            let mut first_ask = None;

            for diff in &msg.diffs {
                match diff.side {
                    Side::Bid if first_bid.is_none() => first_bid = Some(diff.price_tick),
                    Side::Ask if first_ask.is_none() => first_ask = Some(diff.price_tick),
                    _ => {}
                }
            }

            // Initialise les anchors au milieu de la capacité pour permettre l'expansion
            if let Some(bid) = first_bid {
                self.bid_anchor = bid + (self.bids.len() / 2) as i64;
            }
            if let Some(ask) = first_ask {
                self.ask_anchor = ask - (self.asks.len() / 2) as i64;
            }
        }
    }

    /// Convertit un prix en ticks en index pour le vecteur des bids
    #[inline]
    fn bid_price_to_index(&self, price: Price) -> Option<usize> {
        let offset = self.bid_anchor - price;
        if offset >= 0 && (offset as usize) < self.bids.len() {
            Some(offset as usize)
        } else {
            None
        }
    }

    /// Convertit un index du vecteur des bids en prix en ticks
    #[inline]
    fn bid_index_to_price(&self, index: usize) -> Price {
        self.bid_anchor - index as i64
    }

    /// Convertit un prix en ticks en index pour le vecteur des asks
    #[inline]
    fn ask_price_to_index(&self, price: Price) -> Option<usize> {
        let offset = price - self.ask_anchor;
        if offset >= 0 && (offset as usize) < self.asks.len() {
            Some(offset as usize)
        } else {
            None
        }
    }

    /// Convertit un index du vecteur des asks en prix en ticks
    #[inline]
    fn ask_index_to_price(&self, index: usize) -> Price {
        self.ask_anchor + index as i64
    }

    /// Agrandit le vecteur des bids si nécessaire
    fn expand_bids_if_needed(&mut self, price: Price) {
        let offset = self.bid_anchor - price;
        if offset < 0 {
            // Prix au-dessus de l'anchor, décaler l'anchor vers le haut
            let shift = (-offset) as usize;
            let mut new_bids = vec![0.0; shift];
            new_bids.extend_from_slice(&self.bids);
            self.bids = new_bids;
            self.bid_anchor += shift as i64;
        } else if (offset as usize) >= self.bids.len() {
            // Prix en dessous, étendre le vecteur
            let new_len = (offset as usize) + 100; // Ajoute 100 niveaux de marge
            self.bids.resize(new_len, 0.0);
        }
    }

    /// Agrandit le vecteur des asks si nécessaire
    fn expand_asks_if_needed(&mut self, price: Price) {
        let offset = price - self.ask_anchor;
        if offset < 0 {
            // Prix en dessous de l'anchor, décaler l'anchor vers le bas
            let shift = (-offset) as usize;
            let mut new_asks = vec![0.0; shift];
            new_asks.extend_from_slice(&self.asks);
            self.asks = new_asks;
            self.ask_anchor -= shift as i64;
        } else if (offset as usize) >= self.asks.len() {
            // Prix au-dessus, étendre le vecteur
            let new_len = (offset as usize) + 100; // Ajoute 100 niveaux de marge
            self.asks.resize(new_len, 0.0);
        }
    }

    /// Retourne le meilleur bid (prix le plus élevé) - O(1) grâce au cache
    #[inline]
    pub fn best_bid(&self) -> Option<(Price, Qty)> {
        self.cached_best_bid
    }

    /// Retourne le meilleur ask (prix le plus bas) - O(1) grâce au cache
    #[inline]
    pub fn best_ask(&self) -> Option<(Price, Qty)> {
        self.cached_best_ask
    }

    /// Recalcule le meilleur bid en parcourant le vecteur
    fn recalc_best_bid(&mut self) {
        self.cached_best_bid = None;
        for (idx, &qty) in self.bids.iter().enumerate() {
            if qty > 0.0 {
                let price = self.bid_index_to_price(idx);
                self.cached_best_bid = Some((price, qty));
                return;
            }
        }
    }

    /// Recalcule le meilleur ask en parcourant le vecteur
    fn recalc_best_ask(&mut self) {
        self.cached_best_ask = None;
        for (idx, &qty) in self.asks.iter().enumerate() {
            if qty > 0.0 {
                let price = self.ask_index_to_price(idx);
                self.cached_best_ask = Some((price, qty));
                return;
            }
        }
    }

    /// Calcule le mid-price en ticks - O(1)
    #[inline]
    pub fn mid_price_ticks(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some((bid as f64 + ask as f64) / 2.0),
            _ => None,
        }
    }

    /// Calcule le mid-price en prix réel - O(1)
    #[inline]
    pub fn mid_price(&self) -> Option<f64> {
        self.mid_price_ticks().map(|mid_ticks| mid_ticks * self.tick_size)
    }

    /// Calcule l'orderbook imbalance au meilleur niveau - O(1)
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

    /// Calcule l'orderbook imbalance sur plusieurs niveaux - O(depth)
    pub fn orderbook_imbalance_depth(&self, depth: usize) -> Option<f64> {
        let mut total_bid_size = 0.0;
        let mut total_ask_size = 0.0;
        let mut bid_count = 0;
        let mut ask_count = 0;

        // Accumule les bids
        for &qty in self.bids.iter() {
            if qty > 0.0 {
                total_bid_size += qty;
                bid_count += 1;
                if bid_count >= depth {
                    break;
                }
            }
        }

        // Accumule les asks
        for &qty in self.asks.iter() {
            if qty > 0.0 {
                total_ask_size += qty;
                ask_count += 1;
                if ask_count >= depth {
                    break;
                }
            }
        }

        if bid_count == 0 || ask_count == 0 {
            return None;
        }

        let total = total_bid_size + total_ask_size;
        if total > 1e-9 {
            Some((total_bid_size - total_ask_size) / total)
        } else {
            None
        }
    }

    /// Met à jour le L2Book avec un message L2UpdateMsg
    /// Retourne true si le checksum est valide, false sinon
    pub fn update(&mut self, msg: &L2UpdateMsg, symbol: &str) -> bool {
        // Initialise les anchors si c'est le premier update
        self.initialize_anchors(msg);

        // Applique chaque différence
        for diff in &msg.diffs {
            match diff.side {
                Side::Bid => {
                    // Agrandit le vecteur si nécessaire
                    self.expand_bids_if_needed(diff.price_tick);

                    if let Some(idx) = self.bid_price_to_index(diff.price_tick) {
                        let old_qty = self.bids[idx];
                        self.bids[idx] = diff.size;

                        // Met à jour le cache si nécessaire
                        if let Some((best_price, _)) = self.cached_best_bid {
                            if diff.price_tick > best_price && diff.size > 0.0 {
                                // Nouveau meilleur bid
                                self.cached_best_bid = Some((diff.price_tick, diff.size));
                            } else if diff.price_tick == best_price && diff.size == 0.0 {
                                // Le meilleur bid a été supprimé, recalculer
                                self.recalc_best_bid();
                            } else if diff.price_tick == best_price && diff.size != old_qty {
                                // Mise à jour du meilleur bid
                                self.cached_best_bid = Some((diff.price_tick, diff.size));
                            }
                        } else if diff.size > 0.0 {
                            // Pas de cache, calculer
                            self.recalc_best_bid();
                        }
                    }
                }
                Side::Ask => {
                    // Agrandit le vecteur si nécessaire
                    self.expand_asks_if_needed(diff.price_tick);

                    if let Some(idx) = self.ask_price_to_index(diff.price_tick) {
                        let old_qty = self.asks[idx];
                        self.asks[idx] = diff.size;

                        // Met à jour le cache si nécessaire
                        if let Some((best_price, _)) = self.cached_best_ask {
                            if diff.price_tick < best_price && diff.size > 0.0 {
                                // Nouveau meilleur ask
                                self.cached_best_ask = Some((diff.price_tick, diff.size));
                            } else if diff.price_tick == best_price && diff.size == 0.0 {
                                // Le meilleur ask a été supprimé, recalculer
                                self.recalc_best_ask();
                            } else if diff.price_tick == best_price && diff.size != old_qty {
                                // Mise à jour du meilleur ask
                                self.cached_best_ask = Some((diff.price_tick, diff.size));
                            }
                        } else if diff.size > 0.0 {
                            // Pas de cache, calculer
                            self.recalc_best_ask();
                        }
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
    fn verify_checksum(&self, symbol: &str, seq: u64, expected_checksum: u32) -> bool {
        let (bb, _) = self.best_bid().unwrap_or((0, 0.0));
        let (aa, _) = self.best_ask().unwrap_or((0, 0.0));
        let payload = format!("{}|{}|{}|{}", symbol, seq, bb, aa);

        let mut hasher = Adler32::new();
        hasher.write_slice(payload.as_bytes());
        let computed = hasher.checksum();

        computed == expected_checksum
    }

    /// Calcule le spread en ticks - O(1)
    #[inline]
    pub fn spread_ticks(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calcule le spread en prix réel - O(1)
    #[inline]
    pub fn spread(&self) -> Option<f64> {
        self.spread_ticks().map(|s| s as f64 * self.tick_size)
    }

    /// Retourne le nombre de niveaux côté bid - O(n)
    pub fn bid_depth(&self) -> usize {
        self.bids.iter().filter(|&&qty| qty > 0.0).count()
    }

    /// Retourne le nombre de niveaux côté ask - O(n)
    pub fn ask_depth(&self) -> usize {
        self.asks.iter().filter(|&&qty| qty > 0.0).count()
    }

    /// Retourne les N meilleurs niveaux de bids - O(n) mais optimisé
    pub fn top_bids(&self, n: usize) -> Vec<(Price, Qty)> {
        let mut result = Vec::with_capacity(n);
        for (idx, &qty) in self.bids.iter().enumerate() {
            if qty > 0.0 {
                let price = self.bid_index_to_price(idx);
                result.push((price, qty));
                if result.len() >= n {
                    break;
                }
            }
        }
        result
    }

    /// Retourne les N meilleurs niveaux d'asks - O(n) mais optimisé
    pub fn top_asks(&self, n: usize) -> Vec<(Price, Qty)> {
        let mut result = Vec::with_capacity(n);
        for (idx, &qty) in self.asks.iter().enumerate() {
            if qty > 0.0 {
                let price = self.ask_index_to_price(idx);
                result.push((price, qty));
                if result.len() >= n {
                    break;
                }
            }
        }
        result
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
    fn test_optimized_book_basic() {
        let mut book = L2Book::new(0.1, 0.001);

        // Créer un message de bootstrap
        let msg = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 0,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 1000, size: 10.0 },
                L2Diff { side: Side::Ask, price_tick: 1002, size: 8.0 },
            ],
            checksum: 0, // Le checksum sera vérifié séparément
        };

        book.update(&msg, "BTC-USDT");

        assert_eq!(book.best_bid(), Some((1000, 10.0)));
        assert_eq!(book.best_ask(), Some((1002, 8.0)));
        assert_eq!(book.mid_price_ticks(), Some(1001.0));
    }

    #[test]
    fn test_optimized_book_update() {
        let mut book = L2Book::new(0.1, 0.001);

        // Bootstrap
        let bootstrap = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 0,
            seq: 1,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 1000, size: 10.0 },
                L2Diff { side: Side::Bid, price_tick: 999, size: 8.0 },
                L2Diff { side: Side::Ask, price_tick: 1002, size: 8.0 },
                L2Diff { side: Side::Ask, price_tick: 1003, size: 6.0 },
            ],
            checksum: 0,
        };

        book.update(&bootstrap, "BTC-USDT");

        // Update: nouveau meilleur bid
        let update = L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: "BTC-USDT".to_string(),
            ts: 1,
            seq: 2,
            diffs: vec![
                L2Diff { side: Side::Bid, price_tick: 1001, size: 12.0 },
            ],
            checksum: 0,
        };

        book.update(&update, "BTC-USDT");

        assert_eq!(book.best_bid(), Some((1001, 12.0)));
    }
}
