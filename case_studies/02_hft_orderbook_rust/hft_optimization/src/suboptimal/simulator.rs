use std::{collections::HashMap, time::{SystemTime, UNIX_EPOCH}};
use rand_distr::{Distribution, Normal, LogNormal};
use adler::Adler32;

use crate::common::types::{Price, Qty, Side};
use crate::common::messages::{L2Diff, L2UpdateMsg, MsgType};
use crate::suboptimal::book::L2Book;

/// Configuration du simulateur
pub struct SimConfig {
    pub symbol: String,
    pub tick_size: f64,
    pub lot_size: f64,
    pub depth: usize,      // nb de niveaux par côté
    pub dt_ms: u64,        // cadence d'updates
    pub sigma_daily: f64,  // sigma annualisée (par ex 60%/an)
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            symbol: "BTC-USDT".to_string(),
            tick_size: 0.1,
            lot_size: 0.001,
            depth: 20,
            dt_ms: 100,       // 10 Hz
            sigma_daily: 0.60, // 60% annualisé
        }
    }
}

/// Simulateur de carnet d'ordres L2
pub struct LOBSimulator {
    cfg: SimConfig,
    book: L2Book,
    mid_tick_f: f64,      // mid en ticks (float pour brownien)
    spread_rng: Normal<f64>,
    brownian_rng: Normal<f64>,
    size_logn: LogNormal<f64>,
    decr_factor: f64,     // décroissance des tailles avec profondeur
    seq: u64,
}

impl LOBSimulator {
    /// Crée un nouveau simulateur avec la configuration par défaut
    pub fn new() -> Self {
        Self::with_config(SimConfig::default())
    }

    /// Crée un nouveau simulateur avec une configuration personnalisée
    pub fn with_config(cfg: SimConfig) -> Self {
        // Convertit sigma_journalier -> sigma_dt
        // Brownien discret : dS = sigma * sqrt(dt) * N(0,1)
        // Ici on travaille en ticks, donc on bouge "mid_tick_f" directement
        let steps_per_day = (1000.0 / cfg.dt_ms as f64) * 60.0 * 60.0 * 24.0;
        let sigma_per_step = cfg.sigma_daily / steps_per_day.sqrt();

        let mut sim = LOBSimulator {
            book: L2Book {
                seq: 0,
                tick_size: cfg.tick_size,
                lot_size: cfg.lot_size,
                bids: HashMap::new(),
                asks: HashMap::new(),
            },
            cfg,
            mid_tick_f: 650_000.0, // 65000.0 / 0.1
            spread_rng: Normal::new(2.0, 0.8).unwrap(), // spread moyen ~2 ticks
            brownian_rng: Normal::new(0.0, sigma_per_step).unwrap(),
            size_logn: LogNormal::new(-1.2, 0.6).unwrap(), // tailles ~ lognormale
            decr_factor: 0.92,
            seq: 0,
        };

        // Seed initial : book bootstrapé
        sim.rebuild_full_book_from_state();
        sim
    }

    /// Retourne le timestamp actuel en nanosecondes
    #[inline]
    fn now_ns() -> i64 {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        (now.as_secs() as i64) * 1_000_000_000i64 + (now.subsec_nanos() as i64)
    }

    /// Calcule le checksum du livre actuel
    fn checksum(&self) -> u32 {
        let (bb, _) = self.book.best_bid().unwrap_or((0, 0.0));
        let (aa, _) = self.book.best_ask().unwrap_or((0, 0.0));
        let payload = format!("{}|{}|{}|{}", self.cfg.symbol, self.seq, bb, aa);
        let mut hasher = Adler32::new();
        hasher.write_slice(payload.as_bytes());
        hasher.checksum()
    }

    /// Avance le processus brownien du mid-price
    fn step_brownian(&mut self) {
        // Brownien sur le mid (en ticks)
        let d = self.brownian_rng.sample(&mut rand::thread_rng());
        self.mid_tick_f += d * 10.0; // échelle (10) pour rendre vivant à 10Hz
    }

    /// Échantillonne un spread stochastique
    fn sample_spread_ticks(&mut self) -> i64 {
        // Spread stochastique >= 1 tick
        let s = self.spread_rng.sample(&mut rand::thread_rng()).round();
        s.max(1.0).min(5.0) as i64
    }

    /// Calcule la taille cible pour un niveau donné
    fn target_level_size(&mut self, level: usize) -> f64 {
        // Taille ~ lognormale * décroissance^level, tronquée au lot
        let base = self.size_logn.sample(&mut rand::thread_rng());
        let dec = self.decr_factor.powi(level as i32);
        (base * dec).max(self.cfg.lot_size)
    }

    /// Reconstruit un carnet complet autour du mid-price actuel
    fn rebuild_full_book_from_state(&mut self) {
        // Reconstruit un carnet "idéal" autour du mid + spread échantillonné
        let spread = self.sample_spread_ticks();
        let mid_tick_i = self.mid_tick_f.round() as i64;

        self.book.bids.clear();
        self.book.asks.clear();

        // best bid / ask centrés autour du mid (attention parité du spread)
        let half_spread_down = spread / 2;
        let half_spread_up = spread - half_spread_down;

        // Bids
        for k in 0..self.cfg.depth {
            let p = mid_tick_i - half_spread_down - k as i64;
            let sz = self.target_level_size(k);
            self.book.bids.insert(p, sz);
        }
        // Asks
        for k in 0..self.cfg.depth {
            let p = mid_tick_i + half_spread_up + k as i64;
            let sz = self.target_level_size(k);
            self.book.asks.insert(p, sz);
        }

        self.seq = self.seq.wrapping_add(1);
        self.book.seq = self.seq;
    }

    /// Construit un "target book" et renvoie la liste des diffs vs l'état courant,
    /// en appliquant ces diffs dans le book local (cohérence du flux).
    fn make_and_apply_diffs(&mut self) -> Vec<L2Diff> {
        // Snapshot courant
        let cur_b = self.book.bids.clone();
        let cur_a = self.book.asks.clone();

        // Nouveau book cible
        let spread = self.sample_spread_ticks();
        let mid_tick_i = self.mid_tick_f.round() as i64;
        let half_spread_down = spread / 2;
        let half_spread_up = spread - half_spread_down;

        let mut tgt_b: HashMap<Price, Qty> = HashMap::new();
        let mut tgt_a: HashMap<Price, Qty> = HashMap::new();

        for k in 0..self.cfg.depth {
            tgt_b.insert(
                mid_tick_i - half_spread_down - k as i64,
                self.target_level_size(k),
            );
            tgt_a.insert(
                mid_tick_i + half_spread_up + k as i64,
                self.target_level_size(k),
            );
        }

        // Diffs = (deletes + upserts) nécessaires pour aller de cur -> tgt
        let eps = 1e-9;
        let mut diffs: Vec<L2Diff> = Vec::new();

        // Deletes (présent en cur, absent en tgt) + size changes
        for (p, s) in cur_b.iter() {
            match tgt_b.get(p) {
                Some(ns) if (ns - s).abs() > eps => {
                    diffs.push(L2Diff {
                        side: Side::Bid,
                        price_tick: *p,
                        size: *ns,
                    });
                }
                None => {
                    diffs.push(L2Diff {
                        side: Side::Bid,
                        price_tick: *p,
                        size: 0.0,
                    });
                }
                _ => {}
            }
        }
        for (p, s) in cur_a.iter() {
            match tgt_a.get(p) {
                Some(ns) if (ns - s).abs() > eps => {
                    diffs.push(L2Diff {
                        side: Side::Ask,
                        price_tick: *p,
                        size: *ns,
                    });
                }
                None => {
                    diffs.push(L2Diff {
                        side: Side::Ask,
                        price_tick: *p,
                        size: 0.0,
                    });
                }
                _ => {}
            }
        }

        // Upserts pour niveaux nouveaux (absents en cur)
        for (p, ns) in tgt_b.iter() {
            if !cur_b.contains_key(p) {
                diffs.push(L2Diff {
                    side: Side::Bid,
                    price_tick: *p,
                    size: *ns,
                });
            }
        }
        for (p, ns) in tgt_a.iter() {
            if !cur_a.contains_key(p) {
                diffs.push(L2Diff {
                    side: Side::Ask,
                    price_tick: *p,
                    size: *ns,
                });
            }
        }

        // Appliquer diffs dans le book local (maintient la cohérence du serveur)
        for d in &diffs {
            match d.side {
                Side::Bid => {
                    if d.size == 0.0 {
                        self.book.bids.remove(&d.price_tick);
                    } else {
                        self.book.bids.insert(d.price_tick, d.size);
                    }
                }
                Side::Ask => {
                    if d.size == 0.0 {
                        self.book.asks.remove(&d.price_tick);
                    } else {
                        self.book.asks.insert(d.price_tick, d.size);
                    }
                }
            }
        }

        diffs
    }

    /// Premier message: bootstrap (full book)
    pub fn bootstrap_update(&mut self) -> L2UpdateMsg {
        // Construire un "full diff" depuis un book vide :
        let empty = L2Book {
            seq: 0,
            tick_size: self.book.tick_size,
            lot_size: self.book.lot_size,
            bids: HashMap::new(),
            asks: HashMap::new(),
        };
        let _old_book = std::mem::replace(&mut self.book, empty);

        // Rebuild a partir de l'etat (mid, spread, etc.)
        self.rebuild_full_book_from_state();

        // Génère les diffs depuis le book vide
        let mut full_diffs = Vec::new();
        for (p, sz) in &self.book.bids {
            full_diffs.push(L2Diff {
                side: Side::Bid,
                price_tick: *p,
                size: *sz,
            });
        }
        for (p, sz) in &self.book.asks {
            full_diffs.push(L2Diff {
                side: Side::Ask,
                price_tick: *p,
                size: *sz,
            });
        }

        self.seq = self.seq.wrapping_add(1);
        self.book.seq = self.seq;

        L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: self.cfg.symbol.clone(),
            ts: Self::now_ns(),
            seq: self.seq,
            diffs: full_diffs,
            checksum: self.checksum(),
        }
    }

    /// Tick: avance le processus et renvoie un message l2update
    pub fn next_update(&mut self) -> L2UpdateMsg {
        self.step_brownian();
        let diffs = self.make_and_apply_diffs();

        self.seq = self.seq.wrapping_add(1);
        self.book.seq = self.seq;

        L2UpdateMsg {
            msg_type: MsgType::L2Update,
            symbol: self.cfg.symbol.clone(),
            ts: Self::now_ns(),
            seq: self.seq,
            diffs,
            checksum: self.checksum(),
        }
    }

    /// Retourne la cadence d'update en millisecondes
    pub fn dt_ms(&self) -> u64 {
        self.cfg.dt_ms
    }
}

impl Default for LOBSimulator {
    fn default() -> Self {
        Self::new()
    }
}
