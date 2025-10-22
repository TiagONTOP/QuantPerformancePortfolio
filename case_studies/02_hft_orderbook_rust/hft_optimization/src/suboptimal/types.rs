use serde::{Deserialize, Serialize};

/// Type pour les prix en ticks (nombre entier)
pub type Price = i64;

/// Type pour les quantités (taille agrégée)
pub type Qty = f64;

/// Côté du carnet d'ordres (Bid = achat, Ask = vente)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Bid,
    Ask,
}
