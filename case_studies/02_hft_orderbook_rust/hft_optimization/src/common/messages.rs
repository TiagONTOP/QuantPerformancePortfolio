use serde::{Deserialize, Deserializer, Serialize};
use crate::common::types::{Price, Side};

/// Type de message pour les mises à jour L2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MsgType {
    #[serde(rename = "l2update")]
    L2Update,
}

/// Différence individuelle pour une mise à jour L2
/// Représente un changement de prix/taille sur un niveau de l'orderbook
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct L2Diff {
    pub side: Side,
    #[serde(rename = "price")]
    pub price_tick: Price,
    #[serde(deserialize_with = "non_negative_f64")]
    pub size: f64, // >= 0.0 ; 0.0 => delete ; >0.0 => upsert
}

/// Message complet de mise à jour L2
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct L2UpdateMsg {
    #[serde(rename = "type")]
    pub msg_type: MsgType,
    pub symbol: String,
    pub ts: i64, // timestamp en nanosecondes epoch
    pub seq: u64,
    #[serde(rename = "d")]
    pub diffs: Vec<L2Diff>,
    pub checksum: u32, // adler32 (bestBid/bestAsk)
}

/// Validation stricte: pas de taille négative ou NaN
fn non_negative_f64<'de, D>(de: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let v = f64::deserialize(de)?;
    if v.is_sign_negative() {
        return Err(serde::de::Error::custom("size must be >= 0.0"));
    }
    if v.is_nan() {
        return Err(serde::de::Error::custom("size must not be NaN"));
    }
    Ok(v)
}
