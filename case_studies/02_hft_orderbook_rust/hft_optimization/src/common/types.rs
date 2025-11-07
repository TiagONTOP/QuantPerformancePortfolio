use serde::{Deserialize, Serialize};

/// Type for prices in ticks (integer)
pub type Price = i64;

/// Type for quantities (aggregated size)
pub type Qty = f64;

/// Side of the orderbook (Bid = buy, Ask = sell)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Bid,
    Ask,
}
