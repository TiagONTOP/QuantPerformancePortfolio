/// Common types and messages shared between implementations
pub mod common;

/// Suboptimal L2 orderbook implementation (HashMap)
pub mod suboptimal;

/// Optimized L2 orderbook implementation (Ring buffer + bitset)
pub mod optimized;

// Re-export common types for external use
pub use common::{Price, Qty, Side, MsgType, L2Diff, L2UpdateMsg};
