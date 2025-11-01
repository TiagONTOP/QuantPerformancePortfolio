/// Module containing base types (now in common::types)
#[deprecated(note = "Use crate::common::types instead")]
pub mod types;

/// Module containing L2 message structures (now in common::messages)
#[deprecated(note = "Use crate::common::messages instead")]
pub mod messages;

/// Module containing the L2Book structure and methods (HashMap version)
pub mod book;

/// Module containing the orderbook simulator
pub mod simulator;

// Re-export common types for compatibility
pub use crate::common::{Price, Qty, Side, MsgType, L2Diff, L2UpdateMsg};
pub use book::L2Book;
pub use simulator::{SimConfig, LOBSimulator};
