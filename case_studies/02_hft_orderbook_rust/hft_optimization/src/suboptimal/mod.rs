/// Module contenant les types de base (maintenant dans common::types)
#[deprecated(note = "Use crate::common::types instead")]
pub mod types;

/// Module contenant les structures de messages L2 (maintenant dans common::messages)
#[deprecated(note = "Use crate::common::messages instead")]
pub mod messages;

/// Module contenant la structure L2Book et ses méthodes (version HashMap)
pub mod book;

/// Module contenant le simulateur de carnet d'ordres
pub mod simulator;

// Ré-exporte les types communs pour compatibilité
pub use crate::common::{Price, Qty, Side, MsgType, L2Diff, L2UpdateMsg};
pub use book::L2Book;
pub use simulator::{SimConfig, LOBSimulator};
