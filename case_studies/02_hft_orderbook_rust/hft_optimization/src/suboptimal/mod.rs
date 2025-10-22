/// Module contenant les types de base (Price, Qty, Side)
pub mod types;

/// Module contenant les structures de messages L2
pub mod messages;

/// Module contenant la structure L2Book et ses méthodes
pub mod book;

/// Module contenant le simulateur de carnet d'ordres
pub mod simulator;

// Ré-exporte les types et structures principales pour faciliter l'usage
pub use types::{Price, Qty, Side};
pub use messages::{MsgType, L2Diff, L2UpdateMsg};
pub use book::L2Book;
pub use simulator::{SimConfig, LOBSimulator};
