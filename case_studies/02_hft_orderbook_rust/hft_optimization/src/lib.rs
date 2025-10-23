/// Types et messages communs partagés entre les implémentations
pub mod common;

/// Implémentation suboptimale du carnet d'ordres L2 (HashMap)
pub mod suboptimal;

/// Implémentation optimisée du carnet d'ordres L2 (Vec)
pub mod optimized;

// Ré-exporte les types communs pour faciliter l'usage externe
pub use common::{Price, Qty, Side, MsgType, L2Diff, L2UpdateMsg};
