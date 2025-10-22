/// Module principal contenant l'implémentation du carnet d'ordres L2
pub mod suboptimal;

// Ré-exporte les types principaux pour faciliter l'usage externe
pub use suboptimal::{L2Book, L2UpdateMsg, L2Diff, Side, Price, Qty};
