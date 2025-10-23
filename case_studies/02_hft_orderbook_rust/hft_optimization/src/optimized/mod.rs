/// Implémentation optimisée du carnet d'ordres L2
/// Utilise des Vec au lieu de HashMap pour un accès O(1) aux niveaux de prix
pub mod book;

pub use book::L2Book;
