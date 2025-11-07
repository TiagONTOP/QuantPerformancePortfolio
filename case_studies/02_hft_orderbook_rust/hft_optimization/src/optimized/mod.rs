/// Optimized L2 orderbook implementation
/// Uses Vec instead of HashMap for O(1) access to price levels
pub mod book;

pub use book::L2Book;
