use hft_optimisation::suboptimal::LOBSimulator;
use hft_optimisation::common::messages::L2UpdateMsg;
use plotters::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::error::Error;

/// Configuration enum to easily switch between optimized and suboptimal implementations.
///
/// This allows you to run the same simulation with different orderbook implementations
/// by simply changing a single line in main() without modifying any other logic.
///
/// Usage:
/// - `BookType::Optimized` - High-performance implementation with optimized data structures
/// - `BookType::Suboptimal` - Reference implementation for comparison
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum BookType {
    Optimized,
    Suboptimal,
}

/// Wrapper enum to hold either book type and provide a unified interface.
///
/// This enum abstracts over the two implementations and dispatches method calls
/// to the appropriate underlying book type. This allows the simulation logic
/// to remain identical regardless of which implementation is used.
enum Book {
    Optimized(hft_optimisation::optimized::L2Book),
    Suboptimal(hft_optimisation::suboptimal::book::L2Book),
}

impl Book {
    /// Creates a new book of the specified type with given parameters
    fn new(book_type: BookType, tick_size: f64, lot_size: f64) -> Self {
        match book_type {
            BookType::Optimized => Book::Optimized(hft_optimisation::optimized::L2Book::new(tick_size, lot_size)),
            BookType::Suboptimal => Book::Suboptimal(hft_optimisation::suboptimal::book::L2Book::new(tick_size, lot_size)),
        }
    }

    /// Updates the orderbook with a new L2 update message
    fn update(&mut self, upd: &L2UpdateMsg, venue: &str) {
        match self {
            Book::Optimized(book) => { book.update(upd, venue); },
            Book::Suboptimal(book) => { book.update(upd, venue); },
        }
    }

    /// Returns the top N bid levels (price_tick, quantity)
    fn top_bids(&self, depth: usize) -> Vec<(i64, f64)> {
        match self {
            Book::Optimized(book) => book.top_bids(depth),
            Book::Suboptimal(book) => book.top_bids(depth),
        }
    }

    /// Returns the top N ask levels (price_tick, quantity)
    fn top_asks(&self, depth: usize) -> Vec<(i64, f64)> {
        match self {
            Book::Optimized(book) => book.top_asks(depth),
            Book::Suboptimal(book) => book.top_asks(depth),
        }
    }
}

/// Lightweight snapshot structure for plotting (avoids cloning entire L2Book)
struct PlotSnapshot {
    top_bids: Vec<(i64, f64)>, // (price_tick, qty)
    top_asks: Vec<(i64, f64)>,
    mid_price_dollars: Option<f64>,
}

/// Captured BBO information for later validation/logging
struct BboRecord {
    best_bid_tick: Option<i64>,
    best_bid_qty: Option<f64>,
    best_bid_dollars: Option<f64>,
    best_ask_tick: Option<i64>,
    best_ask_qty: Option<f64>,
    best_ask_dollars: Option<f64>,
    mid_price_dollars: Option<f64>,
}

/// Generates a color with linear interpolation between two colors
fn interpolate_color(ratio: f64, start: RGBColor, end: RGBColor) -> RGBColor {
    let r = (start.0 as f64 + (end.0 as f64 - start.0 as f64) * ratio) as u8;
    let g = (start.1 as f64 + (end.1 as f64 - start.1 as f64) * ratio) as u8;
    let b = (start.2 as f64 + (end.2 as f64 - start.2 as f64) * ratio) as u8;
    RGBColor(r, g, b)
}

/// Generates a color for bids (light blue to dark blue)
/// depth_ratio: 0.0 = best bid (dark blue), 1.0 = furthest bid (light blue)
fn bid_color(depth_ratio: f64) -> RGBColor {
    let light_blue = RGBColor(173, 216, 230); // Light blue
    let dark_blue = RGBColor(0, 0, 139);     // Dark blue
    interpolate_color(depth_ratio, dark_blue, light_blue)
}

/// Generates a color for asks (light red to dark red)
/// depth_ratio: 0.0 = best ask (dark red), 1.0 = furthest ask (light red)
fn ask_color(depth_ratio: f64) -> RGBColor {
    let light_red = RGBColor(255, 182, 193); // Light red
    let dark_red = RGBColor(139, 0, 0);     // Dark red
    interpolate_color(depth_ratio, dark_red, light_red)
}

/// Extracts a snapshot of the book that is internally consistent (mid derived from captured BBO)
fn capture_snapshot(book: &Book, depth_levels: usize, tick_size: f64) -> (PlotSnapshot, BboRecord) {
    let top_bids = book.top_bids(depth_levels);
    let top_asks = book.top_asks(depth_levels);

    let best_bid = top_bids.first().copied();
    let best_ask = top_asks.first().copied();
    let best_bid_tick = best_bid.map(|(p, _)| p);
    let best_ask_tick = best_ask.map(|(p, _)| p);
    let best_bid_qty = best_bid.map(|(_, q)| q);
    let best_ask_qty = best_ask.map(|(_, q)| q);
    let best_bid_dollars = best_bid_tick.map(|tick| tick as f64 * tick_size);
    let best_ask_dollars = best_ask_tick.map(|tick| tick as f64 * tick_size);

    let mid_price_dollars = match (best_bid_tick, best_ask_tick) {
        (Some(bid), Some(ask)) if bid < ask => {
            Some(((bid as f64 + ask as f64) * 0.5) * tick_size)
        }
        _ => None,
    };

    let snapshot = PlotSnapshot {
        top_bids,
        top_asks,
        mid_price_dollars,
    };

    let bbo = BboRecord {
        best_bid_tick,
        best_bid_qty,
        best_bid_dollars,
        best_ask_tick,
        best_ask_qty,
        best_ask_dollars,
        mid_price_dollars,
    };

    (snapshot, bbo)
}

fn main() -> Result<(), Box<dyn Error>> {
    // ============================================================
    // CONFIGURATION: Change BookType here to switch implementation
    // ============================================================
    // BookType::Optimized   -> Uses the high-performance optimized implementation
    // BookType::Suboptimal  -> Uses the reference suboptimal implementation
    let book_type = BookType::Suboptimal;

    // Configuration
    let num_snapshots = 200;     // Number of snapshots to capture (increased for more detail)
    let sample_every = 100;      // Take a snapshot every N updates
    let depth_levels = 10;       // Number of price levels to display
    let output_file = "orderbook_timeseries.png";
    let bbo_log_file = "orderbook_bbo_log.csv";

    println!("=== Orderbook Simulation ===");
    println!("Implementation: {:?}", book_type);
    println!("Generating {} orderbook snapshots (1 every {} updates)...", num_snapshots, sample_every);

    // Custom configuration with much higher volatility for a highly visible Brownian motion
    let tick_size = 0.1;
    let lot_size = 0.001;
    let sim_config = hft_optimisation::suboptimal::simulator::SimConfig {
        symbol: "BTC-USDT".to_string(),
        tick_size,
        lot_size,
        depth: 20,
        dt_ms: 100,
        sigma_daily: 10.0,  // Very high volatility (1000% annualized for a highly visible Brownian motion)
    };

    // Initialize simulator with custom configuration
    let mut sim = LOBSimulator::with_config(sim_config);

    // Create a book with the same parameters using the selected book type
    let mut book = Book::new(book_type, tick_size, lot_size);

    // Bootstrap (full book)
    let boot = sim.bootstrap_update();

    // Apply bootstrap
    book.update(&boot, "SIM");

    // Collect snapshots (take 1 snapshot every sample_every updates)
    let mut snapshots: Vec<PlotSnapshot> = Vec::new();
    let mut bbo_log: Vec<BboRecord> = Vec::new();
    let (snapshot, bbo_record) = capture_snapshot(&book, depth_levels, tick_size);
    snapshots.push(snapshot);
    bbo_log.push(bbo_record);

    for i in 1..num_snapshots {
        // Perform sample_every updates
        for _ in 0..sample_every {
            let upd = sim.next_update();
            book.update(&upd, "SIM");
        }
        // Save the snapshot (only necessary data, not entire book)
        let (snapshot, bbo_record) = capture_snapshot(&book, depth_levels, tick_size);
        snapshots.push(snapshot);
        bbo_log.push(bbo_record);

        if i % 50 == 0 {
            println!("  Progress: {}/{} snapshots", i, num_snapshots);
        }
    }

    println!("Snapshots collected. Writing BBO log to {}...", bbo_log_file);
    write_bbo_log(bbo_log_file, &bbo_log)?;
    println!("BBO log saved. Generating chart...");

    // Find min/max prices for Y-axis scale (in dollars)
    let mut min_price_dollars = f64::MAX;
    let mut max_price_dollars = f64::MIN;
    let mut max_qty = 0.0f64;

    for snapshot in &snapshots {
        for (price_tick, qty) in snapshot.top_bids.iter().chain(snapshot.top_asks.iter()) {
            let price_dollars = *price_tick as f64 * tick_size;
            min_price_dollars = min_price_dollars.min(price_dollars);
            max_price_dollars = max_price_dollars.max(price_dollars);
            max_qty = max_qty.max(*qty);
        }
    }

    // Validate Y-range (check if any orders were found)
    if min_price_dollars == f64::MAX || max_price_dollars == f64::MIN {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "No orders found in snapshots, cannot determine price range.",
        )));
    }

    // Add a margin
    let price_range = max_price_dollars - min_price_dollars;
    min_price_dollars -= price_range / 10.0;
    max_price_dollars += price_range / 10.0;

    // Create the chart
    let root = BitMapBackend::new(output_file, (1600, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Orderbook Time Series", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(80)
        .build_cartesian_2d(
            0f64..(num_snapshots as f64),
            min_price_dollars..max_price_dollars
        )?;

    chart.configure_mesh()
        .x_desc("Snapshot (Time)")
        .y_desc("Price (USD)")
        .draw()?;

    // Draw rectangles for each price level at each timestamp
    for (t, snapshot) in snapshots.iter().enumerate() {
        // Draw bids (from top to bottom, from best price to worst)
        for (i, (price_tick, qty)) in snapshot.top_bids.iter().enumerate() {
            let depth_ratio = i as f64 / depth_levels.max(1) as f64;
            let color = bid_color(depth_ratio);

            // Convert price to dollars
            let price_dollars = *price_tick as f64 * tick_size;

            // Normalize width based on quantity (avoid division by zero)
            let width = if max_qty > 0.0 {
                (qty / max_qty) * 0.8
            } else {
                0.0
            };

            // Rectangle centered on the timestamp
            let x_start = t as f64 + 0.5 - width / 2.0;
            let x_end = t as f64 + 0.5 + width / 2.0;

            // Rectangle height in dollars (0.5 * tick_size for adjacent levels to touch)
            let rect_height = 0.5 * tick_size;

            chart.draw_series(std::iter::once(Rectangle::new([
                (x_start, price_dollars - rect_height),
                (x_end, price_dollars + rect_height)
            ], color.filled())))?;
        }

        // Draw asks (from bottom to top, from best price to worst)
        for (i, (price_tick, qty)) in snapshot.top_asks.iter().enumerate() {
            let depth_ratio = i as f64 / depth_levels.max(1) as f64;
            let color = ask_color(depth_ratio);

            // Convert price to dollars
            let price_dollars = *price_tick as f64 * tick_size;

            // Normalize width based on quantity (avoid division by zero)
            let width = if max_qty > 0.0 {
                (qty / max_qty) * 0.8
            } else {
                0.0
            };

            // Rectangle centered on the timestamp
            let x_start = t as f64 + 0.5 - width / 2.0;
            let x_end = t as f64 + 0.5 + width / 2.0;

            // Rectangle height in dollars (0.5 * tick_size for adjacent levels to touch)
            let rect_height = 0.5 * tick_size;

            chart.draw_series(std::iter::once(Rectangle::new([
                (x_start, price_dollars - rect_height),
                (x_end, price_dollars + rect_height)
            ], color.filled())))?;
        }
    }

    // Draw a line for the mid-price over time (in dollars)
    let mid_prices: Vec<(f64, f64)> = snapshots.iter().enumerate()
        .filter_map(|(t, snapshot)| {
            snapshot.mid_price_dollars.map(|mid_dollars| {
                // +0.5 keeps the mid-line centered inside the time bucket
                (t as f64 + 0.5, mid_dollars)
            })
        })
        .collect();

    chart.draw_series(LineSeries::new(
        mid_prices,
        &BLACK,
    ))?.label("Mid Price").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    println!("Chart saved to: {}", output_file);
    println!("BBO log saved to: {}", bbo_log_file);
    println!("Number of snapshots: {}", num_snapshots);
    println!("Min price: ${:.2}, Max price: ${:.2}", min_price_dollars, max_price_dollars);
    println!("Max quantity: {:.4}", max_qty);

    Ok(())
}

fn write_bbo_log(path: &str, entries: &[BboRecord]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "snapshot_index,best_bid_tick,best_bid_qty,best_bid_usd,best_ask_tick,best_ask_qty,best_ask_usd,mid_usd"
    )?;

    for (idx, entry) in entries.iter().enumerate() {
        let bid_tick = entry
            .best_bid_tick
            .map(|v| v.to_string())
            .unwrap_or_default();
        let bid_qty = entry
            .best_bid_qty
            .map(|v| format!("{:.8}", v))
            .unwrap_or_default();
        let bid_usd = entry
            .best_bid_dollars
            .map(|v| format!("{:.8}", v))
            .unwrap_or_default();

        let ask_tick = entry
            .best_ask_tick
            .map(|v| v.to_string())
            .unwrap_or_default();
        let ask_qty = entry
            .best_ask_qty
            .map(|v| format!("{:.8}", v))
            .unwrap_or_default();
        let ask_usd = entry
            .best_ask_dollars
            .map(|v| format!("{:.8}", v))
            .unwrap_or_default();

        let mid_usd = entry
            .mid_price_dollars
            .map(|v| format!("{:.8}", v))
            .unwrap_or_default();

        writeln!(
            writer,
            "{idx},{},{},{},{},{},{},{}",
            bid_tick, bid_qty, bid_usd, ask_tick, ask_qty, ask_usd, mid_usd
        )?;
    }

    Ok(())
}
