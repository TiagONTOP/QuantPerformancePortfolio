use hft_optimisation::suboptimal::LOBSimulator;
use plotters::prelude::*;
use std::error::Error;

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
    let dark_blue = RGBColor(0, 0, 139);      // Dark blue
    interpolate_color(depth_ratio, dark_blue, light_blue)
}

/// Generates a color for asks (light red to dark red)
/// depth_ratio: 0.0 = best ask (dark red), 1.0 = furthest ask (light red)
fn ask_color(depth_ratio: f64) -> RGBColor {
    let light_red = RGBColor(255, 182, 193); // Light red
    let dark_red = RGBColor(139, 0, 0);      // Dark red
    interpolate_color(depth_ratio, dark_red, light_red)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Configuration
    let num_snapshots = 200;     // Number of snapshots to capture (increased for more detail)
    let sample_every = 100;      // Take a snapshot every N updates
    let depth_levels = 10;       // Number of price levels to display
    let output_file = "orderbook_timeseries.png";

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

    // Create a book with the same parameters
    let mut book = hft_optimisation::suboptimal::book::L2Book::new(tick_size, lot_size);

    // Bootstrap (full book)
    let boot = sim.bootstrap_update();

    // Apply bootstrap
    book.update(&boot, "SIM");

    // Collect snapshots (take 1 snapshot every sample_every updates)
    let mut snapshots = Vec::new();
    snapshots.push(book.clone());

    for i in 1..num_snapshots {
        // Perform sample_every updates
        for _ in 0..sample_every {
            let upd = sim.next_update();
            book.update(&upd, "SIM");
        }
        // Save the snapshot
        snapshots.push(book.clone());

        if i % 50 == 0 {
            println!("  Progress: {}/{} snapshots", i, num_snapshots);
        }
    }

    println!("Snapshots collected. Generating chart...");

    // Find min/max prices for Y-axis scale (in dollars)
    let mut min_price_dollars = f64::MAX;
    let mut max_price_dollars = f64::MIN;
    let mut max_qty = 0.0f64;

    for snapshot in &snapshots {
        let top_bids = snapshot.top_bids(depth_levels);
        let top_asks = snapshot.top_asks(depth_levels);

        for (price_tick, qty) in top_bids.iter().chain(top_asks.iter()) {
            let price_dollars = *price_tick as f64 * tick_size;
            min_price_dollars = min_price_dollars.min(price_dollars);
            max_price_dollars = max_price_dollars.max(price_dollars);
            max_qty = max_qty.max(*qty);
        }
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
        let top_bids = snapshot.top_bids(depth_levels);
        let top_asks = snapshot.top_asks(depth_levels);

        // Draw bids (from top to bottom, from best price to worst)
        for (i, (price_tick, qty)) in top_bids.iter().enumerate() {
            let depth_ratio = i as f64 / depth_levels.max(1) as f64;
            let color = bid_color(depth_ratio);

            // Convert price to dollars
            let price_dollars = *price_tick as f64 * tick_size;

            // Normalize width based on quantity
            let width = (qty / max_qty) * 0.8; // 0.8 to leave some space

            // Rectangle centered on the timestamp
            let x_start = t as f64 + 0.5 - width / 2.0;
            let x_end = t as f64 + 0.5 + width / 2.0;

            // Rectangle height in dollars (0.4 * tick_size to maintain the same visual thickness)
            let rect_height = 0.4 * tick_size;

            chart.draw_series(std::iter::once(Rectangle::new([
                (x_start, price_dollars - rect_height),
                (x_end, price_dollars + rect_height)
            ], color.filled())))?;
        }

        // Draw asks (from bottom to top, from best price to worst)
        for (i, (price_tick, qty)) in top_asks.iter().enumerate() {
            let depth_ratio = i as f64 / depth_levels.max(1) as f64;
            let color = ask_color(depth_ratio);

            // Convert price to dollars
            let price_dollars = *price_tick as f64 * tick_size;

            // Normalize width based on quantity
            let width = (qty / max_qty) * 0.8;

            // Rectangle centered on the timestamp
            let x_start = t as f64 + 0.5 - width / 2.0;
            let x_end = t as f64 + 0.5 + width / 2.0;

            // Rectangle height in dollars
            let rect_height = 0.4 * tick_size;

            chart.draw_series(std::iter::once(Rectangle::new([
                (x_start, price_dollars - rect_height),
                (x_end, price_dollars + rect_height)
            ], color.filled())))?;
        }
    }

    // Draw a line for the mid-price over time (in dollars)
    let mid_prices: Vec<(f64, f64)> = snapshots.iter().enumerate()
        .filter_map(|(t, snapshot)| {
            snapshot.mid_price_ticks().map(|mid_tick| {
                let mid_dollars = mid_tick * tick_size;
                (t as f64, mid_dollars)
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
    println!("Number of snapshots: {}", num_snapshots);
    println!("Min price: ${:.2}, Max price: ${:.2}", min_price_dollars, max_price_dollars);
    println!("Max quantity: {:.4}", max_qty);

    Ok(())
}
