use hft_optimisation::suboptimal::LOBSimulator;
use plotters::prelude::*;
use std::error::Error;

/// Génère une couleur avec interpolation linéaire entre deux couleurs
fn interpolate_color(ratio: f64, start: RGBColor, end: RGBColor) -> RGBColor {
    let r = (start.0 as f64 + (end.0 as f64 - start.0 as f64) * ratio) as u8;
    let g = (start.1 as f64 + (end.1 as f64 - start.1 as f64) * ratio) as u8;
    let b = (start.2 as f64 + (end.2 as f64 - start.2 as f64) * ratio) as u8;
    RGBColor(r, g, b)
}

/// Génère une couleur pour les bids (bleu clair à bleu foncé)
/// depth_ratio: 0.0 = meilleur bid (bleu foncé), 1.0 = bid le plus éloigné (bleu clair)
fn bid_color(depth_ratio: f64) -> RGBColor {
    let light_blue = RGBColor(173, 216, 230); // Bleu clair
    let dark_blue = RGBColor(0, 0, 139);      // Bleu foncé
    interpolate_color(depth_ratio, dark_blue, light_blue)
}

/// Génère une couleur pour les asks (rouge clair à rouge foncé)
/// depth_ratio: 0.0 = meilleur ask (rouge foncé), 1.0 = ask le plus éloigné (rouge clair)
fn ask_color(depth_ratio: f64) -> RGBColor {
    let light_red = RGBColor(255, 182, 193); // Rouge clair
    let dark_red = RGBColor(139, 0, 0);      // Rouge foncé
    interpolate_color(depth_ratio, dark_red, light_red)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Configuration
    let num_snapshots = 200;     // Nombre de snapshots à capturer (augmenté pour plus de détails)
    let sample_every = 100;      // Prendre un snapshot tous les N updates
    let depth_levels = 10;       // Nombre de niveaux de prix à afficher
    let output_file = "orderbook_timeseries.png";

    println!("Génération de {} snapshots de l'orderbook (1 toutes les {} updates)...", num_snapshots, sample_every);

    // Configuration personnalisée avec beaucoup plus de volatilité pour un mouvement brownien très visible
    let tick_size = 0.1;
    let lot_size = 0.001;
    let sim_config = hft_optimisation::suboptimal::simulator::SimConfig {
        symbol: "BTC-USDT".to_string(),
        tick_size,
        lot_size,
        depth: 20,
        dt_ms: 100,
        sigma_daily: 10.0,  // Volatilité très augmentée (1000% annualisé pour un mouvement brownien très visible)
    };

    // Initialisation du simulateur avec configuration personnalisée
    let mut sim = LOBSimulator::with_config(sim_config);

    // Créer un livre avec les mêmes paramètres
    let mut book = hft_optimisation::suboptimal::book::L2Book::new(tick_size, lot_size);

    // Bootstrap (full book)
    let boot = sim.bootstrap_update();

    // Application du bootstrap
    book.update(&boot, "SIM");

    // Collecte des snapshots (prendre 1 snapshot tous les sample_every updates)
    let mut snapshots = Vec::new();
    snapshots.push(book.clone());

    for i in 1..num_snapshots {
        // Faire sample_every updates
        for _ in 0..sample_every {
            let upd = sim.next_update();
            book.update(&upd, "SIM");
        }
        // Sauvegarder le snapshot
        snapshots.push(book.clone());

        if i % 50 == 0 {
            println!("  Progress: {}/{} snapshots", i, num_snapshots);
        }
    }

    println!("Snapshots collectés. Génération du graphique...");

    // Trouver les prix min/max pour l'échelle Y (en dollars)
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

    // Ajouter une marge
    let price_range = max_price_dollars - min_price_dollars;
    min_price_dollars -= price_range / 10.0;
    max_price_dollars += price_range / 10.0;

    // Création du graphique
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

    // Dessiner les rectangles pour chaque niveau de prix à chaque timestamp
    for (t, snapshot) in snapshots.iter().enumerate() {
        let top_bids = snapshot.top_bids(depth_levels);
        let top_asks = snapshot.top_asks(depth_levels);

        // Dessiner les bids (du haut vers le bas, du meilleur prix au moins bon)
        for (i, (price_tick, qty)) in top_bids.iter().enumerate() {
            let depth_ratio = i as f64 / depth_levels.max(1) as f64;
            let color = bid_color(depth_ratio);

            // Convertir le prix en dollars
            let price_dollars = *price_tick as f64 * tick_size;

            // Normaliser la largeur en fonction de la quantité
            let width = (qty / max_qty) * 0.8; // 0.8 pour laisser un peu d'espace

            // Rectangle centré sur le timestamp
            let x_start = t as f64 + 0.5 - width / 2.0;
            let x_end = t as f64 + 0.5 + width / 2.0;

            // Hauteur du rectangle en dollars (0.4 * tick_size pour garder la même épaisseur visuelle)
            let rect_height = 0.4 * tick_size;

            chart.draw_series(std::iter::once(Rectangle::new([
                (x_start, price_dollars - rect_height),
                (x_end, price_dollars + rect_height)
            ], color.filled())))?;
        }

        // Dessiner les asks (du bas vers le haut, du meilleur prix au moins bon)
        for (i, (price_tick, qty)) in top_asks.iter().enumerate() {
            let depth_ratio = i as f64 / depth_levels.max(1) as f64;
            let color = ask_color(depth_ratio);

            // Convertir le prix en dollars
            let price_dollars = *price_tick as f64 * tick_size;

            // Normaliser la largeur en fonction de la quantité
            let width = (qty / max_qty) * 0.8;

            // Rectangle centré sur le timestamp
            let x_start = t as f64 + 0.5 - width / 2.0;
            let x_end = t as f64 + 0.5 + width / 2.0;

            // Hauteur du rectangle en dollars
            let rect_height = 0.4 * tick_size;

            chart.draw_series(std::iter::once(Rectangle::new([
                (x_start, price_dollars - rect_height),
                (x_end, price_dollars + rect_height)
            ], color.filled())))?;
        }
    }

    // Dessiner une ligne pour le mid-price au fil du temps (en dollars)
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

    println!("Graphique sauvegardé dans: {}", output_file);
    println!("Nombre de snapshots: {}", num_snapshots);
    println!("Prix min: ${:.2}, Prix max: ${:.2}", min_price_dollars, max_price_dollars);
    println!("Quantité max: {:.4}", max_qty);

    Ok(())
}
