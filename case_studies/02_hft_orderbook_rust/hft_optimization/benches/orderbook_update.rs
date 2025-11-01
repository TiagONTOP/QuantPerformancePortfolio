use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hft_optimisation::suboptimal::{LOBSimulator, book::L2Book};

/// Benchmark for L2Book update() function
///
/// This benchmark tests orderbook update performance
/// with L2Update messages from the simulator.
fn bench_orderbook_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("orderbook_update");

    // Simulator configuration
    let mut sim = LOBSimulator::new();
    let tick_size = 0.1;
    let lot_size = 0.001;

    // Initial bootstrap
    let boot = sim.bootstrap_update();
    let mut book = L2Book::new(tick_size, lot_size);
    book.update(&boot, "SIM");

    // Test message generation
    let num_messages = 100;
    let mut messages = Vec::with_capacity(num_messages);
    for _ in 0..num_messages {
        messages.push(sim.next_update());
    }

    // Benchmark: single update
    group.bench_function("single_update", |b| {
        let mut book_clone = book.clone();
        let mut msg_idx = 0;

        b.iter(|| {
            let msg = &messages[msg_idx % messages.len()];
            black_box(book_clone.update(black_box(msg), black_box("SIM")));
            msg_idx += 1;
        });
    });

    // Benchmark: batch updates (10 updates)
    group.bench_function("batch_10_updates", |b| {
        b.iter(|| {
            let mut book_clone = book.clone();
            for i in 0..10 {
                let msg = &messages[i % messages.len()];
                black_box(book_clone.update(black_box(msg), black_box("SIM")));
            }
        });
    });

    // Benchmark: batch updates (100 updates)
    group.bench_function("batch_100_updates", |b| {
        b.iter(|| {
            let mut book_clone = book.clone();
            for msg in &messages {
                black_box(book_clone.update(black_box(msg), black_box("SIM")));
            }
        });
    });

    group.finish();
}

/// Benchmark for update() function with different diff sizes
fn bench_orderbook_update_by_diff_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("orderbook_update_by_diff_size");

    let tick_size = 0.1;
    let lot_size = 0.001;

    // Test with different depths (affects number of diffs)
    for depth in [5, 10, 20, 50].iter() {
        let sim_config = hft_optimisation::suboptimal::simulator::SimConfig {
            symbol: "BTC-USDT".to_string(),
            tick_size,
            lot_size,
            depth: *depth,
            dt_ms: 100,
            sigma_daily: 0.60,
        };

        let mut sim = LOBSimulator::with_config(sim_config);
        let boot = sim.bootstrap_update();
        let mut book = L2Book::new(tick_size, lot_size);
        book.update(&boot, "SIM");

        // Message generation
        let mut messages = Vec::with_capacity(50);
        for _ in 0..50 {
            messages.push(sim.next_update());
        }

        group.bench_with_input(
            BenchmarkId::new("depth", depth),
            depth,
            |b, _depth| {
                let mut msg_idx = 0;
                let mut book_clone = book.clone();

                b.iter(|| {
                    let msg = &messages[msg_idx % messages.len()];
                    black_box(book_clone.update(black_box(msg), black_box("SIM")));
                    msg_idx += 1;
                });
            },
        );
    }

    group.finish();
}

/// Benchmark for L2Book calculation functions
fn bench_orderbook_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("orderbook_calculations");

    // Setup
    let mut sim = LOBSimulator::new();
    let tick_size = 0.1;
    let lot_size = 0.001;
    let boot = sim.bootstrap_update();
    let mut book = L2Book::new(tick_size, lot_size);
    book.update(&boot, "SIM");

    // Benchmark: best_bid
    group.bench_function("best_bid", |b| {
        b.iter(|| {
            black_box(book.best_bid());
        });
    });

    // Benchmark: best_ask
    group.bench_function("best_ask", |b| {
        b.iter(|| {
            black_box(book.best_ask());
        });
    });

    // Benchmark: mid_price
    group.bench_function("mid_price", |b| {
        b.iter(|| {
            black_box(book.mid_price());
        });
    });

    // Benchmark: orderbook_imbalance
    group.bench_function("orderbook_imbalance", |b| {
        b.iter(|| {
            black_box(book.orderbook_imbalance());
        });
    });

    // Benchmark: orderbook_imbalance_depth(5)
    group.bench_function("orderbook_imbalance_depth_5", |b| {
        b.iter(|| {
            black_box(book.orderbook_imbalance_depth(5));
        });
    });

    // Benchmark: orderbook_imbalance_depth(10)
    group.bench_function("orderbook_imbalance_depth_10", |b| {
        b.iter(|| {
            black_box(book.orderbook_imbalance_depth(10));
        });
    });

    // Benchmark: top_bids(10)
    group.bench_function("top_bids_10", |b| {
        b.iter(|| {
            black_box(book.top_bids(10));
        });
    });

    // Benchmark: top_asks(10)
    group.bench_function("top_asks_10", |b| {
        b.iter(|| {
            black_box(book.top_asks(10));
        });
    });

    group.finish();
}

/// Benchmark for LOBSimulator
fn bench_simulator(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulator");

    // Benchmark: next_update
    group.bench_function("next_update", |b| {
        let mut sim = LOBSimulator::new();
        let boot = sim.bootstrap_update();
        let tick_size = 0.1;
        let lot_size = 0.001;
        let mut book = L2Book::new(tick_size, lot_size);
        book.update(&boot, "SIM");

        b.iter(|| {
            black_box(sim.next_update());
        });
    });

    // Benchmark: bootstrap_update
    group.bench_function("bootstrap_update", |b| {
        b.iter(|| {
            let mut sim = LOBSimulator::new();
            black_box(sim.bootstrap_update());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_orderbook_update,
    bench_orderbook_update_by_diff_size,
    bench_orderbook_calculations,
    bench_simulator
);
criterion_main!(benches);
