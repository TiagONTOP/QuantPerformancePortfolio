use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hft_optimisation::suboptimal::{LOBSimulator};
use hft_optimisation::common::{L2UpdateMsg};

/// Comparative benchmark between optimized (Ring buffer) and suboptimal (HashMap) implementations
fn bench_update_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("update_comparison");

    // Simulator configuration
    let mut sim = LOBSimulator::new();
    let tick_size = 0.1;
    let lot_size = 0.001;

    // Initial bootstrap
    let boot = sim.bootstrap_update();

    // Test message generation
    let num_messages = 100;
    let mut messages = Vec::with_capacity(num_messages);
    for _ in 0..num_messages {
        messages.push(sim.next_update());
    }

    // Benchmark: HashMap (suboptimal) - single update
    group.bench_function("hashmap_single_update", |b| {
        let mut book = hft_optimisation::suboptimal::book::L2Book::new(tick_size, lot_size);
        book.update(&boot, "SIM");
        let mut msg_idx = 0;

        b.iter(|| {
            let msg = &messages[msg_idx % messages.len()];
            black_box(book.update(black_box(msg), black_box("SIM")));
            msg_idx += 1;
        });
    });

    // Benchmark: Vec (optimized) - single update
    group.bench_function("vec_single_update", |b| {
        let mut book = hft_optimisation::optimized::book::L2Book::new(tick_size, lot_size);
        book.update(&boot, "SIM");
        let mut msg_idx = 0;

        b.iter(|| {
            let msg = &messages[msg_idx % messages.len()];
            black_box(book.update(black_box(msg), black_box("SIM")));
            msg_idx += 1;
        });
    });

    // Benchmark: HashMap - batch 100 updates
    group.bench_function("hashmap_batch_100", |b| {
        b.iter(|| {
            let mut book = hft_optimisation::suboptimal::book::L2Book::new(tick_size, lot_size);
            book.update(&boot, "SIM");
            for msg in &messages {
                black_box(book.update(black_box(msg), black_box("SIM")));
            }
        });
    });

    // Benchmark: Vec - batch 100 updates
    group.bench_function("vec_batch_100", |b| {
        b.iter(|| {
            let mut book = hft_optimisation::optimized::book::L2Book::new(tick_size, lot_size);
            book.update(&boot, "SIM");
            for msg in &messages {
                black_box(book.update(black_box(msg), black_box("SIM")));
            }
        });
    });

    group.finish();
}

/// Benchmark for read operations (best_bid, best_ask, mid_price)
fn bench_read_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_operations");

    // Setup
    let mut sim = LOBSimulator::new();
    let tick_size = 0.1;
    let lot_size = 0.001;
    let boot = sim.bootstrap_update();

    // Create books and apply bootstrap
    let mut book_hashmap = hft_optimisation::suboptimal::book::L2Book::new(tick_size, lot_size);
    book_hashmap.update(&boot, "SIM");

    let mut book_vec = hft_optimisation::optimized::book::L2Book::new(tick_size, lot_size);
    book_vec.update(&boot, "SIM");

    // Benchmark: HashMap - best_bid
    group.bench_function("hashmap_best_bid", |b| {
        b.iter(|| {
            black_box(book_hashmap.best_bid());
        });
    });

    // Benchmark: Vec - best_bid
    group.bench_function("vec_best_bid", |b| {
        b.iter(|| {
            black_box(book_vec.best_bid());
        });
    });

    // Benchmark: HashMap - best_ask
    group.bench_function("hashmap_best_ask", |b| {
        b.iter(|| {
            black_box(book_hashmap.best_ask());
        });
    });

    // Benchmark: Vec - best_ask
    group.bench_function("vec_best_ask", |b| {
        b.iter(|| {
            black_box(book_vec.best_ask());
        });
    });

    // Benchmark: HashMap - mid_price
    group.bench_function("hashmap_mid_price", |b| {
        b.iter(|| {
            black_box(book_hashmap.mid_price());
        });
    });

    // Benchmark: Vec - mid_price
    group.bench_function("vec_mid_price", |b| {
        b.iter(|| {
            black_box(book_vec.mid_price());
        });
    });

    // Benchmark: HashMap - orderbook_imbalance
    group.bench_function("hashmap_imbalance", |b| {
        b.iter(|| {
            black_box(book_hashmap.orderbook_imbalance());
        });
    });

    // Benchmark: Vec - orderbook_imbalance
    group.bench_function("vec_imbalance", |b| {
        b.iter(|| {
            black_box(book_vec.orderbook_imbalance());
        });
    });

    // Benchmark: HashMap - top_bids(10)
    group.bench_function("hashmap_top_bids_10", |b| {
        b.iter(|| {
            black_box(book_hashmap.top_bids(10));
        });
    });

    // Benchmark: Vec - top_bids(10)
    group.bench_function("vec_top_bids_10", |b| {
        b.iter(|| {
            black_box(book_vec.top_bids(10));
        });
    });

    group.finish();
}

/// Benchmark with different orderbook depths
fn bench_depth_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("depth_scaling");

    let tick_size = 0.1;
    let lot_size = 0.001;

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

        // Message generation
        let mut messages = Vec::with_capacity(50);
        for _ in 0..50 {
            messages.push(sim.next_update());
        }

        // HashMap
        group.bench_with_input(
            BenchmarkId::new("hashmap", depth),
            depth,
            |b, _| {
                let mut book = hft_optimisation::suboptimal::book::L2Book::new(tick_size, lot_size);
                book.update(&boot, "SIM");
                let mut msg_idx = 0;

                b.iter(|| {
                    let msg = &messages[msg_idx % messages.len()];
                    black_box(book.update(black_box(msg), black_box("SIM")));
                    msg_idx += 1;
                });
            },
        );

        // Vec
        group.bench_with_input(
            BenchmarkId::new("vec", depth),
            depth,
            |b, _| {
                let mut book = hft_optimisation::optimized::book::L2Book::new(tick_size, lot_size);
                book.update(&boot, "SIM");
                let mut msg_idx = 0;

                b.iter(|| {
                    let msg = &messages[msg_idx % messages.len()];
                    black_box(book.update(black_box(msg), black_box("SIM")));
                    msg_idx += 1;
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_update_comparison,
    bench_read_operations,
    bench_depth_scaling
);
criterion_main!(benches);
