# Optimisation du Orderbook HFT - Guide Complet

## Vue d'ensemble du projet

Ce projet implémente et compare deux versions d'un carnet d'ordres L2 (Level 2 Order Book) pour du trading haute fréquence :

1. **Version Suboptimale** (`src/suboptimal/`) : Utilise HashMap pour le stockage
2. **Version Optimisée** (`src/optimized/`) : Utilise Vec avec indexation directe

## Structure du projet

```
src/
├── common/              # Types et messages partagés
│   ├── types.rs         # Price, Qty, Side
│   ├── messages.rs      # L2UpdateMsg, L2Diff, MsgType
│   └── mod.rs
│
├── suboptimal/          # Implémentation HashMap (baseline)
│   ├── book.rs          # L2Book avec HashMap<Price, Qty>
│   ├── simulator.rs     # LOBSimulator (générateur de données)
│   ├── types.rs         # (deprecated, re-export de common::types)
│   ├── messages.rs      # (deprecated, re-export de common::messages)
│   └── mod.rs
│
├── optimized/           # Implémentation Vec (optimisée)
│   ├── book.rs          # L2Book avec Vec<Qty> + cache
│   └── mod.rs
│
└── lib.rs               # Point d'entrée de la bibliothèque

benches/
├── orderbook_update.rs            # Benchmarks de la version suboptimale
├── optimized_vs_suboptimal.rs     # Benchmarks comparatifs
└── README.md                       # Guide d'utilisation des benchmarks

src/bin/
└── plot_orderbook.rs              # Génération de visualisations
```

## Technique d'optimisation : Vec avec indexation directe

### Principe

Au lieu de stocker les niveaux de prix dans un HashMap :
```rust
HashMap<Price, Qty>  // HashMap de (prix_en_ticks, quantité)
```

On utilise un Vec où **l'indice correspond au prix** :
```rust
Vec<Qty>  // Vec[index] = quantité au prix (anchor ± index)
```

### Conversion prix ↔ index

```rust
// Pour les bids (prix décroissants)
bid_anchor = 650_000    // Prix de référence
Index 0 = prix 650_000  // Meilleur bid
Index 1 = prix 649_999
Index 2 = prix 649_998
...

// Pour les asks (prix croissants)
ask_anchor = 650_010    // Prix de référence
Index 0 = prix 650_010  // Meilleur ask
Index 1 = prix 650_011
Index 2 = prix 650_012
...
```

### Avantages de cette approche

1. **Accès O(1)** : `price_to_index()` est une simple soustraction
2. **Cache CPU** : Données contigües en mémoire = excellent cache locality
3. **Pas de hash** : Pas de calcul de hash, pas de collisions
4. **Lectures ultra-rapides** : Cache du best_bid/best_ask pour O(1)

## Résultats de performance

### Résumé des gains

| Opération | Amélioration | Impact |
|-----------|--------------|--------|
| **Update single** | 1.6x plus rapide | Critique pour HFT |
| **Update batch 100** | 1.9x plus rapide | Traitement par lot |
| **best_bid()** | 134x plus rapide | 🚀 Lecture critique |
| **mid_price()** | 511x plus rapide | 🚀🚀🚀 Sub-nanoseconde |
| **Profondeur 50** | 3.25x plus rapide | Scalabilité excellente |

### Détails complets

Voir [OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md) pour l'analyse complète.

## Installation et utilisation

### Prérequis

```bash
# Rust 1.80+
rustc --version

# Dépendances système
# - Windows: MSVC ou GNU
# - Linux: gcc
# - macOS: clang
```

### Compilation

```bash
# Mode debug
cargo build

# Mode release (optimisations complètes)
cargo build --release
```

### Tests

```bash
# Tous les tests
cargo test

# Tests spécifiques
cargo test --lib optimized
cargo test --lib suboptimal
```

### Benchmarks

```bash
# Benchmark de base (suboptimal)
cargo bench --bench orderbook_update

# Benchmark comparatif (optimized vs suboptimal)
cargo bench --bench optimized_vs_suboptimal

# Benchmark spécifique
cargo bench --bench optimized_vs_suboptimal -- update_comparison
```

Les résultats sont sauvegardés dans `target/criterion/` avec des rapports HTML.

## Utilisation du code

### Exemple : Version suboptimale (HashMap)

```rust
use hft_optimisation::suboptimal::{LOBSimulator, book::L2Book};

fn main() {
    // Créer un simulateur
    let mut sim = LOBSimulator::new();

    // Créer un orderbook
    let mut book = L2Book::new(0.1, 0.001);

    // Bootstrap (initialisation)
    let boot = sim.bootstrap_update();
    book.update(&boot, "BTC-USDT");

    // Mises à jour continues
    for _ in 0..1000 {
        let update = sim.next_update();
        book.update(&update, "BTC-USDT");

        // Lire les données
        if let Some((bid_price, bid_qty)) = book.best_bid() {
            println!("Best bid: {} @ {}", bid_price, bid_qty);
        }

        if let Some(mid) = book.mid_price() {
            println!("Mid price: ${:.2}", mid);
        }
    }
}
```

### Exemple : Version optimisée (Vec)

```rust
use hft_optimisation::suboptimal::LOBSimulator;
use hft_optimisation::optimized::book::L2Book;

fn main() {
    // Créer un simulateur
    let mut sim = LOBSimulator::new();

    // Créer un orderbook optimisé avec capacité initiale
    let mut book = L2Book::with_capacity(0.1, 0.001, 2000);

    // Bootstrap
    let boot = sim.bootstrap_update();
    book.update(&boot, "BTC-USDT");

    // Mises à jour continues (même API qu'avant)
    for _ in 0..1000 {
        let update = sim.next_update();
        book.update(&update, "BTC-USDT");

        // API identique à la version suboptimale
        if let Some(mid) = book.mid_price() {
            println!("Mid price: ${:.2}", mid);
        }
    }
}
```

### Visualisation du orderbook

```bash
# Générer une visualisation du orderbook
cargo run --bin plot_orderbook

# Le graphique est sauvegardé dans orderbook_timeseries.png
```

## API du L2Book

### Création

```rust
// Capacité par défaut (1000 niveaux)
let book = L2Book::new(tick_size, lot_size);

// Capacité personnalisée
let book = L2Book::with_capacity(tick_size, lot_size, 2000);
```

### Mise à jour

```rust
// Retourne true si le checksum est valide
let is_valid = book.update(&msg, "SYMBOL");
```

### Lectures - O(1) avec version optimisée

```rust
// Meilleurs prix
let (bid_price, bid_qty) = book.best_bid()?;
let (ask_price, ask_qty) = book.best_ask()?;

// Prix dérivés
let mid_price_ticks = book.mid_price_ticks()?;  // En ticks
let mid_price_usd = book.mid_price()?;          // En dollars

// Spread
let spread_ticks = book.spread_ticks()?;
let spread_usd = book.spread()?;

// Imbalance (ratio bid/ask)
let imbalance = book.orderbook_imbalance()?;        // Meilleur niveau
let imbalance_5 = book.orderbook_imbalance_depth(5)?;  // Sur 5 niveaux

// Profondeur
let bid_depth = book.bid_depth();  // Nombre de niveaux bid
let ask_depth = book.ask_depth();  // Nombre de niveaux ask

// Top N niveaux
let top_10_bids = book.top_bids(10);
let top_10_asks = book.top_asks(10);
```

## Configuration du simulateur

```rust
use hft_optimisation::suboptimal::simulator::SimConfig;

let config = SimConfig {
    symbol: "BTC-USDT".to_string(),
    tick_size: 0.1,          // 0.1 USD par tick
    lot_size: 0.001,         // Taille minimum de lot
    depth: 20,               // 20 niveaux de chaque côté
    dt_ms: 100,              // Mise à jour toutes les 100ms
    sigma_daily: 0.60,       // Volatilité 60% annualisée
};

let mut sim = LOBSimulator::with_config(config);
```

## Trade-offs et choix d'implémentation

### Quand utiliser la version optimisée (Vec) ?

✅ **OUI** si :
- Latence critique (< 1 µs par update)
- Lectures fréquentes du best_bid/best_ask
- Carnet avec 10-50 niveaux de profondeur
- Range de prix raisonnable (< 10,000 ticks)

❌ **NON** si :
- Prix très dispersés (range > 100,000 ticks)
- Carnet très sparse (peu de niveaux actifs)
- Pas de contrainte de latence
- Prototypage rapide

### Quand utiliser la version suboptimale (HashMap) ?

✅ **OUI** si :
- Prix peuvent être très dispersés
- Carnet sparse avec gaps importants
- Pas besoin d'optimisation extrême
- Simplicité et maintenance prioritaires

## Métriques de performance

### Hardware de référence
- CPU : Intel/AMD moderne (2020+)
- RAM : 16GB+
- OS : Windows/Linux/macOS

### Latences typiques (version optimisée)

```
Operation              Latency    Throughput
────────────────────────────────────────────────
update()               0.97 µs    ~1.03M ops/s
best_bid()             1.19 ns    ~840M ops/s
mid_price()            0.65 ns    ~1.5B ops/s
orderbook_imbalance()  0.67 ns    ~1.5B ops/s
top_bids(10)           402 ns     ~2.5M ops/s
```

### Consommation CPU estimée

Pour un système HFT typique :
- 10,000 updates/sec
- 100,000 lectures mid-price/sec

**Version HashMap** : ~5% CPU
**Version Vec** : **~1% CPU** (5x moins)

## Limitations connues

### Version optimisée (Vec)

1. **Expansion dynamique** : Resize du Vec peut causer un spike de latence
   - Mitigation : Pré-allouer avec `with_capacity()`

2. **Mémoire fixe** : Utilise plus de mémoire si le carnet est sparse
   - Mitigation : Ajuster la capacité initiale

3. **Range de prix** : Performances dégradées si range > capacité
   - Mitigation : Augmenter la capacité ou utiliser HashMap

## Benchmarks et validation

### Exécuter les benchmarks

```bash
# Benchmark complet avec rapport HTML
cargo bench

# Benchmark rapide (10 secondes)
cargo bench --bench optimized_vs_suboptimal -- --quick

# Benchmark spécifique
cargo bench -- update_comparison/vec_single_update
```

### Valider la correction

```bash
# Tests unitaires
cargo test

# Tests d'intégration
cargo test --test '*'

# Tests de non-régression
cargo test --release
```

## Prochaines optimisations possibles

1. **SIMD** : Utiliser AVX2/AVX-512 pour les calculs vectoriels
2. **Zero-copy** : Éliminer les allocations dans les chemins chauds
3. **Lock-free** : Support multi-thread sans locks
4. **Custom allocator** : Pool de mémoire pré-alloué
5. **Inline assembly** : Optimiser les chemins critiques

## Références et documentation

- [OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md) - Résultats détaillés
- [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) - Analyse des benchmarks
- [benches/README.md](benches/README.md) - Guide des benchmarks
- [Criterion.rs](https://github.com/bheisler/criterion.rs) - Framework de benchmarking

## License

Ce projet est un exemple éducatif pour l'optimisation de systèmes HFT.

## Contact et support

Pour toute question sur l'implémentation ou les optimisations, consulter :
- La documentation inline dans le code
- Les tests unitaires pour des exemples d'usage
- Les benchmarks pour des cas d'usage réels
