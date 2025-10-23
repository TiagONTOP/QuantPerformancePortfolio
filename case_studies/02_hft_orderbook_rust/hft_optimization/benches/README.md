# Benchmarks pour HFT Optimization

Ce répertoire contient les benchmarks pour tester les performances du carnet d'ordres L2.

## Installation

Les benchmarks utilisent [Criterion](https://github.com/bheisler/criterion.rs), qui est déjà inclus dans les dépendances de développement.

## Exécution des benchmarks

### Exécuter tous les benchmarks

```bash
cargo bench --bench orderbook_update
```

### Exécuter un benchmark spécifique

```bash
# Benchmark de la fonction update()
cargo bench --bench orderbook_update -- orderbook_update/single_update

# Benchmark par taille de profondeur
cargo bench --bench orderbook_update -- orderbook_update_by_diff_size

# Benchmark des calculs sur le L2Book
cargo bench --bench orderbook_update -- orderbook_calculations

# Benchmark du simulateur
cargo bench --bench orderbook_update -- simulator
```

## Résultats des benchmarks

Les résultats sont sauvegardés dans `target/criterion/` et incluent :
- Des rapports HTML détaillés avec graphiques
- Des statistiques de performance
- Des comparaisons avec les exécutions précédentes

Pour visualiser les résultats HTML :
```bash
open target/criterion/report/index.html  # macOS/Linux
start target/criterion/report/index.html # Windows
```

## Description des benchmarks

### `orderbook_update`
Teste les performances de la fonction `L2Book::update()` :
- **single_update** : mise à jour unique du carnet d'ordres
- **batch_10_updates** : 10 mises à jour consécutives
- **batch_100_updates** : 100 mises à jour consécutives

### `orderbook_update_by_diff_size`
Teste l'impact de la profondeur du carnet sur les performances :
- Teste avec des profondeurs de 5, 10, 20, et 50 niveaux
- Permet d'identifier si le nombre de niveaux affecte significativement les performances

### `orderbook_calculations`
Teste les performances des fonctions de calcul sur le L2Book :
- `best_bid()` : obtenir le meilleur bid
- `best_ask()` : obtenir le meilleur ask
- `mid_price()` : calculer le mid price
- `orderbook_imbalance()` : calculer l'imbalance au meilleur niveau
- `orderbook_imbalance_depth()` : calculer l'imbalance sur plusieurs niveaux
- `top_bids()` / `top_asks()` : obtenir les N meilleurs niveaux

### `simulator`
Teste les performances du simulateur LOBSimulator :
- `next_update()` : génération d'un message L2Update
- `bootstrap_update()` : génération du message de bootstrap initial

## Résultats typiques

Sur une machine moderne, les performances attendues sont :

| Opération | Temps moyen | Throughput |
|-----------|-------------|------------|
| `single_update` | ~2.5 µs | ~400,000 updates/sec |
| `batch_100_updates` | ~155 µs | ~645,000 updates/sec |
| `best_bid` | ~164 ns | ~6M ops/sec |
| `mid_price` | ~660 ns | ~1.5M ops/sec |
| `simulator::next_update` | ~11 µs | ~90,000 updates/sec |

## Optimisations potentielles

Basé sur les résultats des benchmarks, voici quelques pistes d'optimisation :

1. **HashMap vs BTreeMap** : Tester si un BTreeMap améliorerait `best_bid()` et `best_ask()`
2. **Caching** : Mettre en cache le best_bid/best_ask pour éviter les itérations
3. **SIMD** : Utiliser des instructions SIMD pour les calculs d'imbalance
4. **Allocation** : Réduire les allocations dans les fonctions critiques

## Profilage

Pour un profilage plus approfondi, utilisez :

```bash
# Avec flamegraph (nécessite cargo-flamegraph)
cargo flamegraph --bench orderbook_update

# Avec perf (Linux uniquement)
cargo bench --bench orderbook_update -- --profile-time=10
```
