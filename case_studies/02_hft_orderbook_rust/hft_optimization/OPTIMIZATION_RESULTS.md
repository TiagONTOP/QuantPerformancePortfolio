# Résultats d'Optimisation - Orderbook Vec vs HashMap

Date : 2025-10-23

## Vue d'ensemble

Ce document compare les performances de deux implémentations du carnet d'ordres L2 :
- **HashMap (Suboptimal)** : Stockage des niveaux de prix dans un HashMap<Price, Qty>
- **Vec (Optimized)** : Stockage des niveaux de prix dans un Vec<Qty> avec indexation directe

## Résultats principaux

### 1. Performance des mises à jour (update)

| Opération | HashMap | Vec | Amélioration |
|-----------|---------|-----|--------------|
| **Single update** | 1.56 µs | **0.97 µs** | **1.6x plus rapide** |
| **Batch 100 updates** | 167.55 µs | **89.70 µs** | **1.87x plus rapide** |

**Analyse** : L'implémentation Vec est **60-87% plus rapide** pour les opérations de mise à jour. Cela s'explique par :
- Accès O(1) au lieu de O(log n) ou hash lookup
- Pas de calcul de hash
- Meilleure localité du cache (accès séquentiel)

### 2. Performance des opérations de lecture

| Opération | HashMap | Vec | Amélioration |
|-----------|---------|-----|--------------|
| **best_bid()** | 160.42 ns | **1.19 ns** | **134x plus rapide** 🚀 |
| **best_ask()** | 164.98 ns | **1.17 ns** | **141x plus rapide** 🚀 |
| **mid_price()** | 332.24 ns | **0.65 ns** | **511x plus rapide** 🚀🚀🚀 |
| **orderbook_imbalance()** | 365.50 ns | **0.67 ns** | **545x plus rapide** 🚀🚀🚀 |
| **top_bids(10)** | 209.18 ns | 402.05 ns | 1.9x plus lent ⚠️ |

**Analyse** :
- ✅ **Lecture des meilleurs prix** : Amélioration **extrême** (100-500x) grâce au cache
- ✅ **Calculs dérivés** (mid_price, imbalance) : Sub-nanoseconde grâce au cache
- ⚠️ **Top N levels** : Légèrement plus lent car nécessite un scan du Vec (mais reste < 500 ns)

### 3. Performance selon la profondeur du carnet

#### Profondeur = 5 niveaux

| Implémentation | Temps |
|----------------|-------|
| HashMap | 954.83 ns |
| Vec | **703.41 ns** |
| **Amélioration** | **1.36x** |

#### Profondeur = 10 niveaux

| Implémentation | Temps |
|----------------|-------|
| HashMap | 1.047 µs |
| Vec | **0.751 µs** |
| **Amélioration** | **1.39x** |

#### Profondeur = 20 niveaux

| Implémentation | Temps |
|----------------|-------|
| HashMap | 1.476 µs |
| Vec | **0.788 µs** |
| **Amélioration** | **1.87x** |

#### Profondeur = 50 niveaux

| Implémentation | Temps |
|----------------|-------|
| HashMap | 3.434 µs |
| Vec | **1.055 µs** |
| **Amélioration** | **3.25x** 🚀 |

**Analyse** : L'avantage de Vec **augmente avec la profondeur** du carnet. À 50 niveaux, Vec est **3.25x plus rapide**.

## Graphique de comparaison

```
Performance Comparison (lower is better)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Update Operations:
Single update       HashMap: ████████ 1.56 µs
                    Vec:     █████ 0.97 µs  (1.6x faster)

Batch 100 updates   HashMap: ████████████████ 167.55 µs
                    Vec:     █████████ 89.70 µs  (1.87x faster)

Read Operations:
best_bid()          HashMap: ████████████████ 160.42 ns
                    Vec:     ▏1.19 ns  (134x faster)

mid_price()         HashMap: ████████████████████████████████ 332.24 ns
                    Vec:     ▏0.65 ns  (511x faster)

Depth Scaling:
depth=50            HashMap: ████████████████████████████████ 3.43 µs
                    Vec:     ██████████ 1.06 µs  (3.25x faster)
```

## Architecture de l'implémentation optimisée

### Stratégie de stockage

```rust
pub struct L2Book {
    bid_anchor: Price,    // Prix de référence pour les bids
    ask_anchor: Price,    // Prix de référence pour les asks

    bids: Vec<Qty>,       // Index = (bid_anchor - price_tick)
    asks: Vec<Qty>,       // Index = (price_tick - ask_anchor)

    // Cache pour O(1) read operations
    cached_best_bid: Option<(Price, Qty)>,
    cached_best_ask: Option<(Price, Qty)>,
}
```

### Avantages de cette approche

1. **Accès O(1)** : Conversion directe prix → index
2. **Cache CPU-friendly** : Les prix proches sont contigus en mémoire
3. **Cache des best prices** : Lecture en temps constant
4. **Pas d'allocation** : Réutilisation du Vec pré-alloué
5. **Scalabilité** : Performance proportionnellement meilleure avec plus de niveaux

### Trade-offs

| Aspect | HashMap | Vec |
|--------|---------|-----|
| Insertion/Update | O(1) avg, O(n) worst | O(1) constant |
| Lecture best price | O(n) | **O(1)** ✅ |
| Mémoire | Dynamique (sparse) | Statique (dense) |
| Localité cache | Mauvaise | **Excellente** ✅ |
| Expansion dynamique | Facile | Nécessite resize |

## Throughput calculé

### Mises à jour par seconde

| Implémentation | Single update | Batch |
|----------------|---------------|-------|
| HashMap | ~640,000 updates/s | ~596,000 updates/s |
| **Vec** | **~1,030,000 updates/s** | **~1,115,000 updates/s** |

### Lectures par seconde

| Opération | HashMap | Vec |
|-----------|---------|-----|
| best_bid() | ~6.2M reads/s | **~840M reads/s** |
| mid_price() | ~3.0M reads/s | **~1.5B reads/s** |

## Recommandations

### ✅ Utiliser Vec (Optimized) quand :

1. **Latence critique** : Besoin de < 1 µs par update
2. **Lecture intensive** : Beaucoup d'accès aux best prices
3. **Carnet profond** : Plus de 20 niveaux de profondeur
4. **Trading haute fréquence** : Chaque nanoseconde compte

### ⚠️ Utiliser HashMap (Suboptimal) quand :

1. **Prix très dispersés** : Range de prix > 10,000 ticks
2. **Carnet sparse** : Peu de niveaux actifs
3. **Simplicité** : Pas besoin d'optimisation extrême
4. **Prototypage rapide**

## Cas d'usage réel : Market Making HFT

### Scénario typique
- Mise à jour du carnet : 10,000 fois/seconde
- Lecture du mid-price : 100,000 fois/seconde
- Profondeur utilisée : 20 niveaux

### Performance HashMap
```
Updates:  10,000 × 1.56 µs = 15.6 ms/s = 1.56% CPU
Reads:    100,000 × 332 ns = 33.2 ms/s = 3.32% CPU
Total:    4.88% CPU
```

### Performance Vec (Optimized)
```
Updates:  10,000 × 0.97 µs = 9.7 ms/s = 0.97% CPU
Reads:    100,000 × 0.65 ns = 0.065 ms/s = 0.0065% CPU
Total:    0.98% CPU
```

**Économie** : **4.98x moins de CPU** utilisé !

## Tests de validation

Tous les tests unitaires passent pour les deux implémentations :

```bash
cargo test
```

### Tests d'équivalence

✅ Bootstrap update identique
✅ Sequential updates identiques
✅ Best bid/ask identiques
✅ Mid-price identiques
✅ Orderbook imbalance identiques
✅ Checksum validation identique

## Conclusion

L'implémentation optimisée avec Vec offre des **gains de performance spectaculaires** :

- **1.6-1.9x plus rapide** pour les updates
- **100-500x plus rapide** pour les lectures
- **Scalabilité excellente** avec la profondeur
- **Consommation CPU réduite de 5x**

Cette implémentation est **production-ready** pour des systèmes HFT à faible latence et convient parfaitement aux stratégies de market making nécessitant des accès ultra-rapides au carnet d'ordres.

## Prochaines étapes d'optimisation

1. **SIMD** : Utiliser des instructions vectorielles pour les calculs d'imbalance sur plusieurs niveaux
2. **Zero-copy** : Éviter les allocations dans top_bids()/top_asks()
3. **Atomic operations** : Support multi-thread lock-free
4. **Memory pool** : Pré-alloquer les messages pour éviter les allocations
5. **Branch prediction hints** : Optimiser les chemins chauds avec likely/unlikely

## Références

- Code source : `src/optimized/book.rs`
- Benchmarks : `benches/optimized_vs_suboptimal.rs`
- Documentation : `benches/README.md`
