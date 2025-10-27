# FFT Autocorrelation: Rust + Python Performance Case Study

## 🎯 Objectif du Projet

Ce projet démontre la puissance de **l'intégration Rust-Python** via **PyO3** et **Maturin** pour créer des extensions Python ultra-performantes qui **surpassent largement** les implémentations Python pures, même celles utilisant des bibliothèques optimisées comme **SciPy**.

### Le Défi

Implémenter le calcul d'autocorrélation via FFT (Fast Fourier Transform) de manière à **battre à plate couture** l'implémentation de référence de SciPy, qui est déjà elle-même hautement optimisée et utilise des backends C/Fortran performants.

### La Solution

Combiner :
- **La puissance de Rust** : performances natives, optimisations agressives, sécurité mémoire
- **La simplicité de Python** : facilité d'utilisation, écosystème riche, déploiement universel
- **PyO3** : bindings Rust ↔ Python avec overhead minimal
- **Maturin** : packaging automatique et publication de wheels Python

### Les Résultats

**Performance finale vs SciPy (implémentation Python optimisée) :**

| Taille | SciPy (ms) | Rust (ms) | **Speedup** |
|--------|------------|-----------|-------------|
| 100    | 0.236      | 0.005     | **44.9x** ⚡⚡⚡ |
| 1,000  | 0.318      | 0.129     | **2.5x**  |
| 10,000 | 1.121      | 0.237     | **4.7x** ⚡ |
| 50,000 | 6.680      | 0.743     | **9.0x** ⚡⚡ |

**Conclusion : De 2.5x à 45x plus rapide que SciPy !** 🚀

---

## 📁 Structure du Projet

```
03_fft_autocorrelation/
├── README.md                      # Ce fichier
├── TESTS.md                       # Documentation des tests unitaires
├── BENCHMARKS.md                  # Résultats détaillés des benchmarks
│
├── suboptimal/                    # Implémentation Python de référence
│   ├── __init__.py
│   └── processing.py              # Version Python avec SciPy (optimisée)
│
├── optimized/                     # Implémentation Rust + PyO3
│   ├── Cargo.toml                 # Configuration Rust
│   ├── pyproject.toml             # Configuration Python/Maturin
│   ├── src/
│   │   └── lib.rs                 # Code Rust optimisé (315 lignes)
│   ├── README.md                  # Documentation du module Rust
│   ├── OPTIMIZATION_SUMMARY.md    # Historique des optimisations v1
│   ├── OPTIMIZATION_V2_SUMMARY.md # Détails des optimisations v2
│   └── BUILD_AND_RUN.md           # Instructions de compilation
│
└── tests/                         # Tests et benchmarks
    ├── test_unit.py               # Tests unitaires (correctness)
    └── test_benchmark.py          # Tests de performance
```

---

## 🔧 Technologies Utilisées

### Rust
- **rustfft / realfft** : Implémentation FFT pure Rust
- **PyO3** : Bindings Rust ↔ Python
- **numpy crate** : Intégration avec NumPy arrays
- **rayon** : Parallélisation data-parallèle
- **once_cell** : Cache thread-safe pour plans FFT

### Python
- **Maturin** : Build system pour extensions Rust
- **NumPy** : Arrays numériques
- **Pandas** : Manipulation de séries temporelles
- **SciPy** : Implémentation de référence (signal.correlate)

---

## 🚀 Quick Start

### Prérequis

```bash
# Rust (https://rustup.rs/)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python 3.8+
python --version

# Environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows
```

### Installation

```bash
# 1. Aller dans le dossier optimized
cd optimized

# 2. Compiler et installer le module Rust
maturin develop --release --strip

# 3. Tester
cd ../tests
python test_unit.py        # Tests unitaires
python test_benchmark.py   # Benchmarks de performance
```

### Utilisation

```python
import fft_autocorr
import numpy as np

# Générer des données
data = np.random.randn(10000)

# Calculer l'autocorrélation
result = fft_autocorr.compute_autocorrelation(data, max_lag=50)

print(f"Shape: {result.shape}")  # (50,)
print(f"First 5 values: {result[:5]}")
```

---

## 💡 Pourquoi Rust + PyO3 ?

### Avantages de Rust

1. **Performance native**
   - Compilation ahead-of-time
   - Optimisations agressives (LTO, inlining, vectorisation)
   - Zéro overhead d'interprétation

2. **Contrôle mémoire**
   - Gestion manuelle sans GC
   - Allocations explicites
   - Cache-friendly data structures

3. **Sécurité**
   - Pas de segfaults
   - Pas de data races
   - Vérifications à la compilation

4. **Parallélisme**
   - Rayon pour data-parallelism facile
   - Thread-safe par design

### Avantages de PyO3

1. **Zero-copy**
   - Accès direct aux buffers NumPy
   - Pas de conversion Python ↔ Rust

2. **API ergonomique**
   - Macros pour exposer fonctions Rust
   - Types Python mappés automatiquement

3. **GIL release**
   - Calculs sans bloquer Python
   - Concurrence native

4. **Packaging simple**
   - Maturin build wheels automatiquement
   - Compatible PyPI

### Avantages de Maturin

1. **Build automatisé**
   - Détection de la toolchain Rust
   - Compilation optimisée par défaut

2. **Distribution facile**
   - Wheels multi-plateformes
   - Installation via `pip install`

3. **Développement rapide**
   - `maturin develop` pour itération rapide
   - Hot-reload en mode dev

---

## 📊 Méthodologie d'Optimisation

### Phase 1 : Implémentation Naïve (v0)

**Problème :** Plus lent que SciPy pour grandes arrays (0.4-0.5x)

**Causes :**
- FFT complexe (C2C) au lieu de réelle (R2C)
- Tailles FFT en puissance de 2 (trop grandes)
- Multiples allocations et copies
- Pas de cache de plans FFT

### Phase 2 : Optimisation Algorithmique (v1)

**Optimisations :**
1. Real FFT (R2C/C2R) → gain 2x
2. Tailles 2357-smooth → gain 1.6x
3. Cache de plans FFT → gain 10-20%
4. Sélection adaptative direct/FFT → gain 10-20x (petits max_lag)

**Résultat :** 3.6-21x plus rapide que SciPy ✓

### Phase 3 : Optimisation Micro (v2)

**Optimisations supplémentaires :**
1. Pool de buffers thread-local → zéro allocation après warmup
2. LTO + codegen-units=1 → meilleur inlining
3. Loop unrolling 4-way → meilleur pipelining CPU
4. Parallélisation (rayon) → exploitation multi-core
5. Single-pass mean/variance → -33% bande passante mémoire

**Résultat final :** 2.5-45x plus rapide que SciPy ✓✓

---

## 🎓 Leçons Apprises

### 1. Rust n'est pas magique
- Une implémentation naïve peut être **plus lente** que Python+C
- Il faut **comprendre le problème** et optimiser intelligemment

### 2. L'algorithme prime sur l'implémentation
- Direct O(n·k) bat FFT O(n log n) pour petits max_lag
- La sélection adaptative est cruciale

### 3. Les allocations tuent les performances
- Buffer reuse → gain massif
- Thread-local storage évite la contention

### 4. La parallélisation a un coût
- Overhead visible pour petits problèmes
- Calibration des seuils essentielle

### 5. Le profiling est indispensable
- Mesurer avant d'optimiser
- Benchmarks sur hardware réel
- Warmup pour éliminer biais de cache

---

## 📖 Documentation Complète

- **[TESTS.md](TESTS.md)** : Tests unitaires, validation, résultats
- **[BENCHMARKS.md](BENCHMARKS.md)** : Benchmarks détaillés, comparaisons, analyse
- **[optimized/README.md](optimized/README.md)** : Documentation utilisateur du module
- **[optimized/OPTIMIZATION_SUMMARY.md](optimized/OPTIMIZATION_SUMMARY.md)** : Optimisations v1
- **[optimized/OPTIMIZATION_V2_SUMMARY.md](optimized/OPTIMIZATION_V2_SUMMARY.md)** : Optimisations v2
- **[optimized/BUILD_AND_RUN.md](optimized/BUILD_AND_RUN.md)** : Instructions de build

---

## 🔮 Perspectives d'Amélioration

### Court terme
- [ ] SIMD explicite avec `std::simd` (nightly) → +10-30%
- [ ] Calibration automatique des seuils par profiling
- [ ] API batch pour traiter plusieurs séries → +2-5x

### Moyen terme
- [ ] Backend FFT multi-thread (FFTW, MKL) → +1.5-3x grandes arrays
- [ ] Support GPU via cuFFT → +10-100x très grandes arrays
- [ ] Implémentation PACF (partial autocorrelation)

### Long terme
- [ ] Distribution de wheels optimisés par architecture (AVX2, AVX-512, ARM NEON)
- [ ] Support async pour intégration dans workflows concurrents
- [ ] Bindings pour d'autres langages (Julia, R, Node.js)

---

## 📄 Licence

Ce projet fait partie du portfolio quant-performance-portfolio.

---

## 🙏 Remerciements

- **SciPy** pour l'implémentation de référence
- **PyO3** et **Maturin** pour rendre Rust accessible à Python
- **rustfft** pour une implémentation FFT pure Rust performante

---

## 📞 Contact & Contributions

Ce projet est un case study démonstratif. Pour des questions ou suggestions :
- Ouvrir une issue sur le repository
- Contribuer via pull request

**Résumé : Ce projet prouve qu'avec Rust + PyO3, on peut créer des extensions Python qui non seulement égalent, mais dépassent largement les implémentations C/Fortran optimisées, tout en restant simple à utiliser depuis Python ! 🚀**
