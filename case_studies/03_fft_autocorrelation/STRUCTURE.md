# Structure du Projet FFT Autocorrelation

## 📁 Organisation Complète

```
03_fft_autocorrelation/
│
├── README.md                           # ⭐ Documentation principale du projet
├── TESTS.md                            # 📋 Documentation des tests unitaires
├── BENCHMARKS.md                       # 📊 Résultats des benchmarks détaillés
├── STRUCTURE.md                        # 📁 Ce fichier
│
├── suboptimal/                         # 🐍 Implémentation Python de référence
│   ├── __init__.py
│   └── processing.py                   # Version SciPy optimisée (74 lignes)
│
├── optimized/                          # ⚡ Implémentation Rust haute performance
│   ├── Cargo.toml                      # Configuration Rust + dépendances
│   ├── pyproject.toml                  # Configuration Python/Maturin
│   ├── .cargo/
│   │   └── config.toml                 # Flags de compilation agressifs
│   ├── src/
│   │   └── lib.rs                      # Code Rust optimisé (315 lignes)
│   │
│   ├── README.md                       # Documentation utilisateur
│   ├── BUILD_AND_RUN.md                # Instructions de build
│   ├── OPTIMIZATION_SUMMARY.md         # Historique optimisations v1
│   └── OPTIMIZATION_V2_SUMMARY.md      # Détails optimisations v2
│
├── tests/                              # 🧪 Suite de tests complète
│   ├── __init__.py
│   ├── README.md                       # Doc tests
│   ├── test_unit.py                    # Tests unitaires (correctness)
│   └── test_benchmark.py               # Benchmarks de performance
│
└── .venv/                              # Environnement virtuel Python
```

---

## 📖 Guide de Navigation

### Pour Comprendre le Projet

1. **[README.md](README.md)** - Commencez ici !
   - Vue d'ensemble
   - Objectifs du projet
   - Résultats principaux
   - Quick start

2. **[suboptimal/processing.py](suboptimal/processing.py)** - Implémentation de référence
   - Version Python pure avec SciPy
   - ~70 lignes, simple et lisible
   - Utilisée comme baseline pour comparaisons

3. **[optimized/src/lib.rs](optimized/src/lib.rs)** - Implémentation Rust
   - ~315 lignes de Rust optimisé
   - PyO3 bindings pour Python
   - Toutes les optimisations appliquées

### Pour Valider la Correctness

1. **[TESTS.md](TESTS.md)** - Documentation des tests
   - 4 catégories de tests
   - Méthodologie de validation
   - Résultats attendus

2. **[tests/test_unit.py](tests/test_unit.py)** - Tests unitaires
   - Exécuter pour valider
   - Comparaison Python vs Rust
   - Tous les edge cases

### Pour Analyser les Performances

1. **[BENCHMARKS.md](BENCHMARKS.md)** - Résultats détaillés
   - Comparaisons exhaustives
   - Breakdown des temps d'exécution
   - Évolution v0 → v1 → v2

2. **[tests/test_benchmark.py](tests/test_benchmark.py)** - Benchmarks automatisés
   - Exécuter pour mesurer
   - Différentes configurations
   - Résultats statistiques

### Pour Comprendre les Optimisations

1. **[optimized/OPTIMIZATION_SUMMARY.md](optimized/OPTIMIZATION_SUMMARY.md)** - Phase 1
   - Diagnostic de la version naïve
   - Optimisations algorithmiques
   - Passage de 0.4x à 3.6x

2. **[optimized/OPTIMIZATION_V2_SUMMARY.md](optimized/OPTIMIZATION_V2_SUMMARY.md)** - Phase 2
   - Optimisations micro
   - Buffer pool, LTO, parallel
   - Passage de 3.6x à 9.0x

### Pour Compiler et Tester

1. **[optimized/BUILD_AND_RUN.md](optimized/BUILD_AND_RUN.md)** - Instructions build
   - Commandes complètes
   - Options de compilation
   - Troubleshooting

2. **[tests/README.md](tests/README.md)** - Lancer les tests
   - Commandes rapides
   - Prérequis

---

## 🎯 Flux de Travail Typique

### Développeur Python (Utilisateur)

```bash
# 1. Installer le module
cd optimized
maturin develop --release --strip

# 2. Utiliser en Python
python
>>> import fft_autocorr
>>> result = fft_autocorr.compute_autocorrelation(data, max_lag=50)
```

**Documentation :** [README.md](README.md), [optimized/README.md](optimized/README.md)

### Développeur Rust (Contributeur)

```bash
# 1. Modifier le code Rust
nano optimized/src/lib.rs

# 2. Tester
cd optimized
cargo test
maturin develop --release

# 3. Valider
cd ../tests
python test_unit.py
python test_benchmark.py
```

**Documentation :** [optimized/src/lib.rs](optimized/src/lib.rs) (commentaires), [OPTIMIZATION_V2_SUMMARY.md](optimized/OPTIMIZATION_V2_SUMMARY.md)

### Chercheur (Analyse)

```bash
# 1. Lire la méthodologie
cat BENCHMARKS.md

# 2. Reproduire les benchmarks
python tests/test_benchmark.py

# 3. Analyser les résultats
# Voir BENCHMARKS.md pour interprétation
```

**Documentation :** [BENCHMARKS.md](BENCHMARKS.md), [TESTS.md](TESTS.md)

---

## 📊 Métriques du Projet

### Lignes de Code

| Composant | Lignes | Commentaires | Ratio Doc/Code |
|-----------|--------|--------------|----------------|
| suboptimal/processing.py | 74 | 48 | 65% |
| optimized/src/lib.rs | 315 | 120 | 38% |
| tests/test_unit.py | 280 | 50 | 18% |
| tests/test_benchmark.py | 220 | 40 | 18% |
| **Documentation .md** | ~3500 | - | - |

**Total Code :** ~900 lignes
**Total Documentation :** ~3500 lignes
**Ratio Global Doc/Code :** **3.9:1** (excellente documentation !)

### Fichiers par Catégorie

**Code Source :** 4 fichiers
- 1 Python (suboptimal)
- 1 Rust (optimized)
- 2 Tests

**Documentation :** 9 fichiers Markdown
- 1 README principal
- 2 docs tests/benchmarks
- 6 docs techniques (optimized/)

**Configuration :** 4 fichiers
- 2 Cargo/pyproject
- 1 .cargo/config
- 1 .gitignore

---

## 🔄 Dépendances entre Fichiers

```
README.md
  ├─→ TESTS.md
  ├─→ BENCHMARKS.md
  ├─→ suboptimal/processing.py
  └─→ optimized/
      ├─→ src/lib.rs
      ├─→ README.md
      ├─→ BUILD_AND_RUN.md
      ├─→ OPTIMIZATION_SUMMARY.md
      └─→ OPTIMIZATION_V2_SUMMARY.md

tests/
  ├─→ test_unit.py → suboptimal/ + optimized/
  └─→ test_benchmark.py → suboptimal/ + optimized/

TESTS.md → tests/test_unit.py
BENCHMARKS.md → tests/test_benchmark.py
```

---

## 🎓 Ordre de Lecture Recommandé

### Pour Découvrir (20 min)

1. [README.md](README.md) (5 min)
2. [BENCHMARKS.md](BENCHMARKS.md) - Résultats uniquement (5 min)
3. Exécuter `python tests/test_unit.py` (5 min)
4. Exécuter `python tests/test_benchmark.py` (5 min)

### Pour Comprendre (1h)

1. [README.md](README.md) complet (10 min)
2. [suboptimal/processing.py](suboptimal/processing.py) (10 min)
3. [optimized/src/lib.rs](optimized/src/lib.rs) - parcourir (20 min)
4. [OPTIMIZATION_V2_SUMMARY.md](optimized/OPTIMIZATION_V2_SUMMARY.md) (20 min)

### Pour Maîtriser (3h)

1. Tout ci-dessus
2. [TESTS.md](TESTS.md) complet (20 min)
3. [BENCHMARKS.md](BENCHMARKS.md) complet (30 min)
4. [OPTIMIZATION_SUMMARY.md](optimized/OPTIMIZATION_SUMMARY.md) (30 min)
5. [optimized/src/lib.rs](optimized/src/lib.rs) ligne par ligne (1h)

---

## 🚀 Commandes Essentielles

### Setup Initial

```bash
# Créer environnement
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate (Windows)

# Installer dépendances
pip install numpy pandas scipy maturin

# Compiler Rust
cd optimized
maturin develop --release --strip
cd ..
```

### Tests

```bash
# Tests unitaires
python tests/test_unit.py

# Benchmarks
python tests/test_benchmark.py

# Les deux
python tests/test_unit.py && python tests/test_benchmark.py
```

### Développement

```bash
# Modifier Rust
nano optimized/src/lib.rs

# Recompiler
cd optimized && maturin develop --release && cd ..

# Tester rapidement
python -c "import fft_autocorr; print(fft_autocorr.compute_autocorrelation([1,2,3,4,5], 2))"
```

---

## 📝 Conventions de Nommage

### Fichiers

- **README.md** : Documentation principale d'un dossier
- **CAPSLOCK.md** : Documentation importante au niveau racine
- **test_*.py** : Fichiers de test
- **processing.py** : Implémentation de fonctions métier
- **lib.rs** : Point d'entrée Rust

### Fonctions

- **Python :** `snake_case`
  - `compute_autocorrelation()`

- **Rust :** `snake_case`
  - `compute_autocorr_fft()`
  - `autocorr_direct_norm()`

### Versions

- **v0** : Implémentation naïve Rust (historique)
- **v1** : Première optimisation (Real FFT, cache plans)
- **v2** : Seconde optimisation (buffers, parallel, LTO)

---

**Résumé : Le projet est organisé de manière professionnelle avec une séparation claire entre code source (suboptimal/ et optimized/), tests (tests/), et documentation (fichiers .md à la racine et dans optimized/). La documentation représente 3.9x le volume de code, assurant une excellente maintenabilité et compréhension. 📚**
