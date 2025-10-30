# Guide d'installation et configuration

Ce projet utilise **Poetry** pour la gestion des dépendances Python et **Cargo** pour les projets Rust. Chaque case study a son propre environnement isolé.

## Prérequis

### Python & Poetry

```bash
# Python 3.9+ requis
python --version

# Installer Poetry (si pas déjà installé)

curl -sSL https://install.python-poetry.org | python3 -
# or
pip install poetry

# or
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# Vérifier l'installation
poetry --version
```

### Rust (pour les case studies avec composants Rust)

```bash
# Installer Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Ou sur Windows: télécharger depuis https://rustup.rs/

# Vérifier l'installation
rustc --version
cargo --version
```

## Installation par case study

### 01 - Polars vs Pandas

```bash
cd case_studies/01_polars_vs_pandas

# Installer les dépendances avec Poetry
poetry config virtualenvs.in-project true
poetry install

# Activer l'environnement virtuel
.venv\Scripts\activate

```

### 02 - HFT Order Book (Rust)

```bash
cd case_studies/02_hft_orderbook_rust/hft_optimization

# Compiler et exécuter
cargo build --release

# then 
cargo run --release
# run benchmarks
cargo bench
# run test
cargo test
```

### 03 - FFT Autocorrelation (Python + Rust)

#### Version Python (suboptimal)

```bash
cd case_studies/03_fft_autocorrelation

# Installer les dépendances
poetry install

# Lancer les tests
poetry run pytest tests/
```

#### Version Rust optimisée (PyO3)

```bash
cd case_studies/03_fft_autocorrelation/optimized

# Installer Maturin si nécessaire
pip install maturin

# Compiler et installer le module Rust
maturin develop --release --strip

# Revenir au dossier parent et tester
cd ..
poetry run pytest tests/test_benchmark.py
```

### 04 - GPU Monte Carlo

```bash
cd case_studies/04_gpu_monte_carlo

# Installation de base (sans GPU)
poetry install

# Installation avec support GPU CUDA 12
poetry install --with gpu-cuda12

# Installation avec support GPU CUDA 11
poetry install --with gpu-cuda11

# Installation avec toutes les dépendances de développement
poetry install --with dev,test,profiling

# Lancer les tests
poetry run pytest
```

## Commandes Poetry utiles

```bash
# Installer toutes les dépendances
poetry install

# Ajouter une nouvelle dépendance
poetry add numpy

# Ajouter une dépendance de développement
poetry add --group dev pytest

# Mettre à jour les dépendances
poetry update

# Afficher les dépendances installées
poetry show

# Exporter vers requirements.txt (si nécessaire)
poetry export -f requirements.txt --output requirements.txt

# Supprimer l'environnement virtuel
poetry env remove python

# Lister les environnements
poetry env list
```

## Structure des environnements

Chaque projet a son propre environnement virtuel créé par Poetry :

```
quant-performance-portfolio/
├── case_studies/
│   ├── 01_polars_vs_pandas/
│   │   ├── pyproject.toml          # Configuration Poetry
│   │   └── .venv/                  # Environnement virtuel (créé par Poetry)
│   │
│   ├── 02_hft_orderbook_rust/
│   │   └── hft_optimization/
│   │       ├── Cargo.toml          # Configuration Cargo
│   │       └── target/             # Artefacts de build Rust
│   │
│   ├── 03_fft_autocorrelation/
│   │   ├── pyproject.toml          # Config Poetry pour version Python
│   │   ├── .venv/                  # Env virtuel Python
│   │   ├── suboptimal/             # Version Python pure
│   │   └── optimized/
│   │       ├── Cargo.toml          # Config Rust
│   │       ├── pyproject.toml      # Config Maturin
│   │       └── target/             # Build Rust
│   │
│   └── 04_gpu_monte_carlo/
│       ├── pyproject.toml          # Configuration Poetry
│       └── .venv/                  # Environnement virtuel
```

## Configuration Poetry globale

Pour configurer Poetry pour créer les environnements virtuels dans chaque projet :

```bash
# Créer les .venv dans le dossier du projet (recommandé)
poetry config virtualenvs.in-project true

# Vérifier la configuration
poetry config --list
```

## Troubleshooting

### Poetry ne trouve pas Python

```bash
# Spécifier la version de Python
poetry env use python3.10
# ou
poetry env use /usr/bin/python3.10
```

### Problème avec les dépendances GPU

Si vous n'avez pas de GPU NVIDIA, n'installez pas les groupes `gpu-cuda*` :

```bash
cd case_studies/04_gpu_monte_carlo
poetry install  # Sans --with gpu-cuda12
```

### Réinitialiser un environnement

```bash
# Supprimer l'environnement
poetry env remove python

# Réinstaller
poetry install
```

## Avantages de cette approche

1. **Isolation** : Chaque projet a ses propres dépendances, pas de conflits
2. **Reproductibilité** : `poetry.lock` garantit les mêmes versions partout
3. **Modernité** : Poetry gère automatiquement les versions et résout les conflits
4. **Simplicité** : Une seule commande `poetry install` pour tout configurer
5. **Cohérence** : Même approche pour tous les case studies Python

## Notes pour les projets Rust

Les projets Rust utilisent Cargo nativement :
- Chaque projet Rust a son `Cargo.toml`
- Les builds sont isolés dans `target/`
- Pas besoin de configuration supplémentaire
- Maturin (pour PyO3) s'intègre avec Poetry pour les projets hybrides
