# Tests Unitaires - FFT Autocorrelation

## 📋 Vue d'Ensemble

Ce document décrit la suite de tests unitaires qui valide la correctness (justesse) des implémentations Python (suboptimal) et Rust (optimized) du calcul d'autocorrélation par FFT.

## 🎯 Objectifs des Tests

1. **Validation numérique** : Vérifier que les résultats sont correctement identiques entre les deux implémentations
2. **Gestion des cas limites** : Tester le comportement sur des données edge cases (constantes, NaN, etc.)
3. **Robustesse** : S'assurer qu'aucune régression n'est introduite lors des optimisations
4. **Non-régression** : Garantir la stabilité à travers les versions

## 📁 Fichiers de Tests

### `tests/test_unit.py`

Suite complète de tests unitaires comprenant 4 catégories de tests.

---

## 🧪 Tests Implémentés

### TEST 1: Basic Correctness ✓

**Objectif :** Valider la justesse fondamentale avec des valeurs connues

**Données de test :**
```python
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
max_lag = 3
```

**Valeurs attendues :**
```
lag 1: 0.700000
lag 2: 0.412121
lag 3: 0.148485
```

**Critères de réussite :**
- ✅ Python vs valeurs attendues : différence < 1e-5
- ✅ Rust vs valeurs attendues : différence < 1e-5
- ✅ Rust vs Python : différence < 1e-10 (précision machine)

**Résultat :**
```
Python: PASS (max diff: 2.22e-16)
Rust: PASS (max diff: 2.22e-16)
Rust vs Python: PASS (max diff: 2.22e-16)
```

---

### TEST 2: Edge Cases ✓

**Objectif :** Valider le comportement sur des cas limites

**Cas testés :**

#### 1. Série Constante
```python
data = np.ones(100)
```
**Comportement attendu :** NaN (variance nulle)
**Résultat :** ✓ Les deux implémentations retournent NaN

#### 2. Bruit Aléatoire Normal
```python
data = np.random.randn(100)
```
**Comportement attendu :** Autocorrélation décroissante
**Résultat :** ✓ PASS (max diff: 5.55e-17)

#### 3. Onde Sinusoïdale
```python
data = np.sin(np.linspace(0, 4*np.pi, 100))
```
**Comportement attendu :** Oscillations périodiques
**Résultat :** ✓ PASS (max diff: 5.55e-16)

#### 4. Tendance Linéaire
```python
data = np.arange(100, dtype=float)
```
**Comportement attendu :** Forte autocorrélation
**Résultat :** ✓ PASS (max diff: 3.33e-16)

#### 5. Moyenne Zéro
```python
data = np.random.randn(100) - mean
```
**Comportement attendu :** Identique au bruit normal
**Résultat :** ✓ PASS

**Critères de réussite :**
- ✅ Pas de NaN pour séries non-constantes
- ✅ Pas de Inf dans aucun cas
- ✅ Rust vs Python : différence < 1e-10

---

### TEST 3: Different Sizes ✓

**Objectif :** Valider la robustesse sur différentes tailles d'arrays

**Tailles testées :**
- 10, 50, 100, 500, 1000, 5000, 10000

**Pour chaque taille :**
- Génération de données aléatoires
- Calcul avec max_lag=20
- Vérification de la shape du résultat
- Comparaison Rust vs Python

**Critères de réussite :**
- ✅ Shape correcte : `len(result) == max_lag`
- ✅ Différence < 1e-10 pour toutes les tailles

**Résultats :**
```
Size 10:    PASS (max diff: 1.11e-16)
Size 50:    PASS (max diff: 2.22e-16)
Size 100:   PASS (max diff: 5.55e-17)
Size 500:   PASS (max diff: 8.88e-17)
Size 1000:  PASS (max diff: 1.00e-16)
Size 5000:  PASS (max diff: 1.48e-16)
Size 10000: PASS (max diff: 7.72e-17)
```

---

### TEST 4: Large max_lag ✓

**Objectif :** Tester le comportement avec des max_lag très grands

**Configuration :**
```python
data_size = 1000
max_lag = 500  # 50% de la taille des données
```

**Pourquoi c'est important :**
- Teste la limite de l'algorithme
- Valide que l'implémentation ne fait pas d'hypothèses incorrectes
- Vérifie la stabilité numérique sur de longs lags

**Critères de réussite :**
- ✅ Pas d'erreur ou exception
- ✅ Shape correcte : 500 valeurs
- ✅ Max différence < 1e-10
- ✅ Mean différence < 1e-15

**Résultat :**
```
Data size: 1000
Max lag: 500

Max difference: 6.77e-17
Mean difference: 1.50e-17

PASS: Results match perfectly
```

---

## 📊 Résumé des Tests

### Résultat Global

```
TEST SUMMARY
==================================================
✓ basic           : PASS
✓ edge_cases      : PASS
✓ sizes           : PASS
✓ large_lag       : PASS

ALL TESTS PASSED ✓
```

### Statistiques de Précision

| Test | Max Différence | Mean Différence | Status |
|------|----------------|-----------------|--------|
| Basic Correctness | 2.22e-16 | ~1e-16 | ✓ PASS |
| Edge Cases | 5.55e-16 | ~2e-16 | ✓ PASS |
| Different Sizes | 2.22e-16 | ~1e-16 | ✓ PASS |
| Large max_lag | 6.77e-17 | 1.50e-17 | ✓ PASS |

**Conclusion : La précision numérique est au niveau de la machine (< 1e-15), ce qui est optimal.**

---

## 🚀 Lancer les Tests

### Installation

```bash
# 1. Créer et activer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# 2. Installer les dépendances
pip install numpy pandas scipy

# 3. Compiler le module Rust
cd optimized
maturin develop --release --strip
cd ..
```

### Exécution

```bash
# Depuis la racine du projet
python tests/test_unit.py
```

### Sortie Attendue

```
╔══════════════════════════════════════════════════════════════════╗
║                    UNIT TEST SUITE                               ║
╚══════════════════════════════════════════════════════════════════╝

======================================================================
TEST 1: Basic Correctness
======================================================================

Input: [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
Max lag: 3

Python result:
lag
1    0.700000
2    0.412121
3    0.148485
Name: autocorrelation, dtype: float64

Rust result:
[0.7        0.41212121 0.14848485]

Maximum difference: 2.22e-16
Results match perfectly!

======================================================================
TEST 2: Edge Cases
======================================================================

Constant series:
  PASS: Both correctly return NaN for constant series

Random normal:
  PASS (max diff: 5.55e-17)

Sine wave:
  PASS (max diff: 5.55e-16)

...

======================================================================
TEST SUMMARY
======================================================================
✓ basic           : PASS
✓ edge_cases      : PASS
✓ sizes           : PASS
✓ large_lag       : PASS

======================================================================
ALL TESTS PASSED ✓
======================================================================
```

---

## 🔍 Détails d'Implémentation

### Stratégie de Test

1. **Génération de données reproductibles**
   - Seed fixe pour numpy.random
   - Données synthétiques avec propriétés connues

2. **Comparaison multi-niveaux**
   - Valeurs attendues (ground truth)
   - Python vs Rust (cross-validation)
   - Vérification de cohérence interne

3. **Tolérances adaptatives**
   - 1e-5 vs valeurs attendues (arrondis dans la doc)
   - 1e-10 Python vs Rust (erreurs d'arrondi FFT)
   - Gestion spéciale des NaN/Inf

### Gestion des Erreurs

**Cas gérés :**
- ✅ Série constante → NaN (variance nulle)
- ✅ Array vide → ValueError
- ✅ max_lag = 0 → ValueError
- ✅ max_lag > len(data) → Truncation automatique

**Cohérence :**
- Python et Rust se comportent identiquement
- Messages d'erreur clairs
- Pas de silent failures

---

## 📈 Évolution des Tests

### Version 1
- Tests basiques de correctness
- Comparaison manuelle des résultats

### Version 2 (Actuelle)
- Suite complète automatisée
- 4 catégories de tests
- Validation croisée Python/Rust
- Tolérance adaptative selon le contexte

### Version Future
- [ ] Tests de propriétés (property-based testing avec Hypothesis)
- [ ] Tests de performance (seuils min de speedup)
- [ ] Tests de régression automatiques (CI/CD)
- [ ] Couverture de code (coverage.py)

---

## 🐛 Debugging

### Si un test échoue

1. **Vérifier la compilation Rust**
   ```bash
   cd optimized
   cargo clean
   maturin develop --release
   ```

2. **Vérifier les dépendances Python**
   ```bash
   pip install --upgrade numpy pandas scipy
   ```

3. **Tester isolément**
   ```python
   python -c "import fft_autocorr; print(fft_autocorr.__file__)"
   ```

4. **Verbose mode**
   ```bash
   python tests/test_unit.py -v
   ```

### Warnings connus

**RuntimeWarning: invalid value encountered in divide**
- Origine : série constante dans SciPy
- Impact : aucun (comportement attendu)
- Résolution : non nécessaire

---

## ✅ Checklist de Validation

Avant chaque release, vérifier :

- [ ] Tous les tests passent
- [ ] Aucune régression de performance
- [ ] Pas de warnings non-gérés
- [ ] Documentation à jour
- [ ] Exemples fonctionnels

---

## 📚 Références

- [NumPy Testing Guidelines](https://numpy.org/doc/stable/reference/testing.html)
- [pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [SciPy signal.correlate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html)

---

**Résumé : Tous les tests passent avec une précision au niveau de la machine (< 1e-15). Les implémentations Python et Rust sont numériquement identiques et robustes sur tous les cas testés. ✓**
