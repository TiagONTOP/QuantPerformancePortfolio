# `BENCHMARKS.MD` : Analyse Quantitative des Performances

## 1. Objectif

Ce document présente et analyse les résultats des benchmarks de performance comparant l'implémentation de référence `suboptimal` (Python/Scipy) à l'implémentation `optimized` (Rust/PyO3).

L'objectif est de quantifier le gain de performance (speedup) dans différents scénarios et de relier ces gains aux optimisations architecturales spécifiques (stratégie adaptative, parallélisme, gestion de la mémoire) décrites dans `STRUCTURE.md`.

## 2. Configuration du Test

* **Matériel :**
    * **CPU :** Intel Core i7 4770 @ 4.1 GHz (Overclocké)
    * **RAM :** 16 Go DDR3 @ 2400 MHz
* **Logiciel :**
    * Python 3.12+
    * `pytest`
    * `scipy` (utilisé par le *baseline*)
    * `rustc` (utilisé pour compiler le module optimisé en mode `release`)
* **Méthodologie :**
    * Les tests sont exécutés à l'aide de `pytest`.
    * La durée est mesurée avec `time.perf_counter()`.
    * Pour réduire la variance, chaque benchmark est exécuté `n_iterations = 10` fois. Le temps rapporté est la **médiane** de ces exécutions.
    * Le **Speedup** est calculé comme `Temps_Python / Temps_Rust`.

---

## 3. Résultats des Benchmarks

### 3.1. Test 1 : Performance selon la Taille de la Série (Taille `n` variable, `max_lag = 50`)

Ce test évalue l'impact de la taille de la série d'entrée (`n`) pour un `max_lag` faible et fixe.

| Taille (n) | Python (Scipy) | Rust (Optimisé) | **Speedup** | Méthode Rust (Analyse) |
| ---: | ---: | ---: | :---: | :--- |
| 100 | 0.291 ms | 0.004 ms | **73.57x** | `Direct` |
| 1,000 | 0.360 ms | 0.038 ms | **9.46x** | `Direct` |
| 10,000 | 0.981 ms | 0.190 ms | **5.16x** | `Direct` |
| 50,000 | 6.479 ms | 0.719 ms | **9.01x** | `Direct` |

#### Analyse (Test 1)

1.  **Speedup de 73.57x (n=100) :** Ce résultat est la preuve la plus claire de la **surcharge de Python (overhead)**. Pour une si petite série, le coût de calcul est négligeable. Le temps de Scipy (0.291 ms) est presque entièrement dominé par les appels de fonction, les conversions de types NumPy, et les allocations de buffers internes. La version Rust, étant pré-compilée et utilisant l'algorithme `Direct` (simple boucle), a une surcharge quasi nulle (4 microsecondes).

2.  **Stratégie Adaptative en Action :** L'analyse de l'heuristique (`autocorr_adaptive`) montre que pour *toutes* les tailles de ce test, le coût estimé $O(nk)$ de la méthode `Direct` est inférieur au coût $O(m \log m)$ de la FFT.
    * Ex: Pour `n=50000, k=50`, le coût direct (avec marge) est $\approx (50000 \cdot 50 \cdot 0.25) \cdot 1.2 = 750,000$ unités.
    * Le coût FFT (pour $m = \text{next\_fast\_len}(99999) = 100000$) est $\approx 100000 \cdot \log_2(100000) \approx 1,661,000$.
    * `750,000 < 1,661,000`. Rust sélectionne donc **correctement** la méthode `Direct` (parallélisée avec `rayon`) et surpasse largement Scipy qui utilise la FFT (inutilement coûteuse ici).
3.  **Speedup Optimal à 9.01-9.46x (n=1000 et n=50000) :** Ces deux points montrent le meilleur équilibre entre la surcharge de Python et la charge de calcul. Le speedup de ~9x représente le gain "pur" de Rust pour l'algorithme `Direct` parallélisé, sans être dominé par l'overhead (n=100) ni par les effets de cache (n=10000).

---

### 3.2. Test 2 : Performance selon le Lag (Taille `n = 10 000` fixe, `max_lag` variable)

Ce test est crucial car il met à l'épreuve le **point de bascule** de l'heuristique adaptative.

| Max Lag (k) | Python (Scipy) | Rust (Optimisé) | **Speedup** | Méthode Rust (Analyse) |
| ---: | ---: | ---: | :---: | :--- |
| 10 | 0.949 ms | 0.120 ms | **7.91x** | `Direct` |
| 50 | 0.968 ms | 0.186 ms | **5.20x** | `Direct` |
| 100 | 0.892 ms | 0.393 ms | **2.27x** | **`FFT`** |
| 200 | 0.921 ms | 0.361 ms | **2.55x** | **`FFT`** |
| 500 | 1.018 ms | 0.378 ms | **2.69x** | **`FFT`** |

#### Analyse (Test 2)

1.  **Stabilité de Scipy :** Le temps de Python/Scipy est presque constant (entre 0.89ms et 1.02ms). C'est normal : il utilise *toujours* la FFT $O(m \log m)$, où $m \approx 2n$. Le `max_lag` n'a quasiment aucun impact sur son temps de calcul.

2.  **Point de Bascule de Rust (Le Crossover) :** Le comportement de Rust est radicalement différent et prouve l'efficacité de l'heuristique.
    * **Pour k=10 à 50 :** Le temps de Rust augmente (de 0.120ms à 0.186ms). C'est le comportement attendu de l'algorithme `Direct` ($O(nk)$) : le temps de calcul est proportionnel à `max_lag`.
    * **Le Basculement (k=100) :** L'heuristique `autocorr_adaptive` détecte que le coût de `Direct` (pour $k=100$) dépasse le coût de `FFT`.
        * **Calcul de l'Heuristique (`n=10k`, `k=100`) :**
        * Taille FFT `m = next_fast_len(19999) = 20160`.
        * Coût `FFT` $\approx (20160 \cdot \log_2(20160)) + 1000 \approx 289,288$
        * Coût `Direct` $\approx (10000 \cdot 100 \cdot 0.25) \cdot 1.2 = 300,000$
        * `if 300,000 < 289,288` est **FAUX**. L'algorithme bascule donc vers **`FFT`**.
    * **Pour k=100 à 500 :** Le temps de Rust (0.361ms à 0.393ms) redevient stable, tout comme Scipy, car il utilise désormais aussi la FFT.

3.  **Performance de Rust-FFT vs Scipy-FFT :** Même lorsque les deux implémentations utilisent la FFT (k=100, k=200, et k=500), la version Rust est **2.27x à 2.69x plus rapide**. Ce gain s'explique par :
    * Le `PLAN_CACHE` (amortit le coût de setup).
    * L'utilisation de `realfft` (R2C), ~2x plus efficace que la FFT complexe.
    * Le parallélisme `rayon` sur le calcul du spectre de puissance.
    * L'utilisation de tailles de FFT "lisses" (`next_fast_len`).

---

### 3.3. Test 3 : Appels Répétés (Cache)

Ce test mesure l'efficacité en appelant la fonction 100 fois avec les *mêmes* paramètres (`n=10 000`, `max_lag=50`).

| Métrique | Python (Scipy) | Rust (Optimisé) | Speedup |
| :--- | ---: | ---: | :---: |
| Temps Total (100 appels) | 95.0 ms | 19.1 ms | 4.97x |
| Temps Moyen par Appel | 0.950 ms/appel | 0.191 ms/appel | **4.97x** |

#### Analyse (Test 3)

* Les paramètres (`n=10000`, `k=50`) forcent l'utilisation de la méthode **`Direct`** (comme vu au Test 3.2).
* La méthode `Direct` n'utilise *pas* le `PLAN_CACHE` ni le `BUFFER_POOL` (qui sont spécifiques à la FFT).
* Le speedup soutenu de **4.97x** (très proche du 5.16x observé en Test 3.1) confirme que la performance n'est pas un artefact, mais un gain structurel et robuste.
* Ce gain provient de l'efficacité brute de `rayon` (parallélisme), du déroulement de boucle, et de la faible surcharge d'appel, maintenue même lors d'appels répétés (grâce à la localité du cache CPU, etc.).

## 4. Conclusion Générale

1.  **Supériorité Totale :** Le module Rust est plus performant que le baseline Scipy dans *tous les scénarios testés*, avec un speedup allant de **2.27x** (pire cas, FFT vs FFT) à **73.57x** (meilleur cas, surcharge Python minimale).
2.  **L'Algorithme Adaptatif est la Clé :** Le gain de performance le plus important (5x à 73x) provient de l'utilisation de `autocorr_adaptive`. L'implémentation `Direct` ($O(nk)$) parallélisée surpasse massivement la méthode FFT de Scipy lorsque `max_lag` est faible, ce qui est un cas d'utilisation très courant en finance.
3.  **Optimisations FFT Efficaces :** Même lorsque Rust doit utiliser la FFT, son implémentation (cache de plans, R2C, `rayon`) est **2.27x à 2.69x** plus rapide que celle, déjà optimisée, de Scipy.