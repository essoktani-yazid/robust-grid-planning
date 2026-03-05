# Référence API — Robust Grid Planning

> Documentation complète des endpoints du service de planification et de l'API programmatique des modules Python.

- **URL de base (Développement) :** `http://localhost:8080/api/v1`
- **Format d'échange :** `application/json`
- **Documentation interactive (Swagger / OpenAPI) :** `http://localhost:8080/docs` *(FastAPI OpenAPI Specification)*

---

## Authentification

Les requêtes vers les endpoints de calcul distribué ou de benchmarking nécessitent un jeton JWT valide dans l'en-tête HTTP :

```http
Authorization: Bearer <votre_token_jwt>
```

---

## Endpoints Principaux

### 1. Planification Heuristique de Chemin (A* / UCS / Greedy)

Calcule le chemin optimal ou sous-optimal entre les points de départ et d'arrivée spécifiés sur une grille 2D configurée avec obstacles.

- **URL :** `/plan/astar`
- **Méthode :** `POST`
- **Authentification requise :** Non (Accès public pour calcul rapide)

#### Corps de la requête (Request Body)

```json
{
  "grid": {
    "width": 15,
    "height": 15,
    "start": [1, 1],
    "goal": [13, 13],
    "obstacles": [
      [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8],
      [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [10, 11], [10, 12], [10, 13], [10, 14]
    ]
  },
  "algorithm": "astar",
  "heuristic": "manhattan",
  "weight": 1.0
}
```

#### Réponse de succès (`200 OK`)

```json
{
  "status": "success",
  "timestamp": "2026-10-25T17:30:00Z",
  "data": {
    "success": true,
    "cost": 24.0,
    "length": 25,
    "nodes_expanded": 82,
    "max_open_size": 28,
    "execution_time_ms": 2.45,
    "path": [
      [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [4, 3], [4, 4],
      [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [5, 9], [6, 9],
      [7, 9], [8, 9], [9, 9], [9, 10], [9, 11], [9, 12], [9, 13],
      [10, 13], [11, 13], [12, 13], [13, 13]
    ]
  }
}
```

---

### 2. Résolution Stochastique MDP (Value & Policy Iteration)

Résout le Processus Décisionnel de Markov (MDP) associé à l'environnement pour dériver la fonction de valeur optimale $V^*(s)$ et la politique globale $\pi^*(s)$ sous niveau de bruit stochastique paramétrable.

- **URL :** `/markov/solve`
- **Méthode :** `POST`
- **Authentification requise :** Oui

#### Corps de la requête (Request Body)

```json
{
  "grid": {
    "width": 10,
    "height": 10,
    "start": [1, 1],
    "goal": [8, 8],
    "obstacles": [[3, 3], [3, 4], [3, 5], [6, 2], [6, 3], [6, 4]]
  },
  "method": "value_iteration",
  "gamma": 0.9,
  "noise": 0.2,
  "theta": 0.01,
  "max_iterations": 1000
}
```

#### Réponse de succès (`200 OK`)

```json
{
  "status": "success",
  "timestamp": "2026-10-25T17:30:00Z",
  "data": {
    "method": "value_iteration",
    "converged": true,
    "iterations": 27,
    "execution_time_s": 0.0842,
    "gamma": 0.9,
    "noise": 0.2,
    "policy_path_length": 15,
    "policy_path": [
      [1, 1], [1, 2], [1, 3], [1, 4], [2, 4], [2, 5], [2, 6],
      [3, 6], [4, 6], [5, 6], [5, 7], [6, 7], [7, 7], [8, 7], [8, 8]
    ],
    "sample_values": {
      "(1, 1)": 42.15,
      "(8, 7)": 88.50,
      "(8, 8)": 100.0
    }
  }
}
```

---

### 3. Évaluation Empirique par Simulation Monte-Carlo

Simule un grand nombre d'épisodes en milieu stochastique sous la politique sélectionnée afin d'évaluer statistiquement la résilience et le taux de succès vers l'état cible.

- **URL :** `/simulation/monte-carlo`
- **Méthode :** `POST`
- **Authentification requise :** Oui

#### Corps de la requête (Request Body)

```json
{
  "grid": {
    "width": 15,
    "height": 15,
    "start": [1, 1],
    "goal": [13, 13]
  },
  "gamma": 0.9,
  "noise": 0.2,
  "n_episodes": 1000,
  "max_steps": 500,
  "use_policy": true
}
```

#### Réponse de succès (`200 OK`)

```json
{
  "status": "success",
  "timestamp": "2026-10-25T17:30:00Z",
  "data": {
    "n_episodes": 1000,
    "success_rate_percent": 86.40,
    "avg_reward": 78.32,
    "std_reward": 14.21,
    "avg_steps": 28.65,
    "std_steps": 7.42,
    "min_steps": 24,
    "max_steps": 94,
    "execution_time_s": 1.15
  }
}
```

---

### 4. Consultation des Benchmarks Expérimentaux (E.1 à E.4)

Récupère la liste paginée et les métriques détaillées issues des protocoles expérimentaux exécutés.

- **URL :** `/experiments/results`
- **Méthode :** `GET`
- **Paramètres de requête (Query Params) :**
  - `experiment` *(optionnel, ex: `E.1`, `E.4`)* : Filtre par identifiant d'expérience
  - `page` *(optionnel, défaut: 1)* : Numéro de la page
  - `size` *(optionnel, défaut: 10)* : Nombre de résultats par page

#### Réponse de succès (`200 OK`)

```json
{
  "totalElements": 12,
  "totalPages": 2,
  "items": [
    {
      "experiment": "E.1",
      "grid": "Moyenne (15x15)",
      "algorithm": "A*",
      "cost": 24.0,
      "nodes_expanded": 82,
      "max_open_size": 28,
      "execution_time_ms": 2.45,
      "path_length": 25
    },
    {
      "experiment": "E.1",
      "grid": "Moyenne (15x15)",
      "algorithm": "UCS",
      "cost": 24.0,
      "nodes_expanded": 178,
      "max_open_size": 54,
      "execution_time_ms": 5.12,
      "path_length": 25
    },
    {
      "experiment": "E.4",
      "grid": "Moyenne (15x15)",
      "metric": "A* vs Markov (bruit=0.2)",
      "astar_success_rate": 68.20,
      "markov_success_rate": 86.40,
      "delta_robustness": "+18.20%"
    }
  ]
}
```

---

## Codes de statut et d'erreurs

| Code HTTP | Signification | Description |
| :--- | :--- | :--- |
| `200 OK` | Succès | Le calcul ou la consultation de ressources a abouti avec succès. |
| `201 Created` | Créé | Une simulation longue durée ou une tâche de benchmark a été initiée. |
| `400 Bad Request` | Requête invalide | Coordonnées hors limites, points de départ/but superposés ou paramètres numériques invalides. |
| `401 Unauthorized` | Non autorisé | Jeton d'authentification Bearer JWT manquant, invalide ou expiré. |
| `404 Not Found` | Introuvable | Résultat expérimental ou ressource demandée inexistant. |
| `422 Unprocessable` | Erreur de validation | Format de grille incohérent ou obstacles placés directement sur le départ ou le but. |
| `500 Internal Error`| Erreur serveur | Exception non gérée lors de la boucle de calcul numérique. |

---

## Référence de l'API Programmatique Python

Pour une utilisation directe sous forme de bibliothèque Python, les classes et méthodes principales sont structurées comme suit :

### 1. `grid.Grid`
- `Grid(width: int, height: int)` : Initialise une grille 2D vide de dimensions $W \times H$.
- `add_obstacle(x: int, y: int) -> None` : Déclare un obstacle infranchissable à la position $(x, y)$.
- `add_rectangular_obstacle(x1: int, y1: int, x2: int, y2: int) -> None` : Remplit un polygone rectangulaire d'obstacles.
- `set_start(x: int, y: int)` / `set_goal(x: int, y: int)` : Définit les cellules d'origine et de destination.
- `is_valid(x: int, y: int) -> bool` : Vérifie si la cellule est franchissable et dans les bornes.
- `get_neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]` : Renvoie les voisins accessibles en 4-voisinage.

### 2. `astar.PathFinder`
- `PathFinder(grid: Grid)` : Instancie le résolveur heuristique lié à une grille.
- `astar(heuristic=None, weight=1.0) -> SearchStats` : Exécute l'algorithme $A^*$ (ou Weighted $A^*$).
- `ucs() -> SearchStats` : Exécute la recherche à coût uniforme ($h(n) = 0$).
- `greedy(heuristic=None) -> SearchStats` : Exécute la recherche gloutonne Best-First ($f(n) = h(n)$).

### 3. `markov.MarkovChain`
- `MarkovChain(grid: Grid, gamma: float = 0.9, noise: float = 0.2)` : Initialise le Processus Décisionnel de Markov.
- `value_iteration(theta: float = 0.01, max_iterations: int = 1000) -> Dict` : Calcule $V^*(s)$ et extrait $\pi^*(s)$ via les équations d'optimalité de Bellman.
- `policy_iteration(theta: float = 0.01, max_iterations: int = 100) -> Dict` : Évalue et améliore itérativement la politique.
- `get_policy_path(start: Tuple[int, int], max_steps: int = 100) -> List[Tuple[int, int]]` : Trace le chemin nominal dicté par la politique optimale.

### 4. `simulation.MonteCarloSimulator`
- `MonteCarloSimulator(grid: Grid, mdp: MarkovChain)` : Instancie le moteur stochastique.
- `simulate_episode(start: Tuple[int, int], max_steps: int = 500, use_policy: bool = True) -> Dict` : Simule une trajectoire individuelle avec dérives aléatoires.
- `run_simulations(start: Tuple[int, int], n_episodes: int = 1000, max_steps: int = 500, use_policy: bool = True, verbose: bool = True) -> Dict` : Exécute une campagne Monte-Carlo et agrège les distributions statistiques.
- `compare_noise_levels(start: Tuple[int, int], noise_levels: List[float], n_episodes: int = 500) -> Dict` : Compare les métriques sur une série de valeurs de bruit $\varepsilon$.

---
[⬅ Retour au README principal](../README.md)
