# Mission AstroDynamics — Eagle-1

## Résultat

Le pilote retenu est un agent PPO optimisé avec Optuna. Son évaluation finale sur
100 épisodes indépendants donne :

- récompense moyenne : **268,98** ;
- écart-type : **37,46** ;
- taux d'atterrissage réussi : **96 %** ;
- pire épisode : **78,40** ;
- longueur moyenne : **263 pas** ;
- proxy carburant moyen : **35,79**.

## Démarche d'optimisation

La campagne Optuna se déroule en trois temps, portés par trois modules
réexécutables ([`optuna_search.py`](optuna_search.py),
[`optuna_select.py`](optuna_select.py), [`optuna_final.py`](optuna_final.py)) :

1. **Recherche large** — 120 essais TPE sur dix hyperparamètres. `gamma` est
   échantillonné sur `1-gamma` en loi log-uniforme, ce qui explore uniformément
   l'horizon effectif de 50 à 10 000 pas. Chaque essai est entraîné sur **deux
   seeds** et noté par leur moyenne, pour ne pas optimiser le bruit
   d'initialisation. Un `MedianPruner` a élagué 70 essais sur 120.
2. **Raffinement de `gamma`** — grille de 9 valeurs centrée sur l'optimum
   mesuré, régulière en horizon effectif, chaque point entraîné sur trois seeds.
3. **Validation de robustesse** — les finalistes sont réentraînés sur cinq
   seeds neuves à 300 000 pas. Le classement retient la moyenne inter-seeds et
   la pire seed comme arbitre.

Le réglage final utilise `gamma=0.99976`, `learning_rate≈1,69e-3`,
`ent_coef≈7,29e-3`, `n_epochs=20` et `clip_range=0.1`. Il est entraîné sur
1 500 000 pas avec trois seeds, la meilleure étant choisie sur le jeu de
sélection — jamais sur les 100 épisodes réservés.

## Livrables

| Livrable | Emplacement |
|---|---|
| Notebook | [`NB.ipynb`](NB.ipynb) |
| Vidéo | [`artifacts/video/eagle1-success-episode-0.mp4`](artifacts/video/eagle1-success-episode-0.mp4) |
| API | [`api/app.py`](api/app.py) |
| GUI | [`gui/app.py`](gui/app.py) |
| Dashboard | [`dashboard/app.py`](dashboard/app.py) |
| Modèle | [`artifacts/models/ppo_optuna/best_model.zip`](artifacts/models/ppo_optuna/best_model.zip) |

## Installation

Le projet utilise Python 3.12 et `uv` :

```bash
uv sync --dev
```

## Lancement

### Démonstration complète — commande recommandée

Depuis la racine du projet, cette commande démarre les trois services, contrôle
leur disponibilité et ouvre automatiquement leurs pages dans le navigateur :

```bash
uv run python launch_demo.py
```

Le lanceur affiche les URL et conserve les sorties dans
`ASTRODYNAMICS/artifacts/demo_logs/<date>-<heure>/`. `Ctrl+C` arrête ensemble
l'API, la GUI et le dashboard. Si un port par défaut est occupé, le prochain
port libre est sélectionné automatiquement.

En environnement distant ou sans navigateur :

```bash
uv run python launch_demo.py --no-browser
```

Les ports sont personnalisables si nécessaire, par exemple :

```bash
uv run python launch_demo.py --api-port 8010 --gui-port 8511 --dashboard-port 8512
```

### Lancement manuel

### API

```bash
uv run uvicorn ASTRODYNAMICS.api.app:app
```

- service : `http://localhost:8000` ;
- documentation : `http://localhost:8000/docs`.

### GUI

Dans un deuxième terminal, après avoir lancé l'API :

```bash
uv run streamlit run ASTRODYNAMICS/gui/app.py
```

Interface : `http://localhost:8501`.

### Dashboard

```bash
uv run streamlit run ASTRODYNAMICS/dashboard/app.py --server.port 8502
```

Dashboard : `http://localhost:8502`.

## Tests

```bash
uv run pytest ASTRODYNAMICS/api/tests ASTRODYNAMICS/gui/tests ASTRODYNAMICS/dashboard/tests ASTRODYNAMICS/tests -q
```

Les tests vérifient notamment :

- le chargement unique du modèle par l'API ;
- les entrées valides et invalides ;
- le flux JSON `état → action` ;
- une partie complète réussie via HTTP ;
- la sauvegarde d'un run GUI ;
- la disponibilité des métriques et trajectoires du dashboard ;
- les fonctions pures de la campagne Optuna : reparamétrisation de `gamma`,
  grille de raffinement, déduplication des finalistes et cloisonnement des
  seeds (`ASTRODYNAMICS/tests/`).

## Données et logs

`artifacts/` contient les modèles, configurations, courbes TensorBoard,
évaluations, figures, trajectoires et vidéo. Le dashboard utilise notamment :

- `experiment_registry.csv` ;
- `evaluations/*/episodes.csv` ;
- `evaluations/*/steps.csv` ;
- `evaluations/*/during_training/evaluations.npz` ;
- `optuna/ppo_lunarlander/trials.csv` ;
- `optuna/ppo_lunarlander/parameter_importance.csv` ;
- `optuna/ppo_lunarlander/gamma_focus/` ;
- `optuna/ppo_lunarlander/robustness.csv` ;
- `gui_runs/runs.csv` après utilisation de la GUI.

Les bases `study.db` ne sont pas lues par le dashboard : elles conservent les
études Optuna pour pouvoir les reprendre ou les rejouer hors notebook.
