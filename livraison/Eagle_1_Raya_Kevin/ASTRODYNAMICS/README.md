# Mission AstroDynamics — Eagle-1

## Résultat

Le pilote retenu est un agent PPO avec `gamma=0.999`. Son évaluation finale sur
100 épisodes indépendants donne :

- récompense moyenne : **227,39** ;
- écart-type : **48,37** ;
- taux d'atterrissage réussi : **95 %** ;
- longueur moyenne : **457 pas** ;
- proxy carburant moyen : **67,31**.

## Livrables

| Livrable | Emplacement |
|---|---|
| Notebook | [`NB.ipynb`](NB.ipynb) |
| Vidéo | [`artifacts/video/eagle1-success-episode-0.mp4`](artifacts/video/eagle1-success-episode-0.mp4) |
| API | [`api/app.py`](api/app.py) |
| GUI | [`gui/app.py`](gui/app.py) |
| Dashboard | [`dashboard/app.py`](dashboard/app.py) |
| Modèle | [`artifacts/models/ppo_gamma_extended/best_model.zip`](artifacts/models/ppo_gamma_extended/best_model.zip) |

## Installation

Le projet utilise Python 3.12 et `uv` :

```bash
uv sync --dev
```

## Lancement

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
uv run pytest ASTRODYNAMICS/api/tests ASTRODYNAMICS/gui/tests ASTRODYNAMICS/dashboard/tests -q
```

Les tests vérifient notamment :

- le chargement unique du modèle par l'API ;
- les entrées valides et invalides ;
- le flux JSON `état → action` ;
- une partie complète réussie via HTTP ;
- la sauvegarde d'un run GUI ;
- la disponibilité des métriques et trajectoires du dashboard.

## Données et logs

`artifacts/` contient les modèles, configurations, courbes TensorBoard,
évaluations, figures, trajectoires et vidéo. Le dashboard utilise notamment :

- `experiment_registry.csv` ;
- `evaluations/*/episodes.csv` ;
- `evaluations/*/steps.csv` ;
- `evaluations/*/during_training/evaluations.npz` ;
- `gui_runs/runs.csv` après utilisation de la GUI.

