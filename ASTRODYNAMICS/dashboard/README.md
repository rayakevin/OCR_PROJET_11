# Dashboard Eagle-1

Le dashboard Streamlit lit les fichiers produits pendant les entraînements et
les évaluations. Il ne lance aucun modèle : son rôle est uniquement d'expliquer
les résultats déjà enregistrés dans `ASTRODYNAMICS/artifacts/`.

## Lancement

```bash
uv run streamlit run ASTRODYNAMICS/dashboard/app.py --server.port 8502
```

Dashboard : `http://localhost:8502`.

## Les quatre onglets

| Onglet | Question traitée |
|---|---|
| **Synthèse** | Le pilote atteint-il l'objectif de 200 points et fait-il mieux que les baselines ? |
| **Apprentissage** | Comment les performances ont-elles évolué entre les phases et pendant les entraînements ? |
| **Épisodes** | Quels vols réussissent ou échouent, avec quelle récompense, quelle consommation et quelles actions ? |
| **Optuna** | Quels essais ont été menés et quels hyperparamètres ont compté dans le choix final ? |

La partie « Épisodes » fournit les interactions principales : choix de
l'évaluation, filtre par issue et plage de récompense. Elle affiche aussi la
récompense par épisode, une moyenne glissante sur dix épisodes, le proxy de
carburant et la distribution des quatre actions.

## Sources de données

- `experiment_registry.csv` pour la vue d'ensemble ;
- `evaluations/*/episodes.csv` pour les résultats par épisode ;
- `evaluations/*/during_training/evaluations.npz` pour les courbes ;
- `optuna/ppo_lunarlander/*.csv` pour la recherche et la robustesse.

Le choix de composants graphiques natifs Streamlit garde le code compact et
suffit au besoin d'une application locale.

## Tests

```bash
uv run pytest ASTRODYNAMICS/dashboard/tests -q
```

Les tests contrôlent les schémas de données, le calcul de la moyenne glissante
et le chargement complet de l'application avec `streamlit.testing`.
