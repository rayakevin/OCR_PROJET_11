# Dashboard Eagle-1

Le dashboard Streamlit raconte le passage de la politique aléatoire au pilote
PPO final. Il lit directement les CSV, JSON et fichiers `EvalCallback` produits
par le notebook et les runs enregistrés par la GUI.

## Lancement

```bash
uv run streamlit run ASTRODYNAMICS/dashboard/app.py
```

## Contenu

- indicateurs finaux : moyenne, écart-type, réussite, durée et carburant ;
- comparaison dynamique des expériences et hyperparamètres ;
- courbes d'apprentissage ;
- filtres par algorithme, phase, résultat et récompense ;
- analyse des 100 épisodes finaux ;
- exploration pas à pas d'une trajectoire ;
- suivi des runs lancés depuis la GUI.

## Storytelling

1. **Vue mission** : mesure le gain par rapport aux baselines.
2. **Apprentissage** : montre quand et comment les modèles progressent.
3. **Épisodes** : vérifie que la moyenne ne masque pas les échecs.
4. **Trajectoire** : explique les décisions au cours d'un vol.
5. **Runs GUI** : relie l'évaluation hors ligne à l'utilisation locale.

## Tests

```bash
uv run pytest ASTRODYNAMICS/dashboard/tests -q
```

