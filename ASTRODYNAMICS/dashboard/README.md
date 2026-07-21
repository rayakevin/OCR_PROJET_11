# Dashboard Eagle-1

Le dashboard Streamlit raconte le passage de la politique aléatoire au pilote
PPO final. Il lit directement les CSV, JSON et fichiers `EvalCallback` produits
par le notebook et les runs enregistrés par la GUI.

## Lancement

```bash
uv run streamlit run ASTRODYNAMICS/dashboard/app.py --server.port 8502
```

Dashboard : `http://localhost:8502`. Le port est précisé explicitement car
Streamlit prendrait sinon le port 8501, déjà utilisé par la GUI.

## Contenu

- indicateurs finaux : moyenne, écart-type, réussite, durée et carburant ;
- comparaison dynamique des expériences et hyperparamètres ;
- courbes d'apprentissage ;
- suivi de la recherche Optuna : essais TPE, recherche fine de `gamma`,
  importance des paramètres et validation multi-seed ;
- filtres par algorithme, phase, résultat et récompense ;
- analyse des 100 épisodes finaux ;
- exploration pas à pas d'une trajectoire ;
- suivi des runs lancés depuis la GUI.

## Storytelling

1. **Vue mission** : mesure le gain par rapport aux baselines.
2. **Apprentissage** : montre quand et comment les modèles progressent.
3. **Optuna** : rend visible la recherche large, le raffinement de `gamma`
   autour de l'optimum mesuré et la robustesse du choix final sur cinq seeds.
   Le losange marque la pire seed de chaque finaliste : c'est ce point, et non
   la moyenne, qui distingue un réglage fiable d'un réglage chanceux.
4. **Épisodes** : vérifie que la moyenne ne masque pas les échecs.
5. **Trajectoire** : explique les décisions au cours d'un vol.
6. **Runs GUI** : relie l'évaluation hors ligne à l'utilisation locale.

## Tests

```bash
uv run pytest ASTRODYNAMICS/dashboard/tests -q
```
