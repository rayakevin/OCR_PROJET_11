# Exercice 2 — Q-Learning avec une Q-table

## Objectif

Ce sous-projet implémente le Q-Learning sans bibliothèque d'algorithmes sur
`FrozenLake-v1`. Il rend explicites la Q-table, l'exploration epsilon-greedy et
la mise à jour de Bellman.

La [documentation technique](DOCUMENTATION_TECHNIQUE.html) détaille
l'architecture du notebook, ses cellules et ses artefacts.

## Déroulé

1. création du lac glissant 4 × 4 ;
2. initialisation d'une Q-table de forme 16 × 4 ;
3. entraînement pendant 20 000 épisodes ;
4. visualisation de la moyenne glissante ;
5. évaluation sans exploration sur 1 000 seeds réservées ;
6. lecture de la politique et génération de deux GIF.

## Résultat observé

L'exécution enregistrée obtient :

- **51,3 %** de réussite sur les 1 000 derniers épisodes d'entraînement, avec
  encore 5 % d'exploration ;
- **72,9 %** sur 1 000 épisodes d'évaluation sans exploration.

Le taux n'atteint pas 100 % car `is_slippery=True` rend les transitions
stochastiques.

## Fichiers

| Fichier | Rôle |
|---|---|
| `NB_EXO_2.ipynb` | Notebook principal, exécuté sans erreur |
| `q_table_frozenlake.npy` | Table apprise, 16 états × 4 actions |
| `frozenlake_reussite.gif` | Exemple reproductible d'arrivée au but |
| `frozenlake_echec.gif` | Exemple reproductible d'épisode sans réussite |
| `README.md` | Résumé d'utilisation |
| `DOCUMENTATION_TECHNIQUE.html` | Explication détaillée et autonome |

## Exécution

Depuis la racine du projet :

```bash
uv sync --dev
cd EXERCICE_2
uv run jupyter lab NB_EXO_2.ipynb
```

Réexécution non interactive :

```bash
uv run jupyter nbconvert --to notebook --execute --inplace NB_EXO_2.ipynb \
  --ExecutePreprocessor.timeout=600
```

## Limites

- la méthode suppose un nombre fini et réduit d'états et d'actions ;
- les hyperparamètres sont choisis pour ce lac précis ;
- l'évaluation mesure une politique déterministe dans un environnement
  stochastique ;
- la Q-table ne peut pas représenter directement les observations continues
  de CartPole, ce qui motive l'exercice 3.
