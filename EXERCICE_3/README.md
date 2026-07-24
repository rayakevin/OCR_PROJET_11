# Exercice 3 — Deep Q-Network sur CartPole

## Objectif

Ce sous-projet remplace la Q-table par un réseau de neurones. Il implémente
d'abord un DQN pédagogique en PyTorch, puis compare ce résultat à deux versions
Stable-Baselines3.

La [documentation technique](DOCUMENTATION_TECHNIQUE.html) détaille
l'architecture du notebook, ses classes, ses fonctions et ses artefacts.

## Déroulé

1. définition du réseau `DQN` ;
2. création du `ReplayBuffer` ;
3. sélection epsilon-greedy et mise à jour par lots ;
4. synchronisation périodique du réseau cible ;
5. entraînement manuel pendant 1 000 épisodes ;
6. comparaison avec deux DQN SB3 entraînés pendant 50 000 pas ;
7. sauvegarde des poids et du meilleur modèle.

## Résultats observés

Les trois politiques sont évaluées sans exploration sur les mêmes 100 seeds :

| Politique | Récompense moyenne | Écart-type |
|---|---:|---:|
| DQN manuel PyTorch | 293,9 | 95,8 |
| DQN SB3 par défaut | 199,9 | 54,0 |
| DQN SB3 réglé | 500,0 | 0,0 |

`dqn_cartpole.zip` contient désormais le **DQN réglé**, c'est-à-dire le modèle
qui produit le résultat final annoncé.

## Fichiers

| Fichier | Rôle |
|---|---|
| `NB_EXO_3.ipynb` | Notebook principal, exécuté sans erreur |
| `dqn_manuel_cartpole.pt` | Poids du réseau PyTorch pédagogique |
| `dqn_cartpole.zip` | Modèle SB3 réglé et rechargeable |
| `logs/` | Journaux TensorBoard des essais et des deux configurations finales |
| `README.md` | Résumé d'utilisation |
| `DOCUMENTATION_TECHNIQUE.html` | Explication détaillée et autonome |

## Exécution

Depuis la racine du projet :

```bash
uv sync --dev
cd EXERCICE_3
uv run jupyter lab NB_EXO_3.ipynb
```

Réexécution non interactive :

```bash
uv run jupyter nbconvert --to notebook --execute --inplace NB_EXO_3.ipynb \
  --ExecutePreprocessor.timeout=1800
```

Les courbes TensorBoard se consultent avec :

```bash
uv run tensorboard --logdir EXERCICE_3/logs
```

## Limites

- une seule seed d'entraînement est comparée ;
- les 100 seeds servent à réduire le bruit d'évaluation, pas celui de
  l'apprentissage ;
- le DQN manuel privilégie la lisibilité à l'optimisation ;
- les résultats illustrent l'effet des hyperparamètres sur CartPole, sans
  établir un classement général.
