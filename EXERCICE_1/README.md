# Exercice 1 — Découverte de Gymnasium

## Objectif

Ce sous-projet introduit la boucle d'interaction du Reinforcement Learning avec
`CartPole-v1`. Il inspecte l'environnement, mesure une politique aléatoire,
visualise un épisode puis compare PPO, A2C et DQN sur un même budget.

La [documentation technique](DOCUMENTATION_TECHNIQUE.html) détaille
l'architecture du notebook, ses cellules et ses artefacts.

## Déroulé

1. lecture des espaces d'observation et d'action ;
2. baseline aléatoire sur plusieurs épisodes ;
3. capture d'un épisode en images ;
4. entraînement de PPO, A2C et DQN pendant 50 000 pas ;
5. évaluation sur 20 épisodes et comparaison graphique ;
6. création du GIF de l'agent retenu.

## Résultat observé

Dans l'exécution enregistrée, les trois agents entraînés atteignent une
récompense moyenne de **500 ± 0**, soit le maximum de `CartPole-v1`. PPO est
retenu en cas d'égalité car il apparaît en premier dans la comparaison. Le GIF
de seed 42 montre **30 pas** pour la politique aléatoire contre **500 pas** pour
l'agent entraîné.

## Fichiers

| Fichier | Rôle |
|---|---|
| `NB_EXO_1.ipynb` | Notebook principal, exécuté sans erreur |
| `agent_aleatoire.gif` | Baseline visuelle, seed 42 |
| `agent_entraine.gif` | Épisode de l'agent retenu, même seed |
| `README.md` | Résumé d'utilisation |
| `DOCUMENTATION_TECHNIQUE.html` | Explication détaillée et autonome |

Les modèles restent en mémoire pendant le notebook. Cet exercice conserve les
deux GIF ; la sauvegarde de modèles sera traitée dans l'exercice 3.

## Exécution

Depuis la racine du projet :

```bash
uv sync --dev
cd EXERCICE_1
uv run jupyter lab NB_EXO_1.ipynb
```

Réexécution non interactive :

```bash
uv run jupyter nbconvert --to notebook --execute --inplace NB_EXO_1.ipynb \
  --ExecutePreprocessor.timeout=900
```

## Limites

- une seule seed d'entraînement est utilisée par algorithme ;
- le budget identique ne compense pas la sensibilité aux hyperparamètres ;
- DQN utilise des réglages adaptés à CartPole, contrairement à PPO et A2C ;
- CartPole est un environnement court et ne permet pas de généraliser le
  classement à d'autres problèmes.
