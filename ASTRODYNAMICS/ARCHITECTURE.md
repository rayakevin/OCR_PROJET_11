# Architecture — Mission Eagle-1

Ce document décrit l'organisation technique du projet : les flux de données, la
responsabilité de chaque composant, les contrats entre eux et les décisions
structurantes. Pour l'installation et le lancement, voir [README.md](README.md).

## 1. Vue d'ensemble

Le projet entraîne un pilote automatique d'atterrissage sur `LunarLander-v3`,
puis l'expose par trois applications locales. Un principe gouverne l'ensemble :
**la logique d'apprentissage reste côté backend, jamais dans les interfaces.**

```
                        ┌─────────────────────────────────────────┐
   ENTRAÎNEMENT         │  NB.ipynb  +  optuna_search / _select /  │
   (hors ligne)         │             _final                       │
                        └───────────────────┬─────────────────────┘
                                            │ produit
                                            ▼
                        ┌─────────────────────────────────────────┐
   ARTEFACTS            │  artifacts/  (modèle, configs, logs,     │
   (source de vérité)   │  évaluations, études Optuna, courbes)    │
                        └──────┬───────────────────────┬──────────┘
                              │ lit le modèle          │ lit les logs
                              ▼                         ▼
              ┌───────────────────────┐   ┌───────────────────────────┐
   SERVICES   │  API FastAPI          │   │  Dashboard Streamlit       │
              │  état → action        │   │  suivi des expériences     │
              └───────────┬───────────┘   └───────────────────────────┘
                         │ HTTP /play
                         ▼
              ┌───────────────────────┐
              │  GUI Streamlit        │
              │  visualise une partie │
              └───────────────────────┘
```

Deux natures d'échange coexistent :

- **hors ligne** : le notebook et les modules Optuna entraînent et évaluent, en
  écrivant tout dans `artifacts/`. C'est lent, reproductible, exécuté une fois.
- **en ligne** : l'API sert le modèle, la GUI la consomme par HTTP, le dashboard
  lit les logs. C'est interactif, léger, relançable à volonté.

`artifacts/` est la **frontière** entre les deux : le seul point de contact. Un
service ne dépend jamais du notebook, seulement de ses sorties sur disque.

## 2. Composants et responsabilités

| Composant | Fichier | Rôle | Ne fait pas |
|---|---|---|---|
| Notebook | `NB.ipynb` | Démarche scientifique complète : exploration, baselines, sensibilité, optimisation, évaluation finale, vidéo | Servir des requêtes |
| Recherche Optuna | `optuna_search.py` | 120 essais TPE, gamma en log-uniforme, 2 seeds/essai, élagage | Sélectionner le modèle final |
| Sélection | `optuna_select.py` | Raffinement de gamma, validation multi-seed des finalistes | Entraîner le modèle long |
| Entraînement final | `optuna_final.py` | 3 seeds à 1,5 M pas, choix sur le jeu de sélection | Toucher au jeu de test |
| API | `api/app.py` | Charger le modèle une fois, valider l'état, renvoyer une action | Créer un environnement |
| GUI | `gui/app.py` | Jouer une partie, demander chaque action à l'API, visualiser | Charger un modèle |
| Dashboard | `dashboard/app.py` | Lire les logs, agréger, tracer, filtrer | Écrire des logs |
| Lanceur | `launch_demo.py` | Démarrer les trois services, gérer les ports, ouvrir le navigateur | Contenir de la logique métier |
| Validation | `validate_mission.py` | 22 contrôles bout-en-bout du livrable | Remplacer les tests unitaires |

## 3. Flux principal : état → action

C'est le contrat central de la mission, exigé par le cahier des charges.

```
observation (8 flottants)
   → StateRequest (Pydantic : longueur 8, valeurs finies)
   → np.asarray(..., dtype=float32)
   → PPO.predict(deterministic=True)
   → action ∈ {0, 1, 2, 3}
   → ActionResponse (action, action_label, model_id)
```

La conversion `JSON → NumPy float32 → tenseur` est faite par
`predict_action()` ([api/app.py](api/app.py)). Le modèle est chargé **une seule
fois** au démarrage via le `lifespan` FastAPI, jamais par requête.

La GUI reconstitue une partie complète en appelant `POST /play` à chaque pas,
puis rejoue le vol de façon déterministe (même seed, mêmes actions). Elle
n'importe jamais `stable_baselines3` — vérifié par un contrôle automatique.

## 4. Contrats de données

`artifacts/` suit une convention stable, à trois niveaux de granularité :

| Niveau | Fichier | Contenu | Consommateur |
|---|---|---|---|
| Expérience | `experiment_registry.csv` | 1 ligne/évaluation : algo, seed, scores, réussite | Dashboard, `validate_mission` |
| Configuration | `configs/<id>.json` | Hyperparamètres complets, durée, chemins, versions | Dashboard (comparaison A/B) |
| Épisode | `evaluations/<id>/episodes.csv` | 1 ligne/épisode : score, longueur, issue, état final, actions, carburant | Dashboard, tests |
| Pas de temps | `evaluations/<id>/steps.csv` | 1 ligne/pas : état, action, récompense, valeurs PPO | Dashboard (trajectoire) |
| Apprentissage | `evaluations/<id>/during_training/evaluations.npz` | Courbes EvalCallback | Dashboard |
| Optuna | `optuna/ppo_lunarlander/*.csv` + `study.db` | Essais, grille gamma, robustesse, importance | Dashboard, tests |

Les identifiants relient les niveaux : un `evaluation_id` retrouve sa config,
ses épisodes, ses pas et sa courbe. Le principe d'écriture est un **upsert** :
réévaluer un identifiant remplace sa ligne au lieu d'en créer une seconde.

**Contrat de chemin** : les applications déduisent leurs chemins de leur propre
position (`Path(__file__).parents[...]`), jamais d'un chemin absolu figé. Le
modèle servi est en outre surchargeable par la variable `EAGLE1_MODEL_PATH`, et
l'URL de l'API côté GUI par `EAGLE1_API_URL`.

## 5. La campagne Optuna en trois modules

La recherche d'hyperparamètres est le cœur méthodologique et vit hors du
notebook, dans trois modules réexécutables et parallélisables. Voir
[optuna_search.py](optuna_search.py) pour le détail.

1. **Recherche large** — 120 essais TPE sur dix hyperparamètres. Trois choix
   structurants : `gamma` échantillonné sur `1-gamma` en loi log-uniforme (pour
   explorer uniformément l'horizon effectif) ; chaque essai noté sur la
   **moyenne de deux seeds** (pour ne pas optimiser le bruit d'initialisation) ;
   `MedianPruner` qui interrompt les essais sous la médiane.
2. **Raffinement de gamma** — grille de 9 valeurs **centrée sur l'optimum
   mesuré**, régulière en horizon effectif, chaque point sur trois seeds.
3. **Validation puis entraînement final** — les finalistes sont réentraînés sur
   cinq seeds neuves ; le gagnant est entraîné sur 1,5 M pas avec trois seeds,
   la meilleure étant choisie sur le **jeu de sélection**, jamais sur le jeu de
   test réservé.

Le cloisonnement des seeds est strict et vérifié par des tests unitaires :
recherche `2000+`, sélection `5000+`, évaluation finale `10000+`. Aucune
décision n'est prise en regardant les épisodes qui serviront à annoncer le
résultat.

## 6. Décisions techniques justifiables

| Décision | Raison |
|---|---|
| DQN comme baseline imposée | Espace d'actions discret ; DQN y est l'algorithme de référence |
| PPO comme candidat retenu | Acteur-critique stable, meilleure robustesse mesurée sur cet environnement |
| `gamma` sur `1-gamma` en log | Le comportement de PPO se joue dans les dernières décimales de gamma |
| Multi-seed par essai | Un score sur une seule seed mesure autant le bruit que le réglage |
| Élagage médian | Économise ~60 % du budget en coupant les essais divergents tôt |
| Évaluation en `deterministic=True` | Consistance : le mode stochastique révèle un meilleur comportement latent mais avec une variance ingérable |
| FastAPI | Validation Pydantic, typage, documentation OpenAPI automatique |
| Streamlit (GUI + dashboard) | Application Python locale, pas de dépendance serveur externe |
| Modèle chargé au `lifespan` | Chargement unique, latence d'inférence compatible avec une partie interactive |

## 7. Tests

Deux niveaux, exécutés ensemble :

- **intégration** (`api/tests`, `gui/tests`, `dashboard/tests`, 11 tests) :
  chargement du modèle, entrées valides/invalides, run complet via HTTP,
  sauvegarde d'un run GUI, disponibilité des sources du dashboard ;
- **unitaire** (`tests/`, 22 tests) : fonctions pures de la campagne Optuna —
  reparamétrisation de gamma, grille de raffinement, déduplication des
  finalistes, cloisonnement des seeds.

`validate_mission.py` ajoute 22 contrôles de niveau livrable (notebook exécuté
sans erreur, seuil de 200 dépassé, vidéo dans la durée, sources présentes).

```bash
uv run pytest ASTRODYNAMICS/api/tests ASTRODYNAMICS/gui/tests ASTRODYNAMICS/dashboard/tests ASTRODYNAMICS/tests -q
uv run python ASTRODYNAMICS/validate_mission.py
```
