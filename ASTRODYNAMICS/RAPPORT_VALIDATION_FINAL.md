# Rapport de validation final — Mission Eagle-1

## 1. Verdict

La production technique de la mission est **conforme** au cahier des charges consolidé.

- Les cinq livrables demandés existent.
- Le modèle final dépasse le seuil de 200 sur 100 épisodes.
- L'API, la GUI et le dashboard fonctionnent localement.
- La vidéo respecte la durée et montre un atterrissage réussi.
- Les tests automatisés passent.
- Les logs nécessaires au dashboard sont disponibles aux niveaux expérience,
  épisode et pas de temps.

Deux actions administratives restent à confirmer par l'étudiant avant dépôt :

1. vérifier que `Raya_Kevin` correspond bien à l'ordre officiel `Nom_Prenom` ;
2. vérifier que `072026` correspond au mois réel de démarrage du projet.

Ces deux points ne concernent ni le code ni la performance du système.

## 2. Résultat final du pilote

| Indicateur | Résultat | Critère | Statut |
|---|---:|---:|:---:|
| Algorithme retenu | PPO, `gamma=0.999` | Choix justifié | ✅ |
| Épisodes finaux | 100 | 100 | ✅ |
| Récompense moyenne | 227,39 | > 200 | ✅ |
| Écart-type | 48,37 | Documenté | ✅ |
| Médiane | 239,50 | Documentée | ✅ |
| Taux de réussite | 95 % | Mesuré | ✅ |
| Longueur moyenne | 457,46 pas | Mesurée | ✅ |
| Proxy carburant moyen | 67,31 | Mesuré | ✅ |
| Meilleur modèle | `ppo_gamma_extended/best_model.zip` | Sauvegardé | ✅ |

Preuve : `artifacts/evaluations/ppo_gamma_extended_final_100/summary.json`.

## 3. Validation du notebook

| Exigence | Preuve | Statut |
|---|---|:---:|
| Notebook `.ipynb` propre et commenté | `NB.ipynb`, 46 cellules | ✅ |
| Toutes les cellules de code sont exécutées | 24/24 cellules avec compteur d'exécution | ✅ |
| Aucune erreur enregistrée | audit JSON du notebook | ✅ |
| Exploration de `LunarLander-v3` | sections 2 et 3 | ✅ |
| Description des huit observations | tableau et sortie de `reset()` | ✅ |
| Description des quatre actions | tableau et affichage exécutable | ✅ |
| Explication de `terminated` / `truncated` | section d'exploration | ✅ |
| Explication de la récompense et du carburant | section « récompense » | ✅ |
| Agent aléatoire | −213,04 ± 122,86 sur 50 épisodes | ✅ |
| Baseline DQN par défaut | 29,85 ± 132,90 sur 50 épisodes | ✅ |
| Utilisation explicite de `evaluate_policy` | cellule baseline DQN | ✅ |
| Justification de DQN | espace d'actions discret | ✅ |
| Étude et justification de PPO | acteur/critique et ressource fournie | ✅ |
| Un hyperparamètre modifié à la fois | `gamma`, `learning_rate`, `n_steps` | ✅ |
| Expériences comparées et interprétées | tableaux et graphiques | ✅ |
| TensorBoard / EvalCallback | `artifacts/tensorboard` et courbes `.npz` | ✅ |
| Meilleur modèle sauvegardé | callback et fichier ZIP | ✅ |
| Évaluation finale sur 100 épisodes | section 13 | ✅ |
| Moyenne finale > 200 | 227,39 | ✅ |
| Formats destinés à l'API documentés | section 14, JSON → float32 → action | ✅ |
| Instructions API/GUI/dashboard | conclusion du notebook | ✅ |
| Analyse des limites | conclusion | ✅ |

## 4. Validation de la journalisation

| Niveau | Données disponibles | Statut |
|---|---|:---:|
| Expérience | algorithme, hyperparamètres, seed, durée, scores | ✅ |
| Épisode | score, longueur, issue, état final, actions, carburant | ✅ |
| Pas de temps | états avant/après, action, récompense, probabilités/valeurs | ✅ |
| TensorBoard | métriques internes SB3 | ✅ |
| Registre global | `artifacts/experiment_registry.csv` | ✅ |
| Configuration | `artifacts/configs/*.json` | ✅ |
| Runs GUI | résumé et télémétrie sauvegardés automatiquement | ✅ |

Le fichier final `steps.csv` contient plus de 10 000 décisions et peut alimenter
directement les vues trajectoire du dashboard.

## 5. Validation de l'API

| Exigence | Preuve | Statut |
|---|---|:---:|
| Code source Python | `api/app.py` | ✅ |
| FastAPI choisi et justifiable | validation Pydantic + OpenAPI | ✅ |
| Flux état → action | `POST /predict` et `POST /play` | ✅ |
| État de huit valeurs | contrainte Pydantic min/max = 8 | ✅ |
| Valeurs finies | validateur dédié | ✅ |
| Action entière 0 à 3 | schéma de réponse | ✅ |
| JSON → NumPy `float32` | `predict_action()` | ✅ |
| Modèle chargé une seule fois | lifespan FastAPI | ✅ |
| Route de santé | `GET /health` | ✅ |
| Documentation interactive | `/docs` | ✅ |
| Chemin du modèle configurable | `EAGLE1_MODEL_PATH` | ✅ |
| Tests d'entrées valides/invalides | `api/tests/test_api.py` | ✅ |
| Run complet via l'API | seed 10000, score > 200 | ✅ |
| Code structuré et documenté | module, schémas, docstrings, README | ✅ |

## 6. Validation de la GUI

| Exigence | Preuve | Statut |
|---|---|:---:|
| Code source Python | `gui/app.py` | ✅ |
| Choix Streamlit justifié | `gui/README.md` | ✅ |
| Affichage d'une partie | rendu `rgb_array` en direct | ✅ |
| Actions obtenues via API | appel `/play` à chaque pas | ✅ |
| Aucune logique RL dans la GUI | aucun import Stable-Baselines3 | ✅ |
| État et récompense visibles | métriques temps réel | ✅ |
| Résumé du vol | score, pas, carburant, issue | ✅ |
| Graphiques | récompense cumulée et actions | ✅ |
| Logs du run | JSON et CSV dans `gui_runs/` | ✅ |
| Test d'une partie réussie | `gui/tests/test_gui.py` | ✅ |

## 7. Validation du dashboard

| Exigence | Preuve | Statut |
|---|---|:---:|
| Code source Python | `dashboard/app.py` | ✅ |
| Choix Streamlit justifié | `dashboard/README.md` | ✅ |
| Courbes de récompense | vues mission et apprentissage | ✅ |
| Moyenne et écart-type sur 100 épisodes | indicateurs principaux | ✅ |
| Runs gagnés / perdus | issue et taux de réussite | ✅ |
| Nombre de pas avant atterrissage | métrique et table épisode | ✅ |
| Comparaison des hyperparamètres | registre des expériences | ✅ |
| Distribution des actions | vue trajectoire | ✅ |
| Consommation estimée | scatter carburant / récompense | ✅ |
| Filtres interactifs | algorithme, phase, issue, récompense | ✅ |
| Graphiques dynamiques | Plotly | ✅ |
| Storytelling | cinq onglets ordonnés | ✅ |
| Inspection d'un épisode | sélecteur et télémétrie pas à pas | ✅ |
| Runs GUI | onglet dédié | ✅ |
| Tests des sources | `dashboard/tests/test_dashboard.py` | ✅ |

## 8. Validation de la vidéo

| Exigence | Résultat | Statut |
|---|---:|:---:|
| Format MP4 | H.264/MP4 lisible | ✅ |
| Durée entre 20 et 30 secondes | 25,31 s | ✅ |
| Partie sans erreur | lecture et inspection visuelle | ✅ |
| Manœuvre visible | descente et corrections visibles | ✅ |
| Atterrissage entre les drapeaux | dernière séquence contrôlée | ✅ |
| Récompense de victoire | seed 10000 : 216,89, récompense terminale +100 | ✅ |

## 9. Tests et exécution locale

Commande exécutée :

```bash
uv run pytest ASTRODYNAMICS/api/tests ASTRODYNAMICS/gui/tests ASTRODYNAMICS/dashboard/tests -q
```

Résultat : **10 tests réussis**. Les quatre avertissements affichés proviennent
de dépréciations de Starlette/Box2D et ne correspondent pas à des erreurs.

Smoke tests :

| Service | Résultat |
|---|:---:|
| API Uvicorn | HTTP 200 ✅ |
| GUI Streamlit | HTTP 200 ✅ |
| Dashboard Streamlit | HTTP 200 ✅ |

Validation globale :

```bash
uv run python ASTRODYNAMICS/validate_mission.py
```

Résultat : **15/15 contrôles réussis**.

## 10. Livrables et remise

| N° | Livrable | Statut |
|---:|---|:---:|
| 1 | Notebook mission | ✅ |
| 2 | Vidéo MP4 | ✅ |
| 3 | API Python | ✅ |
| 4 | Interface graphique Python | ✅ |
| 5 | Tableau de bord Python | ✅ |

Une structure prête à zipper est préparée dans
`livraison/Eagle_1_Raya_Kevin/`. Elle contient les cinq fichiers nommés, les
sources, le modèle, les données utiles, les dépendances et la documentation.

## 11. Préparation du bilan mentor

Les éléments suivants sont documentés et prêts à être expliqués oralement :

- choix de DQN puis sélection expérimentale de PPO ;
- rôle de `gamma` et effet observé ;
- séparation entraînement / sélection / évaluation finale ;
- flux et architecture de l'API ;
- formatage JSON, NumPy et action ;
- séparation backend / frontend ;
- choix de Streamlit et storytelling du dashboard ;
- difficultés PyTorch/Triton ;
- limites de la simulation et du proxy carburant ;
- points forts et axes d'amélioration.

La validation de la prestation orale elle-même reste naturellement à effectuer
pendant la session avec le mentor.

