# Cahier des charges — Pilote automatique Eagle-1

## 1. Finalité

AstroDynamics souhaite automatiser l'atterrissage du module Eagle-1 afin d'améliorer la sécurité, la précision d'atterrissage et la maîtrise de la consommation de carburant.

Le projet doit fournir un prototype local fondé sur l'apprentissage par renforcement. Le système doit entraîner puis exposer un agent capable de piloter `LunarLander-v3`, montrer une réussite, fournir des métriques exploitables et rendre les résultats accessibles par une API, une interface graphique et un tableau de bord interactif.

## 2. Sources prises en compte

Ce cahier des charges consolide :

- le texte de la mission « Pilotez l'atterrisseur lunaire Eagle-1 » ;
- le « Brief de Mission : Pilote Automatique pour l'atterrissage d'Eagle-1 » ;
- la fiche d'autoévaluation « P11 - AIE : FAE » ;
- les critères d'évaluation complémentaires communiqués par le mentor ;
- la consigne complémentaire de journaliser un maximum de données utiles au dashboard ;
- les ressources Gymnasium, Stable-Baselines3, FastAPI, Streamlit, Gradio, Looker Studio et l'article consacré aux hyperparamètres PPO.

La version retenue pour toute la réalisation est exclusivement `LunarLander-v3`.

## 3. Périmètre

### Inclus

- exploration et documentation de `LunarLander-v3` ;
- entraînement d'une baseline DQN Stable-Baselines3 ;
- optimisation expérimentale des hyperparamètres ;
- évaluation fiable et reproductible ;
- sauvegarde du meilleur modèle ;
- journalisation détaillée des entraînements et évaluations ;
- notebook complet et commenté ;
- vidéo d'une réussite ;
- API locale d'inférence ;
- GUI locale montrant une partie ;
- dashboard interactif de suivi des performances ;
- tests ou exemples de fonctionnement de l'API ;
- documentation d'installation et d'utilisation.

### Hors périmètre obligatoire

- entraînement sur un environnement physique réel ;
- déploiement sur un serveur distant ;
- dépendance obligatoire à un service externe ;
- utilisation obligatoire de PPO ;
- optimisation parfaite de la consommation de carburant au-delà du mécanisme de récompense et des métriques disponibles.

## 4. Contraintes générales

| Référence | Exigence | Priorité | Validation attendue |
|---|---|---:|---|
| CG-01 | Le langage principal est Python. | Obligatoire | Le code livré est en Python. |
| CG-02 | Le prototype utilise PyTorch et/ou Stable-Baselines3. | Obligatoire | Dépendances et imports vérifiés. |
| CG-03 | L'environnement est `LunarLander-v3`. | Obligatoire | Toutes les créations d'environnement utilisent cette version. |
| CG-04 | API, GUI et dashboard sont exécutables localement sur une machine standard. | Obligatoire | Lancement local documenté et testé. |
| CG-05 | Le code est lisible, structuré, commenté et documenté. | Obligatoire | Relecture et contrôle des livrables. |
| CG-06 | Les expériences sont reproductibles autant que possible. | Obligatoire | Seeds, versions et configurations enregistrées. |
| CG-07 | Le délai indicatif du brief est de dix jours ouvrables. | Contrainte projet | Planning adapté et priorisation des livrables. |

## 5. Exigences relatives à l'environnement RL

| Référence | Exigence | Priorité | Validation attendue |
|---|---|---:|---|
| RL-01 | Le notebook explore l'espace d'observation. | Obligatoire | Forme, type, bornes et sens des huit variables présentés. |
| RL-02 | Le notebook explore l'espace d'actions. | Obligatoire | Les quatre actions discrètes sont décrites. |
| RL-03 | Les retours de `reset()` et `step()` sont expliqués. | Obligatoire | État, récompense, fin, troncature et informations illustrés. |
| RL-04 | Le choix de l'algorithme est justifié. | Obligatoire | DQN est relié explicitement à l'espace d'actions discret. |
| RL-05 | Une baseline DQN utilise initialement les paramètres par défaut. | Obligatoire | Configuration et modèle de référence présents. |
| RL-06 | La baseline est évaluée sur au moins 50 épisodes avec `evaluate_policy`. | Obligatoire | Moyenne et écart-type documentés. |
| RL-07 | Les hyperparamètres sont optimisés et les expériences sont documentées. | Obligatoire | Table comparative et conclusions dans le notebook. |
| RL-08 | Le meilleur modèle est sauvegardé. | Obligatoire | Fichier rechargeable et métadonnées associées. |
| RL-09 | Le modèle final dépasse une récompense moyenne de 200 sur 100 épisodes. | Obligatoire | Rapport d'évaluation final et données par épisode. |
| RL-10 | La stabilité est analysée, pas seulement le meilleur épisode. | Obligatoire | Écart-type, distribution et taux de réussite présentés. |
| RL-11 | La généralisation est contrôlée avec des épisodes/seeds d'évaluation distincts. | Recommandé fort | Seeds enregistrées et protocole décrit. |

Le seuil de validation retenu est le plus strict des documents : moyenne strictement supérieure à 200 sur 100 épisodes. La mention « 150-200 » de la fiche d'autoévaluation n'abaisse pas cet objectif.

## 6. Exigences d'expérimentation

| Référence | Exigence | Priorité | Validation attendue |
|---|---|---:|---|
| EXP-01 | Une performance initiale est établie avant optimisation. | Obligatoire | Résultats de baseline datés et sauvegardés. |
| EXP-02 | Un seul hyperparamètre est modifié à la fois pendant l'analyse initiale. | Recommandé fort | Comparaisons contrôlées dans le notebook. |
| EXP-03 | Les paramètres modifiés et leurs effets sont documentés. | Obligatoire | Configuration, résultat et interprétation par expérience. |
| EXP-04 | TensorBoard ou un équivalent suit l'entraînement. | Obligatoire | Journaux consultables et courbes comparatives. |
| EXP-05 | Les modèles prometteurs sont sauvegardés. | Obligatoire | Checkpoints ou meilleurs modèles conservés. |
| EXP-06 | Le protocole d'évaluation reste comparable entre les runs. | Obligatoire | Même nombre d'épisodes, mêmes métriques et seeds maîtrisées. |
| EXP-07 | La durée et les ressources des entraînements sont suivies. | Recommandé | Durée et nombre de timesteps dans les métadonnées. |

### Hyperparamètres concernés

Pour DQN, les paramètres prioritaires sont notamment `learning_rate`, `gamma` et `buffer_size`, puis `batch_size`, `learning_starts`, `train_freq`, `gradient_steps`, `target_update_interval` et les paramètres d'exploration.

La ressource PPO fournit des indications sur `n_steps`, le nombre d'epochs, `batch_size`, `clip_range`, `gamma`, `gae_lambda`, `ent_coef` et `learning_rate`. Ces indications ne s'appliquent que si PPO est étudié en option. Elles ne constituent pas une grille de réglage DQN.

## 7. Exigences de journalisation et de données

### Objectif

Le système de logs doit permettre de reproduire les expériences, comparer les modèles, analyser les échecs et alimenter directement le dashboard.

### Données par expérience

| Champ minimal | Description |
|---|---|
| `experiment_id` | Identifiant unique du run. |
| `experiment_name` | Nom lisible. |
| `timestamp` | Date et heure. |
| `algorithm` | Algorithme utilisé. |
| `env_id` | Doit valoir `LunarLander-v3`. |
| `seed` | Seed d'entraînement. |
| `hyperparameters` | Configuration complète sérialisée. |
| `total_timesteps` | Nombre de pas d'entraînement. |
| `training_duration_s` | Durée de l'entraînement. |
| `model_path` | Chemin du modèle sauvegardé. |
| `mean_reward` | Moyenne d'évaluation. |
| `std_reward` | Écart-type d'évaluation. |
| `success_rate` | Part des épisodes réussis. |

### Données par épisode

| Champ minimal | Description |
|---|---|
| `experiment_id` | Lien vers l'expérience. |
| `episode_id` | Identifiant de l'épisode. |
| `evaluation_seed` | Seed utilisée. |
| `total_reward` | Récompense cumulée. |
| `episode_length` | Nombre de pas. |
| `outcome` | Réussite, crash ou troncature. |
| `final_x`, `final_y` | Position finale. |
| `final_vx`, `final_vy` | Vitesse finale. |
| `final_angle` | Angle final. |
| `left_contact`, `right_contact` | Contacts finaux. |
| `action_0_count` à `action_3_count` | Distribution des actions. |
| `fuel_proxy` | Estimation documentée de la consommation à partir des actions moteurs. |

### Données détaillées par pas

Pour les évaluations et démonstrations sélectionnées : état complet, action, récompense immédiate, récompense cumulée, valeurs Q si disponibles, `terminated` et `truncated`.

### Formats

- TensorBoard pour les séries d'apprentissage ;
- CSV ou Parquet pour les tables destinées à l'analyse et au dashboard ;
- JSON pour les configurations et les contrats de l'API ;
- modèle Stable-Baselines3 dans son format de sauvegarde natif.

### Validation

- Les logs sont créés automatiquement.
- Ils restent lisibles hors du notebook.
- Un identifiant relie configurations, modèle, épisodes et trajectoires.
- Les données personnelles ou secrets ne sont pas inscrits dans les logs.

## 8. Exigences du notebook

| Référence | Exigence | Priorité | Validation attendue |
|---|---|---:|---|
| NB-01 | Livrer un notebook `.ipynb` propre, commenté et exécutable. | Obligatoire | Exécution complète sans erreur. |
| NB-02 | Présenter exploration, choix de l'algorithme, entraînement, optimisation et évaluation finale. | Obligatoire | Toutes les sections sont présentes. |
| NB-03 | Documenter le formatage des données pour l'API. | Obligatoire | Section dédiée états/actions/récompenses et JSON/NumPy/tenseurs. |
| NB-04 | Présenter la baseline et sa récompense moyenne. | Obligatoire | Évaluation sur au moins 50 épisodes. |
| NB-05 | Documenter chaque expérience d'optimisation. | Obligatoire | Paramètre modifié, résultat et interprétation. |
| NB-06 | Présenter l'évaluation finale sur 100 épisodes. | Obligatoire | Moyenne, écart-type et métriques complémentaires. |
| NB-07 | Expliquer les limites et la généralisation. | Obligatoire | Analyse critique finale. |
| NB-08 | Donner les instructions d'utilisation des autres livrables. | Obligatoire | Commandes et exemples présents ou référencés. |

## 9. Contrat fonctionnel de l'API

### Flux principal

`état LunarLander-v3 → validation et formatage → modèle RL → action discrète`

### Entrée minimale

Un document JSON contenant huit nombres finis, dans l'ordre documenté de l'observation `LunarLander-v3`.

Exemple indicatif :

```json
{
  "state": [0.0, 1.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0],
  "deterministic": true
}
```

### Sortie minimale

```json
{
  "action": 2
}
```

L'action doit être un entier compris entre 0 et 3. La réponse peut aussi fournir le libellé de l'action, l'identifiant/version du modèle et les valeurs Q lorsqu'elles sont disponibles.

### Exigences

| Référence | Exigence | Priorité | Validation attendue |
|---|---|---:|---|
| API-01 | L'API accepte un état et renvoie une action. | Obligatoire | Appel valide démontré. |
| API-02 | Les entrées sont contrôlées : longueur, type et valeurs finies. | Obligatoire | Tests des entrées valides et invalides. |
| API-03 | Le modèle est chargé côté backend. | Obligatoire | Aucun modèle RL dans le frontend. |
| API-04 | Le code est structuré et documenté. | Obligatoire | Modules, docstrings et instructions présents. |
| API-05 | L'API permet à l'agent d'effectuer un run complet. | Obligatoire | Test d'intégration ou exemple exécutable. |
| API-06 | L'architecture est justifiable. | Obligatoire | Flux et responsabilités documentés. |
| API-07 | Une route de santé est disponible. | Recommandé fort | Vérification du service et du modèle chargé. |

FastAPI est l'outil recommandé en raison de la validation des schémas, du typage et de la documentation OpenAPI automatique. Flask reste acceptable si les mêmes contrôles sont mis en œuvre.

## 10. Exigences de la GUI

| Référence | Exigence | Priorité | Validation attendue |
|---|---|---:|---|
| GUI-01 | La GUI affiche une partie jouée par l'agent. | Obligatoire | Démonstration locale complète. |
| GUI-02 | Les actions de l'agent proviennent de l'API. | Obligatoire | Vérification du flux frontend → API. |
| GUI-03 | La logique RL reste côté backend. | Obligatoire | Aucun chargement du modèle dans la GUI. |
| GUI-04 | Le choix de Streamlit, Gradio ou autre est justifié. | Obligatoire | Justification dans la documentation/bilan. |
| GUI-05 | L'état de l'épisode et la récompense sont visibles. | Recommandé fort | Interface lisible pendant la partie. |

## 11. Exigences du tableau de bord

Le tableau de bord peut utiliser Streamlit, Gradio ou Looker Studio. Une solution locale en Python facilite la contrainte d'absence de dépendance à un serveur externe.

### Métriques minimales

- récompenses par épisode ;
- récompense moyenne et écart-type sur 100 épisodes ;
- moyenne glissante de récompense ;
- nombre et taux de runs gagnés/perdus ;
- nombre de pas avant atterrissage ;
- comparaison de la baseline et des expériences optimisées ;
- distribution des actions et indicateur de consommation des moteurs.

### Interactivité

Au moins un filtre ou un graphique dynamique doit être disponible. Les filtres recommandés portent sur l'expérience, les hyperparamètres, la seed, le résultat de l'épisode et la plage de récompense.

### Storytelling attendu

Le dashboard doit permettre de comprendre :

1. le niveau de départ ;
2. l'évolution pendant l'entraînement ;
3. l'effet des réglages ;
4. la stabilité du meilleur modèle ;
5. ses réussites, ses échecs et son utilisation des moteurs.

| Référence | Exigence | Priorité | Validation attendue |
|---|---|---:|---|
| DB-01 | Les métriques pertinentes sont affichées. | Obligatoire | Contrôle visuel et cohérence avec les fichiers de logs. |
| DB-02 | Une interaction est implémentée. | Obligatoire | Filtre ou graphique dynamique fonctionnel. |
| DB-03 | Les courbes de récompense et métriques de synthèse sont présentes. | Obligatoire | Dashboard testé sur les résultats finaux. |
| DB-04 | Les choix de design suivent un storytelling justifiable. | Obligatoire | Parcours visuel documenté. |

## 12. Exigences de la vidéo

| Référence | Exigence | Priorité | Validation attendue |
|---|---|---:|---|
| VID-01 | Le format est MP4 ou un lien YouTube accepté par la plateforme. | Obligatoire | Fichier ou lien lisible. |
| VID-02 | La durée est comprise entre 20 et 30 secondes. | Obligatoire | Durée vérifiée. |
| VID-03 | La séquence montre une partie sans erreur. | Obligatoire | Lecture complète. |
| VID-04 | L'agent réussit l'atterrissage et obtient une récompense de victoire. | Obligatoire | Réussite observable et score associé. |

`gymnasium.wrappers.RecordVideo` est l'outil recommandé pour l'enregistrement.

## 13. Outils et technologies

| Besoin | Outil principal | Alternatives ou compléments |
|---|---|---|
| Environnement RL | Gymnasium `LunarLander-v3` | Box2D, pygame pour le rendu |
| Agent RL | Stable-Baselines3 DQN | PyTorch pour inspection/personnalisation ; PPO facultatif |
| Évaluation | `evaluate_policy` | Boucle d'évaluation personnalisée pour les logs détaillés |
| Suivi d'entraînement | TensorBoard | CSV/Parquet et graphiques du notebook |
| Analyse | NumPy, pandas, Matplotlib | Seaborn, Plotly |
| API | FastAPI | Flask |
| Validation API | Pydantic | Validation manuelle avec Flask |
| Tests API | pytest, client de test FastAPI | Exemples/scripts reproductibles |
| GUI | Streamlit | Gradio |
| Dashboard | Streamlit + Plotly | Gradio, Looker Studio |
| Vidéo | `RecordVideo` | imageio/ffmpeg si nécessaire |
| Notebook | Jupyter/Google Colab | VS Code Jupyter local |

## 14. Tests et validation technique

### Tests minimaux

- création et fermeture de `LunarLander-v3` ;
- chargement du modèle final ;
- prédiction sur un état valide ;
- rejet d'un état de mauvaise longueur ;
- rejet des chaînes, valeurs manquantes, `NaN` et infinis ;
- garantie que l'action appartient à `{0, 1, 2, 3}` ;
- sauvegarde et rechargement sans changement de prédiction déterministe ;
- run complet dont chaque action est obtenue auprès de l'API ;
- lecture des logs par le dashboard ;
- lancement local de la GUI et affichage d'une partie.

### Critères non fonctionnels

- temps de réponse d'inférence compatible avec une partie interactive locale ;
- chargement unique du modèle au démarrage de l'API ;
- messages d'erreur compréhensibles ;
- chemins configurables et absence de chemins personnels figés ;
- absence de secrets dans le dépôt et dans les logs ;
- dépendances et instructions d'installation explicites.

## 15. Livrables attendus

| Numéro indicatif | Livrable | Format | Contenu obligatoire |
|---:|---|---|---|
| 1 | Notebook mission | `.ipynb` ou lien Colab | Démarche complète, résultats, formatage API et conclusions |
| 2 | Vidéo | `.mp4` ou lien YouTube | 20–30 s, atterrissage réussi et récompense positive de victoire |
| 3 | API | `.py` et fichiers associés | État → action, validation, documentation, tests/exemples |
| 4 | Interface graphique | `.py` et fichiers associés | Visualisation d'une partie via l'API |
| 5 | Tableau de bord | `.py` ou lien Looker Studio | Métriques, courbes et interaction |

Les artefacts nécessaires au fonctionnement — meilleur modèle, configurations, fichiers de résultats et dépendances — doivent accompagner les livrables même s'ils ne constituent pas un livrable autonome.

## 16. Convention de remise

Le dossier ZIP suit la forme :

`Titre_du_projet_nom_prenom`

Les fichiers suivent la forme :

`Nom_Prenom_numéro_nom_du_livrable_mmaaaa`

Exemple fourni : `Janek_Meriem_1_Notebook_012025`.

La numérotation exacte doit être homogène pour les cinq livrables. La date correspond au mois et à l'année de démarrage du projet.

## 17. Éléments à savoir justifier lors du bilan

- le choix de DQN pour un espace d'actions discret ;
- le protocole de baseline, d'optimisation et d'évaluation ;
- la signification de la moyenne, de l'écart-type et du taux de réussite ;
- le choix et l'effet des hyperparamètres testés ;
- l'architecture et les responsabilités de l'API ;
- les conversions entre JSON, NumPy, tenseurs et action entière ;
- le choix du système de GUI et de dashboard ;
- le storytelling et le design des graphiques ;
- les difficultés rencontrées et les solutions retenues ;
- les points forts, limites et axes d'amélioration.

## 18. Définition globale de « terminé »

Le projet est considéré comme terminé lorsque toutes les conditions suivantes sont réunies :

- [ ] `LunarLander-v3` est exploré et documenté.
- [ ] La baseline DQN par défaut est entraînée et évaluée sur au moins 50 épisodes.
- [ ] Les expériences d'optimisation sont enregistrées, comparées et commentées.
- [ ] Le meilleur modèle est sauvegardé et rechargeable.
- [ ] Sa récompense moyenne dépasse 200 sur 100 épisodes d'évaluation.
- [ ] Les métriques de stabilité et de réussite sont présentées.
- [ ] Les formats de données destinés à l'API sont documentés.
- [ ] L'API accepte un état valide et renvoie une action valide.
- [ ] Un épisode complet peut utiliser l'API.
- [ ] Les tests ou exemples de l'API fonctionnent.
- [ ] La GUI affiche une partie jouée par l'agent en utilisant l'API.
- [ ] Le dashboard lit les logs, affiche les métriques et propose une interaction.
- [ ] La vidéo dure entre 20 et 30 secondes et montre une réussite.
- [ ] Les cinq livrables s'exécutent ou se consultent conformément aux instructions.
- [ ] Le notebook s'exécute de bout en bout et raconte la démarche complète.
- [ ] Les choix techniques sont préparés pour être justifiés au mentor.
- [ ] La fiche d'autoévaluation est complétée.
- [ ] Les fichiers sont nommés et regroupés selon la convention de remise.

