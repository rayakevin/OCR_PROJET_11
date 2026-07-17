# Démarche de réalisation — Pilote automatique Eagle-1

## 1. Objet du document

Ce document décrit l'ordre recommandé pour réaliser la mission Eagle-1, depuis la préparation de l'environnement jusqu'à la livraison. Il sert de feuille de route opérationnelle.

L'objectif est de produire un agent d'apprentissage par renforcement capable de réussir l'atterrissage dans `LunarLander-v3`, puis de rendre ses décisions et ses performances accessibles au moyen d'un notebook, d'une API, d'une interface graphique, d'un tableau de bord et d'une vidéo.

## 2. Principes directeurs

- Utiliser `LunarLander-v3` avec son espace d'actions discret.
- Choisir DQN comme algorithme principal et justifier ce choix.
- Construire rapidement une baseline fonctionnelle avant toute optimisation.
- Modifier autant que possible un seul hyperparamètre à la fois.
- Séparer strictement entraînement, évaluation et démonstration.
- Évaluer avec des seeds et des épisodes distincts de l'entraînement.
- Journaliser les expériences dès le premier run exploitable.
- Conserver les modèles et les résultats intermédiaires prometteurs.
- Placer la logique RL dans le backend/API, jamais dans la GUI.
- Produire des livrables exécutables localement sur une machine standard.

## 3. Phase 0 — Préparer et stabiliser le projet

### Actions

1. Vérifier la version de Python et les versions de Gymnasium, Stable-Baselines3, PyTorch, Box2D et des outils de visualisation.
2. Corriger le conflit PyTorch/Triton actuellement observé lors de la création de `SB3_DQN`.
3. Vérifier qu'un script minimal peut successivement :
   - créer `LunarLander-v3` ;
   - instancier un DQN Stable-Baselines3 ;
   - effectuer quelques pas d'entraînement ;
   - sauvegarder puis recharger le modèle.
4. Fixer des seeds pour Python, NumPy, PyTorch, Gymnasium et Stable-Baselines3.
5. Préparer une arborescence séparant au minimum : notebook, modèles, logs, données d'évaluation, vidéo, API, GUI, dashboard et tests.
6. Figer les dépendances et documenter la commande d'installation.

### Validation de phase

- Le noyau ne plante plus lors de l'instanciation ou de l'entraînement de DQN.
- Un test minimal d'entraînement, de sauvegarde et de rechargement fonctionne.
- L'environnement peut être reconstruit à partir des fichiers du projet.

## 4. Phase 1 — Explorer `LunarLander-v3`

### Actions

1. Présenter la mission et l'objectif RL dans le notebook.
2. Afficher les espaces d'observation et d'action.
3. Décrire les huit composantes d'un état :
   - position horizontale ;
   - position verticale ;
   - vitesse horizontale ;
   - vitesse verticale ;
   - angle ;
   - vitesse angulaire ;
   - contact du pied gauche ;
   - contact du pied droit.
4. Décrire les quatre actions discrètes : aucune action, moteur latéral gauche, moteur principal et moteur latéral droit.
5. Montrer et commenter les retours de `reset()` et de `step()` : état, récompense, `terminated`, `truncated` et `info`.
6. Examiner les bornes, types, formes et formats des données.
7. Jouer quelques épisodes aléatoires et observer les trajectoires, récompenses et conditions de fin.
8. Expliquer à haut niveau le calcul de la récompense : progression vers la zone cible, stabilité, contacts, réussite, crash et coût d'utilisation des moteurs.

### Validation de phase

- Le notebook explique clairement ce que l'agent voit, ce qu'il peut faire et ce que renvoie l'environnement.
- Les formats nécessaires à l'entraînement et à la future API sont identifiés.

## 5. Phase 2 — Concevoir la journalisation

La journalisation doit être conçue avant l'entraînement afin d'alimenter sans retraitement complexe le notebook, les comparaisons d'expériences et le dashboard.

### Niveau expérience

Pour chaque run, enregistrer au minimum :

- identifiant unique et nom de l'expérience ;
- date et durée ;
- algorithme et seed ;
- version de l'environnement et versions des bibliothèques ;
- totalité des hyperparamètres ;
- nombre de timesteps ;
- fréquence d'évaluation ;
- chemin du modèle et statut de meilleur modèle ;
- récompense moyenne, médiane et écart-type d'évaluation ;
- taux de réussite et nombre de crashs.

### Niveau épisode

Pour chaque épisode d'évaluation, enregistrer au minimum :

- identifiants de l'expérience et de l'épisode ;
- seed d'évaluation ;
- récompense cumulée ;
- nombre de pas ;
- résultat final : réussite, crash ou troncature ;
- position, vitesse, angle et contacts finaux ;
- nombre d'utilisations de chaque action ;
- consommation de carburant estimée à partir des actions moteurs ;
- durée de l'épisode.

### Niveau pas de temps

Pour les épisodes de démonstration ou d'analyse détaillée, enregistrer :

- numéro du pas ;
- huit composantes de l'état ;
- action choisie ;
- récompense immédiate et récompense cumulée ;
- valeurs Q si elles sont accessibles ;
- indicateurs `terminated` et `truncated`.

Le logging par pas n'a pas besoin d'être activé pour tous les pas d'entraînement : il peut être réservé aux évaluations afin de maîtriser l'espace disque et le temps d'exécution.

### Formats

- TensorBoard pour le suivi de l'apprentissage.
- CSV ou Parquet pour les expériences, épisodes et trajectoires.
- JSON pour les configurations, résumés et échanges avec l'API.
- Noms de colonnes et unités stables pendant tout le projet.

### Validation de phase

- Un run court produit automatiquement les fichiers attendus.
- Les fichiers peuvent être relus sans dépendre du notebook.
- Les colonnes nécessaires aux indicateurs du dashboard sont présentes.

## 6. Phase 3 — Établir les performances de référence

### 6.1 Agent aléatoire

1. Évaluer un agent aléatoire afin de disposer d'un point de comparaison simple.
2. Consigner récompense moyenne, écart-type, longueur moyenne et taux de réussite.

### 6.2 Baseline DQN

1. Justifier DQN par la nature discrète de l'espace d'actions.
2. Instancier `stable_baselines3.DQN` avec `MlpPolicy` et les paramètres par défaut.
3. Entraîner sans chercher immédiatement la meilleure performance.
4. Suivre l'entraînement avec TensorBoard et les logs structurés.
5. Sauvegarder le modèle de référence.
6. L'évaluer avec `evaluate_policy` sur au moins 50 épisodes.
7. Documenter la moyenne, l'écart-type, la durée d'entraînement et le taux de réussite.

### Validation de phase

- Le pipeline complet entraînement → sauvegarde → chargement → évaluation fonctionne.
- La performance initiale sur au moins 50 épisodes est calculée et commentée.
- Les résultats de l'agent aléatoire et de la baseline sont comparables.

## 7. Phase 4 — Optimiser l'agent

### Méthode

1. Définir une expérience de contrôle correspondant à la baseline.
2. Choisir une métrique principale : récompense moyenne d'évaluation.
3. Suivre également l'écart-type, le taux de réussite, la longueur des épisodes et l'utilisation des moteurs.
4. Modifier un seul hyperparamètre à la fois lors des premières expériences afin d'en isoler l'effet.
5. Utiliser le même protocole d'évaluation pour toutes les expériences.
6. Sauvegarder automatiquement le meilleur modèle au moyen d'une callback d'évaluation.
7. Arrêter ou écarter les expériences manifestement non prometteuses selon une règle documentée.
8. Comparer les expériences dans TensorBoard et dans une table récapitulative du notebook.

### Hyperparamètres DQN à examiner

- `learning_rate` ;
- `gamma` ;
- `buffer_size` ;
- `learning_starts` ;
- `batch_size` ;
- `train_freq` ;
- `gradient_steps` ;
- `target_update_interval` ;
- `exploration_fraction` ;
- `exploration_final_eps` ;
- architecture du réseau si les réglages précédents ne suffisent pas.

`n_steps`, `gae_lambda` et `clip_range` concernent PPO et ne doivent pas être présentés comme des hyperparamètres DQN. PPO peut être étudié comme comparaison facultative, mais ne remplace pas la baseline DQN demandée.

### Validation de phase

- Chaque expérience a une configuration, des résultats et une conclusion.
- Le meilleur modèle est sauvegardé avec ses hyperparamètres et ses métadonnées.
- Le choix final est fondé sur plusieurs épisodes, pas sur une réussite isolée.

## 8. Phase 5 — Réaliser l'évaluation finale

### Protocole

1. Charger le meilleur modèle sauvegardé.
2. Utiliser une politique déterministe.
3. Évaluer exactement sur 100 épisodes indépendants.
4. Utiliser des seeds d'évaluation conservées dans les résultats.
5. Enregistrer les métriques par épisode et un résumé global.
6. Calculer au minimum : moyenne, écart-type, médiane, minimum, maximum, taux de réussite et longueur moyenne.
7. Vérifier que la récompense moyenne dépasse 200 points.
8. Analyser les échecs, la variabilité et les limites du modèle.

### Validation de phase

- Récompense moyenne strictement supérieure à 200 sur 100 épisodes.
- Performance suffisamment stable et non issue d'un épisode exceptionnel.
- Meilleur modèle reproductible, sauvegardé et rechargeable.

## 9. Phase 6 — Préparer le modèle pour l'exploitation

1. Geler le modèle final et sa configuration.
2. Définir le contrat de données : un état JSON contenant exactement huit nombres finis produit une action entière comprise entre 0 et 3.
3. Centraliser le chargement du modèle et la fonction de prédiction.
4. Définir clairement les conversions JSON → NumPy → entrée du modèle et sortie du modèle → entier JSON.
5. Prévoir des messages d'erreur explicites pour les états invalides.
6. Vérifier que le comportement de la fonction d'inférence est identique à celui utilisé pendant l'évaluation finale.

## 10. Phase 7 — Développer et tester l'API

### Actions

1. Utiliser FastAPI ou Flask ; FastAPI est recommandé pour la validation des schémas et la documentation automatique.
2. Créer un endpoint de santé.
3. Créer un endpoint acceptant un état et renvoyant une action.
4. Ajouter, si utile, un endpoint permettant de jouer ou de résumer un épisode complet.
5. Charger le modèle une seule fois au démarrage de l'API.
6. Documenter les formats d'entrée, de sortie et les erreurs possibles.
7. Ajouter des tests unitaires ou des exemples reproductibles.
8. Démontrer un run complet dans lequel les actions proviennent de l'API.

### Validation de phase

- L'API respecte le flux RL état → action.
- Elle accepte et valide le format LunarLander-v3.
- Un épisode complet peut être piloté par des appels à l'API.
- Les tests ou exemples d'utilisation réussissent.

## 11. Phase 8 — Construire la GUI et le dashboard

### GUI

1. Choisir Streamlit ou Gradio et justifier ce choix.
2. Afficher une partie jouée par l'agent.
3. Obtenir les décisions auprès de l'API.
4. Afficher au minimum l'image de l'environnement, l'action, la récompense et l'état de l'épisode.
5. Ne dupliquer aucune logique de décision RL dans l'interface.

### Dashboard

1. Lire les journaux structurés produits pendant les entraînements et évaluations.
2. Construire un récit visuel : progression de la baseline vers le modèle final, stabilité, réussites et compromis avec la consommation.
3. Afficher au minimum :
   - courbes de récompense ;
   - moyenne glissante et écart-type ;
   - récompense moyenne et écart-type sur 100 épisodes ;
   - réussites contre échecs ;
   - nombre de pas avant atterrissage ;
   - distribution des actions ou utilisation des moteurs ;
   - comparaison des expériences et hyperparamètres.
4. Ajouter au moins un filtre ou un graphique dynamique, par exemple par expérience, seed, résultat ou plage de récompense.
5. Justifier les choix de design et la hiérarchie des informations.

### Validation de phase

- La GUI visualise effectivement une partie complète.
- Le dashboard contient des métriques pertinentes et une interaction.
- Le parcours visuel permet de comprendre pourquoi le modèle final a été retenu.

## 12. Phase 9 — Produire la vidéo

1. Utiliser `gymnasium.wrappers.RecordVideo` ou un mécanisme équivalent.
2. Parcourir des seeds d'évaluation jusqu'à identifier une réussite représentative, sans modifier le modèle.
3. Enregistrer un atterrissage réussi avec récompense de victoire.
4. Monter ou sélectionner une séquence finale de 20 à 30 secondes.
5. Vérifier le format MP4, la lisibilité et la durée.

## 13. Phase 10 — Finaliser le notebook et les livrables

### Notebook final

Le notebook doit présenter une narration complète :

1. contexte et objectif ;
2. installation, imports et reproductibilité ;
3. exploration de l'environnement ;
4. formats de données ;
5. agent aléatoire ;
6. justification de DQN ;
7. baseline par défaut et évaluation sur au moins 50 épisodes ;
8. système de journalisation ;
9. protocole et résultats d'optimisation ;
10. évaluation finale sur 100 épisodes ;
11. analyse des résultats et limites ;
12. sauvegarde et utilisation du modèle ;
13. formatage destiné à l'API ;
14. instructions pour exécuter API, GUI et dashboard.

### Contrôle final

1. Exécuter le notebook de bout en bout dans un environnement propre.
2. Exécuter les tests de l'API.
3. Lancer localement l'API, la GUI et le dashboard.
4. Vérifier la vidéo et le modèle sauvegardé.
5. Compléter la fiche d'autoévaluation.
6. Préparer les justifications demandées pour la session avec le mentor.
7. Nommer les livrables selon la convention demandée.
8. Regrouper uniquement les livrables pertinents dans le ZIP final.

## 14. Préparation du bilan avec le mentor

Préparer une synthèse couvrant :

- choix de DQN et protocole expérimental ;
- architecture et flux de données de l'API ;
- formatage des états, actions et récompenses ;
- choix de la GUI et du dashboard ;
- storytelling et design des visualisations ;
- difficultés rencontrées, notamment l'environnement logiciel ;
- points forts du projet ;
- limites du modèle et pistes d'amélioration ;
- connaissances à approfondir après le projet.

