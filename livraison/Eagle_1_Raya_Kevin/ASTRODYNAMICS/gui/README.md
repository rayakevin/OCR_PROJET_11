# GUI Eagle-1

Cette interface Streamlit affiche une partie `LunarLander-v3` jouée par le
pilote automatique. Elle ne charge jamais le modèle : chaque action est obtenue
auprès de l'API FastAPI avec la route `/play`.

Streamlit a été retenu car il permet de combiner rendu d'images, télémétrie,
graphiques et contrôles interactifs dans une application Python locale. Les
composants pourront aussi être réutilisés pour le dashboard.

## Lancement

Ouvrir deux terminaux depuis la racine du projet.

Terminal 1 — API :

```bash
uv run uvicorn ASTRODYNAMICS.api.app:app
```

Terminal 2 — GUI :

```bash
uv run streamlit run ASTRODYNAMICS/gui/app.py
```

L'interface est disponible par défaut sur `http://localhost:8501`.

## Fonctionnalités

- contrôle de la disponibilité de l'API ;
- choix de la seed ;
- choix de la durée de visualisation, de 5 à 60 secondes (10 s par défaut) ;
- calcul des actions par l'API, puis rejeu déterministe du même vol ;
- affichage de l'action, de la récompense et de l'altitude ;
- bilan du vol et graphiques affichés **en haut de page**, pour être lus sans
  avoir à faire défiler la télémétrie ;
- résumé de la réussite, du nombre de pas et du proxy carburant ;
- graphiques de récompense et de distribution des actions ;
- sauvegarde automatique du résumé et de la télémétrie dans
  `ASTRODYNAMICS/artifacts/gui_runs/`.

## Test

```bash
uv run pytest ASTRODYNAMICS/gui/tests -q
```

Le test joue une réussite complète via l'API et contrôle les fichiers destinés
au dashboard.

## Durée de visualisation

La durée choisie ne modifie ni la politique ni la simulation. La GUI demande d'abord
une action à l'API pour chaque pas, enregistre ces actions, puis rejoue la même seed
avec les mêmes actions. Le rythme d'affichage est calculé automatiquement pour viser
la durée demandée sans exposer un réglage technique de FPS.

La cadence vise 24 images par seconde : sur un vol d'environ 240 pas rejoué en
10 secondes, la totalité des pas est affichée. À la cadence précédente de 10
images par seconde, plus de la moitié des pas étaient sautés et la descente
paraissait saccadée.

## Habillage de la visualisation

Le rendu de Gymnasium est volontairement minimal : ciel noir uni, sol blanc, et
aucune indication de la cible visée. `enhance_frame()` le retouche à la volée,
pour environ 6 ms par image :

- le ciel noir est remplacé par un dégradé sombre, ce qui donne de la profondeur
  sans toucher au sol ni au module — seuls les pixels quasi noirs sont modifiés ;
- l'aire d'atterrissage visée est matérialisée entre les deux drapeaux par une
  ligne claire surmontée d'un halo, ce qui rend la réussite ou l'échec lisible
  d'un coup d'œil.

Les coordonnées de l'aire sont **déduites des constantes de `lunar_lander`**, et
non codées en dur : si Gymnasium change sa géométrie, le repère suit.

Deux pistes ont été essayées puis écartées, mesures à l'appui :

- **augmenter la résolution native** en multipliant `VIEWPORT_W`, `VIEWPORT_H` et
  `SCALE` par un même facteur. La géométrie du monde est bien préservée, mais
  `SCALE` dimensionne aussi le module dans le monde physique : la trajectoire
  diverge (récompense −62 contre −210 sur une même séquence d'actions) ;
- **agrandir l'image côté Python** avant affichage. À 130 ms par image, cela
  plafonnait le rejeu à 7 images par seconde, soit l'inverse de l'effet
  recherché. Le navigateur agrandit très bien lui-même, et l'image est affichée
  dans une colonne centrale pour rester proche de sa résolution d'origine.
