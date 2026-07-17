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
- visualisation en direct de l'épisode ;
- affichage de l'action, de la récompense et de l'altitude ;
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

