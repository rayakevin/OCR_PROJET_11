# GUI Eagle-1

Cette interface Streamlit affiche une partie `LunarLander-v3` pilotée par le
modèle servi dans l'API. Elle ne charge jamais Stable-Baselines3 : chaque action
est obtenue avec `POST /play`.

## Lancement

Dans un premier terminal :

```bash
uv run uvicorn ASTRODYNAMICS.api.app:app
```

Puis dans un second :

```bash
uv run streamlit run ASTRODYNAMICS/gui/app.py
```

L'interface est disponible par défaut sur `http://localhost:8501`.

## Fonctionnement

1. L'utilisateur choisit une seed et une durée d'affichage.
2. `play_episode()` exécute l'épisode et demande chaque action à l'API.
3. Les actions sont rejouées avec la même seed dans un environnement
   `rgb_array`. Ce rejeu permet de régler la durée d'affichage sans modifier la
   trajectoire.
4. Le résultat, la télémétrie et les actions sont affichés puis sauvegardés dans
   `ASTRODYNAMICS/artifacts/gui_runs/`.

L'écran utilise volontairement le rendu natif de Gymnasium. Les métriques
visibles pendant le rejeu sont le pas, l'action, la récompense cumulée et
l'altitude. Le bilan ajoute le nombre de pas et un proxy simple de carburant.

L'URL par défaut de l'API peut être remplacée avec `EAGLE1_API_URL`.

## Tests

```bash
uv run pytest ASTRODYNAMICS/gui/tests -q
```

Les tests jouent un épisode complet via le client FastAPI, vérifient le rejeu et
les fichiers écrits, puis chargent l'écran initial avec `streamlit.testing`.
