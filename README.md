# Projet 11 — Pilote automatique Eagle-1

Le livrable de mission se trouve dans [`ASTRODYNAMICS/`](ASTRODYNAMICS/README.md).

La documentation complète de l'architecture et des flux est disponible dans
[`ASTRODYNAMICS/DOCUMENTATION_PERSONNELLE.html`](ASTRODYNAMICS/DOCUMENTATION_PERSONNELLE.html).

Résultat final sur `LunarLander-v3` : **227,39 ± 48,37** sur 100 épisodes,
avec **95 % d'atterrissages réussis**.

Installation :

```bash
uv sync --dev
```

## Démonstration complète

Une seule commande démarre l'API, la GUI et le dashboard, attend qu'ils soient
prêts puis ouvre les trois pages dans le navigateur :

```bash
uv run python launch_demo.py
```

Adresses utilisées par défaut :

- documentation API : `http://127.0.0.1:8000/docs` ;
- GUI de simulation : `http://127.0.0.1:8501` ;
- dashboard : `http://127.0.0.1:8502`.

`Ctrl+C` arrête proprement les trois services. Pour une machine sans navigateur :

```bash
uv run python launch_demo.py --no-browser
```

Les sorties de chaque démonstration sont enregistrées dans
`ASTRODYNAMICS/artifacts/demo_logs/`. Si un port est déjà occupé, le lanceur
sélectionne automatiquement un autre port libre et affiche l'URL correspondante.

Tests :

```bash
uv run pytest ASTRODYNAMICS/api/tests ASTRODYNAMICS/gui/tests ASTRODYNAMICS/dashboard/tests -q
```
