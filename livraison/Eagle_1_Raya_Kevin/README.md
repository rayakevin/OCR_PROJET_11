# Projet 11 — Pilote automatique Eagle-1

Le livrable de mission se trouve dans [`ASTRODYNAMICS/`](ASTRODYNAMICS/README.md).

Résultat final sur `LunarLander-v3` : **227,39 ± 48,37** sur 100 épisodes,
avec **95 % d'atterrissages réussis**.

Installation :

```bash
uv sync --dev
```

Tests :

```bash
uv run pytest ASTRODYNAMICS/api/tests ASTRODYNAMICS/gui/tests ASTRODYNAMICS/dashboard/tests -q
```
