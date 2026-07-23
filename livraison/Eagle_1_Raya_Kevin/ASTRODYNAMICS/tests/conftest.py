"""Rend les modules Optuna importables par leur nom court.

`optuna_select` et `optuna_final` s'importent mutuellement par nom court
(`from optuna_search import ...`), comme lors d'un lancement en script ou depuis
le notebook. Ajouter le dossier ASTRODYNAMICS au chemin reproduit ce contexte
sans modifier le style d'import des modules de production.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
