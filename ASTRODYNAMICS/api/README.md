# API Eagle-1

Cette API charge une seule fois le meilleur modèle PPO puis transforme un état
`LunarLander-v3` de huit valeurs en une action discrète comprise entre 0 et 3.

## Lancement

Depuis la racine du projet :

```bash
uv run uvicorn ASTRODYNAMICS.api.app:app --reload
```

La documentation interactive est alors disponible sur
`http://127.0.0.1:8000/docs`.

## Exemple

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"state":[0.0,1.0,0.0,-0.1,0.0,0.0,0.0,0.0]}'
```

Réponse :

```json
{
  "action": 0,
  "action_label": "ne rien faire",
  "model_id": "ppo_optuna"
}
```

## Corps de requête

| Champ | Type | Défaut | Rôle |
|---|---|---|---|
| `state` | liste de 8 nombres | — | Observation `LunarLander-v3`, dans l'ordre `x`, `y`, `vx`, `vy`, angle, vitesse angulaire, contact gauche, contact droit. |
| `deterministic` | booléen | `true` | `true` : action la plus probable selon la politique. `false` : action échantillonnée, donc variable d'un appel à l'autre. |

L'état est refusé avec un code `422` si sa longueur n'est pas exactement 8 ou
s'il contient une valeur non finie (`NaN`, `inf`).

## Routes

- `GET /health` : état du service et du modèle ;
- `POST /predict` : route principale `état → action` ;
- `POST /play` : alias destiné à une boucle de simulation.

Le chemin du modèle peut être remplacé avec la variable d'environnement
`EAGLE1_MODEL_PATH`. Le champ `model_id` de la réponse reste l'identifiant
nominal de la mission : il n'est pas relu depuis le fichier chargé.

## Tests

```bash
uv run pytest ASTRODYNAMICS/api/tests -q
```

Les tests couvrent la santé du service, les formats valides et invalides, ainsi
qu'un épisode complet réussi dont chaque action est obtenue via l'API.
