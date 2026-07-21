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
  "action": 2,
  "action_label": "moteur principal",
  "model_id": "ppo_optuna"
}
```

## Routes

- `GET /health` : état du service et du modèle ;
- `POST /predict` : route principale `état → action` ;
- `POST /play` : alias destiné à une boucle de simulation.

Le chemin du modèle peut être remplacé avec la variable d'environnement
`EAGLE1_MODEL_PATH`.

## Tests

```bash
uv run pytest ASTRODYNAMICS/api/tests -q
```

Les tests couvrent la santé du service, les formats valides et invalides, ainsi
qu'un épisode complet réussi dont chaque action est obtenue via l'API.
