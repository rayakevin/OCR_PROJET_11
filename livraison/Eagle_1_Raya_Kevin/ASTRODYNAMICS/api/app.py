"""API FastAPI exposant la politique PPO entraînée pour LunarLander-v3."""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

# Dans l'environnement local du projet, Triton doit être chargé avant
# Box2D/pygame. L'import reste facultatif pour assurer la portabilité CPU.
try:
    import triton  # noqa: F401
except ImportError:
    pass

import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator
from stable_baselines3 import PPO


ENV_ID = "LunarLander-v3"
MODEL_ID = "ppo_optuna"
ACTION_LABELS = {
    0: "ne rien faire",
    1: "moteur d'orientation gauche",
    2: "moteur principal",
    3: "moteur d'orientation droit",
}

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "models"
    / MODEL_ID
    / "best_model.zip"
)


class StateRequest(BaseModel):
    """État LunarLander-v3 reçu par l'API."""

    state: Annotated[list[float], Field(min_length=8, max_length=8)]
    deterministic: bool = True

    @field_validator("state")
    @classmethod
    def state_must_be_finite(cls, state: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in state):
            raise ValueError("Toutes les composantes de l'état doivent être finies.")
        return state


class ActionResponse(BaseModel):
    """Décision renvoyée au client."""

    action: int = Field(ge=0, le=3)
    action_label: str
    model_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str
    environment: str


def get_model_path() -> Path:
    """Autorise le remplacement du modèle par la variable EAGLE1_MODEL_PATH."""
    return Path(os.getenv("EAGLE1_MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser().resolve()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le réseau une seule fois au démarrage du serveur."""
    model_path = get_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle Eagle-1 introuvable : {model_path}")

    app.state.model = PPO.load(model_path, device="cpu")
    app.state.model_path = model_path
    yield
    app.state.model = None


app = FastAPI(
    title="Eagle-1 Autopilot API",
    description="Transforme un état LunarLander-v3 en action du pilote PPO.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> HealthResponse:
    """Vérifie que l'API et le modèle sont prêts."""
    return HealthResponse(
        status="ok",
        model_loaded=request.app.state.model is not None,
        model_id=MODEL_ID,
        environment=ENV_ID,
    )


def predict_action(payload: StateRequest, request: Request) -> ActionResponse:
    """Convertit le JSON en float32, appelle PPO et formate l'action."""
    observation = np.asarray(payload.state, dtype=np.float32)
    action, _ = request.app.state.model.predict(
        observation,
        deterministic=payload.deterministic,
    )
    action_id = int(np.asarray(action).item())
    return ActionResponse(
        action=action_id,
        action_label=ACTION_LABELS[action_id],
        model_id=MODEL_ID,
    )


@app.post("/predict", response_model=ActionResponse, tags=["agent"])
def predict(payload: StateRequest, request: Request) -> ActionResponse:
    """Route principale respectant le flux RL état → action."""
    return predict_action(payload, request)


@app.post("/play", response_model=ActionResponse, tags=["agent"])
def play(payload: StateRequest, request: Request) -> ActionResponse:
    """Alias explicite de /predict destiné aux clients de simulation."""
    return predict_action(payload, request)
