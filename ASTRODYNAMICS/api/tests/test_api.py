"""Tests unitaires et test d'intégration de l'API Eagle-1."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import gymnasium as gym
from stable_baselines3 import PPO

from ASTRODYNAMICS.api.app import DEFAULT_MODEL_PATH, ENV_ID, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_id": "ppo_optuna",
        "environment": "LunarLander-v3",
    }


def test_predict_returns_a_valid_action(client):
    response = client.post(
        "/predict",
        json={"state": [0.0, 1.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["action"] in {0, 1, 2, 3}
    assert isinstance(body["action_label"], str)
    assert body["model_id"] == "ppo_optuna"


@pytest.mark.parametrize(
    "invalid_state",
    [
        [0.0] * 7,
        [0.0] * 9,
        [0.0, 1.0, "invalide", 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
)
def test_predict_rejects_invalid_states(client, invalid_state):
    response = client.post("/predict", json={"state": invalid_state})
    assert response.status_code == 422


@pytest.mark.parametrize("invalid_number", ["NaN", "Infinity", "-Infinity"])
def test_predict_rejects_non_finite_numbers(client, invalid_number):
    payload = (
        '{"state":[0.0,1.0,'
        + invalid_number
        + ',0.0,0.0,0.0,0.0,0.0]}'
    )
    response = client.post(
        "/predict",
        content=payload,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_predict_rejects_a_missing_state(client):
    assert client.post("/predict", json={}).status_code == 422


def test_saved_model_keeps_the_same_deterministic_prediction(client):
    state = np.array([0.0, 1.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    api_action = client.post("/predict", json={"state": state.tolist()}).json()["action"]
    reloaded = PPO.load(DEFAULT_MODEL_PATH, device="cpu")
    reloaded_action, _ = reloaded.predict(state, deterministic=True)

    assert int(np.asarray(reloaded_action).item()) == api_action


def test_full_successful_episode_is_driven_by_the_api(client):
    """Chaque décision d'un épisode réel passe par HTTP, jamais par le modèle local."""
    env = gym.make(ENV_ID)
    observation, _ = env.reset(seed=10_000)
    terminated = truncated = False
    total_reward = 0.0
    final_reward = 0.0
    steps = 0

    while not (terminated or truncated):
        response = client.post(
            "/play",
            json={"state": observation.tolist(), "deterministic": True},
        )
        assert response.status_code == 200

        action = response.json()["action"]
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        final_reward = float(reward)
        steps += 1

    env.close()

    assert np.isclose(final_reward, 100.0)  # atterrissage, et non crash
    assert total_reward > 200.0
    assert steps < 1_000
