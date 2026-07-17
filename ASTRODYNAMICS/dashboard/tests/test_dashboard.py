"""Vérifie que les données attendues par le dashboard sont disponibles."""

from ASTRODYNAMICS.dashboard.app import (
    FINAL_EVALUATION_ID,
    load_episodes,
    load_learning_curves,
    load_registry,
    load_steps,
)


def test_dashboard_loads_final_metrics():
    registry = load_registry()
    final = registry[registry["evaluation_id"] == FINAL_EVALUATION_ID].iloc[0]

    assert final["mean_reward"] > 200
    assert final["n_episodes"] == 100
    assert final["success_rate"] >= 0.90


def test_dashboard_loads_episode_and_step_data():
    episodes = load_episodes(FINAL_EVALUATION_ID)
    steps = load_steps(FINAL_EVALUATION_ID)

    assert len(episodes) == 100
    assert episodes["episode_id"].nunique() == 100
    assert len(steps) > 10_000
    assert {"action", "reward", "next_state_1", "next_state_4"} <= set(steps.columns)


def test_dashboard_loads_learning_curves():
    curves = load_learning_curves()

    assert not curves.empty
    assert {"dqn_default", "ppo_default", "ppo_gamma_extended"} <= set(curves["experiment"])
    assert curves["timestep"].max() >= 400_000

