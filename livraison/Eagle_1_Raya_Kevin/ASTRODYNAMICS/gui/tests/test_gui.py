"""Teste la logique de simulation utilisée par Streamlit."""

import pandas as pd
from fastapi.testclient import TestClient

from ASTRODYNAMICS.api.app import app
from ASTRODYNAMICS.gui.app import check_api, play_episode, save_run


def test_gui_plays_and_saves_a_successful_episode(tmp_path):
    with TestClient(app) as client:
        health = check_api("", client=client)
        result, steps = play_episode("", seed=10_000, client=client)

    assert health["model_loaded"] is True
    assert result.success is True
    assert result.total_reward > 200
    assert len(steps) == result.steps
    assert {"action", "cumulative_reward", "x", "y", "angle"} <= set(steps.columns)

    run_dir = save_run(result, steps, output_dir=tmp_path)
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "steps.csv").exists()

    registry = pd.read_csv(tmp_path / "runs.csv")
    assert len(registry) == 1
    assert bool(registry.iloc[0]["success"])

