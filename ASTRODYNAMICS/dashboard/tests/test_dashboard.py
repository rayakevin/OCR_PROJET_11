"""Vérifie que les données attendues par le dashboard sont disponibles."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from ASTRODYNAMICS.dashboard.app import (
    FINAL_EVALUATION_ID,
    build_reward_history,
    load_episodes,
    load_learning_curves,
    load_optuna_results,
    load_registry,
)


def test_dashboard_loads_final_metrics():
    registry = load_registry()
    final = registry[registry["evaluation_id"] == FINAL_EVALUATION_ID].iloc[0]

    assert final["mean_reward"] > 200
    assert final["n_episodes"] == 100
    assert final["success_rate"] >= 0.90


def test_dashboard_loads_episode_data():
    episodes = load_episodes(FINAL_EVALUATION_ID)

    assert len(episodes) == 100
    assert episodes["episode_id"].nunique() == 100
    assert {"action_0_count", "action_1_count", "action_2_count", "action_3_count"} <= set(
        episodes.columns
    )


def test_reward_history_contains_a_rolling_mean():
    history = build_reward_history(load_episodes(FINAL_EVALUATION_ID))

    assert list(history.columns) == ["Récompense", "Moyenne glissante"]
    assert history["Moyenne glissante"].notna().all()
    assert history.iloc[0]["Moyenne glissante"] == history.iloc[0]["Récompense"]


def test_dashboard_loads_learning_curves():
    curves = load_learning_curves()

    assert not curves.empty
    assert {"dqn_default", "ppo_default", "ppo_gamma_extended", "ppo_optuna"} <= set(curves["experiment"])
    assert curves["timestep"].max() >= 800_000


def test_dashboard_loads_optuna_results():
    results = load_optuna_results()
    trials = results["trials"]

    # La recherche note chaque essai sur plusieurs seeds et élague les moins bons :
    # les deux états doivent donc être présents et lisibles par le dashboard.
    assert len(trials.query("state == 'COMPLETE'")) >= 40
    assert len(trials.query("state == 'PRUNED'")) > 0
    assert "params_gamma" in trials.columns
    assert (trials["user_attrs_n_train_seeds"].dropna() >= 2).all()

    assert len(results["gamma"]) >= 5
    assert {"gamma", "mean", "std"} <= set(results["gamma"].columns)
    assert {"mean", "std", "min"} <= set(results["robustness_summary"].columns)
    assert set(results["importance"]["parameter"]) >= {"learning_rate", "ent_coef"}


def test_dashboard_renders_without_exception():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    rendered = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not rendered.exception
    assert [tab.label for tab in rendered.tabs] == [
        "Synthèse",
        "Apprentissage",
        "Épisodes",
        "Optuna",
    ]
