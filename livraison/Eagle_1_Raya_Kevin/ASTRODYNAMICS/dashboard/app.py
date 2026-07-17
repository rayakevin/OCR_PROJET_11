"""Dashboard Streamlit alimenté par les logs réels de la mission Eagle-1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
FINAL_EVALUATION_ID = "ppo_gamma_extended_final_100"


def experiment_phase(evaluation_id: str) -> str:
    """Regroupe les nombreuses mesures dans des étapes lisibles."""
    if evaluation_id == "random_baseline" or "baseline" in evaluation_id:
        return "Baseline"
    if "screen" in evaluation_id:
        return "Screening"
    if "selection" in evaluation_id:
        return "Sélection"
    if "tuning" in evaluation_id:
        return "Optimisation"
    if "final" in evaluation_id:
        return "Finale"
    return "Autre"


def load_registry(artifacts_dir: Path = ARTIFACTS_DIR) -> pd.DataFrame:
    path = Path(artifacts_dir) / "experiment_registry.csv"
    registry = pd.read_csv(path)
    registry["phase"] = registry["evaluation_id"].map(experiment_phase)
    registry["success_percent"] = 100 * registry["success_rate"]
    return registry


def list_episode_evaluations(artifacts_dir: Path = ARTIFACTS_DIR) -> list[str]:
    evaluation_dir = Path(artifacts_dir) / "evaluations"
    return sorted(path.parent.name for path in evaluation_dir.glob("*/episodes.csv"))


def load_episodes(evaluation_id: str, artifacts_dir: Path = ARTIFACTS_DIR) -> pd.DataFrame:
    path = Path(artifacts_dir) / "evaluations" / evaluation_id / "episodes.csv"
    return pd.read_csv(path)


def load_steps(evaluation_id: str, artifacts_dir: Path = ARTIFACTS_DIR) -> pd.DataFrame:
    path = Path(artifacts_dir) / "evaluations" / evaluation_id / "steps.csv"
    return pd.read_csv(path)


def load_learning_curves(artifacts_dir: Path = ARTIFACTS_DIR) -> pd.DataFrame:
    """Fusionne les fichiers EvalCallback dans une seule table."""
    rows = []
    pattern = Path(artifacts_dir).glob("evaluations/*/during_training/evaluations.npz")
    for path in pattern:
        data = np.load(path)
        means = data["results"].mean(axis=1)
        stds = data["results"].std(axis=1)
        for timestep, mean, std in zip(data["timesteps"], means, stds):
            rows.append({
                "experiment": path.parents[1].name,
                "timestep": int(timestep),
                "mean_reward": float(mean),
                "std_reward": float(std),
            })
    return pd.DataFrame(rows)


def load_gui_runs(artifacts_dir: Path = ARTIFACTS_DIR) -> pd.DataFrame:
    path = Path(artifacts_dir) / "gui_runs" / "runs.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def render_overview(registry: pd.DataFrame) -> None:
    st.subheader("De la baseline au pilote final")
    st.write(
        "La lecture se fait de gauche à droite : politiques de référence, essais "
        "contrôlés, sélection sur seeds communes, puis validation sur 100 épisodes."
    )

    fig = px.bar(
        registry.sort_values("mean_reward"),
        x="mean_reward",
        y="evaluation_id",
        color="algorithm",
        error_x="std_reward",
        orientation="h",
        labels={"mean_reward": "Récompense moyenne", "evaluation_id": "Évaluation"},
        height=max(450, 34 * len(registry)),
    )
    fig.add_vline(x=200, line_dash="dash", line_color="#2ca02c", annotation_text="objectif 200")
    st.plotly_chart(fig, use_container_width=True)

    fig_tradeoff = px.scatter(
        registry,
        x="mean_fuel_proxy",
        y="mean_reward",
        color="success_percent",
        size="n_episodes",
        hover_name="evaluation_id",
        labels={
            "mean_fuel_proxy": "Proxy carburant moyen",
            "mean_reward": "Récompense moyenne",
            "success_percent": "Réussite (%)",
        },
        title="Compromis performance, réussite et utilisation des moteurs",
    )
    fig_tradeoff.add_hline(y=200, line_dash="dash", line_color="#2ca02c")
    st.plotly_chart(fig_tradeoff, use_container_width=True)


def render_learning(artifacts_dir: Path) -> None:
    curves = load_learning_curves(artifacts_dir)
    if curves.empty:
        st.warning("Aucune courbe EvalCallback disponible.")
        return

    experiments = st.multiselect(
        "Expériences à comparer",
        sorted(curves["experiment"].unique()),
        default=[name for name in ["dqn_default", "ppo_default", "ppo_gamma_extended"]
                 if name in set(curves["experiment"])],
    )
    selected = curves[curves["experiment"].isin(experiments)]
    fig = px.line(
        selected,
        x="timestep",
        y="mean_reward",
        color="experiment",
        markers=True,
        labels={"timestep": "Transitions", "mean_reward": "Récompense moyenne"},
        title="Progression mesurée pendant l'entraînement",
    )
    fig.add_hline(y=200, line_dash="dash", line_color="#2ca02c")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(selected.sort_values(["experiment", "timestep"]), use_container_width=True)


def render_episodes(artifacts_dir: Path) -> None:
    evaluations = list_episode_evaluations(artifacts_dir)
    default_index = evaluations.index(FINAL_EVALUATION_ID) if FINAL_EVALUATION_ID in evaluations else 0
    evaluation_id = st.selectbox("Évaluation", evaluations, index=default_index)
    episodes = load_episodes(evaluation_id, artifacts_dir)

    outcomes = st.multiselect(
        "Issue",
        sorted(episodes["outcome"].unique()),
        default=sorted(episodes["outcome"].unique()),
    )
    minimum, maximum = float(episodes["total_reward"].min()), float(episodes["total_reward"].max())
    reward_range = st.slider(
        "Plage de récompense",
        min_value=minimum,
        max_value=maximum,
        value=(minimum, maximum),
    )
    filtered = episodes[
        episodes["outcome"].isin(outcomes)
        & episodes["total_reward"].between(*reward_range)
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("Épisodes filtrés", len(filtered))
    col2.metric("Score moyen", f"{filtered['total_reward'].mean():.1f}" if len(filtered) else "—")
    col3.metric("Réussite", f"{100 * filtered['success'].mean():.1f}%" if len(filtered) else "—")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(filtered, x="total_reward", color="outcome", nbins=20,
                           title="Distribution des récompenses")
        fig.add_vline(x=200, line_dash="dash", line_color="#2ca02c")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(
            filtered,
            x="fuel_proxy",
            y="total_reward",
            color="outcome",
            hover_data=["episode_id", "seed", "episode_length"],
            title="Score et utilisation des moteurs",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(filtered, use_container_width=True)


def render_trajectory(artifacts_dir: Path) -> None:
    available = [
        evaluation_id
        for evaluation_id in list_episode_evaluations(artifacts_dir)
        if (Path(artifacts_dir) / "evaluations" / evaluation_id / "steps.csv").exists()
    ]
    if not available:
        st.warning("Aucune trajectoire détaillée disponible.")
        return

    evaluation_id = st.selectbox("Jeu de trajectoires", available)
    steps = load_steps(evaluation_id, artifacts_dir)
    episode_id = st.selectbox("Épisode", sorted(steps["episode_id"].unique()))
    episode = steps[steps["episode_id"] == episode_id]

    fig = go.Figure()
    for column, label in [("next_state_1", "Altitude"), ("next_state_3", "Vitesse verticale"),
                          ("next_state_4", "Angle")]:
        fig.add_trace(go.Scatter(x=episode["step"], y=episode[column], mode="lines", name=label))
    fig.update_layout(title="Télémétrie normalisée", xaxis_title="Pas", yaxis_title="Valeur")
    st.plotly_chart(fig, use_container_width=True)

    action_counts = episode["action_label"].value_counts().rename_axis("action").reset_index(name="count")
    st.plotly_chart(px.bar(action_counts, x="action", y="count", title="Décisions de l'agent"),
                    use_container_width=True)
    st.dataframe(episode, use_container_width=True)


def render_gui_runs(artifacts_dir: Path) -> None:
    runs = load_gui_runs(artifacts_dir)
    if runs.empty:
        st.info("Aucun run GUI enregistré pour l'instant. Lancez un épisode depuis la GUI.")
        return
    st.dataframe(runs.sort_values("run_id", ascending=False), use_container_width=True)
    fig = px.bar(runs, x="run_id", y="total_reward", color="outcome",
                 title="Performances observées depuis la GUI")
    fig.add_hline(y=200, line_dash="dash", line_color="#2ca02c")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Dashboard Eagle-1", page_icon="📊", layout="wide")
    st.title("📊 Eagle-1 — Tableau de bord de mission")
    st.caption("Suivi interactif des entraînements, évaluations, trajectoires et simulations GUI.")

    registry = load_registry(ARTIFACTS_DIR)
    final_row = registry[registry["evaluation_id"] == FINAL_EVALUATION_ID].iloc[0]

    cols = st.columns(5)
    cols[0].metric("Score final", f"{final_row['mean_reward']:.1f}", delta=f"objectif +{final_row['mean_reward'] - 200:.1f}")
    cols[1].metric("Écart-type", f"{final_row['std_reward']:.1f}")
    cols[2].metric("Réussite", f"{final_row['success_percent']:.0f}%")
    cols[3].metric("Pas moyens", f"{final_row['mean_episode_length']:.0f}")
    cols[4].metric("Proxy carburant", f"{final_row['mean_fuel_proxy']:.1f}")

    with st.sidebar:
        st.header("Filtres globaux")
        algorithms = st.multiselect(
            "Algorithme",
            sorted(registry["algorithm"].unique()),
            default=sorted(registry["algorithm"].unique()),
        )
        phases = st.multiselect(
            "Phase",
            sorted(registry["phase"].unique()),
            default=sorted(registry["phase"].unique()),
        )
    filtered_registry = registry[
        registry["algorithm"].isin(algorithms) & registry["phase"].isin(phases)
    ]

    overview, learning, episodes, trajectory, gui_runs = st.tabs([
        "Vue mission", "Apprentissage", "Épisodes", "Trajectoire", "Runs GUI"
    ])
    with overview:
        render_overview(filtered_registry)
    with learning:
        render_learning(ARTIFACTS_DIR)
    with episodes:
        render_episodes(ARTIFACTS_DIR)
    with trajectory:
        render_trajectory(ARTIFACTS_DIR)
    with gui_runs:
        render_gui_runs(ARTIFACTS_DIR)


if __name__ == "__main__":
    main()

