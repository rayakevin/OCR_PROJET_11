"""Dashboard Streamlit alimenté par les logs réels de la mission Eagle-1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
OPTUNA_DIR = ARTIFACTS_DIR / "optuna" / "ppo_lunarlander"
FINAL_EVALUATION_ID = "ppo_optuna_final_100"
BRAND_COLORS = ["#176780", "#c26a2e", "#667985", "#2f765f", "#815d78"]

APP_STYLES = """
<style>
    :root {
        --ad-ink: #17242d;
        --ad-navy: #12384a;
        --ad-blue: #176780;
        --ad-orange: #c26a2e;
        --ad-line: #cbd5da;
        --ad-surface: #f4f6f7;
    }
    .stApp { background: var(--ad-surface); color: var(--ad-ink); }
    [data-testid="stHeader"] { background: rgba(244, 246, 247, 0.96); }
    [data-testid="stSidebar"] { background: #e8edef; border-right: 1px solid var(--ad-line); }
    .block-container { max-width: 1480px; padding-top: 2rem; padding-bottom: 3rem; }
    .mission-header { border-top: 4px solid var(--ad-orange); border-bottom: 1px solid var(--ad-line); padding: 1rem 0 1.1rem; margin-bottom: 1.4rem; }
    .mission-header .kicker { color: var(--ad-blue); font-size: .74rem; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; }
    .mission-header h1 { color: var(--ad-navy); font-size: 2.1rem; font-weight: 650; letter-spacing: -.02em; margin: .25rem 0; }
    .mission-header p { color: #52636d; margin: 0; font-size: .96rem; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid var(--ad-line); border-radius: 2px; padding: .75rem .9rem; }
    [data-testid="stMetricLabel"] { color: #5b6a73; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--ad-line); }
    .stTabs [data-baseweb="tab"] { border-radius: 0; padding: .65rem 1rem; }
    .stTabs [aria-selected="true"] { color: var(--ad-navy); border-bottom: 3px solid var(--ad-orange); }
    h2, h3 { color: var(--ad-navy); font-weight: 650; }
</style>
"""


def render_header() -> None:
    """Affiche le mot-symbole et le contexte de mission."""
    st.markdown(APP_STYLES, unsafe_allow_html=True)
    st.markdown(
        """
        <header class="mission-header">
            <div class="kicker">AstroDynamics / Mission Analytics</div>
            <h1>Eagle-1 — Analyse des performances</h1>
            <p>Entraînements, évaluations, trajectoires et simulations opérateur</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig, *, height: int | None = None):
    """Applique une charte graphique commune aux figures Plotly."""
    title_text = fig.layout.title.text
    has_title = bool(title_text and str(title_text).strip())
    fig.update_layout(
        template="plotly_white",
        colorway=BRAND_COLORS,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "color": "#263843", "size": 13},
        legend_title_text="",
        margin={"l": 40, "r": 24, "t": 62 if has_title else 24, "b": 40},
        hoverlabel={"bgcolor": "#ffffff", "font_color": "#17242d"},
    )
    if has_title:
        fig.update_layout(
            title={
                "text": title_text,
                "font": {"color": "#12384a", "size": 18},
                "x": 0.01,
                "xanchor": "left",
            }
        )
    else:
        # Plotly peut afficher le mot « undefined » si le champ titre est absent.
        fig.update_layout(title={"text": ""})
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="#e2e8eb", zerolinecolor="#cbd5da")
    fig.update_yaxes(gridcolor="#e2e8eb", zerolinecolor="#cbd5da")
    return fig


def experiment_phase(evaluation_id: str) -> str:
    """Regroupe les nombreuses mesures dans des étapes lisibles.

    L'ordre des tests compte : un identifiant peut cumuler plusieurs mots-clés
    (par exemple `ppo_optuna_gamma_t003_s242_robustness`), et c'est le premier
    test satisfait qui décide de la phase.
    """
    if "baseline" in evaluation_id:
        return "Baseline"
    if "robustness" in evaluation_id:
        return "Robustesse"
    if "optuna_trial" in evaluation_id or "gamma_focus" in evaluation_id:
        return "Recherche Optuna"
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


def load_optuna_results(artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, pd.DataFrame]:
    """Charge les tables produites par la campagne Optuna.

    `trials` reçoit une colonne `params_gamma` dérivée : la recherche
    échantillonne `1 - gamma` en loi log-uniforme, mais c'est gamma lui-même
    qu'on veut lire sur les graphiques.
    """
    optuna_dir = (
        OPTUNA_DIR
        if Path(artifacts_dir) == ARTIFACTS_DIR
        else Path(artifacts_dir) / "optuna" / "ppo_lunarlander"
    )
    paths = {
        "trials": optuna_dir / "trials.csv",
        "gamma": optuna_dir / "gamma_focus" / "trials.csv",
        "robustness": optuna_dir / "robustness.csv",
        "robustness_summary": optuna_dir / "robustness_summary.csv",
        "importance": optuna_dir / "parameter_importance.csv",
    }
    tables = {
        name: pd.read_csv(path) if path.exists() else pd.DataFrame()
        for name, path in paths.items()
    }
    trials = tables["trials"]
    if not trials.empty and "params_one_minus_gamma" in trials:
        trials["params_gamma"] = 1.0 - trials["params_one_minus_gamma"]
    return tables


def load_selected_config(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    """Lit le réglage retenu à l'issue de la validation multi-seed."""
    path = Path(artifacts_dir) / "optuna" / "ppo_lunarlander" / "selected_config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def render_overview(registry: pd.DataFrame) -> None:
    st.subheader("De la baseline au pilote final")
    st.write(
        "Les évaluations sont classées par récompense moyenne croissante : les "
        "politiques de référence se trouvent en bas, le pilote final en haut. "
        "Le filtre « Phase » isole une étape précise de la mission."
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
        color_discrete_sequence=BRAND_COLORS,
    )
    fig.add_vline(x=200, line_dash="dash", line_color="#2ca02c", annotation_text="objectif 200")
    st.plotly_chart(style_figure(fig), width="stretch")

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
        color_continuous_scale=[[0, "#d9e2e6"], [0.5, "#4f91a6"], [1, "#12384a"]],
    )
    fig_tradeoff.add_hline(y=200, line_dash="dash", line_color="#2ca02c")
    st.plotly_chart(style_figure(fig_tradeoff), width="stretch")


def render_learning(artifacts_dir: Path) -> None:
    curves = load_learning_curves(artifacts_dir)
    if curves.empty:
        st.warning("Aucune courbe EvalCallback disponible.")
        return

    experiments = st.multiselect(
        "Expériences à comparer",
        sorted(curves["experiment"].unique()),
        default=[name for name in ["dqn_default", "ppo_default", "ppo_gamma_extended", "ppo_optuna"]
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
        color_discrete_sequence=BRAND_COLORS,
    )
    fig.add_hline(y=200, line_dash="dash", line_color="#2ca02c")
    st.plotly_chart(style_figure(fig), width="stretch")
    st.dataframe(selected.sort_values(["experiment", "timestep"]), width="stretch")


def render_optuna(artifacts_dir: Path) -> None:
    """Raconte la recherche large, le raffinement de gamma et la sélection robuste."""
    results = load_optuna_results(artifacts_dir)
    trials = results["trials"]
    gamma = results["gamma"]
    robustness = results["robustness"]
    robust_summary = results["robustness_summary"]
    importance = results["importance"]
    selected = load_selected_config(artifacts_dir)
    if trials.empty or gamma.empty or robustness.empty:
        st.warning("Les artefacts Optuna ne sont pas encore disponibles.")
        return

    complete = trials[trials["state"] == "COMPLETE"].copy()
    pruned = trials[trials["state"] == "PRUNED"]

    st.subheader("Recherche reproductible des hyperparamètres PPO")
    st.write(
        "Une recherche TPE explore dix paramètres conjointement. Chaque essai est "
        "entraîné sur deux seeds et noté par leur moyenne : un score obtenu sur une "
        "seule seed mesurerait autant le tirage d'initialisation que le réglage. "
        "Une grille affine ensuite gamma autour de l'optimum mesuré, puis les "
        "finalistes sont réentraînés sur cinq seeds neuves."
    )

    cols = st.columns(4)
    cols[0].metric("Essais aboutis", len(complete))
    cols[1].metric("Essais élagués", len(pruned), help="Interrompus par le MedianPruner.")
    if selected:
        cols[2].metric("Gamma retenu", f"{selected['parameters']['gamma']:.5f}")
        cols[3].metric(
            "Moyenne sur 5 seeds",
            f"{selected['robust_mean_reward']:.1f}",
            delta=f"pire seed {selected['robust_min_reward']:.0f}",
        )

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            complete,
            x="number",
            y="value",
            color="params_gamma",
            hover_data=["params_learning_rate", "params_ent_coef", "user_attrs_score_std"],
            labels={
                "number": "Essai",
                "value": "Moyenne sur 2 seeds",
                "params_gamma": "Gamma",
            },
            title=f"Recherche large TPE — {len(complete)} essais aboutis",
            color_continuous_scale=[[0, "#d9e2e6"], [1, "#12384a"]],
        )
        fig.add_hline(y=200, line_dash="dash", line_color="#2f765f")
        st.plotly_chart(style_figure(fig), width="stretch")
    with right:
        # Barres d'erreur : chaque point de grille est mesuré sur plusieurs seeds,
        # donc sa dispersion fait partie du résultat, pas du décor.
        grid = gamma.sort_values("gamma")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=grid["gamma"],
                y=grid["mean"],
                error_y=dict(type="data", array=grid["std"], visible=True),
                mode="lines+markers",
                line=dict(color="#176780"),
                name="Moyenne inter-seeds",
            )
        )
        fig.add_hline(y=200, line_dash="dash", line_color="#2f765f")
        if selected:
            fig.add_vline(
                x=selected["parameters"]["gamma"],
                line_dash="dot",
                line_color="#c26a2e",
                annotation_text="retenu",
            )
        fig.update_layout(
            title="Grille ciblée sur gamma, centrée sur l'optimum mesuré",
            xaxis_title="Gamma",
            yaxis_title="Récompense moyenne",
        )
        st.plotly_chart(style_figure(fig), width="stretch")

    left, right = st.columns(2)
    with left:
        if not importance.empty:
            fig = px.bar(
                importance.sort_values("importance"),
                x="importance",
                y="parameter",
                orientation="h",
                title="Importance estimée par PED-ANOVA",
                color_discrete_sequence=["#176780"],
            )
            st.plotly_chart(style_figure(fig), width="stretch")
    with right:
        if not robust_summary.empty:
            plot = robust_summary.sort_values("mean")
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=plot["mean"],
                    y=plot["candidate_id"],
                    orientation="h",
                    error_x=dict(type="data", array=plot["std"], visible=True),
                    marker_color="#176780",
                    name="Moyenne sur 5 seeds",
                )
            )
            # La pire seed est le vrai critère de fiabilité : un candidat qui
            # s'effondre une fois sur cinq n'est pas utilisable en mission.
            fig.add_trace(
                go.Scatter(
                    x=plot["min"],
                    y=plot["candidate_id"],
                    mode="markers",
                    marker=dict(color="#c26a2e", symbol="diamond", size=11),
                    name="Pire seed",
                )
            )
            fig.add_vline(x=200, line_dash="dash", line_color="#2f765f")
            fig.update_layout(
                title="Validation des finalistes sur cinq seeds",
                xaxis_title="Récompense moyenne",
                yaxis_title="Candidat",
            )
            st.plotly_chart(style_figure(fig), width="stretch")

    st.caption(
        "Un score de recherche élevé ne garantit rien : le candidat le plus "
        "prometteur peut s'effondrer sur une seed. Le losange marque le pire des "
        "cinq entraînements, seul critère qui distingue un réglage fiable d'un "
        "réglage chanceux."
    )

    displayed_columns = [
        "number", "value", "user_attrs_score_std", "user_attrs_score_min",
        "params_gamma", "params_learning_rate", "params_ent_coef", "params_n_steps",
        "params_batch_size", "params_n_epochs", "params_gae_lambda",
        "params_clip_range", "params_vf_coef", "params_max_grad_norm",
    ]
    st.dataframe(
        complete[[column for column in displayed_columns if column in complete]]
        .sort_values("value", ascending=False),
        width="stretch",
    )


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
                           title="Distribution des récompenses",
                           color_discrete_sequence=BRAND_COLORS)
        fig.add_vline(x=200, line_dash="dash", line_color="#2ca02c")
        st.plotly_chart(style_figure(fig), width="stretch")
    with right:
        fig = px.scatter(
            filtered,
            x="fuel_proxy",
            y="total_reward",
            color="outcome",
            hover_data=["episode_id", "seed", "episode_length"],
            title="Score et utilisation des moteurs",
            color_discrete_sequence=BRAND_COLORS,
        )
        st.plotly_chart(style_figure(fig), width="stretch")
    st.dataframe(filtered, width="stretch")


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
    fig.update_layout(
        title="Télémétrie de l'épisode (unités du simulateur)",
        xaxis_title="Pas",
        yaxis_title="Valeur",
    )
    st.plotly_chart(style_figure(fig), width="stretch")

    action_counts = episode["action_label"].value_counts().rename_axis("action").reset_index(name="count")
    action_fig = px.bar(
        action_counts,
        x="action",
        y="count",
        title="Décisions de l'agent",
        color_discrete_sequence=["#176780"],
    )
    st.plotly_chart(style_figure(action_fig), width="stretch")
    st.dataframe(episode, width="stretch")


def render_gui_runs(artifacts_dir: Path) -> None:
    runs = load_gui_runs(artifacts_dir)
    if runs.empty:
        st.info("Aucun run GUI enregistré pour l'instant. Lancez un épisode depuis la GUI.")
        return
    st.dataframe(runs.sort_values("run_id", ascending=False), width="stretch")
    fig = px.bar(runs, x="run_id", y="total_reward", color="outcome",
                 title="Performances observées depuis la GUI",
                 color_discrete_sequence=BRAND_COLORS)
    fig.add_hline(y=200, line_dash="dash", line_color="#2ca02c")
    st.plotly_chart(style_figure(fig), width="stretch")


def main() -> None:
    st.set_page_config(page_title="Eagle-1 | Mission Analytics", layout="wide")
    render_header()

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

    overview, learning, optuna_tab, episodes, trajectory, gui_runs = st.tabs([
        "Vue mission", "Apprentissage", "Optuna", "Épisodes", "Trajectoire", "Runs GUI"
    ])
    with overview:
        render_overview(filtered_registry)
    with learning:
        render_learning(ARTIFACTS_DIR)
    with optuna_tab:
        render_optuna(ARTIFACTS_DIR)
    with episodes:
        render_episodes(ARTIFACTS_DIR)
    with trajectory:
        render_trajectory(ARTIFACTS_DIR)
    with gui_runs:
        render_gui_runs(ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
