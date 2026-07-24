"""Dashboard Streamlit alimenté par les logs réels de la mission Eagle-1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
OPTUNA_DIR = ARTIFACTS_DIR / "optuna" / "ppo_lunarlander"
FINAL_EVALUATION_ID = "ppo_optuna_final_100"
TARGET_REWARD = 200

APP_STYLES = """
<style>
    .stApp { background: #f7f8f9; color: #17242d; }
    .block-container { max-width: 1480px; padding-top: 2rem; }
    .mission-header { border-top: 4px solid #eb6834; border-bottom: 1px solid #d7dde1; padding: 1rem 0 1.1rem; margin-bottom: 1.4rem; }
    .mission-header .kicker { color: #2a78d6; font-size: .74rem; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; }
    .mission-header h1 { color: #12384a; font-size: 2.1rem; margin: .25rem 0; }
    .mission-header p { color: #52636d; margin: 0; }
    h2, h3 { color: #12384a; }
</style>
"""

# Traduction des identifiants techniques en libellés lisibles.
EXPERIMENT_LABELS = {
    "random": "Politique aléatoire",
    "random_baseline": "Politique aléatoire",
    "dqn_default": "DQN par défaut",
    "ppo_default": "PPO par défaut",
    "ppo_gamma_0999": "PPO — gamma porté à 0.999",
    "ppo_lr_0001": "PPO — pas d'apprentissage 0.001",
    "ppo_nsteps_1024": "PPO — n_steps 1024",
    "ppo_gamma_extended": "PPO gamma=0.999 prolongé",
    "ppo_optuna": "Pilote final (Optuna)",
    "ppo_optuna_s900": "Pilote final — seed 900",
    "ppo_optuna_s901": "Pilote final — seed 901",
    "ppo_optuna_s902": "Pilote final — seed 902",
}

PHASE_ORDER = [
    "1 · Référence aléatoire",
    "2 · Baselines par défaut",
    "3 · Sensibilité (un paramètre à la fois)",
    "4 · Optimisation Optuna",
    "5 · Sélection sur seeds communes",
    "6 · Évaluation finale",
]

OUTCOME_LABELS = {"success": "Réussite", "crash": "Échec", "truncated": "Temps écoulé"}
ACTION_LABELS = {
    "action_0_count": "Ne rien faire",
    "action_1_count": "Orientation gauche",
    "action_2_count": "Moteur principal",
    "action_3_count": "Orientation droite",
}


def experiment_of(evaluation_id: str) -> str:
    """Retrouve l'expérience d'origine à partir d'un identifiant d'évaluation."""
    for suffix in ("_final_100", "_baseline", "_selection", "_screen", "_tuning"):
        if evaluation_id.endswith(suffix):
            return evaluation_id[: -len(suffix)]
    return evaluation_id


def human_label(evaluation_id: str) -> str:
    """Traduit un identifiant d'évaluation en libellé lisible."""
    experiment = experiment_of(evaluation_id)
    base = EXPERIMENT_LABELS.get(experiment, experiment.replace("_", " "))
    suffixes = {
        "_final_100": " — évaluation finale",
        "_selection": " — jeu de sélection",
        "_screen": " — test court",
        "_tuning": " — réglage",
    }
    for suffix, text in suffixes.items():
        if evaluation_id.endswith(suffix):
            return base + text
    return base


def experiment_phase(evaluation_id: str) -> str:
    """Range chaque mesure dans l'étape de mission qui lui correspond."""
    if evaluation_id.startswith("random"):
        return PHASE_ORDER[0]
    if "baseline" in evaluation_id:
        return PHASE_ORDER[1]
    if "screen" in evaluation_id:
        return PHASE_ORDER[2]
    if "tuning" in evaluation_id or "optuna_trial" in evaluation_id:
        return PHASE_ORDER[3]
    if "selection" in evaluation_id:
        return PHASE_ORDER[4]
    if "final" in evaluation_id:
        return PHASE_ORDER[5]
    return "Autre"


def candidate_label(candidate_id: str) -> str:
    """Traduit un identifiant de finaliste Optuna en libellé lisible."""
    if candidate_id.startswith("broad_t"):
        return f"Recherche large — essai {int(candidate_id.removeprefix('broad_t'))}"
    if candidate_id.startswith("gamma_g"):
        return f"Grille gamma — point {int(candidate_id.removeprefix('gamma_g'))}"
    return candidate_id


def render_header() -> None:
    st.markdown(APP_STYLES, unsafe_allow_html=True)
    st.markdown(
        """
        <header class="mission-header">
            <div class="kicker">AstroDynamics / Mission Analytics</div>
            <h1>Eagle-1 — Analyse des performances</h1>
            <p>PPO · LunarLander-v3 · expériences, apprentissage et évaluation finale</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def load_registry(artifacts_dir: Path = ARTIFACTS_DIR) -> pd.DataFrame:
    registry = pd.read_csv(Path(artifacts_dir) / "experiment_registry.csv")
    registry["phase"] = registry["evaluation_id"].map(experiment_phase)
    registry["success_percent"] = 100 * registry["success_rate"]
    registry["label"] = registry["evaluation_id"].map(human_label)
    return registry


def list_episode_evaluations(artifacts_dir: Path = ARTIFACTS_DIR) -> list[str]:
    evaluation_dir = Path(artifacts_dir) / "evaluations"
    return sorted(path.parent.name for path in evaluation_dir.glob("*/episodes.csv"))


def load_episodes(evaluation_id: str, artifacts_dir: Path = ARTIFACTS_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(artifacts_dir) / "evaluations" / evaluation_id / "episodes.csv")


def build_reward_history(episodes: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Prépare la récompense par épisode et sa moyenne glissante."""
    history = (
        episodes.sort_values("episode_id")
        .set_index("episode_id")[["total_reward"]]
        .rename(columns={"total_reward": "Récompense"})
    )
    history["Moyenne glissante"] = history["Récompense"].rolling(
        window=min(window, len(history)),
        min_periods=1,
    ).mean()
    return history


def load_learning_curves(artifacts_dir: Path = ARTIFACTS_DIR) -> pd.DataFrame:
    """Fusionne les fichiers EvalCallback dans une seule table."""
    rows = []
    for path in Path(artifacts_dir).glob("evaluations/*/during_training/evaluations.npz"):
        data = np.load(path)
        means = data["results"].mean(axis=1)
        for timestep, mean in zip(data["timesteps"], means):
            rows.append({
                "experiment": path.parents[1].name,
                "timestep": int(timestep),
                "mean_reward": float(mean),
            })
    return pd.DataFrame(rows)


def load_optuna_results(artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, pd.DataFrame]:
    """Charge les tables produites par la campagne Optuna."""
    optuna_dir = (
        OPTUNA_DIR
        if Path(artifacts_dir) == ARTIFACTS_DIR
        else Path(artifacts_dir) / "optuna" / "ppo_lunarlander"
    )
    paths = {
        "trials": optuna_dir / "trials.csv",
        "gamma": optuna_dir / "gamma_focus" / "trials.csv",
        "robustness_summary": optuna_dir / "robustness_summary.csv",
        "importance": optuna_dir / "parameter_importance.csv",
    }
    tables = {
        name: pd.read_csv(path) if path.exists() else pd.DataFrame()
        for name, path in paths.items()
    }
    trials = tables["trials"]
    if not trials.empty and "params_one_minus_gamma" in trials:
        # La recherche échantillonne 1 - gamma ; on affiche gamma lui-même.
        trials["params_gamma"] = 1.0 - trials["params_one_minus_gamma"]
    return tables


def load_selected_config(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    path = Path(artifacts_dir) / "optuna" / "ppo_lunarlander" / "selected_config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def render_summary(registry: pd.DataFrame, artifacts_dir: Path) -> None:
    """Indicateurs finaux et positionnement face aux points de comparaison."""
    final = registry[registry["evaluation_id"] == FINAL_EVALUATION_ID]
    if final.empty:
        st.warning("Évaluation finale indisponible.")
        return
    final = final.iloc[0]
    episodes = load_episodes(FINAL_EVALUATION_ID, artifacts_dir)

    cols = st.columns(4)
    cols[0].metric("Récompense moyenne", f"{final['mean_reward']:.2f}",
                   delta=f"{final['mean_reward'] - TARGET_REWARD:+.1f} / seuil 200")
    cols[1].metric("Écart-type", f"{final['std_reward']:.2f}")
    cols[2].metric("Médiane", f"{episodes['total_reward'].median():.2f}")
    cols[3].metric("Taux de réussite", f"{final['success_percent']:.0f} %")
    outcome_counts = episodes["outcome"].value_counts()
    st.caption(
        f"{outcome_counts.get('success', 0)} réussites · "
        f"{outcome_counts.get('crash', 0)} crashs · "
        f"{outcome_counts.get('truncated', 0)} épisodes tronqués"
    )

    st.markdown("#### Récompense moyenne par politique")
    reference_ids = [
        "random_baseline", "dqn_default_baseline", "ppo_default_baseline",
        "ppo_gamma_extended_selection", FINAL_EVALUATION_ID,
    ]
    comparison = (
        registry[registry["evaluation_id"].isin(reference_ids)]
        .sort_values("mean_reward")
    )
    st.bar_chart(comparison, x="label", y="mean_reward", horizontal=True)
    st.caption("Objectif de mission : 200. Pilote final en tête.")


def render_phases(registry: pd.DataFrame, artifacts_dir: Path) -> None:
    """Montre la progression étape par étape, puis les courbes d'apprentissage."""
    st.markdown("#### Meilleur résultat par phase")

    ordered = [phase for phase in PHASE_ORDER if phase in set(registry["phase"])]
    rows = []
    for phase in ordered:
        subset = registry[registry["phase"] == phase]
        best = subset.loc[subset["mean_reward"].idxmax()]
        rows.append({
            "Phase": phase,
            "Meilleure config": best["label"],
            "Récompense": round(best["mean_reward"], 2),
            "Réussite": f"{best['success_percent']:.0f} %",
        })
    phase_table = pd.DataFrame(rows)
    st.dataframe(phase_table, width="stretch", hide_index=True)
    st.bar_chart(phase_table, x="Phase", y="Récompense")

    st.markdown("#### Courbes d'apprentissage (EvalCallback)")
    curves = load_learning_curves(artifacts_dir)
    if curves.empty:
        st.warning("Aucune courbe d'entraînement disponible.")
        return

    default = [name for name in ["dqn_default", "ppo_default", "ppo_gamma_extended", "ppo_optuna"]
               if name in set(curves["experiment"])]
    chosen = st.multiselect(
        "Expériences",
        options=sorted(curves["experiment"].unique()),
        default=default,
        format_func=lambda name: EXPERIMENT_LABELS.get(name, name),
    )
    if not chosen:
        st.info("Sélectionnez au moins un entraînement.")
        return

    selected = curves[curves["experiment"].isin(chosen)].copy()
    selected["label"] = selected["experiment"].map(lambda name: EXPERIMENT_LABELS.get(name, name))
    pivot = selected.pivot_table(index="timestep", columns="label", values="mean_reward")
    st.line_chart(pivot)


def render_episodes(artifacts_dir: Path) -> None:
    """Permet d'isoler les réussites et les échecs d'une évaluation."""
    st.markdown("#### Épisodes filtrables")
    evaluations = list_episode_evaluations(artifacts_dir)
    if not evaluations:
        st.warning("Aucune évaluation par épisode n'est disponible.")
        return
    default_index = evaluations.index(FINAL_EVALUATION_ID) if FINAL_EVALUATION_ID in evaluations else 0

    filters = st.columns([2, 2, 3])
    with filters[0]:
        evaluation_id = st.selectbox("Évaluation", evaluations, index=default_index,
                                     format_func=human_label)
    episodes = load_episodes(evaluation_id, artifacts_dir)
    episodes = episodes.assign(issue=episodes["outcome"].map(lambda v: OUTCOME_LABELS.get(v, v)))

    st.markdown("##### Récompense par épisode")
    st.line_chart(build_reward_history(episodes))
    st.caption("La moyenne glissante porte sur les 10 derniers épisodes.")

    with filters[1]:
        available_outcomes = sorted(episodes["outcome"].unique())
        chosen = st.multiselect(
            "Issue de l'épisode", available_outcomes, default=available_outcomes,
            format_func=lambda v: OUTCOME_LABELS.get(v, v),
        )
    with filters[2]:
        minimum = float(episodes["total_reward"].min())
        maximum = float(episodes["total_reward"].max())
        reward_range = st.slider("Plage de récompense", min_value=minimum,
                                 max_value=maximum, value=(minimum, maximum))

    filtered = episodes[
        episodes["outcome"].isin(chosen)
        & episodes["total_reward"].between(*reward_range)
    ]
    if filtered.empty:
        st.warning("Aucun épisode ne correspond à ces filtres.")
        return

    cols = st.columns(5)
    cols[0].metric("Épisodes", f"{len(filtered)} / {len(episodes)}")
    cols[1].metric("Moyenne", f"{filtered['total_reward'].mean():.2f}")
    cols[2].metric("Écart-type", f"{filtered['total_reward'].std(ddof=0):.2f}")
    cols[3].metric("Médiane", f"{filtered['total_reward'].median():.2f}")
    cols[4].metric("Min / Max",
                   f"{filtered['total_reward'].min():.0f} / {filtered['total_reward'].max():.0f}")

    left, right = st.columns(2)
    with left:
        st.markdown("##### Récompense et consommation")
        st.scatter_chart(filtered, x="fuel_proxy", y="total_reward", color="issue")
        st.caption("En bas à droite : les vols coûteux et peu rentables.")
    with right:
        st.markdown("##### Actions utilisées")
        action_columns = [column for column in ACTION_LABELS if column in filtered]
        action_counts = (
            filtered[action_columns]
            .sum()
            .rename(index=ACTION_LABELS)
            .rename_axis("Action")
            .reset_index(name="Utilisations")
        )
        st.bar_chart(action_counts, x="Action", y="Utilisations", horizontal=True)

    table_columns = ["episode_id", "seed", "issue", "total_reward", "episode_length",
                     "fuel_proxy", "final_x", "final_y"]
    st.dataframe(
        filtered[[c for c in table_columns if c in filtered]]
        .rename(columns={
            "episode_id": "Épisode", "seed": "Seed", "issue": "Issue",
            "total_reward": "Récompense", "episode_length": "Pas",
            "fuel_proxy": "Carburant", "final_x": "x final", "final_y": "y final",
        })
        .round(2),
        width="stretch", hide_index=True,
    )


def render_optuna(artifacts_dir: Path) -> None:
    """Recherche large, raffinement de gamma et sélection robuste."""
    results = load_optuna_results(artifacts_dir)
    trials = results["trials"]
    gamma = results["gamma"]
    importance = results["importance"]
    robust_summary = results["robustness_summary"]
    selected = load_selected_config(artifacts_dir)
    if trials.empty or gamma.empty:
        st.warning("Les artefacts Optuna ne sont pas encore disponibles.")
        return

    complete = trials[trials["state"] == "COMPLETE"].copy()
    pruned = trials[trials["state"] == "PRUNED"]

    st.markdown("#### Recherche TPE — 120 essais, 2 seeds/essai, MedianPruner")
    cols = st.columns(4)
    cols[0].metric("Essais menés à terme", len(complete))
    cols[1].metric("Essais élagués", len(pruned))
    if selected:
        cols[2].metric("Gamma retenu", f"{selected['parameters']['gamma']:.5f}")
        cols[3].metric("Robustesse (5 seeds)", f"{selected['robust_mean_reward']:.1f}",
                       delta=f"min {selected['robust_min_reward']:.0f}")

    left, right = st.columns(2)
    with left:
        st.markdown("##### Chaque essai de la recherche")
        st.scatter_chart(complete, x="number", y="value")
        st.caption(f"{len(pruned)} essais élagués non représentés.")
    with right:
        st.markdown("##### Réglage fin de gamma")
        st.line_chart(gamma.sort_values("gamma"), x="gamma", y="mean")
        st.caption("3 seeds/point.")

    left, right = st.columns(2)
    with left:
        if not importance.empty:
            st.markdown("##### Quels réglages comptent vraiment")
            st.bar_chart(importance.sort_values("importance"),
                         x="parameter", y="importance", horizontal=True)
            st.caption("PED-ANOVA · part de variance expliquée par paramètre.")
    with right:
        if not robust_summary.empty:
            st.markdown("##### Les finalistes réentraînés 5 fois")
            plot = robust_summary.sort_values("mean").copy()
            plot["nom"] = plot["candidate_id"].map(candidate_label)
            st.bar_chart(plot, x="nom", y="mean", horizontal=True)
            st.caption("Moyenne sur 5 entraînements.")

    with st.expander("Table des essais aboutis"):
        columns = ["number", "value", "params_gamma", "params_learning_rate",
                   "params_ent_coef", "params_n_steps", "params_batch_size"]
        st.dataframe(
            complete[[c for c in columns if c in complete]]
            .sort_values("value", ascending=False).round(5),
            width="stretch", hide_index=True,
        )


def main() -> None:
    st.set_page_config(page_title="Eagle-1 | Mission Analytics", layout="wide")
    render_header()

    registry = load_registry(ARTIFACTS_DIR)

    summary, phases, episodes, optuna_tab = st.tabs([
        "Synthèse", "Apprentissage", "Épisodes", "Optuna",
    ])
    with summary:
        render_summary(registry, ARTIFACTS_DIR)
    with phases:
        render_phases(registry, ARTIFACTS_DIR)
    with episodes:
        render_episodes(ARTIFACTS_DIR)
    with optuna_tab:
        render_optuna(ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
