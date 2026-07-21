"""GUI Streamlit : visualisation d'un épisode piloté par l'API Eagle-1."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import gymnasium as gym
import httpx
import numpy as np
import pandas as pd
import streamlit as st


ENV_ID = "LunarLander-v3"
DEFAULT_API_URL = os.getenv("EAGLE1_API_URL", "http://127.0.0.1:8000")
DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "gui_runs"

ACTION_LABELS = {
    0: "Ne rien faire",
    1: "Orientation gauche",
    2: "Moteur principal",
    3: "Orientation droite",
}

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
    .block-container { max-width: 1380px; padding-top: 2rem; padding-bottom: 3rem; }
    .mission-header { border-top: 4px solid var(--ad-orange); border-bottom: 1px solid var(--ad-line); padding: 1rem 0 1.1rem; margin-bottom: 1.4rem; }
    .mission-header .kicker { color: var(--ad-blue); font-size: .74rem; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; }
    .mission-header h1 { color: var(--ad-navy); font-size: 2.1rem; font-weight: 650; letter-spacing: -.02em; margin: .25rem 0; }
    .mission-header p { color: #52636d; margin: 0; font-size: .96rem; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid var(--ad-line); border-radius: 2px; padding: .75rem .9rem; }
    [data-testid="stMetricLabel"] { color: #5b6a73; }
    .stButton > button { border-radius: 2px; border: 1px solid var(--ad-navy); font-weight: 650; }
    .stButton > button[kind="primary"] { background: var(--ad-navy); color: #fff; }
    .stButton > button[kind="primary"]:hover { background: var(--ad-blue); border-color: var(--ad-blue); }
    div[data-testid="stImage"] img { border: 1px solid var(--ad-line); }
    h2, h3 { color: var(--ad-navy); font-weight: 650; }
</style>
"""


def render_header() -> None:
    """Affiche une identité sobre inspirée d'un pupitre de contrôle."""
    st.markdown(APP_STYLES, unsafe_allow_html=True)
    st.markdown(
        """
        <header class="mission-header">
            <div class="kicker">AstroDynamics / Flight Control</div>
            <h1>Eagle-1</h1>
            <p>Simulation du pilote automatique d'atterrissage lunaire</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


@dataclass
class EpisodeResult:
    seed: int
    total_reward: float
    steps: int
    outcome: str
    success: bool
    final_reward: float
    fuel_proxy: float


def _endpoint(api_url: str, route: str, external_client) -> str:
    """TestClient attend une route relative ; httpx attend l'URL complète."""
    return route if external_client is not None else f"{api_url.rstrip('/')}{route}"


def check_api(api_url: str, client=None) -> dict:
    """Interroge la route de santé sans connaître le modèle côté GUI."""
    owned_client = client is None
    http_client = client or httpx.Client(timeout=10.0)
    try:
        response = http_client.get(_endpoint(api_url, "/health", client))
        response.raise_for_status()
        return response.json()
    finally:
        if owned_client:
            http_client.close()


def play_episode(
    api_url: str,
    seed: int,
    client=None,
) -> tuple[EpisodeResult, pd.DataFrame]:
    """Joue un épisode ; chaque action est demandée à l'API par HTTP."""
    env = gym.make(ENV_ID)
    observation, _ = env.reset(seed=seed)

    owned_client = client is None
    http_client = client or httpx.Client(timeout=10.0)
    terminated = truncated = False
    total_reward = 0.0
    fuel_proxy = 0.0
    final_reward = 0.0
    step = 0
    rows = []

    try:
        while not (terminated or truncated):
            response = http_client.post(
                _endpoint(api_url, "/play", client),
                json={"state": observation.tolist(), "deterministic": True},
            )
            response.raise_for_status()
            action = int(response.json()["action"])

            next_observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            final_reward = float(reward)
            fuel_proxy += 0.30 if action == 2 else (0.03 if action in (1, 3) else 0.0)

            row = {
                "step": step,
                "action": action,
                "action_label": ACTION_LABELS[action],
                "reward": float(reward),
                "cumulative_reward": total_reward,
                "x": float(next_observation[0]),
                "y": float(next_observation[1]),
                "vx": float(next_observation[2]),
                "vy": float(next_observation[3]),
                "angle": float(next_observation[4]),
                "angular_velocity": float(next_observation[5]),
                "left_contact": float(next_observation[6]),
                "right_contact": float(next_observation[7]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            rows.append(row)

            observation = next_observation
            step += 1
    finally:
        env.close()
        if owned_client:
            http_client.close()

    if truncated:
        outcome = "truncated"
    elif np.isclose(final_reward, 100.0):
        outcome = "success"
    else:
        outcome = "crash"

    result = EpisodeResult(
        seed=seed,
        total_reward=total_reward,
        steps=step,
        outcome=outcome,
        success=outcome == "success",
        final_reward=final_reward,
        fuel_proxy=fuel_proxy,
    )
    return result, pd.DataFrame(rows)


def replay_episode(
    seed: int,
    steps: pd.DataFrame,
    duration_seconds: float,
    on_frame: Callable[[np.ndarray, dict], None],
    target_ui_fps: float = 10.0,
) -> None:
    """Rejoue les actions de l'API pendant la durée de visualisation demandée.

    La simulation est déterministe : même seed et mêmes actions donnent le même vol.
    Le nombre d'images est choisi automatiquement pour ne pas surcharger Streamlit.
    """
    if steps.empty:
        raise ValueError("Aucune action à rejouer.")
    if duration_seconds < 0:
        raise ValueError("La durée de visualisation doit être positive.")

    steps = steps.reset_index(drop=True)
    frame_count = min(
        len(steps),
        max(1, math.ceil(duration_seconds * target_ui_fps)),
    )
    frame_indices = (
        {len(steps) - 1}
        if frame_count == 1
        else set(np.linspace(0, len(steps) - 1, num=frame_count, dtype=int).tolist())
    )

    env = gym.make(ENV_ID, render_mode="rgb_array")
    env.reset(seed=seed)
    terminated = truncated = False
    displayed_frames = 0
    started_at = time.monotonic()

    try:
        for index, row in steps.iterrows():
            if terminated or truncated:
                raise RuntimeError("Le rejeu s'est terminé avant la dernière action enregistrée.")

            _, _, terminated, truncated, _ = env.step(int(row["action"]))
            if index not in frame_indices:
                continue

            if frame_count > 1:
                target_elapsed = duration_seconds * displayed_frames / (frame_count - 1)
                remaining = started_at + target_elapsed - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)

            on_frame(env.render(), row.to_dict())
            displayed_frames += 1

        if not (terminated or truncated):
            raise RuntimeError("Le rejeu n'a pas atteint la fin de l'épisode.")

        if frame_count == 1 and duration_seconds > 0:
            remaining = started_at + duration_seconds - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close()


def save_run(
    result: EpisodeResult,
    steps: pd.DataFrame,
    output_dir: Path = DEFAULT_RUNS_DIR,
) -> Path:
    """Sauvegarde le résumé et la télémétrie pour le dashboard."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_seed{result.seed}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {"run_id": run_id, **asdict(result)}
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    steps.to_csv(run_dir / "steps.csv", index=False)

    registry_path = Path(output_dir) / "runs.csv"
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    registry = pd.concat([registry, pd.DataFrame([summary])], ignore_index=True)
    registry.to_csv(registry_path, index=False)
    return run_dir


def main() -> None:
    st.set_page_config(page_title="Eagle-1 | Flight Control", layout="wide")
    render_header()
    st.caption("La simulation exécute LunarLander-v3 ; chaque décision est fournie par l'API FastAPI.")

    with st.sidebar:
        st.header("Paramètres de vol")
        api_url = st.text_input(
            "URL de l'API",
            DEFAULT_API_URL,
            help="Valeur par défaut issue de la variable d'environnement EAGLE1_API_URL.",
        )
        seed = int(st.number_input("Seed", min_value=0, value=10_000, step=1))
        duration_seconds = st.slider(
            "Durée de visualisation (secondes)",
            min_value=5,
            max_value=60,
            value=10,
            step=5,
            help="Durée du rejeu à l'écran. Elle ne change ni les décisions de "
                 "l'agent ni le résultat du vol, seulement la vitesse d'affichage.",
        )

        if st.button("Vérifier l'API", width="stretch"):
            try:
                health = check_api(api_url)
                st.success(f"API prête — {health['model_id']}")
            except Exception as exc:
                st.error(f"API indisponible : {exc}")

        launch = st.button("Lancer la simulation", type="primary", width="stretch")

    # Le bilan et les courbes sont réservés en haut de page : après un vol, on
    # veut les lire immédiatement, sans faire défiler toute la télémétrie. Ces
    # emplacements restent vides pendant la simulation et sont remplis à la fin.
    status_box = st.empty()
    summary_box = st.empty()
    charts_box = st.empty()

    st.subheader("Vol en direct")
    frame_box = st.empty()
    metric_boxes = st.columns(4)

    if not launch:
        st.info("Démarrez l'API, choisissez une seed puis lancez un épisode.")
        return

    def update_screen(frame: np.ndarray, row: dict) -> None:
        frame_box.image(frame, channels="RGB", width="stretch")
        metric_boxes[0].metric("Pas", row["step"])
        metric_boxes[1].metric("Action", row["action_label"])
        metric_boxes[2].metric("Récompense", f"{row['cumulative_reward']:.1f}")
        metric_boxes[3].metric("Altitude", f"{row['y']:.3f}")

    try:
        with st.spinner("Calcul des décisions via l'API…"):
            result, steps = play_episode(api_url=api_url, seed=seed)
        with st.spinner(f"Visualisation du vol pendant environ {duration_seconds} s…"):
            replay_episode(
                seed=seed,
                steps=steps,
                duration_seconds=float(duration_seconds),
                on_frame=update_screen,
            )
    except Exception as exc:
        st.error(f"La simulation a échoué : {exc}")
        return

    run_dir = save_run(result, steps)

    if result.success:
        status_box.success(f"Atterrissage réussi — récompense {result.total_reward:.1f}")
    else:
        status_box.error(
            f"Épisode terminé : {result.outcome} — récompense {result.total_reward:.1f}"
        )

    with summary_box.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Récompense totale", f"{result.total_reward:.1f}")
        col2.metric("Nombre de pas", result.steps)
        col3.metric("Proxy carburant", f"{result.fuel_proxy:.1f}")

    with charts_box.container():
        chart_col, action_col = st.columns(2)
        with chart_col:
            st.subheader("Récompense cumulée")
            st.line_chart(steps.set_index("step")["cumulative_reward"])
        with action_col:
            st.subheader("Actions utilisées")
            st.bar_chart(steps["action_label"].value_counts())

    with st.expander("Télémétrie détaillée"):
        st.dataframe(steps, width="stretch")
    st.caption(f"Run sauvegardé dans : {run_dir}")


if __name__ == "__main__":
    main()
