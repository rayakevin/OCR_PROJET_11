"""GUI Streamlit : visualisation d'un épisode piloté par l'API Eagle-1."""

from __future__ import annotations

import json
import os
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
    display_every: int = 3,
    on_frame: Callable[[np.ndarray, dict], None] | None = None,
    client=None,
) -> tuple[EpisodeResult, pd.DataFrame]:
    """Joue un épisode ; chaque action est demandée à l'API par HTTP."""
    render_mode = "rgb_array" if on_frame is not None else None
    env = gym.make(ENV_ID, render_mode=render_mode)
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

            if on_frame is not None and step % max(display_every, 1) == 0:
                on_frame(env.render(), row)

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
    st.set_page_config(page_title="Eagle-1", page_icon="🚀", layout="wide")
    st.title("🚀 Eagle-1 — Pilote automatique lunaire")
    st.caption("La GUI visualise l'environnement ; toutes les décisions viennent de l'API FastAPI.")

    with st.sidebar:
        st.header("Simulation")
        api_url = st.text_input("URL de l'API", DEFAULT_API_URL)
        seed = int(st.number_input("Seed", min_value=0, value=10_000, step=1))
        display_every = st.slider("Afficher une image tous les N pas", 1, 10, 3)

        if st.button("Vérifier l'API", use_container_width=True):
            try:
                health = check_api(api_url)
                st.success(f"API prête — {health['model_id']}")
            except Exception as exc:
                st.error(f"API indisponible : {exc}")

        launch = st.button("▶ Lancer un épisode", type="primary", use_container_width=True)

    frame_box = st.empty()
    metric_boxes = st.columns(4)

    if not launch:
        st.info("Démarrez l'API, choisissez une seed puis lancez un épisode.")
        return

    def update_screen(frame: np.ndarray, row: dict) -> None:
        frame_box.image(frame, channels="RGB", use_container_width=True)
        metric_boxes[0].metric("Pas", row["step"])
        metric_boxes[1].metric("Action", row["action_label"])
        metric_boxes[2].metric("Récompense", f"{row['cumulative_reward']:.1f}")
        metric_boxes[3].metric("Altitude", f"{row['y']:.3f}")

    try:
        with st.spinner("Eagle-1 est en vol…"):
            result, steps = play_episode(
                api_url=api_url,
                seed=seed,
                display_every=display_every,
                on_frame=update_screen,
            )
    except Exception as exc:
        st.error(f"La simulation a échoué : {exc}")
        return

    run_dir = save_run(result, steps)
    if result.success:
        st.success(f"Atterrissage réussi — récompense {result.total_reward:.1f}")
    else:
        st.error(f"Épisode terminé : {result.outcome} — récompense {result.total_reward:.1f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Récompense totale", f"{result.total_reward:.1f}")
    col2.metric("Nombre de pas", result.steps)
    col3.metric("Proxy carburant", f"{result.fuel_proxy:.1f}")

    chart_col, action_col = st.columns(2)
    with chart_col:
        st.subheader("Récompense cumulée")
        st.line_chart(steps.set_index("step")["cumulative_reward"])
    with action_col:
        st.subheader("Actions utilisées")
        action_counts = steps["action_label"].value_counts()
        st.bar_chart(action_counts)

    with st.expander("Télémétrie détaillée"):
        st.dataframe(steps, use_container_width=True)
    st.caption(f"Run sauvegardé dans : {run_dir}")


if __name__ == "__main__":
    main()

