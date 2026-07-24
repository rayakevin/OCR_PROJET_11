#!/usr/bin/env python3
"""Raffinement de gamma puis sélection robuste du réglage PPO final.

Ce module consomme l'étude produite par :mod:`optuna_search` et enchaîne les
deux étapes qui séparent « un bon score de recherche » d'« un réglage sur lequel
on peut s'engager » :

**1. Grille ciblée sur gamma.** La recherche large échantillonne gamma en
continu, mais le cahier des charges demande une mesure fine de la zone forte.
La grille est donc centrée sur l'optimum *effectivement mesuré* par la
recherche, et non sur une valeur supposée d'avance : elle balaie les horizons
effectifs voisins de celui du meilleur essai. Chaque point est entraîné sur
plusieurs seeds, car un point de grille noté sur une seule seed reproduirait
exactement le défaut que ce module corrige.

**2. Validation multi-seed des finalistes.** Les meilleurs candidats des deux
sources sont réentraînés sur un budget plus long et sur plusieurs seeds
d'entraînement neuves. Le classement retenu est celui de la moyenne inter-seeds,
avec l'écart-type et le minimum reportés : un réglage qui gagne en moyenne mais
s'effondre sur une seed n'est pas un réglage sur lequel poser une mission.

Utilisation :

    uv run python ASTRODYNAMICS/optuna_select.py --workers 16
    uv run python ASTRODYNAMICS/optuna_select.py --fast
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:  # pragma: no cover
    import triton  # noqa: F401
except ImportError:  # pragma: no cover
    pass

import numpy as np
import optuna
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from optuna_search import (  # noqa: E402
    STUDY_DIR,
    STUDY_NAME,
    build_storage,
    evaluate_deterministic,
    resolved_parameters,
)

GAMMA_DIR = STUDY_DIR / "gamma_focus"

# Seeds de sélection : distinctes de la recherche (2000+) et de la finale (10000+).
SELECTION_EVAL_SEED_START = 5_000
GAMMA_TRAIN_SEEDS = (800, 801, 802)
ROBUST_TRAIN_SEEDS = (242, 342, 442, 542, 642)

GAMMA_STEPS = 150_000
ROBUST_STEPS = 300_000
GAMMA_EVAL_EPISODES = 20
ROBUST_EVAL_EPISODES = 30
TOP_K = 5
GAMMA_GRID_SIZE = 9


def _fast_mode() -> None:
    """Réduit tous les budgets pour valider le code en quelques minutes."""
    global GAMMA_STEPS, ROBUST_STEPS, GAMMA_EVAL_EPISODES, ROBUST_EVAL_EPISODES
    global TOP_K, GAMMA_GRID_SIZE, GAMMA_TRAIN_SEEDS, ROBUST_TRAIN_SEEDS
    GAMMA_STEPS = ROBUST_STEPS = 8_192
    GAMMA_EVAL_EPISODES = ROBUST_EVAL_EPISODES = 4
    TOP_K = 2
    GAMMA_GRID_SIZE = 3
    GAMMA_TRAIN_SEEDS = (800,)
    ROBUST_TRAIN_SEEDS = (242, 342)


def train_and_evaluate(
    parameters: dict,
    train_seed: int,
    total_timesteps: int,
    n_eval_episodes: int,
    seed_start: int,
    save_path: Path | None = None,
) -> dict:
    """Entraîne un réglage sur une seed et renvoie son évaluation."""
    torch.set_num_threads(1)
    train_env = make_vec_env("LunarLander-v3", n_envs=1, seed=train_seed)
    started = time.perf_counter()
    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            seed=train_seed,
            device="cpu",
            verbose=0,
            **parameters,
        )
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
        duration = time.perf_counter() - started
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(save_path)
        mean_reward = evaluate_deterministic(model, n_eval_episodes, seed_start)
    finally:
        train_env.close()

    return {
        "train_seed": train_seed,
        "mean_reward": mean_reward,
        "total_timesteps": total_timesteps,
        "training_duration_s": duration,
    }


def _job(payload: dict) -> dict:
    """Tâche unitaire exécutée dans un processus worker."""
    result = train_and_evaluate(
        parameters=payload["parameters"],
        train_seed=payload["train_seed"],
        total_timesteps=payload["total_timesteps"],
        n_eval_episodes=payload["n_eval_episodes"],
        seed_start=payload["seed_start"],
        save_path=Path(payload["save_path"]) if payload.get("save_path") else None,
    )
    result.update({k: payload[k] for k in ("label", "group") if k in payload})
    return result


def run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    """Exécute les tâches en parallèle sur plusieurs processus."""
    import multiprocessing as mp

    if not jobs:
        return []
    context = mp.get_context("spawn")
    with context.Pool(processes=max(1, min(workers, len(jobs)))) as pool:
        return pool.map(_job, jobs)


def build_gamma_grid(best_gamma: float, size: int) -> list[float]:
    """Construit une grille de gamma centrée sur l'optimum mesuré.

    La grille est régulière en `log(1 - gamma)`, c'est-à-dire en horizon
    effectif : elle explore un facteur 10 de part et d'autre de l'optimum, ce
    qui donne autant de résolution des deux côtés — contrairement à une grille
    régulière en gamma, qui écrase systématiquement le côté des horizons courts.
    """
    center = np.log10(1.0 - best_gamma)
    offsets = np.linspace(-1.0, 1.0, size)
    gammas = [float(1.0 - 10 ** (center + offset)) for offset in offsets]
    return sorted({round(g, 6) for g in gammas if 0.9 < g < 0.99999})


def dedup_key(parameters: dict) -> str:
    """Clé d'unicité tolérante aux écarts numériques négligeables.

    La grille arrondit gamma à six décimales alors que la recherche large le
    garde en pleine précision : sans arrondi commun, deux réglages identiques à
    5e-7 près sur gamma — soit 0,2 % d'horizon effectif — seraient validés deux
    fois, et compteraient à tort comme deux finalistes distincts.
    """
    rounded = {}
    for name, value in parameters.items():
        if name == "gamma":
            # La grille produit déjà gamma arrondi à six décimales ; appliquer le
            # même arrondi à un gamma de recherche large fait tomber les deux sur
            # la même clé lorsqu'ils s'accordent à ce niveau. Les points de grille
            # distincts diffèrent de plus de 1e-4 et restent donc bien séparés.
            rounded[name] = round(value, 6)
        elif isinstance(value, float):
            rounded[name] = float(f"{value:.6g}")
        else:
            rounded[name] = value
    return json.dumps(rounded, sort_keys=True)


def phase_gamma_focus(study: optuna.Study, workers: int) -> pd.DataFrame:
    """Mesure finement gamma autour de l'optimum, sur plusieurs seeds."""
    best = study.best_trial
    base_parameters = resolved_parameters(best.params)
    best_gamma = base_parameters["gamma"]
    grid = build_gamma_grid(best_gamma, GAMMA_GRID_SIZE)
    print(f"Meilleur gamma mesuré : {best_gamma:.5f}")
    print(f"Grille ciblée ({len(grid)} valeurs) : {grid}")

    jobs = []
    for index, gamma in enumerate(grid):
        parameters = {**base_parameters, "gamma": gamma}
        for train_seed in GAMMA_TRAIN_SEEDS:
            jobs.append({
                "parameters": parameters,
                "train_seed": train_seed,
                "total_timesteps": GAMMA_STEPS,
                "n_eval_episodes": GAMMA_EVAL_EPISODES,
                "seed_start": SELECTION_EVAL_SEED_START,
                "label": f"gamma_{index:03d}",
                "group": gamma,
            })

    print(f"Grille gamma : {len(jobs)} entraînements")
    results = run_jobs(jobs, workers)

    rows = pd.DataFrame(results).rename(columns={"group": "gamma", "label": "point_id"})
    GAMMA_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(GAMMA_DIR / "runs.csv", index=False)

    summary = (
        rows.groupby(["point_id", "gamma"])["mean_reward"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    summary.to_csv(GAMMA_DIR / "trials.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def phase_robustness(
    study: optuna.Study, gamma_summary: pd.DataFrame, workers: int
) -> pd.DataFrame:
    """Réentraîne les finalistes sur un budget long et plusieurs seeds neuves."""
    best_parameters = resolved_parameters(study.best_trial.params)

    candidates: dict[str, dict] = {}
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    for trial in sorted(completed, key=lambda t: t.value, reverse=True)[:TOP_K]:
        candidates[f"broad_t{trial.number:03d}"] = {
            "source": "recherche_large",
            "search_mean_reward": float(trial.value),
            "parameters": resolved_parameters(trial.params),
        }

    for _, row in gamma_summary.head(TOP_K).iterrows():
        candidates[f"gamma_g{row['point_id'].split('_')[1]}"] = {
            "source": "grille_gamma",
            "search_mean_reward": float(row["mean"]),
            "parameters": {**best_parameters, "gamma": float(row["gamma"])},
        }

    # Deux sources peuvent proposer le même réglage : on ne le valide qu'une fois.
    unique: dict[str, tuple[str, dict]] = {}
    for candidate_id, candidate in candidates.items():
        key = dedup_key(candidate["parameters"])
        if key not in unique or candidate["search_mean_reward"] > unique[key][1]["search_mean_reward"]:
            unique[key] = (candidate_id, candidate)

    finalists = dict(
        sorted(
            (v for v in unique.values()),
            key=lambda item: item[1]["search_mean_reward"],
            reverse=True,
        )[:TOP_K]
    )
    print(f"Finalistes retenus : {list(finalists)}")

    jobs = []
    for candidate_id, candidate in finalists.items():
        for train_seed in ROBUST_TRAIN_SEEDS:
            jobs.append({
                "parameters": candidate["parameters"],
                "train_seed": train_seed,
                "total_timesteps": ROBUST_STEPS,
                "n_eval_episodes": ROBUST_EVAL_EPISODES,
                "seed_start": SELECTION_EVAL_SEED_START,
                "label": candidate_id,
                "save_path": str(
                    STUDY_DIR / "robustness" / candidate_id / f"seed_{train_seed}.zip"
                ),
            })

    print(f"Validation multi-seed : {len(jobs)} entraînements")
    results = run_jobs(jobs, workers)

    rows = pd.DataFrame(results).rename(columns={"label": "candidate_id"})
    rows.to_csv(STUDY_DIR / "robustness.csv", index=False)

    summary = (
        rows.groupby("candidate_id")["mean_reward"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    for candidate_id, candidate in finalists.items():
        mask = summary["candidate_id"] == candidate_id
        summary.loc[mask, "source"] = candidate["source"]
        summary.loc[mask, "search_mean_reward"] = candidate["search_mean_reward"]
    summary.to_csv(STUDY_DIR / "robustness_summary.csv", index=False)
    print(summary.to_string(index=False))

    winner_id = summary.iloc[0]["candidate_id"]
    winner = finalists[winner_id]
    payload = {
        "winner_id": winner_id,
        "source": winner["source"],
        "parameters": winner["parameters"],
        "search_mean_reward": winner["search_mean_reward"],
        "robust_mean_reward": float(summary.iloc[0]["mean"]),
        "robust_std_reward": float(summary.iloc[0]["std"]),
        "robust_min_reward": float(summary.iloc[0]["min"]),
        "train_seeds": list(ROBUST_TRAIN_SEEDS),
        "robust_timesteps": ROBUST_STEPS,
        "selection_eval_seed_start": SELECTION_EVAL_SEED_START,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (STUDY_DIR / "selected_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    """Enchaîne le raffinement de gamma et la validation multi-seed, puis écrit le réglage retenu."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    if args.fast:
        _fast_mode()

    study = optuna.load_study(study_name=STUDY_NAME, storage=build_storage())
    started = time.perf_counter()
    gamma_summary = phase_gamma_focus(study, args.workers)
    phase_robustness(study, gamma_summary, args.workers)
    print(f"Sélection terminée en {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
