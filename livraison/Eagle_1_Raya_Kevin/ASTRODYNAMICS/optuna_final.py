#!/usr/bin/env python3
"""Entraînement long et évaluation finale du réglage PPO retenu.

Le réglage sélectionné par :mod:`optuna_select` est réentraîné sur un budget
long, avec **plusieurs seeds neuves** — distinctes de celles de la validation de
robustesse, pour ne pas rejouer un tirage déjà favorable.

Le protocole sépare strictement trois jeux de seeds :

- recherche : 2000+ (dans :mod:`optuna_search`) ;
- sélection : 5000+ — c'est ici qu'on choisit **lequel** des entraînements longs
  devient le pilote de la mission ;
- évaluation finale : 10000+ — n'est utilisée qu'une seule fois, sur le seul
  modèle retenu.

Choisir le modèle final d'après ses résultats sur les 100 épisodes réservés
reviendrait à sélectionner sur le jeu de test : la moyenne annoncée serait alors
optimiste. La sélection se fait donc exclusivement sur les seeds 5000+.

Utilisation :

    uv run python ASTRODYNAMICS/optuna_final.py --workers 3
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

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from optuna_search import STUDY_DIR, evaluate_deterministic  # noqa: E402

ENV_ID = "LunarLander-v3"
MISSION_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MISSION_DIR / "artifacts"

FINAL_TRAIN_SEEDS = (900, 901, 902)
SELECTION_SEED_START = 5_000
SELECTION_EPISODES = 50
FINAL_SEED_START = 10_000
FINAL_EPISODES = 100
FINAL_STEPS = 1_500_000

CANDIDATE_PREFIX = "ppo_optuna_s"
WINNER_ID = "ppo_optuna"


def train_final(payload: dict) -> dict:
    """Entraîne un candidat final et l'évalue sur le jeu de sélection."""
    torch.set_num_threads(1)
    experiment_id = payload["experiment_id"]
    parameters = payload["parameters"]
    seed = payload["seed"]
    total_timesteps = payload["total_timesteps"]

    model_dir = ARTIFACTS_DIR / "models" / experiment_id
    monitor_dir = ARTIFACTS_DIR / "monitor" / experiment_id
    eval_dir = ARTIFACTS_DIR / "evaluations" / experiment_id / "during_training"
    for directory in (model_dir, monitor_dir, eval_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(ENV_ID, n_envs=1, seed=seed, monitor_dir=str(monitor_dir))
    eval_env = Monitor(gym.make(ENV_ID))
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(eval_dir),
        eval_freq=25_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=0,
    )

    started = time.perf_counter()
    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            seed=seed,
            device="cpu",
            verbose=0,
            tensorboard_log=str(ARTIFACTS_DIR / "tensorboard" / experiment_id),
            **parameters,
        )
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
        duration = time.perf_counter() - started
        model.save(model_dir / "final_model.zip")
    finally:
        train_env.close()
        eval_env.close()

    # `EvalCallback` conserve le meilleur point de contrôle rencontré : c'est lui
    # qui est évalué, comme pour tous les autres modèles de la mission.
    best_path = model_dir / "best_model.zip"
    path_to_load = best_path if best_path.exists() else model_dir / "final_model.zip"
    best_model = PPO.load(path_to_load, device="cpu")
    selection_score = evaluate_deterministic(
        best_model, SELECTION_EPISODES, SELECTION_SEED_START
    )

    config = {
        "experiment_id": experiment_id,
        "algorithm": "PPO",
        "env_id": ENV_ID,
        "seed": seed,
        "n_envs": 1,
        "total_timesteps": total_timesteps,
        "training_duration_s": duration,
        "hyperparameters": parameters,
        "best_model_path": str(path_to_load),
        "selection_mean_reward": selection_score,
        "selection_seed_start": SELECTION_SEED_START,
        "selection_episodes": SELECTION_EPISODES,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    config_path = ARTIFACTS_DIR / "configs" / f"{experiment_id}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--steps", type=int, default=FINAL_STEPS)
    args = parser.parse_args()

    selected = json.loads((STUDY_DIR / "selected_config.json").read_text())
    parameters = selected["parameters"]
    print(f"Réglage retenu : {selected['winner_id']}")
    print(json.dumps(parameters, ensure_ascii=False, indent=2))

    jobs = [
        {
            "experiment_id": f"{CANDIDATE_PREFIX}{seed}",
            "parameters": parameters,
            "seed": seed,
            "total_timesteps": args.steps,
        }
        for seed in FINAL_TRAIN_SEEDS
    ]

    import multiprocessing as mp

    started = time.perf_counter()
    context = mp.get_context("spawn")
    with context.Pool(processes=max(1, min(args.workers, len(jobs)))) as pool:
        configs = pool.map(train_final, jobs)
    print(f"Entraînements longs terminés en {(time.perf_counter() - started) / 60:.1f} min")

    table = pd.DataFrame(configs)[
        ["experiment_id", "seed", "selection_mean_reward", "training_duration_s"]
    ].sort_values("selection_mean_reward", ascending=False)
    print(table.to_string(index=False))
    table.to_csv(STUDY_DIR / "final_candidates.csv", index=False)

    winner = table.iloc[0]
    print(f"\nModèle retenu sur le jeu de sélection : {winner['experiment_id']}")

    winner_dir = ARTIFACTS_DIR / "models" / WINNER_ID
    winner_dir.mkdir(parents=True, exist_ok=True)
    source_dir = ARTIFACTS_DIR / "models" / winner["experiment_id"]
    for name in ("best_model.zip", "final_model.zip"):
        source = source_dir / name
        if source.exists():
            (winner_dir / name).write_bytes(source.read_bytes())

    payload = {
        "winner_experiment_id": winner["experiment_id"],
        "winner_seed": int(winner["seed"]),
        "selection_mean_reward": float(winner["selection_mean_reward"]),
        "selection_seed_start": SELECTION_SEED_START,
        "candidates": table.to_dict(orient="records"),
        "hyperparameters": parameters,
        "total_timesteps": args.steps,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (STUDY_DIR / "final_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
