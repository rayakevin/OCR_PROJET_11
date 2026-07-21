#!/usr/bin/env python3
"""Recherche d'hyperparamètres PPO pour LunarLander-v3.

Ce module remplace la première campagne Optuna de la mission, qui souffrait de
trois défauts méthodologiques :

1. `gamma` était échantillonné dans un ensemble catégoriel `[0.99, 0.995,
   0.999]`, ce qui interdisait de trouver un optimum entre ces trois points ;
2. chaque essai n'était entraîné que sur **une seule seed**, si bien que le
   score d'un essai mesurait autant le bruit d'initialisation que la qualité du
   réglage — le classement s'inversait ensuite à la validation multi-seed ;
3. aucun élagage n'était appliqué : le budget complet était dépensé même sur des
   configurations manifestement divergentes.

Les trois corrections apportées ici :

1. `gamma` est échantillonné en loi log-uniforme sur `1 - gamma`, la
   paramétrisation usuelle pour un facteur d'actualisation. Elle couvre
   uniformément les ordres de grandeur de l'horizon effectif `1 / (1 - gamma)`,
   soit ici de 50 à 10 000 pas ;
2. chaque essai est entraîné sur plusieurs seeds et noté par la **moyenne** de
   leurs évaluations, ce qui réduit la variance du signal optimisé ;
3. un `MedianPruner` interrompt les essais dont la trajectoire d'apprentissage
   est en dessous de la médiane des essais déjà observés au même stade.

L'étude est persistée dans SQLite, ce qui permet de répartir les essais sur
plusieurs processus travaillant en parallèle sur le même stockage — le schéma
d'optimisation distribuée documenté par Optuna.

Utilisation :

    uv run python ASTRODYNAMICS/optuna_search.py --trials 120 --workers 16
    uv run python ASTRODYNAMICS/optuna_search.py --fast          # test rapide
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Triton doit être importé avant Box2D/pygame dans l'environnement local.
try:  # pragma: no cover - dépend de l'installation locale
    import triton  # noqa: F401
except ImportError:  # pragma: no cover
    pass

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym

ENV_ID = "LunarLander-v3"
MISSION_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MISSION_DIR / "artifacts"
STUDY_DIR = ARTIFACTS_DIR / "optuna" / "ppo_lunarlander"

STUDY_NAME = "ppo_lunarlander"

# Séparation stricte des seeds : la recherche n'observe jamais les épisodes qui
# serviront à la sélection (5000+) ni à l'évaluation finale (10000+).
SEARCH_EVAL_SEED_START = 2_000
TRAIN_SEED_BASE = 700


@dataclass
class SearchConfig:
    """Budget d'une campagne de recherche."""

    n_trials: int = 120
    timesteps_per_seed: int = 100_000
    train_seeds: tuple[int, ...] = (TRAIN_SEED_BASE, TRAIN_SEED_BASE + 1)
    eval_episodes: int = 20
    intermediate_eval_episodes: int = 10
    eval_interval: int = 25_000
    startup_trials: int = 15
    warmup_steps: int = 50_000
    extra_search_space: dict = field(default_factory=dict)

    @property
    def total_timesteps(self) -> int:
        return self.timesteps_per_seed * len(self.train_seeds)


FAST_CONFIG = SearchConfig(
    n_trials=6,
    timesteps_per_seed=8_192,
    train_seeds=(TRAIN_SEED_BASE,),
    eval_episodes=5,
    intermediate_eval_episodes=3,
    eval_interval=4_096,
    startup_trials=2,
    warmup_steps=4_096,
)


def gamma_from_trial(trial: optuna.Trial) -> float:
    """Échantillonne gamma via `1 - gamma` en loi log-uniforme.

    Un tirage uniforme sur `gamma` concentrerait presque toute la masse loin de
    1, alors que le comportement de PPO change surtout dans la dernière décimale.
    Travailler sur `1 - gamma` en échelle logarithmique donne autant de poids à
    l'horizon court (gamma≈0.98) qu'à l'horizon long (gamma≈0.9999).
    """
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 1e-4, 2e-2, log=True)
    return 1.0 - one_minus_gamma


def sample_ppo_parameters(trial: optuna.Trial) -> dict:
    """Espace de recherche PPO, aligné sur la ressource fournie par la mission."""
    parameters = {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 3e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_epochs": trial.suggest_categorical("n_epochs", [3, 5, 10, 20]),
        "gamma": gamma_from_trial(trial),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.90, 0.95, 0.98, 1.0]),
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 2e-2, log=True),
        "vf_coef": trial.suggest_categorical("vf_coef", [0.25, 0.5, 1.0]),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 1.0]),
    }
    # PPO exige que la taille de lot divise le tampon de collecte.
    if parameters["batch_size"] > parameters["n_steps"]:
        parameters["batch_size"] = parameters["n_steps"]
    return parameters


def evaluate_deterministic(model, n_episodes: int, seed_start: int) -> float:
    """Évalue la politique sur des épisodes à seeds fixées.

    Les mêmes seeds sont utilisées pour tous les essais, afin que les scores
    soient comparables entre eux et non tributaires du tirage d'évaluation.
    """
    env = Monitor(gym.make(ENV_ID))
    env.reset(seed=seed_start)
    try:
        mean_reward, _ = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            warn=False,
        )
    finally:
        env.close()
    return float(mean_reward)


def train_and_score_seed(
    parameters: dict,
    train_seed: int,
    config: SearchConfig,
    trial: optuna.Trial | None,
    seed_index: int,
) -> float:
    """Entraîne une seed et renvoie son score, en signalant l'avancement à Optuna.

    L'entraînement est découpé en tranches de `config.eval_interval` pas. Après
    chaque tranche, une évaluation courte est remontée à Optuna par
    `trial.report`, ce qui permet au `MedianPruner` d'interrompre un essai
    manifestement moins bon que la médiane des essais déjà vus au même stade.
    """
    train_env = make_vec_env(ENV_ID, n_envs=1, seed=train_seed)
    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            seed=train_seed,
            device="cpu",
            verbose=0,
            **parameters,
        )
        trained = 0
        while trained < config.timesteps_per_seed:
            chunk = min(config.eval_interval, config.timesteps_per_seed - trained)
            model.learn(total_timesteps=chunk, reset_num_timesteps=(trained == 0))
            trained += chunk

            if trial is not None:
                intermediate = evaluate_deterministic(
                    model,
                    config.intermediate_eval_episodes,
                    SEARCH_EVAL_SEED_START,
                )
                global_step = seed_index * config.timesteps_per_seed + trained
                trial.report(intermediate, step=global_step)
                # L'élagage n'est autorisé qu'après une phase d'échauffement :
                # PPO progresse rarement avant quelques dizaines de milliers de pas.
                if global_step >= config.warmup_steps and trial.should_prune():
                    raise optuna.TrialPruned()

        return evaluate_deterministic(
            model, config.eval_episodes, SEARCH_EVAL_SEED_START
        )
    finally:
        train_env.close()


def make_objective(config: SearchConfig):
    """Construit la fonction objectif : moyenne des scores sur plusieurs seeds."""

    def objective(trial: optuna.Trial) -> float:
        parameters = sample_ppo_parameters(trial)
        started = time.perf_counter()
        scores = []
        for seed_index, train_seed in enumerate(config.train_seeds):
            scores.append(
                train_and_score_seed(parameters, train_seed, config, trial, seed_index)
            )

        trial.set_user_attr("seed_scores", scores)
        trial.set_user_attr("score_std", float(np.std(scores)))
        trial.set_user_attr("score_min", float(np.min(scores)))
        trial.set_user_attr("gamma", parameters["gamma"])
        trial.set_user_attr("duration_s", time.perf_counter() - started)
        trial.set_user_attr("n_train_seeds", len(config.train_seeds))
        return float(np.mean(scores))

    return objective


def build_storage() -> optuna.storages.RDBStorage:
    """Stockage SQLite partagé, tolérant aux accès concurrents des workers."""
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    db_path = (STUDY_DIR / "study.db").resolve()
    return optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path.as_posix()}",
        engine_kwargs={"connect_args": {"timeout": 120}},
    )


def create_or_load_study(config: SearchConfig) -> optuna.Study:
    """Crée l'étude, ou la recharge si des essais existent déjà."""
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=build_storage(),
        sampler=TPESampler(seed=42, n_startup_trials=config.startup_trials),
        pruner=MedianPruner(
            n_startup_trials=config.startup_trials,
            n_warmup_steps=config.warmup_steps,
        ),
        direction="maximize",
        load_if_exists=True,
    )


def run_worker(worker_index: int, n_trials: int, fast: bool) -> None:
    """Point d'entrée d'un processus worker : consomme `n_trials` essais."""
    torch.set_num_threads(1)
    config = FAST_CONFIG if fast else SearchConfig()
    study = optuna.load_study(study_name=STUDY_NAME, storage=build_storage())
    study.optimize(
        make_objective(config),
        n_trials=n_trials,
        catch=(ValueError, RuntimeError),
    )


def export_results(config: SearchConfig) -> dict:
    """Écrit `trials.csv`, l'importance des paramètres et le meilleur essai."""
    import pandas as pd

    study = optuna.load_study(study_name=STUDY_NAME, storage=build_storage())
    trials_df = study.trials_dataframe()
    trials_df.to_csv(STUDY_DIR / "trials.csv", index=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    importance_path = STUDY_DIR / "parameter_importance.csv"
    try:
        from optuna.importance import PedAnovaImportanceEvaluator

        importance = optuna.importance.get_param_importances(
            study, evaluator=PedAnovaImportanceEvaluator()
        )
        pd.DataFrame(
            {"parameter": list(importance), "importance": list(importance.values())}
        ).to_csv(importance_path, index=False)
    except Exception as error:  # pragma: no cover - dépend du nombre d'essais
        print(f"Importance non calculée : {error}")

    best = study.best_trial
    payload = {
        "study_name": STUDY_NAME,
        "n_trials": len(study.trials),
        "n_complete": len(completed),
        "n_pruned": len(pruned),
        "timesteps_per_seed": config.timesteps_per_seed,
        "train_seeds": list(config.train_seeds),
        "eval_episodes": config.eval_episodes,
        "search_eval_seed_start": SEARCH_EVAL_SEED_START,
        "best_trial_number": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_gamma": best.user_attrs.get("gamma"),
        "best_seed_scores": best.user_attrs.get("seed_scores"),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (STUDY_DIR / "best_trial_search.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def resolved_parameters(params: dict) -> dict:
    """Traduit les paramètres bruts d'un essai en arguments PPO utilisables."""
    parameters = {k: v for k, v in params.items() if k != "one_minus_gamma"}
    parameters["gamma"] = 1.0 - params["one_minus_gamma"]
    if parameters.get("batch_size", 0) > parameters.get("n_steps", 1 << 30):
        parameters["batch_size"] = parameters["n_steps"]
    return parameters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=120, help="Nombre total d'essais visés.")
    parser.add_argument("--workers", type=int, default=16, help="Processus parallèles.")
    parser.add_argument("--fast", action="store_true", help="Budget réduit, pour valider le code.")
    parser.add_argument("--export-only", action="store_true", help="N'exporte que les résultats.")
    args = parser.parse_args()

    config = FAST_CONFIG if args.fast else SearchConfig(n_trials=args.trials)
    study = create_or_load_study(config)
    done = len([t for t in study.trials if t.state != optuna.trial.TrialState.RUNNING])
    target = config.n_trials if args.fast else args.trials
    remaining = max(0, target - done)

    print(f"Étude          : {STUDY_NAME}")
    print(f"Essais présents: {done} / {target}")

    if not args.export_only and remaining:
        import multiprocessing as mp

        workers = max(1, min(args.workers, remaining))
        per_worker = [remaining // workers] * workers
        for i in range(remaining % workers):
            per_worker[i] += 1
        print(f"Lancement de {workers} workers ({per_worker[0]}-{per_worker[-1]} essais chacun)")

        context = mp.get_context("spawn")
        processes = [
            context.Process(target=run_worker, args=(i, count, args.fast))
            for i, count in enumerate(per_worker)
            if count
        ]
        started = time.perf_counter()
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        print(f"Recherche terminée en {(time.perf_counter() - started) / 60:.1f} min")

    payload = export_results(config)
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
