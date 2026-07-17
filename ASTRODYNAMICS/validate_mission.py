"""Contrôle automatique des critères techniques de la mission Eagle-1."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def video_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def main() -> int:
    notebook = json.loads((ROOT / "NB.ipynb").read_text(encoding="utf-8"))
    notebook_errors = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    summary_path = ARTIFACTS / "evaluations/ppo_gamma_extended_final_100/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = pd.read_csv(summary_path.parent / "episodes.csv")
    steps = pd.read_csv(summary_path.parent / "steps.csv", nrows=10)
    video = ARTIFACTS / "video/eagle1-success-episode-0.mp4"
    duration = video_duration(video)

    checks = {
        "notebook_executed": all(cell.get("execution_count") is not None for cell in code_cells),
        "notebook_without_error": not notebook_errors,
        "baseline_50_episodes": json.loads(
            (ARTIFACTS / "evaluations/dqn_default_baseline/summary.json").read_text()
        )["n_episodes"] >= 50,
        "final_100_episodes": summary["n_episodes"] == 100 and len(episodes) == 100,
        "mean_reward_above_200": summary["mean_reward"] > 200,
        "best_model_saved": (ARTIFACTS / "models/ppo_gamma_extended/best_model.zip").exists(),
        "detailed_step_logs": {"action", "reward", "next_state_1", "next_state_4"} <= set(steps.columns),
        "video_duration_20_to_30_seconds": 20 <= duration <= 30,
        "video_shows_success_seed": bool(episodes.loc[episodes["seed"] == 10_000, "success"].iloc[0]),
        "api_source": (ROOT / "api/app.py").exists(),
        "gui_source": (ROOT / "gui/app.py").exists(),
        "dashboard_source": (ROOT / "dashboard/app.py").exists(),
        "dashboard_interactive": all(
            token in (ROOT / "dashboard/app.py").read_text()
            for token in ["st.tabs", "st.multiselect", "st.slider", "st.plotly_chart"]
        ),
        "gui_has_no_model_loading": "stable_baselines3" not in (ROOT / "gui/app.py").read_text(),
        "global_documentation": (ROOT / "README.md").exists(),
    }

    failed = [name for name, passed in checks.items() if not passed]
    print(json.dumps({
        "checks": checks,
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "failed": failed,
        "final_mean_reward": summary["mean_reward"],
        "final_std_reward": summary["std_reward"],
        "success_rate": summary["success_rate"],
        "video_duration_s": duration,
    }, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

