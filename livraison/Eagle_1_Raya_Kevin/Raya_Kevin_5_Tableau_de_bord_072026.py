"""Dashboard Streamlit alimenté par les logs réels de la mission Eagle-1.

Le dashboard est conçu pour être lisible par quelqu'un qui découvre le projet :
chaque onglet répond à une question formulée en français courant, chaque
indicateur est accompagné de ce qu'il signifie, et les identifiants techniques
des expériences sont traduits en libellés lisibles.
"""

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
TARGET_REWARD = 200

# Palette validée par scripts/validate_palette.js (mode clair, surface #ffffff).
# Les deux premiers créneaux passent tous les contrôles, y compris la séparation
# en vision daltonienne (ΔE 24,7). Le vert/rouge intuitif pour réussite/échec a
# été écarté des graphiques : il échoue au contrôle deutéranopie (ΔE 4,1). Il ne
# sert donc que dans les textes, toujours accompagné d'une icône.
SERIES_1 = "#2a78d6"   # bleu — série principale / réussite
SERIES_2 = "#eb6834"   # orange — série de comparaison / échec
SERIES_3 = "#1baf7a"   # aqua — troisième série (télémétrie)
NEUTRAL = "#9aa4ac"    # gris de mise en retrait
INK = "#17242d"
TARGET_LINE = "#5b6a73"
BRAND_COLORS = [SERIES_1, SERIES_2, SERIES_3, "#eda100", "#e87ba4"]

APP_STYLES = """
<style>
    :root {
        --ad-ink: #17242d;
        --ad-navy: #12384a;
        --ad-blue: #2a78d6;
        --ad-orange: #eb6834;
        --ad-line: #d7dde1;
        --ad-surface: #f7f8f9;
    }
    .stApp { background: var(--ad-surface); color: var(--ad-ink); }
    [data-testid="stHeader"] { background: rgba(247, 248, 249, 0.96); }
    [data-testid="stSidebar"] { background: #eceff1; border-right: 1px solid var(--ad-line); }
    .block-container { max-width: 1480px; padding-top: 2rem; padding-bottom: 3rem; }
    .mission-header { border-top: 4px solid var(--ad-orange); border-bottom: 1px solid var(--ad-line); padding: 1rem 0 1.1rem; margin-bottom: 1.4rem; }
    .mission-header .kicker { color: var(--ad-blue); font-size: .74rem; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; }
    .mission-header h1 { color: var(--ad-navy); font-size: 2.1rem; font-weight: 650; letter-spacing: -.02em; margin: .25rem 0; }
    .mission-header p { color: #52636d; margin: 0; font-size: .96rem; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid var(--ad-line); border-radius: 3px; padding: .8rem .9rem; }
    [data-testid="stMetricLabel"] { color: #5b6a73; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--ad-line); }
    .stTabs [data-baseweb="tab"] { border-radius: 0; padding: .65rem 1.05rem; }
    .stTabs [aria-selected="true"] { color: var(--ad-navy); border-bottom: 3px solid var(--ad-orange); }
    h2, h3 { color: var(--ad-navy); font-weight: 650; }
    .hero-figure { font-size: 3.4rem; font-weight: 680; color: var(--ad-navy); line-height: 1; letter-spacing: -.03em; }
    .hero-caption { color: #52636d; font-size: .95rem; margin-top: .35rem; }
    .readme { background: #fff; border: 1px solid var(--ad-line); border-left: 3px solid var(--ad-blue); padding: .9rem 1.1rem; margin-bottom: 1.2rem; }
    .readme p { margin: .25rem 0; color: #3d4d57; font-size: .95rem; }
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

METRIC_HELP = {
    "mean": "Moyenne des récompenses sur tous les épisodes. Au-dessus de 200, "
            "l'atterrissage est considéré comme maîtrisé.",
    "std": "Écart-type : à quel point les épisodes se ressemblent. Plus il est "
           "bas, plus le pilote est régulier.",
    "success": "Part des épisodes terminés par un atterrissage stable entre les "
               "drapeaux.",
    "worst": "Récompense du plus mauvais épisode. C'est l'indicateur de "
             "fiabilité : une bonne moyenne peut cacher un vol raté.",
    "length": "Nombre de pas de simulation avant la fin de l'épisode. Plus court "
              "signifie un atterrissage plus direct.",
    "fuel": "Estimation de la consommation, calculée à partir des allumages "
            "moteur. Plus bas est mieux.",
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
    if evaluation_id.endswith("_final_100"):
        return f"{base} — évaluation finale"
    if evaluation_id.endswith("_selection"):
        return f"{base} — jeu de sélection"
    if evaluation_id.endswith("_screen"):
        return f"{base} — test court"
    if evaluation_id.endswith("_tuning"):
        return f"{base} — réglage"
    return base


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
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 40, "r": 24, "t": 68 if has_title else 30, "b": 40},
        hoverlabel={"bgcolor": "#ffffff", "font_color": INK},
        hovermode="closest",
    )
    if has_title:
        fig.update_layout(
            title={
                "text": title_text,
                "font": {"color": "#12384a", "size": 17},
                "x": 0.01,
                "xanchor": "left",
            }
        )
    else:
        # Plotly peut afficher le mot « undefined » si le champ titre est absent.
        fig.update_layout(title={"text": ""})
    if height is not None:
        fig.update_layout(height=height)
    # Grille discrète : elle situe, elle ne doit pas concurrencer les données.
    fig.update_xaxes(gridcolor="#eceff1", zerolinecolor="#d7dde1", linecolor="#d7dde1")
    fig.update_yaxes(gridcolor="#eceff1", zerolinecolor="#d7dde1", linecolor="#d7dde1")
    return fig


def add_target_line(fig, *, axis: str = "y") -> None:
    """Trace le seuil de 200, référence commune à toutes les vues.

    L'étiquette est ancrée du côté le moins chargé de chaque orientation :
    sur un graphique en barres horizontales, le haut est occupé par la barre
    du meilleur candidat, et l'annotation viendrait s'y superposer.
    """
    kwargs = dict(line_dash="dot", line_color=TARGET_LINE, line_width=1.5,
                  annotation_text="objectif 200", annotation_font_size=11,
                  annotation_font_color=TARGET_LINE)
    if axis == "y":
        fig.add_hline(y=TARGET_REWARD, annotation_position="top left", **kwargs)
    else:
        fig.add_vline(x=TARGET_REWARD, annotation_position="bottom right", **kwargs)


def thin_bars(fig, gap: float = 0.45):
    """Amincit les barres : un bloc épais et saturé écrase la lecture."""
    fig.update_layout(bargap=gap)
    return fig


def candidate_label(candidate_id: str) -> str:
    """Traduit un identifiant de finaliste Optuna en libellé lisible."""
    if candidate_id.startswith("broad_t"):
        return f"Recherche large — essai {int(candidate_id.removeprefix('broad_t'))}"
    if candidate_id.startswith("gamma_g"):
        return f"Grille gamma — point {int(candidate_id.removeprefix('gamma_g'))}"
    return candidate_id


def experiment_phase(evaluation_id: str) -> str:
    """Range chaque mesure dans l'étape de mission qui lui correspond.

    L'ordre des tests compte : un identifiant peut cumuler plusieurs mots-clés
    (par exemple `ppo_optuna_gamma_t003_s242_robustness`), et c'est le premier
    test satisfait qui décide de la phase.
    """
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


def load_registry(artifacts_dir: Path = ARTIFACTS_DIR) -> pd.DataFrame:
    path = Path(artifacts_dir) / "experiment_registry.csv"
    registry = pd.read_csv(path)
    registry["phase"] = registry["evaluation_id"].map(experiment_phase)
    registry["success_percent"] = 100 * registry["success_rate"]
    registry["label"] = registry["evaluation_id"].map(human_label)
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


def load_configs(artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, dict]:
    """Charge les configurations d'entraînement, indexées par expérience."""
    config_dir = Path(artifacts_dir) / "configs"
    configs = {}
    for path in sorted(config_dir.glob("*.json")):
        try:
            configs[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return configs


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


def outcome_colors(values) -> dict[str, str]:
    """Associe une couleur validée à chaque issue d'épisode.

    Le couple vert/rouge est écarté : sa séparation en deutéranopie est de
    ΔE 4,1, très en dessous du seuil de 8. Le couple bleu/orange mesure 24,7 et
    reste lisible pour tout le monde.
    """
    mapping = {"success": SERIES_1, "crash": SERIES_2, "truncated": NEUTRAL}
    return {value: mapping.get(value, NEUTRAL) for value in values}


OUTCOME_LABELS = {"success": "Réussite", "crash": "Échec", "truncated": "Temps écoulé"}


def render_essentials(registry: pd.DataFrame, artifacts_dir: Path) -> None:
    """Répond à « ça marche, oui ou non ? » pour un lecteur qui découvre."""
    final = registry[registry["evaluation_id"] == FINAL_EVALUATION_ID]
    if final.empty:
        st.warning("L'évaluation finale n'est pas disponible.")
        return
    final = final.iloc[0]
    episodes = load_episodes(FINAL_EVALUATION_ID, artifacts_dir)

    st.markdown(
        """
        <div class="readme">
        <p><strong>De quoi s'agit-il ?</strong> Un pilote automatique doit poser
        un module lunaire entre deux drapeaux, dans le simulateur
        <code>LunarLander-v3</code>. À chaque instant il choisit parmi quatre
        actions : ne rien faire, ou allumer l'un des trois moteurs.</p>
        <p><strong>Comment lit-on le score ?</strong> Chaque atterrissage rapporte
        une récompense : positive s'il est réussi et économe, négative en cas de
        crash ou de gaspillage de carburant. <strong>La mission est validée
        au-dessus de 200 de moyenne sur 100 atterrissages.</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f'<div class="hero-figure">{final["mean_reward"]:.0f}</div>'
            f'<div class="hero-caption">récompense moyenne sur 100 atterrissages<br>'
            f'objectif : 200 — dépassé de {final["mean_reward"] - TARGET_REWARD:.0f} points</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        cols = st.columns(3)
        cols[0].metric("Régularité (écart-type)", f"± {final['std_reward']:.0f}",
                       help=METRIC_HELP["std"])
        cols[1].metric("Atterrissages réussis", f"{final['success_percent']:.0f} %",
                       help=METRIC_HELP["success"])
        cols[2].metric("Pire épisode", f"{episodes['total_reward'].min():.0f}",
                       help=METRIC_HELP["worst"])
        cols = st.columns(3)
        cols[0].metric("Durée moyenne", f"{final['mean_episode_length']:.0f} pas",
                       help=METRIC_HELP["length"])
        cols[1].metric("Carburant estimé", f"{final['mean_fuel_proxy']:.1f}",
                       help=METRIC_HELP["fuel"])
        cols[2].metric("Épisodes évalués", f"{int(final['n_episodes'])}",
                       help="Tous joués sur des seeds réservées, jamais vues pendant l'entraînement.")

    st.markdown("### Le pilote final face à ses points de comparaison")
    st.caption(
        "Chaque barre est une politique évaluée dans les mêmes conditions. "
        "Le pilote retenu est en couleur, les points de comparaison en gris."
    )

    reference_ids = [
        "random_baseline", "dqn_default_baseline", "ppo_default_baseline",
        "ppo_gamma_extended_selection", FINAL_EVALUATION_ID,
    ]
    comparison = registry[registry["evaluation_id"].isin(reference_ids)].copy()
    comparison = comparison.sort_values("mean_reward")
    # Emphase : une seule série porte l'information, le reste est contexte.
    comparison["couleur"] = np.where(
        comparison["evaluation_id"] == FINAL_EVALUATION_ID, SERIES_1, NEUTRAL
    )

    fig = go.Figure(
        go.Bar(
            x=comparison["mean_reward"],
            y=comparison["label"],
            orientation="h",
            marker_color=comparison["couleur"],
            error_x=dict(type="data", array=comparison["std_reward"], color="#b6bec4", thickness=1.5),
            # Pas d'étiquette chiffrée sur chaque barre : la moustache
            # d'écart-type traverse la zone du texte, et l'axe suffit à situer
            # les valeurs. Le détail exact reste accessible au survol.
            hovertemplate="%{y}<br>moyenne %{x:.1f}<extra></extra>",
        )
    )
    add_target_line(fig, axis="x")
    fig.update_layout(xaxis_title="Récompense moyenne", yaxis_title="", height=380)
    st.plotly_chart(thin_bars(style_figure(fig)), width="stretch")
    st.caption(
        "Les barres d'erreur montrent l'écart-type : une barre longue signifie "
        "des résultats dispersés d'un atterrissage à l'autre."
    )


def render_phases(registry: pd.DataFrame, artifacts_dir: Path) -> None:
    """Montre la progression étape par étape, dans l'ordre chronologique."""
    st.markdown("### Les six étapes de la mission")
    st.caption(
        "La mission progresse par étapes, chacune répondant à une question "
        "précise. Le tableau ci-dessous donne le meilleur résultat atteint à "
        "chaque étape."
    )

    descriptions = {
        PHASE_ORDER[0]: "Que vaut un pilote qui appuie au hasard ? C'est le plancher.",
        PHASE_ORDER[1]: "Que valent DQN et PPO avec leurs réglages d'usine, sans aucun ajustement ?",
        PHASE_ORDER[2]: "Quel effet a un paramètre modifié seul, toutes choses égales par ailleurs ?",
        PHASE_ORDER[3]: "Que trouve une recherche automatique explorant dix paramètres à la fois ?",
        PHASE_ORDER[4]: "Rejoués sur les mêmes épisodes, lequel de tous les candidats gagne ?",
        PHASE_ORDER[5]: "Le modèle retenu, mesuré une seule fois sur 100 épisodes réservés.",
    }

    ordered = [phase for phase in PHASE_ORDER if phase in set(registry["phase"])]
    rows = []
    for phase in ordered:
        subset = registry[registry["phase"] == phase]
        best = subset.loc[subset["mean_reward"].idxmax()]
        rows.append({
            "Étape": phase,
            "Question posée": descriptions.get(phase, ""),
            "Meilleur de l'étape": best["label"],
            "Récompense moyenne": round(best["mean_reward"], 1),
            "Écart-type": round(best["std_reward"], 1),
            "Réussite": f"{best['success_percent']:.0f} %",
        })
    phase_table = pd.DataFrame(rows)
    st.dataframe(phase_table, width="stretch", hide_index=True)

    fig = go.Figure(
        go.Bar(
            x=phase_table["Étape"],
            y=phase_table["Récompense moyenne"],
            marker_color=[NEUTRAL] * (len(phase_table) - 1) + [SERIES_1],
            text=[f"{value:.0f}" for value in phase_table["Récompense moyenne"]],
            textposition="outside",
            hovertemplate="%{x}<br>meilleur : %{y:.1f}<extra></extra>",
        )
    )
    add_target_line(fig)
    fig.update_layout(
        title="Meilleur résultat atteint à chaque étape",
        xaxis_title="", yaxis_title="Récompense moyenne", height=430,
    )
    st.plotly_chart(thin_bars(style_figure(fig), 0.55), width="stretch")

    st.markdown("### Progression pendant l'entraînement")
    st.caption(
        "Chaque point est une évaluation faite pendant l'apprentissage. "
        "L'axe horizontal compte les transitions vues par l'agent : plus on va "
        "à droite, plus il a d'expérience."
    )

    curves = load_learning_curves(artifacts_dir)
    if curves.empty:
        st.warning("Aucune courbe d'entraînement disponible.")
        return

    curves = curves.copy()
    curves["label"] = curves["experiment"].map(lambda name: EXPERIMENT_LABELS.get(name, name))
    default = [name for name in ["dqn_default", "ppo_default", "ppo_gamma_extended", "ppo_optuna"]
               if name in set(curves["experiment"])]
    chosen = st.multiselect(
        "Entraînements affichés",
        options=sorted(curves["experiment"].unique()),
        default=default,
        format_func=lambda name: EXPERIMENT_LABELS.get(name, name),
    )
    if not chosen:
        st.info("Sélectionnez au moins un entraînement.")
        return

    selected = curves[curves["experiment"].isin(chosen)]
    fig = go.Figure()
    for index, experiment in enumerate(chosen):
        part = selected[selected["experiment"] == experiment].sort_values("timestep")
        # Le pilote final est mis en avant ; les autres servent de contexte.
        is_final = experiment.startswith("ppo_optuna")
        fig.add_trace(go.Scatter(
            x=part["timestep"], y=part["mean_reward"], mode="lines+markers",
            name=EXPERIMENT_LABELS.get(experiment, experiment),
            line=dict(width=2.5 if is_final else 2,
                      color=SERIES_1 if is_final else BRAND_COLORS[(index % 4) + 1]),
            marker=dict(size=6),
            hovertemplate="%{fullData.name}<br>%{x:,} transitions<br>%{y:.1f}<extra></extra>",
        ))
    add_target_line(fig)
    fig.update_layout(xaxis_title="Transitions d'entraînement",
                      yaxis_title="Récompense moyenne", height=460)
    st.plotly_chart(style_figure(fig), width="stretch")


def render_comparison(registry: pd.DataFrame, artifacts_dir: Path) -> None:
    """Compare deux modèles terme à terme : chiffres, distribution, réglages."""
    st.markdown("### Comparer deux modèles")
    st.caption(
        "Choisissez deux évaluations : le tableau de bord aligne leurs chiffres, "
        "leurs distributions de récompense et leurs hyperparamètres."
    )

    available = sorted(
        set(registry["evaluation_id"]) & set(list_episode_evaluations(artifacts_dir))
    )
    if len(available) < 2:
        st.warning("Il faut au moins deux évaluations comparables.")
        return

    default_a = FINAL_EVALUATION_ID if FINAL_EVALUATION_ID in available else available[0]
    others = [name for name in available if name != default_a]
    default_b = next((n for n in others if "dqn_default" in n), others[0])

    col_a, col_b = st.columns(2)
    with col_a:
        choice_a = st.selectbox("Modèle A", available,
                                index=available.index(default_a), format_func=human_label)
    with col_b:
        choice_b = st.selectbox("Modèle B", available,
                                index=available.index(default_b), format_func=human_label)

    if choice_a == choice_b:
        st.info("Choisissez deux évaluations différentes pour voir un écart.")
        return

    episodes_a = load_episodes(choice_a, artifacts_dir)
    episodes_b = load_episodes(choice_b, artifacts_dir)

    def describe(frame: pd.DataFrame) -> dict[str, float]:
        return {
            "Récompense moyenne": frame["total_reward"].mean(),
            "Écart-type": frame["total_reward"].std(ddof=0),
            "Médiane": frame["total_reward"].median(),
            "Pire épisode": frame["total_reward"].min(),
            "Meilleur épisode": frame["total_reward"].max(),
            "Réussite (%)": 100 * frame["success"].mean(),
            "Durée moyenne (pas)": frame["episode_length"].mean(),
            "Carburant estimé": frame["fuel_proxy"].mean(),
        }

    stats_a, stats_b = describe(episodes_a), describe(episodes_b)

    st.markdown(f"**A — {human_label(choice_a)}** · **B — {human_label(choice_b)}**")
    # Pour l'écart-type, le carburant et la durée, plus bas est meilleur.
    lower_is_better = {"Écart-type", "Durée moyenne (pas)", "Carburant estimé"}
    for index, (name, value_a) in enumerate(stats_a.items()):
        if index % 4 == 0:
            row = st.columns(4)
        value_b = stats_b[name]
        delta = value_a - value_b
        better = delta < 0 if name in lower_is_better else delta > 0
        row[index % 4].metric(
            name,
            f"{value_a:.1f}",
            delta=f"{delta:+.1f} vs B",
            delta_color="normal" if better else "inverse",
        )

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for frame, name, color in ((episodes_a, choice_a, SERIES_1), (episodes_b, choice_b, SERIES_2)):
            fig.add_trace(go.Histogram(
                x=frame["total_reward"], name=human_label(name), marker_color=color,
                opacity=0.72, nbinsx=24,
                hovertemplate="%{fullData.name}<br>%{x:.0f} → %{y} épisodes<extra></extra>",
            ))
        fig.update_layout(barmode="overlay", title="Distribution des récompenses",
                          xaxis_title="Récompense de l'épisode",
                          yaxis_title="Nombre d'épisodes", height=420)
        add_target_line(fig, axis="x")
        st.plotly_chart(style_figure(fig), width="stretch")
        st.caption(
            "Deux histogrammes superposés. Une distribution resserrée et décalée "
            "vers la droite est le signe d'un pilote à la fois bon et régulier."
        )
    with right:
        curves = load_learning_curves(artifacts_dir)
        experiments = [experiment_of(choice_a), experiment_of(choice_b)]
        subset = curves[curves["experiment"].isin(experiments)]
        if subset.empty:
            st.info("Aucune courbe d'entraînement pour ces deux modèles.")
        else:
            fig = go.Figure()
            for experiment, color in zip(experiments, (SERIES_1, SERIES_2)):
                part = subset[subset["experiment"] == experiment].sort_values("timestep")
                if part.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=part["timestep"], y=part["mean_reward"], mode="lines+markers",
                    name=EXPERIMENT_LABELS.get(experiment, experiment),
                    line=dict(width=2.5, color=color), marker=dict(size=6),
                    hovertemplate="%{fullData.name}<br>%{x:,} transitions<br>%{y:.1f}<extra></extra>",
                ))
            add_target_line(fig)
            fig.update_layout(title="Apprentissage comparé",
                              xaxis_title="Transitions d'entraînement",
                              yaxis_title="Récompense moyenne", height=420)
            st.plotly_chart(style_figure(fig), width="stretch")
            st.caption(
                "Attention aux budgets d'entraînement différents : une courbe "
                "plus courte n'a simplement pas été entraînée aussi longtemps."
            )

    configs = load_configs(artifacts_dir)
    config_a = configs.get(experiment_of(choice_a), {}).get("hyperparameters") or {}
    config_b = configs.get(experiment_of(choice_b), {}).get("hyperparameters") or {}
    if config_a or config_b:
        st.markdown("#### Réglages des deux modèles")
        names = sorted(set(config_a) | set(config_b))
        table = pd.DataFrame({
            "Hyperparamètre": names,
            "A": [config_a.get(name, "défaut") for name in names],
            "B": [config_b.get(name, "défaut") for name in names],
        })
        table["Identique"] = np.where(table["A"].astype(str) == table["B"].astype(str), "oui", "non")
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption(
            "« défaut » signifie que l'hyperparamètre n'a pas été fixé et garde "
            "la valeur par défaut de Stable-Baselines3."
        )


def render_episodes(artifacts_dir: Path) -> None:
    """Permet d'isoler les réussites et les échecs d'une évaluation."""
    st.markdown("### Explorer les épisodes un par un")
    evaluations = list_episode_evaluations(artifacts_dir)
    default_index = evaluations.index(FINAL_EVALUATION_ID) if FINAL_EVALUATION_ID in evaluations else 0

    filters = st.columns([2, 2, 3])
    with filters[0]:
        evaluation_id = st.selectbox("Évaluation", evaluations, index=default_index,
                                     format_func=human_label)
    episodes = load_episodes(evaluation_id, artifacts_dir)
    episodes = episodes.assign(issue=episodes["outcome"].map(lambda v: OUTCOME_LABELS.get(v, v)))

    with filters[1]:
        available_outcomes = sorted(episodes["outcome"].unique())
        chosen = st.multiselect(
            "Issue de l'épisode",
            available_outcomes,
            default=available_outcomes,
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

    total_success = int((episodes["outcome"] == "success").sum())
    total_failure = len(episodes) - total_success
    st.caption(
        f"Sur les {len(episodes)} épisodes de cette évaluation : "
        f"✅ {total_success} réussis · ✖ {total_failure} non réussis. "
        f"Filtre actif : {len(filtered)} épisodes affichés."
    )

    cols = st.columns(5)
    cols[0].metric("Épisodes affichés", len(filtered))
    cols[1].metric("Moyenne", f"{filtered['total_reward'].mean():.1f}", help=METRIC_HELP["mean"])
    cols[2].metric("Écart-type", f"± {filtered['total_reward'].std(ddof=0):.1f}", help=METRIC_HELP["std"])
    cols[3].metric("Médiane", f"{filtered['total_reward'].median():.1f}")
    cols[4].metric("Min / Max",
                   f"{filtered['total_reward'].min():.0f} / {filtered['total_reward'].max():.0f}")

    colors = outcome_colors(episodes["outcome"].unique())
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for outcome in chosen:
            part = filtered[filtered["outcome"] == outcome]
            if part.empty:
                continue
            fig.add_trace(go.Histogram(
                x=part["total_reward"], name=OUTCOME_LABELS.get(outcome, outcome),
                marker_color=colors[outcome], nbinsx=22, opacity=0.85,
                hovertemplate="%{fullData.name}<br>%{x:.0f} → %{y} épisodes<extra></extra>",
            ))
        fig.update_layout(barmode="stack", title="Répartition des récompenses",
                          xaxis_title="Récompense de l'épisode",
                          yaxis_title="Nombre d'épisodes", height=420)
        add_target_line(fig, axis="x")
        st.plotly_chart(style_figure(fig), width="stretch")
    with right:
        fig = go.Figure()
        for outcome in chosen:
            part = filtered[filtered["outcome"] == outcome]
            if part.empty:
                continue
            fig.add_trace(go.Scatter(
                x=part["fuel_proxy"], y=part["total_reward"], mode="markers",
                name=OUTCOME_LABELS.get(outcome, outcome),
                marker=dict(color=colors[outcome], size=9,
                            line=dict(width=1.5, color="#ffffff")),
                customdata=part[["episode_id", "seed", "episode_length"]],
                hovertemplate=("épisode %{customdata[0]} (seed %{customdata[1]})<br>"
                               "récompense %{y:.1f}<br>carburant %{x:.1f}<br>"
                               "%{customdata[2]} pas<extra></extra>"),
            ))
        fig.update_layout(title="Récompense et consommation",
                          xaxis_title="Carburant estimé (moteurs allumés)",
                          yaxis_title="Récompense de l'épisode", height=420)
        add_target_line(fig)
        st.plotly_chart(style_figure(fig), width="stretch")
        st.caption("En bas à droite : les vols coûteux et peu rentables.")

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

    runs = load_gui_runs(artifacts_dir)
    with st.expander(f"Parties lancées depuis la GUI ({len(runs)})"):
        if runs.empty:
            st.info("Aucune partie enregistrée. Lancez un épisode depuis la GUI.")
        else:
            st.dataframe(runs.sort_values("run_id", ascending=False),
                         width="stretch", hide_index=True)


def render_trajectory(artifacts_dir: Path) -> None:
    """Déroule un vol pas à pas : altitude, vitesse, angle et décisions."""
    st.markdown("### Le déroulé d'un vol")
    available = [
        evaluation_id
        for evaluation_id in list_episode_evaluations(artifacts_dir)
        if (Path(artifacts_dir) / "evaluations" / evaluation_id / "steps.csv").exists()
    ]
    if not available:
        st.warning("Aucune trajectoire détaillée disponible.")
        return

    cols = st.columns([2, 1, 2])
    with cols[0]:
        evaluation_id = st.selectbox("Jeu de trajectoires", available, format_func=human_label)
    steps = load_steps(evaluation_id, artifacts_dir)
    episodes = load_episodes(evaluation_id, artifacts_dir)
    with cols[1]:
        episode_id = st.selectbox("Épisode", sorted(steps["episode_id"].unique()))

    episode = steps[steps["episode_id"] == episode_id]
    summary = episodes[episodes["episode_id"] == episode_id]
    if not summary.empty:
        row = summary.iloc[0]
        mark = "✅" if row["outcome"] == "success" else "✖"
        with cols[2]:
            st.metric(
                f"{mark} {OUTCOME_LABELS.get(row['outcome'], row['outcome'])}",
                f"{row['total_reward']:.1f}",
                help="Récompense cumulée de ce vol.",
            )

    st.caption(
        "Les trois courbes décrivent l'état du module à chaque pas. Un bon "
        "atterrissage se lit ainsi : l'altitude descend régulièrement vers zéro, "
        "la vitesse verticale reste faible, et l'angle revient près de zéro."
    )

    fig = go.Figure()
    for column, label, color in [
        ("next_state_1", "Altitude", SERIES_1),
        ("next_state_3", "Vitesse verticale", SERIES_2),
        ("next_state_4", "Angle", SERIES_3),
    ]:
        fig.add_trace(go.Scatter(
            x=episode["step"], y=episode[column], mode="lines", name=label,
            line=dict(width=2, color=color),
            hovertemplate=f"{label}<br>pas %{{x}}<br>%{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(title="Télémétrie de l'épisode (unités du simulateur)",
                      xaxis_title="Pas de simulation", yaxis_title="Valeur",
                      hovermode="x unified", height=430)
    st.plotly_chart(style_figure(fig), width="stretch")

    left, right = st.columns(2)
    with left:
        counts = (episode["action_label"].value_counts()
                  .rename_axis("action").reset_index(name="count").sort_values("count"))
        fig = go.Figure(go.Bar(
            x=counts["count"], y=counts["action"], orientation="h",
            marker_color=SERIES_1,
            text=counts["count"], textposition="outside",
            hovertemplate="%{y}<br>%{x} fois<extra></extra>",
        ))
        fig.update_layout(title="Décisions prises pendant le vol",
                          xaxis_title="Nombre de pas", yaxis_title="", height=330)
        st.plotly_chart(thin_bars(style_figure(fig)), width="stretch")
    with right:
        fig = go.Figure(go.Scatter(
            x=episode["step"], y=episode["cumulative_reward"], mode="lines",
            line=dict(width=2, color=SERIES_1), fill="tozeroy",
            fillcolor="rgba(42,120,214,0.10)",
            hovertemplate="pas %{x}<br>cumul %{y:.1f}<extra></extra>",
        ))
        fig.update_layout(title="Récompense accumulée au fil du vol",
                          xaxis_title="Pas de simulation",
                          yaxis_title="Récompense cumulée", height=330)
        st.plotly_chart(style_figure(fig), width="stretch")

    with st.expander("Données pas à pas"):
        st.dataframe(episode, width="stretch", hide_index=True)


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

    st.markdown("### Comment les réglages ont été trouvés")
    st.caption(
        "Un algorithme de recherche (TPE) propose des combinaisons de réglages, "
        "les teste, et se concentre progressivement sur les zones prometteuses. "
        "Chaque essai est entraîné sur deux seeds différentes et noté par leur "
        "moyenne : un essai jugé sur un seul entraînement mesurerait surtout la "
        "chance du tirage initial."
    )

    cols = st.columns(4)
    cols[0].metric("Essais menés à terme", len(complete))
    cols[1].metric("Essais interrompus", len(pruned),
                   help="Arrêtés en cours de route car nettement sous la médiane. "
                        "Cela libère du temps de calcul pour les essais prometteurs.")
    if selected:
        cols[2].metric("Gamma retenu", f"{selected['parameters']['gamma']:.5f}",
                       help="Le facteur d'actualisation : à quel point l'agent "
                            "tient compte des récompenses lointaines.")
        cols[3].metric("Moyenne sur 5 entraînements", f"{selected['robust_mean_reward']:.1f}",
                       delta=f"pire des 5 : {selected['robust_min_reward']:.0f}",
                       help="Le réglage retenu réentraîné 5 fois de zéro.")

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            complete, x="number", y="value", color="params_gamma",
            hover_data=["params_learning_rate", "params_ent_coef"],
            labels={"number": "Numéro d'essai", "value": "Moyenne sur 2 seeds",
                    "params_gamma": "Gamma"},
            title="Chaque essai de la recherche",
            color_continuous_scale=[[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]],
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=1, color="#ffffff")))
        add_target_line(fig)
        st.plotly_chart(style_figure(fig), width="stretch")
        st.caption(
            f"{len(pruned)} essais supplémentaires ont été interrompus avant la fin "
            "et ne figurent pas ici. La couleur indique la valeur de gamma."
        )
    with right:
        grid = gamma.sort_values("gamma")
        fig = go.Figure(go.Scatter(
            x=grid["gamma"], y=grid["mean"],
            error_y=dict(type="data", array=grid["std"], visible=True, color="#b6bec4"),
            mode="lines+markers", line=dict(color=SERIES_1, width=2),
            marker=dict(size=9),
            hovertemplate="gamma %{x:.5f}<br>moyenne %{y:.1f}<extra></extra>",
        ))
        add_target_line(fig)
        if selected:
            fig.add_vline(x=selected["parameters"]["gamma"], line_dash="dot",
                          line_color=SERIES_2, annotation_text="retenu",
                          annotation_font_color=SERIES_2)
        fig.update_layout(title="Réglage fin de gamma",
                          xaxis_title="Gamma", yaxis_title="Récompense moyenne")
        st.plotly_chart(style_figure(fig), width="stretch")
        st.caption(
            "Chaque point est entraîné sur trois seeds ; la barre verticale "
            "montre l'écart entre elles."
        )

    left, right = st.columns(2)
    with left:
        if not importance.empty:
            ordered = importance.sort_values("importance")
            fig = go.Figure(go.Bar(
                x=ordered["importance"], y=ordered["parameter"], orientation="h",
                marker_color=SERIES_1,
                hovertemplate="%{y}<br>importance %{x:.3f}<extra></extra>",
            ))
            fig.update_layout(title="Quels réglages comptent vraiment",
                              xaxis_title="Importance estimée", yaxis_title="", height=400)
            st.plotly_chart(thin_bars(style_figure(fig)), width="stretch")
            st.caption(
                "Estimation PED-ANOVA : quelle part des écarts de performance "
                "chaque hyperparamètre explique."
            )
    with right:
        if not robust_summary.empty:
            plot = robust_summary.sort_values("mean").copy()
            plot["nom"] = plot["candidate_id"].map(candidate_label)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=plot["mean"], y=plot["nom"], orientation="h",
                error_x=dict(type="data", array=plot["std"], visible=True, color="#b6bec4"),
                marker_color=SERIES_1, name="Moyenne sur 5 entraînements",
                hovertemplate="%{y}<br>moyenne %{x:.1f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=plot["min"], y=plot["nom"], mode="markers",
                marker=dict(color=SERIES_2, symbol="diamond", size=12,
                            line=dict(width=1.5, color="#ffffff")),
                name="Pire des 5 entraînements",
                hovertemplate="%{y}<br>pire seed %{x:.1f}<extra></extra>",
            ))
            add_target_line(fig, axis="x")
            fig.update_layout(title="Les finalistes réentraînés 5 fois",
                              xaxis_title="Récompense moyenne", yaxis_title="", height=400)
            st.plotly_chart(thin_bars(style_figure(fig), 0.5), width="stretch")
            st.caption(
                "Le losange orange est le point décisif : un candidat peut avoir "
                "une bonne moyenne et s'effondrer sur un entraînement. C'est ce "
                "qui distingue un réglage fiable d'un réglage chanceux."
            )

    with st.expander("Tous les essais menés à terme"):
        columns = ["number", "value", "user_attrs_score_std", "user_attrs_score_min",
                   "params_gamma", "params_learning_rate", "params_ent_coef",
                   "params_n_steps", "params_batch_size", "params_n_epochs",
                   "params_gae_lambda", "params_clip_range", "params_vf_coef"]
        st.dataframe(
            complete[[c for c in columns if c in complete]]
            .sort_values("value", ascending=False).round(5),
            width="stretch", hide_index=True,
        )


def main() -> None:
    st.set_page_config(page_title="Eagle-1 | Mission Analytics", layout="wide")
    render_header()

    registry = load_registry(ARTIFACTS_DIR)

    essentials, phases, comparison, episodes, trajectory, optuna_tab = st.tabs([
        "L'essentiel",
        "Étapes de la mission",
        "Comparer deux modèles",
        "Épisodes",
        "Un vol en détail",
        "Recherche des réglages",
    ])
    with essentials:
        render_essentials(registry, ARTIFACTS_DIR)
    with phases:
        render_phases(registry, ARTIFACTS_DIR)
    with comparison:
        render_comparison(registry, ARTIFACTS_DIR)
    with episodes:
        render_episodes(ARTIFACTS_DIR)
    with trajectory:
        render_trajectory(ARTIFACTS_DIR)
    with optuna_tab:
        render_optuna(ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
