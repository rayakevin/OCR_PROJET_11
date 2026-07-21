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

# --- Habillage de la visualisation ------------------------------------------
#
# Le rendu natif de Gymnasium est volontairement minimal : ciel noir uni, sol
# blanc, aucune indication de la cible. On le retouche à la volée pour rendre
# la scène plus lisible, sans jamais toucher à la simulation elle-même.
#
# Deux pistes ont été écartées :
#   - augmenter la résolution native en multipliant VIEWPORT_W/H et SCALE :
#     mesuré, cela change la taille physique du module et donc la trajectoire ;
#   - agrandir l'image côté Python : à 130 ms par image, cela plafonnait le
#     rejeu à 7 images/seconde. Le navigateur agrandit très bien lui-même.

GLOW_HEIGHT = 54  # hauteur, en pixels, du halo au-dessus de l'aire d'atterrissage
_SKY_GRADIENT = None
_PAD_OVERLAY = None


def landing_pad_pixels() -> tuple[int, int, int]:
    """Position de l'aire d'atterrissage en pixels, déduite de l'environnement.

    Les coordonnées viennent des constantes de `lunar_lander` plutôt que d'être
    codées en dur : si Gymnasium change sa géométrie, le repère suit.
    """
    import gymnasium.envs.box2d.lunar_lander as lunar

    chunks = 11
    world_width = lunar.VIEWPORT_W / lunar.SCALE
    world_height = lunar.VIEWPORT_H / lunar.SCALE
    chunk_x = [world_width / (chunks - 1) * i for i in range(chunks)]
    x1 = int(chunk_x[chunks // 2 - 1] * lunar.SCALE)
    x2 = int(chunk_x[chunks // 2 + 1] * lunar.SCALE)
    y = int(lunar.VIEWPORT_H - (world_height / 4) * lunar.SCALE)
    return x1, x2, y


def _build_overlays(height: int, width: int) -> None:
    """Précalcule le dégradé de ciel et le repère d'atterrissage."""
    global _SKY_GRADIENT, _PAD_OVERLAY
    from PIL import Image, ImageDraw

    top = np.array([13, 22, 46], dtype=np.float32)
    bottom = np.array([3, 4, 10], dtype=np.float32)
    ratio = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    _SKY_GRADIENT = (top * (1 - ratio) + bottom * ratio).astype(np.uint8).repeat(width, axis=1)

    x1, x2, pad_y = landing_pad_pixels()
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Une ligne par pixel : dessiner une ligne sur deux laisse des rayures.
    for offset in range(GLOW_HEIGHT):
        alpha = int(46 * (1 - offset / GLOW_HEIGHT) ** 1.6)
        draw.line([(x1, pad_y - offset), (x2, pad_y - offset)], fill=(42, 120, 214, alpha))
    draw.rectangle([x1, pad_y - 1, x2, pad_y + 1], fill=(96, 176, 255, 215))
    _PAD_OVERLAY = overlay


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Remplace le ciel noir par un dégradé et matérialise l'aire visée.

    Coût mesuré : environ 6 ms par image, ce qui laisse la marge nécessaire
    pour un rejeu fluide.
    """
    from PIL import Image

    height, width, _ = frame.shape
    if _SKY_GRADIENT is None or _SKY_GRADIENT.shape[:2] != (height, width):
        _build_overlays(height, width)

    enhanced = frame.copy()
    # Seuls les pixels quasi noirs sont du ciel : le sol et le module sont
    # laissés intacts.
    sky = enhanced.sum(axis=2, dtype=np.int16) < 26
    enhanced[sky] = _SKY_GRADIENT[sky]

    composed = Image.alpha_composite(
        Image.fromarray(enhanced).convert("RGBA"), _PAD_OVERLAY
    )
    return np.asarray(composed.convert("RGB"))

# --- Rendu « arcade » --------------------------------------------------------
#
# Un second style de visualisation, dessiné intégralement à partir de l'état de
# la simulation : relief, position et angle du module, moteur allumé, contacts.
# La physique n'est pas touchée — c'est uniquement une couche d'affichage, et
# les chiffres de la mission restent ceux de `LunarLander-v3`.
#
# Rien d'équivalent n'existe publiquement : les projets LunarLander se
# concentrent sur les algorithmes et gardent le rendu d'origine. Les banques de
# sprites libres (Kenney, CC0) proposent des vaisseaux vus de dessus, qui
# cadrent mal avec une vue de profil munie de pattes d'atterrissage.

ARCADE_W, ARCADE_H = 1200, 800
_ARCADE_BACKGROUND: dict[int, np.ndarray] = {}


def _world_to_pixels(x: float, y: float) -> tuple[float, float]:
    """Convertit des coordonnées monde en pixels de la toile arcade."""
    import gymnasium.envs.box2d.lunar_lander as lunar

    kx = ARCADE_W / (lunar.VIEWPORT_W / lunar.SCALE)
    ky = ARCADE_H / (lunar.VIEWPORT_H / lunar.SCALE)
    return x * kx, ARCADE_H - y * ky


def _arcade_background(env, seed: int) -> np.ndarray:
    """Compose le décor fixe de l'épisode : ciel, étoiles, relief, piste.

    Tout cela est immobile pendant un vol : le calculer une seule fois fait
    tomber le coût par image d'environ 23 ms à quelques millisecondes.
    """
    if seed in _ARCADE_BACKGROUND:
        return _ARCADE_BACKGROUND[seed]

    from PIL import Image, ImageDraw, ImageFilter

    unwrapped = env.unwrapped

    top = np.array([16, 24, 54], dtype=np.float32)
    bottom = np.array([4, 6, 14], dtype=np.float32)
    ratio = np.linspace(0, 1, ARCADE_H, dtype=np.float32)[:, None, None]
    sky = (top * (1 - ratio) + bottom * ratio).astype(np.uint8).repeat(ARCADE_W, axis=1)
    image = Image.fromarray(sky)
    draw = ImageDraw.Draw(image)

    # Champ d'étoiles : trois tailles pour suggérer la profondeur.
    generator = np.random.default_rng(seed)
    for count, size, brightness in ((150, 1, 90), (70, 2, 150), (25, 3, 230)):
        for x, y in zip(generator.uniform(0, ARCADE_W, count),
                        generator.uniform(0, ARCADE_H * 0.8, count)):
            draw.ellipse([x - size, y - size, x + size, y + size],
                         fill=(brightness, brightness, min(255, brightness + 20)))

    # Halo au-dessus de l'aire visée : trapèze qui se resserre, puis flou large.
    # Un rectangle net laisserait une arête visible dans le ciel.
    x1, pad_y = _world_to_pixels(unwrapped.helipad_x1, unwrapped.helipad_y)
    x2, _ = _world_to_pixels(unwrapped.helipad_x2, unwrapped.helipad_y)
    glow = Image.new("RGBA", (ARCADE_W, ARCADE_H), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    height = 190
    for offset in range(height):
        share = offset / height
        inset = (x2 - x1) * 0.30 * share
        glow_draw.line([(x1 + inset, pad_y - offset), (x2 - inset, pad_y - offset)],
                       fill=(70, 155, 255, int(34 * (1 - share) ** 1.5)))
    image = Image.alpha_composite(
        image.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(28))
    ).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Relief. `sky_polys` décrit le CIEL, de la crête jusqu'en haut : le sol est
    # son complément, reconstruit depuis l'arête basse de ces polygones.
    ridge = [polygon[0] for polygon in unwrapped.sky_polys] + [unwrapped.sky_polys[-1][1]]
    ridge_pixels = [_world_to_pixels(x, y) for x, y in ridge]
    draw.polygon(ridge_pixels + [(ARCADE_W, ARCADE_H), (0, ARCADE_H)], fill=(46, 48, 60))
    draw.line(ridge_pixels, fill=(188, 198, 220), width=4)

    draw.line([(x1, pad_y), (x2, pad_y)], fill=(120, 200, 255), width=5)
    for index in range(5):
        beacon = x1 + (x2 - x1) * index / 4
        draw.ellipse([beacon - 4, pad_y - 4, beacon + 4, pad_y + 4], fill=(190, 230, 255))

    _ARCADE_BACKGROUND[seed] = np.asarray(image)
    return _ARCADE_BACKGROUND[seed]


def render_arcade(env, seed: int, action: int, contact: bool) -> np.ndarray:
    """Dessine la scène complète : décor mémorisé plus module et moteurs."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(_arcade_background(env, seed).copy())
    draw = ImageDraw.Draw(image)

    body = env.unwrapped.lander
    cx, cy = _world_to_pixels(body.position.x, body.position.y)
    angle = -body.angle
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    def point(dx: float, dy: float) -> tuple[float, float]:
        return cx + (dx * cos_a - dy * sin_a), cy + (dx * sin_a + dy * cos_a)

    # Flammes d'abord : elles doivent passer derrière la structure.
    if action == 2:
        draw.polygon([point(-16, 26), point(16, 26), point(0, 78)], fill=(255, 170, 60))
        draw.polygon([point(-8, 22), point(8, 22), point(0, 52)], fill=(255, 236, 190))
    elif action in (1, 3):
        side = -1 if action == 3 else 1
        draw.polygon([point(side * 26, -4), point(side * 26, 10), point(side * 62, 4)],
                     fill=(255, 190, 90))

    draw.polygon([point(-30, -22), point(-34, 10), point(0, 26), point(34, 10), point(30, -22)],
                 fill=(126, 116, 214), outline=(196, 190, 255))
    draw.polygon([point(-20, -12), point(-22, 4), point(0, 12), point(22, 4), point(20, -12)],
                 fill=(154, 146, 236))
    draw.ellipse([cx - 12, cy - 14, cx + 12, cy + 10],
                 fill=(210, 232, 255), outline=(120, 150, 200))
    for side in (-1, 1):
        draw.line([point(side * 26, 16), point(side * 44, 46)], fill=(176, 170, 226), width=7)
        draw.line([point(side * 34, 46), point(side * 54, 46)], fill=(196, 190, 255), width=6)

    if contact:
        for side in (-1, 1):
            fx, fy = point(side * 44, 48)
            draw.ellipse([fx - 26, fy - 8, fx + 26, fy + 8], fill=(120, 120, 140))

    return np.asarray(image)


RENDER_STYLES = {
    "Amélioré": "Rendu de Gymnasium, ciel dégradé et aire d'atterrissage matérialisée.",
    "Arcade": "Scène redessinée en 1200×800 : étoiles, relief ombré, module et moteurs.",
    "Natif": "Rendu brut de Gymnasium, sans retouche.",
}


def render_frame(env, row: dict, seed: int, style: str) -> np.ndarray:
    """Produit l'image d'un pas selon le style choisi."""
    if style == "Natif":
        return env.render()
    if style == "Arcade":
        contact = bool(row.get("left_contact", 0) or row.get("right_contact", 0))
        return render_arcade(env, seed, int(row["action"]), contact)
    return enhance_frame(env.render())


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
    target_ui_fps: float = 24.0,
    render_style: str = "Amélioré",
) -> None:
    """Rejoue les actions de l'API pendant la durée de visualisation demandée.

    La simulation est déterministe : même seed et mêmes actions donnent le même vol.

    Le nombre d'images vaut ``min(len(steps), ceil(duration_seconds *
    target_ui_fps))``. À 24 images par seconde, un vol d'environ 260 pas est
    rendu presque intégralement sur 10 secondes ; à 10 images par seconde, plus
    de la moitié des pas étaient sautés et le vol paraissait saccadé.

    Avec une durée nulle, seule la dernière image est rendue — c'est le cas
    utilisé par les tests.

    :param on_frame: appelé avec l'image RGB et la ligne de télémétrie du pas.
    :raises ValueError: si ``steps`` est vide ou la durée négative.
    :raises RuntimeError: si le rejeu ne reproduit pas l'épisode enregistré.
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

    # Le style « Arcade » dessine depuis l'état et n'a pas besoin du rendu
    # natif, mais le mode rgb_array reste demandé pour les deux autres styles.
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

            data = row.to_dict()
            on_frame(render_frame(env, data, seed, render_style), data)
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
        render_style = st.selectbox(
            "Style de rendu",
            list(RENDER_STYLES),
            help="Change uniquement l'affichage : la simulation, les décisions "
                 "de l'agent et le résultat du vol sont identiques.",
        )
        st.caption(RENDER_STYLES[render_style])
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
    # L'image native fait 600x400 : l'étaler sur toute la largeur la rend floue.
    # Une colonne centrale la maintient proche de sa résolution d'origine.
    _, viewport, _ = st.columns([1, 4, 1])
    with viewport:
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
                render_style=render_style,
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
