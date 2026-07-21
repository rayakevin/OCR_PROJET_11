#!/usr/bin/env python3
"""Lance toute la démonstration Eagle-1 depuis un seul terminal.

Services démarrés, avec leur port par défaut :
- API FastAPI sur le port 8000 ;
- GUI Streamlit sur le port 8501 ;
- dashboard Streamlit sur le port 8502.

Ces ports sont surchargeables et un port déjà occupé est automatiquement
remplacé par le premier port libre suivant, sauf avec ``--strict-ports``.
L'URL réellement retenue est affichée au démarrage de chaque service.

Les trois processus sont arrêtés ensemble avec Ctrl+C. Leurs sorties sont
conservées dans ``ASTRODYNAMICS/artifacts/demo_logs/<horodatage>/``, qui
contient ``api.log``, ``gui.log`` et ``dashboard.log``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_ROOT = PROJECT_ROOT / "ASTRODYNAMICS" / "artifacts" / "demo_logs"


@dataclass
class Service:
    """Configuration et processus d'un service de la démonstration."""

    name: str
    command: list[str]
    url: str
    health_url: str
    log_path: Path
    process: subprocess.Popen[bytes] | None = None
    log_file: BinaryIO | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lance la démonstration Eagle-1.")
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port de l'API FastAPI (défaut : 8000). Remplacé s'il est occupé.",
    )
    parser.add_argument(
        "--gui-port",
        type=int,
        default=8501,
        help="Port de la GUI Streamlit (défaut : 8501). Remplacé s'il est occupé.",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8502,
        help="Port du dashboard Streamlit (défaut : 8502). Remplacé s'il est occupé.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="N'ouvre pas automatiquement les trois interfaces dans le navigateur.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=90.0,
        help="Temps maximal de démarrage de chaque service, en secondes.",
    )
    parser.add_argument(
        "--strict-ports",
        action="store_true",
        help="Échoue au lieu de choisir automatiquement un autre port libre.",
    )
    return parser.parse_args()


def port_is_free(port: int) -> bool:
    """Indique si un serveur local peut réserver le port demandé."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def resolve_ports(args: argparse.Namespace) -> None:
    """Conserve les ports demandés ou sélectionne les prochains ports libres."""
    attributes = ["api_port", "gui_port", "dashboard_port"]
    requested = [getattr(args, attribute) for attribute in attributes]
    if len(set(requested)) != len(requested):
        raise RuntimeError("L'API, la GUI et le dashboard doivent utiliser trois ports distincts.")
    if any(port < 1 or port > 65535 for port in requested):
        raise RuntimeError("Les ports doivent être compris entre 1 et 65535.")

    reserved = set(requested)
    selected: set[int] = set()
    for attribute in attributes:
        requested_port = getattr(args, attribute)
        if port_is_free(requested_port):
            selected_port = requested_port
        elif args.strict_ports:
            raise RuntimeError(f"Le port {requested_port} est déjà utilisé.")
        else:
            candidates = range(requested_port + 1, min(requested_port + 101, 65536))
            selected_port = next(
                (
                    port
                    for port in candidates
                    if port not in reserved and port not in selected and port_is_free(port)
                ),
                None,
            )
            if selected_port is None:
                raise RuntimeError(
                    f"Aucun port libre trouvé après le port {requested_port}."
                )
            print(
                f"Port {requested_port} occupé : utilisation automatique "
                f"du port {selected_port}.",
                flush=True,
            )

        setattr(args, attribute, selected_port)
        selected.add(selected_port)


def start_service(service: Service, environment: dict[str, str]) -> None:
    """Démarre un service dans son propre groupe et redirige ses logs."""
    service.log_file = service.log_path.open("wb")
    service.process = subprocess.Popen(
        service.command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=service.log_file,
        stderr=subprocess.STDOUT,
        start_new_session=os.name != "nt",
    )


def health_response_is_valid(service: Service, body: bytes) -> bool:
    """Vérifie aussi que le modèle PPO est chargé dans l'API."""
    if service.name != "API":
        return body.strip().lower() == b"ok"

    payload = json.loads(body)
    return payload.get("status") == "ok" and payload.get("model_loaded") is True


def wait_until_ready(service: Service, timeout: float) -> None:
    """Attend une réponse valide du service ou signale son échec."""
    assert service.process is not None
    deadline = time.monotonic() + timeout
    last_error = "aucune réponse"

    while time.monotonic() < deadline:
        exit_code = service.process.poll()
        if exit_code is not None:
            raise RuntimeError(f"{service.name} s'est arrêté avec le code {exit_code}.")

        try:
            with urlopen(service.health_url, timeout=2) as response:
                body = response.read()
                if response.status == 200 and health_response_is_valid(service, body):
                    return
                last_error = f"réponse HTTP {response.status} inattendue"
        except (OSError, URLError, json.JSONDecodeError) as error:
            last_error = str(error)

        time.sleep(0.5)

    raise RuntimeError(
        f"{service.name} n'est pas prêt après {timeout:g} s ({last_error})."
    )


def stop_services(services: list[Service]) -> None:
    """Arrête tous les enfants, puis force leur arrêt si nécessaire."""
    running = [
        service
        for service in services
        if service.process is not None and service.process.poll() is None
    ]

    for service in reversed(running):
        assert service.process is not None
        if os.name == "nt":
            service.process.terminate()
        else:
            os.killpg(service.process.pid, signal.SIGTERM)

    deadline = time.monotonic() + 8
    while running and time.monotonic() < deadline:
        running = [service for service in running if service.process.poll() is None]
        time.sleep(0.1)

    for service in running:
        assert service.process is not None
        if os.name == "nt":
            service.process.kill()
        else:
            os.killpg(service.process.pid, signal.SIGKILL)

    for service in services:
        if service.log_file is not None:
            service.log_file.close()


def print_log_tail(service: Service, line_count: int = 20) -> None:
    """Affiche la fin du journal pour rendre un échec immédiatement lisible."""
    if service.log_file is not None:
        service.log_file.flush()
    if not service.log_path.exists():
        return

    lines = service.log_path.read_text(errors="replace").splitlines()
    if lines:
        print(f"\n--- Fin de {service.log_path.name} ---", file=sys.stderr)
        print("\n".join(lines[-line_count:]), file=sys.stderr)


def build_services(args: argparse.Namespace, run_directory: Path) -> list[Service]:
    """Construit les commandes des trois services.

    Réutilise ``sys.executable``, c'est-à-dire l'interpréteur qui exécute ce
    script : l'environnement virtuel actif est donc conservé, que le lancement
    passe par ``uv run`` ou par un ``python`` déjà dans le bon environnement.
    """
    python = sys.executable
    host = "127.0.0.1"
    api_url = f"http://{host}:{args.api_port}"
    gui_url = f"http://{host}:{args.gui_port}"
    dashboard_url = f"http://{host}:{args.dashboard_port}"

    streamlit_options = [
        "--server.address",
        host,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    return [
        Service(
            name="API",
            command=[
                python,
                "-m",
                "uvicorn",
                "ASTRODYNAMICS.api.app:app",
                "--host",
                host,
                "--port",
                str(args.api_port),
            ],
            url=f"{api_url}/docs",
            health_url=f"{api_url}/health",
            log_path=run_directory / "api.log",
        ),
        Service(
            name="GUI",
            command=[
                python,
                "-m",
                "streamlit",
                "run",
                "ASTRODYNAMICS/gui/app.py",
                "--server.port",
                str(args.gui_port),
                *streamlit_options,
            ],
            url=gui_url,
            health_url=f"{gui_url}/_stcore/health",
            log_path=run_directory / "gui.log",
        ),
        Service(
            name="Dashboard",
            command=[
                python,
                "-m",
                "streamlit",
                "run",
                "ASTRODYNAMICS/dashboard/app.py",
                "--server.port",
                str(args.dashboard_port),
                *streamlit_options,
            ],
            url=dashboard_url,
            health_url=f"{dashboard_url}/_stcore/health",
            log_path=run_directory / "dashboard.log",
        ),
    ]


def run_demo(args: argparse.Namespace) -> int:
    resolve_ports(args)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_directory = LOGS_ROOT / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    services = build_services(args, run_directory)

    environment = os.environ.copy()
    environment["EAGLE1_API_URL"] = f"http://127.0.0.1:{args.api_port}"

    try:
        for service in services:
            print(f"Démarrage de {service.name}...", flush=True)
            start_service(service, environment)
            wait_until_ready(service, args.startup_timeout)
            print(f"  ✓ {service.name} prêt : {service.url}", flush=True)

        print(f"\nJournaux : {run_directory}")
        print("Démonstration prête. Appuyez sur Ctrl+C pour tout arrêter.\n")

        if not args.no_browser:
            for service in services:
                webbrowser.open_new_tab(service.url)

        while True:
            for service in services:
                assert service.process is not None
                exit_code = service.process.poll()
                if exit_code is not None:
                    raise RuntimeError(
                        f"{service.name} s'est arrêté de manière inattendue "
                        f"avec le code {exit_code}."
                    )
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nArrêt de la démonstration...", flush=True)
        return 0
    except Exception:
        for service in services:
            if service.process is not None:
                print_log_tail(service)
        raise
    finally:
        stop_services(services)


def main() -> int:
    args = parse_args()

    def request_stop(_signal_number: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_stop)

    try:
        return run_demo(args)
    except RuntimeError as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
