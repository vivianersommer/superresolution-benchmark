"""Utilidades compartilhadas: leitura do config.yaml e resolução de caminhos."""
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_config(path: Path | None = None) -> dict:
    path = path or ROOT_DIR / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel: str) -> Path:
    """Resolve um caminho do config relativo à raiz do repositório."""
    return (ROOT_DIR / rel).resolve()
