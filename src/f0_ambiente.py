"""Fase 0 — registra o ambiente e o manifesto dos modelos.

Saídas:
  models/MANIFEST.csv     SHA-256 de cada arquivo em models/
  results/ambiente.json   SO, CPU/GPU, Python, versões de pacotes

Uso: .venv/bin/python src/f0_ambiente.py
"""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata

import cv2
import pandas as pd

from common import load_config, resolve


def sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_manifest(cfg: dict) -> pd.DataFrame:
    models_dir = resolve(cfg["paths"]["models_dir"])
    rows = []
    for p in sorted(models_dir.glob("*.pb")):
        m = re.fullmatch(r"(\w+)_x(\d+)\.pb", p.name)
        rows.append({
            "file": p.name,
            "algorithm": m.group(1).lower() if m else "",
            "scale": int(m.group(2)) if m else "",
            "size_bytes": p.stat().st_size,
            "sha256": sha256(p),
        })
    df = pd.DataFrame(rows)
    df.to_csv(resolve(cfg["paths"]["models_manifest"]), index=False)
    return df


def _cmd(args: list[str]) -> str:
    try:
        env = {**os.environ, "LC_ALL": "C"}
        return subprocess.run(args, capture_output=True, text=True, check=True, env=env).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "indisponível"


def environment_info() -> dict:
    cpu = next((l.split(":", 1)[1].strip() for l in _cmd(["lscpu"]).splitlines()
                if l.startswith("Model name")), platform.processor())
    os_name = next((l.split("=", 1)[1].strip().strip('"') for l in open("/etc/os-release")
                    if l.startswith("PRETTY_NAME")), platform.platform())
    pkgs = ["opencv-contrib-python", "numpy", "pandas", "scipy", "scikit-image",
            "statsmodels", "pingouin", "torch", "torchvision", "lpips",
            "matplotlib", "PyYAML", "tqdm", "pytest"]
    return {
        "os": os_name,
        "kernel": platform.release(),
        "cpu": cpu,
        "cpu_count": _cmd(["nproc"]),
        "gpu": _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader"]),
        "gpu_usage_note": "Todas as inferências rodam em CPU (OpenCV DNN backend padrão).",
        "python": sys.version.split()[0],
        "opencv_build_version": cv2.__version__,
        "opencv_has_dnn_superres": hasattr(cv2, "dnn_superres"),
        "rscript": _cmd(["Rscript", "--version"]),
        "packages": {p: metadata.version(p) for p in pkgs},
    }


def main() -> None:
    cfg = load_config()
    out_dir = resolve(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(cfg)
    print(manifest.to_string(index=False))

    env = environment_info()
    (out_dir / "ambiente.json").write_text(json.dumps(env, indent=2, ensure_ascii=False))
    print(json.dumps(env, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
