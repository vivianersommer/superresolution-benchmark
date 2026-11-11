"""Testes da fase 0: ambiente e manifesto dos modelos."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import ROOT_DIR, load_config, resolve  # noqa: E402
from f0_ambiente import sha256  # noqa: E402

CFG = load_config()


def test_seed():
    assert CFG["seed"] == 42


def test_requirements_pinned():
    lines = [l.strip() for l in (ROOT_DIR / "requirements.txt").read_text().splitlines()]
    pkgs = [l for l in lines if l and not l.startswith(("#", "--"))]
    assert pkgs and all("==" in l for l in pkgs)
    names = {l.split("==")[0].lower() for l in pkgs}
    for required in ["opencv-contrib-python", "numpy", "pandas", "scipy", "scikit-image",
                     "statsmodels", "pingouin", "torch", "lpips", "matplotlib", "pyyaml", "tqdm"]:
        assert required in names, required


def test_manifest_matches_models():
    manifest = pd.read_csv(resolve(CFG["paths"]["models_manifest"]))
    models_dir = resolve(CFG["paths"]["models_dir"])
    assert sorted(manifest["file"]) == sorted(p.name for p in models_dir.glob("*.pb"))
    for row in manifest.itertuples():
        assert sha256(models_dir / row.file) == row.sha256, row.file


@pytest.mark.parametrize("algorithm", CFG["methods"]["dl"])
@pytest.mark.parametrize("scale", CFG["scales"])
def test_models_load_and_upsample_exact_factor(algorithm, scale):
    path = resolve(CFG["paths"]["models_dir"]) / CFG["model_files"][algorithm].format(s=scale)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(path))
    sr.setModel(algorithm, scale)
    lr = np.random.default_rng(CFG["seed"]).integers(0, 256, (24, 36, 3), dtype=np.uint8)
    assert sr.upsample(lr).shape == (24 * scale, 36 * scale, 3)
