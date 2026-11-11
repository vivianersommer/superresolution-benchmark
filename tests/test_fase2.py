"""Testes da fase 2: redução compatível com o MATLAB e reconstruções no tamanho da HR."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import load_config, resolve  # noqa: E402
from f2_reconstrucao import load_model, read_image  # noqa: E402
from imresize_matlab import imresize  # noqa: E402

CFG = load_config()
R = CFG["reconstrucao"]
SAMPLE = pd.read_csv(resolve(CFG["amostragem"]["sample_csv"]))
HR_DIR = resolve(CFG["amostragem"]["hr_dir"])
METHODS = CFG["methods"]["dl"] + CFG["methods"]["classic"]
BENCH = resolve(CFG["benchmarks"]["extract_dir"])


def _bench_cases():
    for st in CFG["benchmarks"]["sets"]:
        for p in sorted((BENCH / st / "HR").glob("*.png")):
            for s in (2, 3, 4):
                yield st, p.stem, s


@pytest.mark.parametrize("bench_set,name,scale", list(_bench_cases()))
def test_imresize_matches_matlab_bit_exact(bench_set, name, scale):
    hr = read_image(BENCH / bench_set / "HR" / f"{name}.png")
    h, w = hr.shape[:2]
    hr = hr[: h - h % scale, : w - w % scale]
    ref = read_image(BENCH / bench_set / "LR_bicubic" / f"X{scale}" / f"{name}x{scale}.png")
    assert np.array_equal(imresize(hr, 1 / scale), ref)


def test_lr_images():
    for s in CFG["scales"]:
        for name in SAMPLE["hr_arquivo"]:
            hr = read_image(HR_DIR / name)
            lr = read_image(resolve(R["lr_dir"]) / f"x{s}" / name)
            assert lr.shape == (hr.shape[0] // s, hr.shape[1] // s, 3)
            assert np.array_equal(lr, imresize(hr, 1 / s)), (name, s)


def test_all_reconstructions_have_hr_size():
    n = 0
    for s in CFG["scales"]:
        for m in METHODS:
            files = sorted(p.name for p in (resolve(R["sr_dir"]) / f"x{s}" / m).glob("*.png"))
            assert files == sorted(SAMPLE["hr_arquivo"]), (s, m)
            for name in files:
                sr = read_image(resolve(R["sr_dir"]) / f"x{s}" / m / name)
                assert sr.shape == read_image(HR_DIR / name).shape, (s, m, name)
                n += 1
    assert n == len(SAMPLE) * len(METHODS) * len(CFG["scales"])


def test_runtime_csv():
    df = pd.read_csv(resolve(R["runtime_csv"]))
    assert len(df) == len(SAMPLE) * len(METHODS) * len(CFG["scales"])
    assert not df.isna().any().any()
    assert not df.duplicated(["image", "method", "scale"]).any()
    assert (df["runtime_ms"] > 0).all()
    assert set(df["family"]) == {"dl", "classic"}


@pytest.mark.parametrize("method", ["fsrcnn", "bicubic"])
def test_reconstruction_deterministic(method):
    s, name = CFG["scales"][0], SAMPLE["hr_arquivo"].iloc[0]
    lr = read_image(resolve(R["lr_dir"]) / f"x{s}" / name)
    if method in CFG["methods"]["dl"]:
        out = load_model(CFG, method, s).upsample(lr)
    else:
        interp = getattr(cv2, CFG["classic_interpolation"][method])
        out = cv2.resize(lr, (lr.shape[1] * s, lr.shape[0] * s), interpolation=interp)
    assert np.array_equal(out, read_image(resolve(R["sr_dir"]) / f"x{s}" / method / name))


def test_grid_exists():
    for ext in CFG["figuras"]["formats"]:
        assert (resolve("results/figs") / f"{R['grid']['name']}.{ext}").stat().st_size > 0
