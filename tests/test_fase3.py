"""Testes da fase 3: definição das métricas, metrics_long.csv e validação no Set5."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import load_config, resolve  # noqa: E402
from f3_metricas import compute_metrics, rgb2y  # noqa: E402

CFG = load_config()
V = CFG["validacao"]
RNG = np.random.default_rng(CFG["seed"])
IMG = RNG.integers(20, 236, (64, 80, 3), dtype=np.uint8)


def test_rgb2y_bt601_range():
    assert rgb2y(np.full((1, 1, 3), 255, np.uint8))[0, 0] == pytest.approx(235.0)
    assert rgb2y(np.zeros((1, 1, 3), np.uint8))[0, 0] == pytest.approx(16.0)
    # Pesos BT.601 (canais em ordem BGR): só o vermelho.
    red = np.zeros((1, 1, 3), np.uint8); red[..., 2] = 255
    assert rgb2y(red)[0, 0] == pytest.approx(16 + 65.481)


def test_psnr_matches_known_mse():
    sr = IMG.copy().astype(np.int16)
    sr = np.clip(sr + 2, 0, 255).astype(np.uint8)   # +2 em todos os canais -> ΔY = 2·219/255
    m = compute_metrics(IMG, sr, 2, with_lpips=False)
    dy = 2 * (65.481 + 128.553 + 24.966) / 255
    assert m["mse_y"] == pytest.approx(dy ** 2)
    assert m["psnr_y"] == pytest.approx(10 * np.log10(255 ** 2 / dy ** 2))


@pytest.mark.parametrize("s", [2, 4])
def test_border_is_shaved(s):
    interior = IMG.copy()
    interior[s + 10, s + 10] = 255 - interior[s + 10, s + 10]
    with_border = interior.copy()
    with_border[:s] = 0; with_border[-s:] = 255; with_border[:, :s] = 0; with_border[:, -s:] = 255
    a = compute_metrics(IMG, interior, s)
    b = compute_metrics(IMG, with_border, s)
    for k in a:
        assert a[k] == pytest.approx(b[k]), k
    # Mudança só na borda: o interior é idêntico, MSE-Y = 0 -> erro explícito (sem sentinela).
    border_only = IMG.copy(); border_only[:s] = 0
    with pytest.raises(ValueError):
        compute_metrics(IMG, border_only, s)


def test_identical_interior_raises_and_ssim_one_otherwise():
    with pytest.raises(ValueError):
        compute_metrics(IMG, IMG, 2)
    near = IMG.copy(); near[10, 10, 0] ^= 1
    m = compute_metrics(IMG, near, 2)
    assert m["ssim_y"] == pytest.approx(1.0, abs=1e-3)
    assert m["lpips"] == pytest.approx(0.0, abs=1e-3)


def test_size_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics(IMG, IMG[:-2], 2, with_lpips=False)


def test_metrics_long():
    df = pd.read_csv(resolve(CFG["metricas"]["metrics_csv"]))
    assert list(df.columns) == ["image", "method", "family", "scale", "psnr_y", "ssim_y", "lpips", "mse_y"]
    methods = CFG["methods"]["dl"] + CFG["methods"]["classic"]
    assert len(df) == CFG["amostragem"]["n_sample"] * len(methods) * len(CFG["scales"]) == 1400
    assert not df.isna().any().any()
    assert np.isfinite(df[["psnr_y", "ssim_y", "lpips", "mse_y"]]).all().all()
    assert not df.duplicated(["image", "method", "scale"]).any()
    assert set(df["method"]) == set(methods) and set(df["scale"]) == set(CFG["scales"])
    assert ((df["family"] == "dl") == df["method"].isin(CFG["methods"]["dl"])).all()
    assert df["ssim_y"].between(-1, 1).all() and (df["lpips"] >= 0).all()
    # PSNR deriva do MSE (R14).
    assert np.allclose(df["psnr_y"], 10 * np.log10(255 ** 2 / df["mse_y"]))


def test_metrics_reproducible_for_one_row():
    import cv2
    df = pd.read_csv(resolve(CFG["metricas"]["metrics_csv"]))
    row = df.iloc[0]
    hr = cv2.imread(str(resolve(CFG["amostragem"]["hr_dir"]) / row["image"]), cv2.IMREAD_UNCHANGED)
    sr = cv2.imread(str(resolve(CFG["reconstrucao"]["sr_dir"]) / f"x{row['scale']}" / row["method"] / row["image"]),
                    cv2.IMREAD_UNCHANGED)
    m = compute_metrics(hr, sr, int(row["scale"]))
    for k in ("psnr_y", "ssim_y", "lpips", "mse_y"):
        assert m[k] == pytest.approx(row[k], rel=1e-9, abs=1e-9), k


def test_validation_criterion_set5_bicubic():
    t = pd.read_csv(resolve(V["table_csv"]))
    crit = t[(t["set"] == V["criterion_set"]) & (t["method"] == V["criterion_method"])]
    assert sorted(crit["scale"]) == sorted(V["scales"])
    assert (crit["diferenca_db"].abs() <= V["tolerance_db"]).all(), crit
