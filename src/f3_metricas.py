"""Fase 3 — métricas (PSNR-Y, SSIM-Y, LPIPS, MSE-Y) e validação do pipeline em Set5/Set14.

Saídas:
  results/metrics_long.csv          image, method, family, scale, psnr_y, ssim_y, lpips, mse_y
  results/validacao_por_imagem.csv  métricas por imagem dos benchmarks
  results/validacao_tabela.csv      PSNR-Y médio vs. publicado, com a diferença

A validação roda antes das métricas do COCO. Se o critério (bicúbica do MATLAB no Set5 a
no máximo 0,1 dB do publicado) falhar, a execução é interrompida.

Uso: .venv/bin/python src/f3_metricas.py
"""
import logging
import time
from functools import lru_cache

import cv2
import lpips
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity
from tqdm import tqdm

from common import load_config, resolve
from f2_reconstrucao import load_model, read_image
from imresize_matlab import imresize

log = logging.getLogger("f3")
CFG = load_config()
M = CFG["metricas"]


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Imagens de 1 canal (ex.: Set14/bridge) viram BGR com 3 canais iguais."""
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def rgb2y(bgr: np.ndarray) -> np.ndarray:
    """Canal Y do YCbCr BT.601 (faixa 16–235), como o rgb2ycbcr do MATLAB."""
    b, g, r = (bgr[..., i].astype(np.float64) for i in range(3))
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return np.floor(y + 0.5) if M["y_round"] else y


def shave(img: np.ndarray, s: int) -> np.ndarray:
    return img[s:-s, s:-s]


@lru_cache(maxsize=1)
def _lpips_model():
    torch.manual_seed(CFG["seed"])
    model = lpips.LPIPS(net=M["lpips"]["net"], version=M["lpips"]["version"], verbose=False)
    return model.eval()


def lpips_dist(hr_bgr: np.ndarray, sr_bgr: np.ndarray) -> float:
    def tensor(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        return float(_lpips_model()(tensor(hr_bgr), tensor(sr_bgr)).item())


def compute_metrics(hr: np.ndarray, sr: np.ndarray, s: int, with_lpips: bool = True) -> dict:
    hr, sr = to_bgr(hr), to_bgr(sr)
    if hr.shape != sr.shape:
        raise ValueError(f"Tamanhos diferentes: HR {hr.shape} vs SR {sr.shape}")
    y_hr, y_sr = shave(rgb2y(hr), s), shave(rgb2y(sr), s)
    mse = float(np.mean((y_hr - y_sr) ** 2))
    if mse == 0:
        raise ValueError("MSE-Y = 0: PSNR indefinido (SR idêntica à HR)")
    L = M["data_range"]
    out = {
        "psnr_y": 10.0 * np.log10(L ** 2 / mse),
        "ssim_y": float(structural_similarity(y_hr, y_sr, data_range=L, **M["ssim"])),
        "mse_y": mse,
    }
    if with_lpips:
        out["lpips"] = lpips_dist(shave(hr, s), shave(sr, s))
    for k, v in out.items():
        if not np.isfinite(v):
            raise ValueError(f"Métrica {k} não finita: {v}")
    return out


# ----------------------------------------------------------------- validação

def _upscale(method: str, lr: np.ndarray, s: int) -> np.ndarray:
    if method == "bicubic_matlab":
        return imresize(lr, s)
    if method in CFG["classic_interpolation"]:
        interp = getattr(cv2, CFG["classic_interpolation"][method])
        return cv2.resize(lr, (lr.shape[1] * s, lr.shape[0] * s), interpolation=interp)
    return _model(method, s).upsample(lr)


@lru_cache(maxsize=None)
def _model(method: str, s: int):
    return load_model(CFG, method, s)


def validate() -> pd.DataFrame:
    v = CFG["validacao"]
    root = resolve(CFG["benchmarks"]["extract_dir"])
    rows = []
    for st in v["sets"]:
        for p in sorted((root / st / "HR").glob("*.png")):
            hr_full = to_bgr(read_image(p))
            for s in v["scales"]:
                h, w = hr_full.shape[:2]
                hr = hr_full[: h - h % s, : w - w % s]
                lr = imresize(hr, 1 / s)
                for m in v["methods"]:
                    sr = _upscale(m, lr, s)
                    assert sr.shape == hr.shape, (st, p.name, m, s, sr.shape, hr.shape)
                    rows.append({"set": st, "image": p.stem, "method": m, "scale": s,
                                 **compute_metrics(hr, sr, s)})
    per_image = pd.DataFrame(rows)
    per_image.to_csv(resolve(v["per_image_csv"]), index=False)

    table = per_image.groupby(["set", "method", "scale"], as_index=False)[["psnr_y", "ssim_y"]].mean()
    pub = v["published_psnr_y"]

    def published(r):
        key = "bicubic" if r["method"] in ("bicubic", "bicubic_matlab") else r["method"]
        return pub.get(r["set"], {}).get(key, {}).get(r["scale"], np.nan)
    table["psnr_y_publicado"] = table.apply(published, axis=1)
    table["diferenca_db"] = table["psnr_y"] - table["psnr_y_publicado"]
    table.to_csv(resolve(v["table_csv"]), index=False)
    return table


def check_criterion(table: pd.DataFrame) -> None:
    v = CFG["validacao"]
    crit = table[(table["set"] == v["criterion_set"]) & (table["method"] == v["criterion_method"])]
    if len(crit) != len(v["scales"]):
        raise RuntimeError("Critério de validação sem todas as escalas")
    bad = crit[crit["diferenca_db"].abs() > v["tolerance_db"]]
    if len(bad):
        raise RuntimeError(f"Validação falhou (|Δ| > {v['tolerance_db']} dB):\n{bad.to_string()}")
    log.info("Critério de validação OK: %s", crit[["scale", "psnr_y", "diferenca_db"]].to_dict("records"))


# ------------------------------------------------------------------- COCO

def coco_metrics() -> pd.DataFrame:
    r = CFG["reconstrucao"]
    sample = pd.read_csv(resolve(CFG["amostragem"]["sample_csv"]))
    hr_dir, sr_root = resolve(CFG["amostragem"]["hr_dir"]), resolve(r["sr_dir"])
    families = {m: f for f in ("dl", "classic") for m in CFG["methods"][f]}
    rows = []
    for name in tqdm(sample["hr_arquivo"], desc="métricas COCO"):
        hr = read_image(hr_dir / name)
        for s in CFG["scales"]:
            for m, fam in families.items():
                try:
                    met = compute_metrics(hr, read_image(sr_root / f"x{s}" / m / name), s)
                except Exception:
                    log.exception("Falha em %s / %s / x%d", name, m, s)
                    raise
                rows.append({"image": name, "method": m, "family": fam, "scale": s, **met})
    cols = ["image", "method", "family", "scale", "psnr_y", "ssim_y", "lpips", "mse_y"]
    df = pd.DataFrame(rows)[cols]
    expected = len(sample) * len(families) * len(CFG["scales"])
    if len(df) != expected or df.isna().any().any():
        raise RuntimeError(f"metrics_long com {len(df)} linhas (esperado {expected}) ou valores ausentes")
    df.to_csv(resolve(M["metrics_csv"]), index=False)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.perf_counter()
    table = validate()
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(table.to_string(index=False))
    log.info("Validação em %.1f s", time.perf_counter() - t0)
    check_criterion(table)

    t0 = time.perf_counter()
    df = coco_metrics()
    log.info("Métricas COCO: %d linhas em %.1f s", len(df), time.perf_counter() - t0)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(df.groupby(["scale", "method"])[["psnr_y", "ssim_y", "lpips"]].mean().to_string())


if __name__ == "__main__":
    main()
