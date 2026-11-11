"""Fase 2 — degradação (HR -> LR por 1/s) e reconstrução com os 7 métodos.

Saídas:
  outputs/lr/x{s}/<id>.png              LR (imresize bicúbico do MATLAB, com antialias)
  outputs/sr/x{s}/<método>/<id>.png     reconstruções, todas no tamanho exato da HR
  results/runtime_long.csv              tempo de reconstrução (ms) por imagem × método × escala
  results/figs/grade_fase2.{png,pdf}    grade visual para conferência

Qualquer falha interrompe a execução (sem valores sentinela).

Uso: .venv/bin/python src/f2_reconstrucao.py
"""
import logging
import shutil
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common import load_config, resolve  # noqa: E402
from imresize_matlab import imresize  # noqa: E402
import plot_style  # noqa: E402

log = logging.getLogger("f2")


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Falha ao ler {path}")
    return img


def write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"Falha ao gravar {path}")


def load_model(cfg: dict, algorithm: str, scale: int):
    path = resolve(cfg["paths"]["models_dir"]) / cfg["model_files"][algorithm].format(s=scale)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(path))
    sr.setModel(algorithm, scale)
    return sr


def make_lr(cfg: dict, sample: pd.DataFrame) -> None:
    r = cfg["reconstrucao"]
    hr_dir, lr_root = resolve(cfg["amostragem"]["hr_dir"]), resolve(r["lr_dir"])
    for s in cfg["scales"]:
        for name in tqdm(sample["hr_arquivo"], desc=f"LR x{s}"):
            hr = read_image(hr_dir / name)
            lr = imresize(hr, 1 / s)
            assert lr.shape == (hr.shape[0] // s, hr.shape[1] // s, 3), (name, s, lr.shape)
            write_image(lr_root / f"x{s}" / name, lr)


def reconstruct(cfg: dict, sample: pd.DataFrame) -> pd.DataFrame:
    r = cfg["reconstrucao"]
    hr_dir, lr_root, sr_root = (resolve(cfg["amostragem"]["hr_dir"]), resolve(r["lr_dir"]),
                                resolve(r["sr_dir"]))
    rows = []
    for s in cfg["scales"]:
        lrs = {n: read_image(lr_root / f"x{s}" / n) for n in sample["hr_arquivo"]}
        hr_shapes = {n: read_image(hr_dir / n).shape for n in sample["hr_arquivo"]}

        methods = [("dl", m) for m in cfg["methods"]["dl"]] + [("classic", m) for m in cfg["methods"]["classic"]]
        for family, method in methods:
            if family == "dl":
                model = load_model(cfg, method, s)
                run = model.upsample
            else:
                interp = getattr(cv2, cfg["classic_interpolation"][method])
                run = lambda lr, interp=interp: cv2.resize(  # noqa: E731
                    lr, (lr.shape[1] * s, lr.shape[0] * s), interpolation=interp)
            first = next(iter(lrs.values()))
            for _ in range(r["warmup_runs"]):
                run(first)

            for name, lr in tqdm(lrs.items(), desc=f"x{s} {method}"):
                try:
                    t0 = time.perf_counter_ns()
                    out = run(lr)
                    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
                except Exception:
                    log.exception("Falha em %s / %s / x%d", name, method, s)
                    raise
                # Nenhum redimensionamento posterior: a saída tem de ter o tamanho da HR.
                assert out.shape == hr_shapes[name], (name, method, s, out.shape, hr_shapes[name])
                assert out.dtype == np.uint8, (name, method, s, out.dtype)
                write_image(sr_root / f"x{s}" / method / name, out)
                rows.append({"image": name, "method": method, "family": family, "scale": s,
                             "lr_altura": lr.shape[0], "lr_largura": lr.shape[1],
                             "runtime_ms": elapsed_ms})
    df = pd.DataFrame(rows)
    out_csv = resolve(r["runtime_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def visual_grid(cfg: dict, sample: pd.DataFrame) -> None:
    plot_style.apply()
    g = cfg["reconstrucao"]["grid"]
    hr_dir, sr_root = resolve(cfg["amostragem"]["hr_dir"]), resolve(cfg["reconstrucao"]["sr_dir"])
    names = list(sample.sort_values("ordem_sorteio")["hr_arquivo"][: g["n_images"]])
    methods = cfg["methods"]["dl"] + cfg["methods"]["classic"]
    cols = ["HR"] + methods
    rows = [(s, n) for s in cfg["scales"] for n in names]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.6 * len(cols), 1.7 * len(rows)))
    c = g["crop_hr_px"]
    for i, (s, n) in enumerate(rows):
        hr = read_image(hr_dir / n)
        y0, x0 = (hr.shape[0] - c) // 2, (hr.shape[1] - c) // 2
        for j, col in enumerate(cols):
            img = hr if col == "HR" else read_image(sr_root / f"x{s}" / col / n)
            ax = axes[i, j]
            ax.imshow(cv2.cvtColor(img[y0:y0 + c, x0:x0 + c], cv2.COLOR_BGR2RGB), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if i == 0:
                ax.set_title(plot_style.LABELS.get(col, col), fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{s}×\n{Path(n).stem[-6:]}", fontsize=8)
    fig.suptitle(f"Fase 2 — recorte central {c}×{c} px da HR", fontsize=10)
    fig.tight_layout()
    plot_style.save(fig, g["name"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    r = cfg["reconstrucao"]
    if r["cv2_num_threads"] is not None:
        cv2.setNumThreads(r["cv2_num_threads"])
    log.info("cv2 threads: %d", cv2.getNumThreads())

    sample = pd.read_csv(resolve(cfg["amostragem"]["sample_csv"]))
    for d in (r["lr_dir"], r["sr_dir"]):
        if resolve(d).exists():
            shutil.rmtree(resolve(d))

    t0 = time.perf_counter()
    make_lr(cfg, sample)
    log.info("LR geradas em %.1f s", time.perf_counter() - t0)
    t0 = time.perf_counter()
    df = reconstruct(cfg, sample)
    log.info("Reconstruções: %d em %.1f s", len(df), time.perf_counter() - t0)
    visual_grid(cfg, sample)
    print(df.groupby(["scale", "method"])["runtime_ms"].median().unstack(0).round(2).to_string())


if __name__ == "__main__":
    main()
