"""Fase 3 — diagnóstico do déficit dos modelos do OpenCV em relação aos artigos (Set5).

Não entra na análise principal. Verifica duas hipóteses de erro do pipeline e uma de
incompatibilidade de degradação:
  1. desalinhamento: PSNR-Y com a saída deslocada de -1, 0 e +1 px;
  2. ordem de canais: entrada BGR (padrão) vs RGB;
  3. degradação: PSNR-Y de cada modelo com LR gerada por imresize do MATLAB (protocolo),
     cv2 INTER_CUBIC e cv2 INTER_AREA.

Saída: results/validacao_diagnostico.csv
Uso: .venv/bin/python src/f3_diagnostico.py
"""
import cv2
import numpy as np
import pandas as pd

from common import load_config, resolve
from f2_reconstrucao import load_model, read_image
from f3_metricas import rgb2y, shave, to_bgr
from imresize_matlab import imresize

CFG = load_config()


def psnr_y(hr, sr, s, dy=0, dx=0):
    a, b = rgb2y(hr), np.roll(rgb2y(sr), (dy, dx), (0, 1))
    m = s + 1  # margem extra para o np.roll não entrar na região avaliada
    return 10 * np.log10(255 ** 2 / np.mean((a[m:-m, m:-m] - b[m:-m, m:-m]) ** 2))


DOWNSCALE = {
    "matlab_imresize_aa": lambda hr, s: imresize(hr, 1 / s),
    "cv2_INTER_CUBIC": lambda hr, s: cv2.resize(hr, (hr.shape[1] // s, hr.shape[0] // s),
                                                interpolation=cv2.INTER_CUBIC),
    "cv2_INTER_AREA": lambda hr, s: cv2.resize(hr, (hr.shape[1] // s, hr.shape[0] // s),
                                               interpolation=cv2.INTER_AREA),
}


def main() -> None:
    root = resolve(CFG["benchmarks"]["extract_dir"]) / "Set5" / "HR"
    rows = []
    for s in CFG["scales"]:
        models = {m: load_model(CFG, m, s) for m in CFG["methods"]["dl"]}
        for p in sorted(root.glob("*.png")):
            full = to_bgr(read_image(p))
            hr = full[: full.shape[0] - full.shape[0] % s, : full.shape[1] - full.shape[1] % s]
            for down, f in DOWNSCALE.items():
                lr = f(hr, s)
                ups = {"bicubic_matlab": imresize(lr, s), **{m: md.upsample(lr) for m, md in models.items()}}
                for m, sr in ups.items():
                    rows.append({"teste": "degradacao", "lr": down, "method": m, "scale": s,
                                 "image": p.stem, "variante": "", "psnr_y": psnr_y(hr, sr, s)})
                if down != "matlab_imresize_aa":
                    continue
                for m, md in models.items():
                    rgb = md.upsample(np.ascontiguousarray(lr[..., ::-1]))[..., ::-1]
                    rows.append({"teste": "canais", "lr": down, "method": m, "scale": s,
                                 "image": p.stem, "variante": "entrada_rgb", "psnr_y": psnr_y(hr, rgb, s)})
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            rows.append({"teste": "deslocamento", "lr": down, "method": m, "scale": s,
                                         "image": p.stem, "variante": f"dy={dy},dx={dx}",
                                         "psnr_y": psnr_y(hr, ups[m], s, dy, dx)})
    df = pd.DataFrame(rows)
    summary = df.groupby(["teste", "lr", "variante", "method", "scale"], as_index=False)["psnr_y"].mean()
    summary.to_csv(resolve("results/validacao_diagnostico.csv"), index=False)
    deg = summary[summary["teste"] == "degradacao"].pivot_table(index=["scale", "lr"], columns="method", values="psnr_y")
    print(deg.round(2).to_string())
    sh = summary[summary["teste"] == "deslocamento"]
    best = sh.loc[sh.groupby(["method", "scale"])["psnr_y"].idxmax(), ["method", "scale", "variante", "psnr_y"]]
    print(best.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
