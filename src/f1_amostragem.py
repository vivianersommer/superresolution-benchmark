"""Fase 1 — inventário, elegibilidade, amostragem aleatória e mod-crop das HR.

Também baixa (se ausente) e extrai Set5/Set14 para a validação da fase 3.

Saídas:
  data/inventario_coco.csv     uma linha por imagem da pasta, com o motivo de exclusão
  data/amostra.csv             100 imagens sorteadas (seed do config)
  data/hr/<id>.png             HR com mod-crop (fora do git)
  data/benchmarks/<Set>/...    benchmarks extraídos (fora do git)
  data/benchmarks_manifest.csv SHA-256 de cada arquivo de benchmark extraído

Uso: .venv/bin/python src/f1_amostragem.py
"""
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import load_config, resolve
from f0_ambiente import sha256

# Leitura sem aplicar orientação EXIF: as dimensões ficam as do arquivo.
# (Nenhuma imagem da pasta tem tag de orientação != 1; verificado na fase 1.)
IMREAD_FLAGS = cv2.IMREAD_UNCHANGED


def inventory(cfg: dict) -> pd.DataFrame:
    a = cfg["amostragem"]
    src = resolve(a["source_dir"])
    rows = []
    for p in sorted(src.glob(a["source_glob"])):
        img = cv2.imread(str(p), IMREAD_FLAGS)
        if img is None:
            raise RuntimeError(f"Falha ao ler {p}")
        h, w = img.shape[:2]
        channels = 1 if img.ndim == 2 else img.shape[2]
        rgb_gray = bool(channels == 3 and (img[..., 0] == img[..., 1]).all()
                        and (img[..., 1] == img[..., 2]).all())
        if channels != 3:
            motivo = f"{channels} canal(is)"
        elif a["exclude_rgb_gray"] and rgb_gray:
            motivo = "cinza salva em RGB (R=G=B)"
        elif min(h, w) < a["min_side"]:
            motivo = f"menor lado < {a['min_side']} px"
        else:
            motivo = ""
        rows.append({"arquivo": p.name, "largura": w, "altura": h, "canais": channels,
                     "rgb_cinza": rgb_gray, "elegivel": motivo == "", "motivo_exclusao": motivo})
    df = pd.DataFrame(rows)
    df.to_csv(resolve(a["inventory_csv"]), index=False)
    return df


def draw_sample(cfg: dict, inv: pd.DataFrame) -> pd.DataFrame:
    a = cfg["amostragem"]
    eligible = inv[inv["elegivel"]].sort_values("arquivo").reset_index(drop=True)
    if len(eligible) < a["n_sample"]:
        raise RuntimeError(f"Só {len(eligible)} elegíveis para n = {a['n_sample']}")
    # Amostragem aleatória simples sem reposição sobre a lista ordenada por nome.
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(eligible), size=a["n_sample"], replace=False)
    sample = eligible.loc[idx, ["arquivo", "largura", "altura"]].reset_index(drop=True)
    sample.insert(0, "ordem_sorteio", np.arange(1, len(sample) + 1))
    src = resolve(a["source_dir"])
    sample["sha256"] = [sha256(src / f) for f in sample["arquivo"]]
    return sample


def modcrop(img: np.ndarray, m: int) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % m, : w - w % m]


def write_hr(cfg: dict, sample: pd.DataFrame) -> pd.DataFrame:
    a = cfg["amostragem"]
    src, hr_dir = resolve(a["source_dir"]), resolve(a["hr_dir"])
    if hr_dir.exists():
        shutil.rmtree(hr_dir)
    hr_dir.mkdir(parents=True)
    names, ws, hs, shas = [], [], [], []
    for f in sample["arquivo"]:
        img = cv2.imread(str(src / f), IMREAD_FLAGS)
        if img is None:
            raise RuntimeError(f"Falha ao ler {f}")
        hr = modcrop(img, a["modcrop"])
        out = hr_dir / (Path(f).stem + ".png")
        if not cv2.imwrite(str(out), hr):
            raise RuntimeError(f"Falha ao gravar {out}")
        names.append(out.name); hs.append(hr.shape[0]); ws.append(hr.shape[1]); shas.append(sha256(out))
    sample = sample.assign(hr_arquivo=names, hr_largura=ws, hr_altura=hs, hr_sha256=shas)
    sample.to_csv(resolve(a["sample_csv"]), index=False)
    return sample


def prepare_benchmarks(cfg: dict) -> pd.DataFrame:
    b = cfg["benchmarks"]
    dl_dir, out_dir = resolve(b["download_dir"]), resolve(b["extract_dir"])
    dl_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dl_dir / Path(b["url"]).name
    if not tar_path.exists():
        print(f"Baixando {b['url']} ...")
        tmp = tar_path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(b["url"], tmp)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Falha ao baixar {b['url']} ({e}). Baixe manualmente para {tar_path} "
                f"(por exemplo, curl -L -o {tar_path} {b['url']}); o SHA-256 será conferido.") from e
        tmp.rename(tar_path)
    digest = sha256(tar_path)
    if b["sha256"] and digest != b["sha256"]:
        raise RuntimeError(f"SHA-256 de {tar_path.name} não confere: {digest}")

    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers()
                   if m.isfile() and len(Path(m.name).parts) > 1
                   and Path(m.name).parts[1] in b["sets"]]
        for s in b["sets"]:
            if (out_dir / s).exists():
                shutil.rmtree(out_dir / s)
        # filter="data" bloqueia caminhos absolutos, links e saídas da pasta de destino.
        tar.extractall(out_dir, members=members, filter="data")

    rows = []
    for s in b["sets"]:
        # O pacote tem a raiz "benchmark/"; move cada conjunto para data/benchmarks/<Set>.
        shutil.move(out_dir / "benchmark" / s, out_dir / s)
        for p in sorted((out_dir / s).rglob("*")):
            if p.is_file():
                img = cv2.imread(str(p), IMREAD_FLAGS)
                if img is None:
                    raise RuntimeError(f"Falha ao ler {p}")
                rows.append({"arquivo": str(p.relative_to(out_dir)), "largura": img.shape[1],
                             "altura": img.shape[0], "canais": 1 if img.ndim == 2 else img.shape[2],
                             "sha256": sha256(p)})
    shutil.rmtree(out_dir / "benchmark")
    df = pd.DataFrame(rows)
    df.to_csv(resolve(b["manifest_csv"]), index=False)
    print(f"{tar_path.name}: sha256 = {digest}")
    return df


def main() -> None:
    cfg = load_config()
    resolve(cfg["paths"]["data_dir"]).mkdir(parents=True, exist_ok=True)

    inv = inventory(cfg)
    print(f"Inventário: {len(inv)} arquivos")
    print(inv.groupby(["canais", "rgb_cinza"]).size().to_string())
    print(inv["motivo_exclusao"].replace("", "elegível").value_counts().to_string())

    sample = write_hr(cfg, draw_sample(cfg, inv))
    print(sample.head(10).to_string(index=False))
    print(sample[["hr_largura", "hr_altura"]].describe().to_string())

    bench = prepare_benchmarks(cfg)
    print(bench.assign(conjunto=bench["arquivo"].str.split("/").str[:2].str.join("/"))
          .groupby("conjunto").size().to_string())


if __name__ == "__main__":
    main()
