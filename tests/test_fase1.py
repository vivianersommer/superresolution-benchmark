"""Testes da fase 1: inventário, elegibilidade, amostragem e mod-crop."""
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import load_config, resolve  # noqa: E402
from f0_ambiente import sha256  # noqa: E402
from f1_amostragem import draw_sample  # noqa: E402

CFG = load_config()
A = CFG["amostragem"]
INV = pd.read_csv(resolve(A["inventory_csv"]), keep_default_na=False)
SAMPLE = pd.read_csv(resolve(A["sample_csv"]))


def test_inventory_covers_source():
    n_files = len(list(resolve(A["source_dir"]).glob(A["source_glob"])))
    assert len(INV) == n_files
    # Toda imagem é elegível ou tem um motivo de exclusão, nunca os dois.
    assert ((INV["motivo_exclusao"] == "") == INV["elegivel"]).all()


def test_sample_size_unique_and_eligible():
    assert len(SAMPLE) == A["n_sample"]
    assert SAMPLE["arquivo"].is_unique
    eligible = set(INV.loc[INV["elegivel"], "arquivo"])
    assert set(SAMPLE["arquivo"]) <= eligible


def test_sample_reproducible():
    again = draw_sample(CFG, INV)
    assert list(again["arquivo"]) == list(SAMPLE["arquivo"])


def test_sample_meets_criteria():
    inv = INV.set_index("arquivo").loc[SAMPLE["arquivo"]]
    assert (inv["canais"] == 3).all()
    assert not inv["rgb_cinza"].astype(bool).any()
    assert (SAMPLE[["largura", "altura"]].min(axis=1) >= A["min_side"]).all()


def test_source_hashes():
    src = resolve(A["source_dir"])
    for row in SAMPLE.itertuples():
        assert sha256(src / row.arquivo) == row.sha256, row.arquivo


def test_hr_modcrop():
    m, hr_dir = A["modcrop"], resolve(A["hr_dir"])
    assert sorted(p.name for p in hr_dir.glob("*.png")) == sorted(SAMPLE["hr_arquivo"])
    for row in SAMPLE.itertuples():
        path = hr_dir / row.hr_arquivo
        assert sha256(path) == row.hr_sha256, row.hr_arquivo
        hr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert hr.shape == (row.hr_altura, row.hr_largura, 3)
        assert row.hr_altura % m == 0 and row.hr_largura % m == 0
        assert row.hr_altura == row.altura - row.altura % m
        assert row.hr_largura == row.largura - row.largura % m
        # Mod-crop só corta: o canto superior esquerdo é idêntico ao original.
        orig = cv2.imread(str(resolve(A["source_dir"]) / row.arquivo), cv2.IMREAD_UNCHANGED)
        assert (orig[: row.hr_altura, : row.hr_largura] == hr).all()
        # A versão 4x reduzida tem pelo menos 64 px no menor lado.
        assert min(hr.shape[:2]) // max(CFG["scales"]) >= 64


def test_benchmarks():
    b = CFG["benchmarks"]
    assert sha256(resolve(b["download_dir"]) / Path(b["url"]).name) == b["sha256"]
    man = pd.read_csv(resolve(b["manifest_csv"]))
    hr = man[man["arquivo"].str.contains("/HR/")]
    assert (hr["arquivo"].str.startswith("Set5/").sum(), hr["arquivo"].str.startswith("Set14/").sum()) == (5, 14)
    root = resolve(b["extract_dir"])
    for row in man.itertuples():
        assert sha256(root / row.arquivo) == row.sha256, row.arquivo
