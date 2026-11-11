"""Testes da fase 4: tabela descritiva, ICs bootstrap, normalidade e figuras."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import load_config, resolve  # noqa: E402
from f4_descritiva import load, median_cis  # noqa: E402

CFG = load_config()
D = CFG["descritiva"]
METRICS = list(CFG["statistics"]["metrics"])
METHODS = CFG["methods"]["dl"] + CFG["methods"]["classic"]
N_CELLS = len(METRICS) * len(CFG["scales"]) * len(METHODS)


def test_descriptive_table():
    t = pd.read_csv(resolve(D["table_csv"]))
    assert len(t) == N_CELLS
    assert "mse_y" not in set(t["metric"])            # MSE fora das análises
    assert not t.isna().any().any()
    assert (t["n"] == CFG["amostragem"]["n_sample"]).all()
    assert (t["min"] <= t["q1"]).all() and (t["q1"] <= t["median"]).all()
    assert (t["median"] <= t["q3"]).all() and (t["q3"] <= t["max"]).all()
    assert np.allclose(t["iqr"], t["q3"] - t["q1"])
    df = pd.read_csv(resolve(CFG["metricas"]["metrics_csv"]))
    for r in t.sample(10, random_state=CFG["seed"]).itertuples():
        x = df.loc[(df["scale"] == r.scale) & (df["method"] == r.method), r.metric]
        assert r.mean == pytest.approx(x.mean()) and r.sd == pytest.approx(x.std(ddof=1))
        assert r.median == pytest.approx(x.median())


def test_median_ci_valid_and_reproducible():
    ci = pd.read_csv(resolve(D["median_ci_csv"]))
    assert len(ci) == N_CELLS
    assert (ci["ci_low"] <= ci["median"]).all() and (ci["median"] <= ci["ci_high"]).all()
    again = median_cis(load())
    assert np.allclose(again[["median", "ci_low", "ci_high"]], ci[["median", "ci_low", "ci_high"]])


def test_normality_table():
    n = pd.read_csv(resolve(D["normality_csv"]))
    assert len(n) == len(METRICS) * len(CFG["scales"]) * (len(METHODS) - 1)
    assert CFG["statistics"]["reference_method"] not in set(n["method"])
    assert n["shapiro_p"].between(0, 1).all() and n["shapiro_W"].between(0, 1).all()


@pytest.mark.parametrize("name", [f"{k}_{m}" for k in ("boxplot", "dif_pareada", "interacao", "qq")
                                  for m in METRICS] + ["qualidade_tempo"])
def test_figures_exist(name):
    for ext in CFG["figuras"]["formats"]:
        p = resolve(D["figs_dir"]) / f"{name}.{ext}"
        assert p.exists() and p.stat().st_size > 0, p
