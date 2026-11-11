"""Testes da fase 5: H1, post hoc, H2, H3 (modelo misto e robustez) e poder."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import load_config, resolve  # noqa: E402
from f5_inferencia import _fit, _prepare, art_align, load, wide, wilcoxon  # noqa: E402

CFG = load_config()
O = CFG["inferencia"]["outputs"]
METRICS = list(CFG["statistics"]["metrics"])
METHODS = CFG["methods"]["dl"] + CFG["methods"]["classic"]
DF = load()


def test_rank_biserial_extremes():
    assert wilcoxon(np.arange(1, 21, dtype=float))["rank_biserial"] == pytest.approx(1.0)
    assert wilcoxon(-np.arange(1, 21, dtype=float))["rank_biserial"] == pytest.approx(-1.0)


def test_friedman():
    fr = pd.read_csv(resolve(O["friedman_csv"]))
    assert len(fr) == len(METRICS) * len(CFG["scales"])
    assert (fr["p_holm"] >= fr["p"] - 1e-300).all() and fr["kendall_w"].between(0, 1).all()
    r = fr.iloc[0]
    w = wide(DF, r["metric"], int(r["scale"]))
    chi2, p = stats.friedmanchisquare(*[w[m] for m in METHODS])
    assert r["chi2"] == pytest.approx(chi2) and r["kendall_w"] == pytest.approx(chi2 / (100 * 6))


def test_posthoc():
    ph = pd.read_csv(resolve(O["posthoc_csv"]))
    assert len(ph) == 21 * len(METRICS) * len(CFG["scales"])
    assert ph.groupby(["metric", "scale"]).size().eq(21).all()
    assert ph["rank_biserial"].between(-1, 1).all()
    assert (ph["ic95_low"] <= ph["mediana_dif_a_menos_b"]).all()
    assert (ph["mediana_dif_a_menos_b"] <= ph["ic95_high"]).all()
    assert (ph["p_holm"] >= ph["p"]).all() and not ph.isna().any().any()


def test_h2():
    h2 = pd.read_csv(resolve(O["h2_csv"]))
    assert len(h2) == len(METRICS) * 8
    assert set(h2["method"]) == set(CFG["inferencia"]["h2_methods"])
    assert h2["pct_modelo_melhor"].between(0, 100).all()
    r = h2.iloc[0]
    w = wide(DF, r["metric"], int(r["scale"]))
    d = (w[r["method"]] - w["bicubic"]).to_numpy()
    assert r["p"] == pytest.approx(stats.wilcoxon(d, method="approx").pvalue)
    assert r["mediana_dif"] == pytest.approx(np.median(d))


def test_h3_contrast_algebra():
    h3 = pd.read_csv(resolve(O["h3_csv"]))
    mixed = h3[h3["modelo"].str.match(r"^(original|log)\|")]
    assert set(mixed["modelo"]) == {"original|principal", "original|sensibilidade",
                                    "log|principal", "log|sensibilidade"}
    for (metric, modelo, method), g in mixed.groupby(["metric", "modelo", "method"]):
        e = g.set_index("efeito")["estimativa"]
        assert e["x4"] == pytest.approx(e["x2"] + e["interacao"], abs=1e-8)
    # Desenho balanceado: contraste do modelo misto = diferença das médias de célula.
    d = DF[(DF["scale"] == 2)].pivot(index="image", columns="method", values="psnr_y")
    est = mixed[(mixed["metric"] == "psnr_y") & (mixed["method"] == "edsr") & (mixed["efeito"] == "x2")]
    assert np.allclose(est["estimativa"], (d["edsr"] - d["bicubic"]).mean(), atol=1e-6)


def test_art_alignment_removes_main_effects():
    data = _prepare(DF, "psnr_y")
    a = art_align(data)
    assert np.allclose(data.assign(a=a).groupby("method", observed=True)["a"].mean(), 0, atol=1e-9)
    assert np.allclose(data.assign(a=a).groupby("scale", observed=True)["a"].mean(), 0, atol=1e-9)


def test_text_outputs_and_power():
    for key in ("mixed_txt", "art_txt", "power_txt"):
        assert resolve(O[key]).stat().st_size > 0
    txt = resolve(O["power_txt"]).read_text()
    dz = float(re.search(r"dz = ([0-9.]+)\.", txt).group(1))
    assert 0.3 < dz < 0.45


def test_principal_model_has_both_random_intercepts():
    """(1|imagem) + (1|imagem:escala): os dois componentes de variância estimados e > 0."""
    res, _ = _fit(_prepare(DF, "psnr_y"), "y ~ C(method) * C(scale)", vc=True, reml=True)
    assert res.k_re2 == 1 and res.k_vc == 1
    assert float(res.cov_re.iloc[0, 0]) > 0 and float(res.vcomp[0]) > 0
    sens, _ = _fit(_prepare(DF, "psnr_y"), "y ~ C(method) * C(scale)", vc=False, reml=True)
    assert sens.k_re2 == 1 and sens.k_vc == 0


def test_random_structure_comparison_uses_reml():
    t = pd.read_csv(resolve(O["random_structure_csv"]))
    assert (t["ajuste"] == "REML").all()
    assert np.allclose(t["delta_AIC"], t["AIC_REML_principal"] - t["AIC_REML_sensibilidade"])


def test_h3_table():
    t = pd.read_csv(resolve(O["h3_table_csv"]))
    assert len(t) == len(METRICS) * len(CFG["inferencia"]["h3_methods"]) * 3
    assert not t[["estimativa", "ep_principal", "ep_sensibilidade", "p_holm_wilcoxon"]].isna().any().any()
    # Conclusões coerentes com os p ajustados.
    assert ((t["conclusao_principal"] == "n.s.") == (t["p_holm_principal"] >= 0.05)).all()
    assert resolve(O["h3_table_md"]).stat().st_size > 0
