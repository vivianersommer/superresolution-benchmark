"""Fase 5 — inferência: H1 (Friedman + post hoc), H2 (contrastes planejados), H3 (interação).

Entrada: results/metrics_long.csv
Saídas:
  results/inferencia_friedman.csv       H1: χ², gl, p, p Holm (6 testes), W de Kendall
  results/inferencia_posthoc.csv        21 pares por métrica × escala: Wilcoxon, Holm, mediana
                                        da diferença, IC 95% bootstrap, rank-biserial
  results/inferencia_h2.csv             4 modelos − bicúbica por escala (8 por métrica, Holm),
                                        % de imagens em que o modelo supera a bicúbica
  results/inferencia_h3_contrastes.csv  contrastes "método − bicúbica" por escala e sua
                                        mudança entre 2× e 4× (modelo misto, ART, Wilcoxon)
  results/modelo_misto.txt              modelos mistos, LRT da interação, resíduos
  results/art_anova.txt                 robustez: ART em Python
  results/poder.txt                     análise de sensibilidade do poder
  results/figs/residuos_<métrica>.*     QQ-plot e resíduos vs. ajustados

Uso: .venv/bin/python src/f5_inferencia.py
"""
import io
import warnings
from contextlib import redirect_stdout
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import plot_style
from common import load_config, resolve
from plot_style import COLORS, LABELS, METHODS, plt

CFG = load_config()
S = CFG["statistics"]
I = CFG["inferencia"]
O = I["outputs"]
METRICS = S["metrics"]
REF = S["reference_method"]
SCALES = CFG["scales"]
ALPHA = S["alpha"]


# ----------------------------------------------------------------- utilidades

def load() -> pd.DataFrame:
    df = pd.read_csv(resolve(CFG["metricas"]["metrics_csv"]))
    if df[list(METRICS)].isna().any().any():
        raise RuntimeError("metrics_long com valores ausentes")
    return df


def wide(df: pd.DataFrame, metric: str, scale: int) -> pd.DataFrame:
    w = df[df["scale"] == scale].pivot(index="image", columns="method", values=metric)[METHODS]
    if w.isna().any().any():
        raise RuntimeError(f"Pareamento incompleto em {metric} x{scale}")
    return w


def holm(p) -> np.ndarray:
    return multipletests(np.asarray(p, dtype=float), alpha=ALPHA, method=I["correction"])[1]


def wilcoxon(d: np.ndarray) -> dict:
    """Wilcoxon pareado bilateral + correlação rank-biserial (mesmo tratamento de zeros)."""
    res = stats.wilcoxon(d, zero_method=I["wilcoxon_zero_method"], method=I["wilcoxon_method"])
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    r_plus, r_minus = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return {"W": float(res.statistic), "z": float(res.zstatistic), "p": float(res.pvalue),
            "n_zeros": int((d == 0).sum()),
            "rank_biserial": float((r_plus - r_minus) / (r_plus + r_minus))}


def boot_median_ci(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(d), size=(S["n_boot"], len(d)))
    meds = np.median(d[idx], axis=1)
    a = (1 - S["ci_level"]) / 2
    return float(np.quantile(meds, a)), float(np.quantile(meds, 1 - a))


def better(d: np.ndarray, metric: str) -> np.ndarray:
    return d > 0 if METRICS[metric]["higher_is_better"] else d < 0


# ------------------------------------------------------------------------ H1

def h1_friedman(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for s in SCALES:
            w = wide(df, metric, s)
            chi2, p = stats.friedmanchisquare(*[w[m].to_numpy() for m in METHODS])
            n, k = w.shape
            rows.append({"metric": metric, "scale": s, "n": n, "k": k, "chi2": chi2, "gl": k - 1,
                         "p": p, "kendall_w": chi2 / (n * (k - 1))})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    out["rejeita_H0"] = out["p_holm"] < ALPHA
    out.to_csv(resolve(O["friedman_csv"]), index=False)
    return out


def posthoc(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for s in SCALES:
            w = wide(df, metric, s)
            block = []
            for a, b in combinations(METHODS, 2):
                d = (w[a] - w[b]).to_numpy()
                lo, hi = boot_median_ci(d, rng)
                block.append({"metric": metric, "scale": s, "metodo_a": a, "metodo_b": b,
                              "mediana_dif_a_menos_b": float(np.median(d)), "ic95_low": lo, "ic95_high": hi,
                              **wilcoxon(d), "pct_a_melhor": 100 * better(d, metric).mean()})
            blk = pd.DataFrame(block)
            blk["p_holm"] = holm(blk["p"])
            rows.append(blk)
    out = pd.concat(rows, ignore_index=True)
    out["rejeita_H0"] = out["p_holm"] < ALPHA
    out.to_csv(resolve(O["posthoc_csv"]), index=False)
    return out


# ------------------------------------------------------------------------ H2

def h2_contrasts(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        block = []
        for s in SCALES:
            w = wide(df, metric, s)
            for m in I["h2_methods"]:
                d = (w[m] - w[REF]).to_numpy()
                lo, hi = boot_median_ci(d, rng)
                block.append({"metric": metric, "scale": s, "contraste": f"{m} - {REF}", "method": m,
                              "mediana_dif": float(np.median(d)), "ic95_low": lo, "ic95_high": hi,
                              "media_dif": float(d.mean()), "dz_observado": float(d.mean() / d.std(ddof=1)),
                              **wilcoxon(d), "pct_modelo_melhor": 100 * better(d, metric).mean(),
                              "pct_empates": 100 * (d == 0).mean()})
        blk = pd.DataFrame(block)
        blk["p_holm"] = holm(blk["p"])   # 8 testes por métrica
        rows.append(blk)
    out = pd.concat(rows, ignore_index=True)
    out["rejeita_H0"] = out["p_holm"] < ALPHA
    hib = out["metric"].map(lambda m: METRICS[m]["higher_is_better"])
    out["direcao"] = np.where(~out["rejeita_H0"], "sem diferença",
                              np.where((out["mediana_dif"] > 0) == hib, "modelo melhor", "bicúbica melhor"))
    out.to_csv(resolve(O["h2_csv"]), index=False)
    return out


# ------------------------------------------------------------------------ H3

def _prepare(df: pd.DataFrame, metric: str, response: str = "y") -> pd.DataFrame:
    d = df[["image", "method", "scale", metric]].rename(columns={metric: response}).copy()
    d["method"] = pd.Categorical(d["method"], categories=[REF] + [m for m in METHODS if m != REF])
    d["scale"] = pd.Categorical(d["scale"], categories=SCALES)
    d["img_scale"] = d["image"] + "_x" + d["scale"].astype(str)
    return d


def _fit(data: pd.DataFrame, formula: str, vc: bool, reml: bool):
    """Modelo misto com (1|imagem) e, se vc, também (1|imagem:escala).

    Ajusta com cada otimizador da lista do config e fica com o ajuste convergido de maior
    log-verossimilhança (um otimizador sozinho pode parar num ótimo pior).
    """
    # re_formula="1" explícito: com vc_formula, o statsmodels omite o intercepto de grupo se
    # re_formula não for informado (o modelo viraria só (1|imagem:escala)).
    kw = {"re_formula": "1", "vc_formula": {"img_scale": "0 + C(scale)"}} if vc else {}
    model = smf.mixedlm(formula, data, groups="image", **kw)
    fits, notes = [], []
    for opt in I["mixed"]["optimizers"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = model.fit(reml=reml, method=opt, maxiter=I["mixed"]["maxiter"])
            except (np.linalg.LinAlgError, ValueError) as e:
                notes.append(f"{opt}: erro {type(e).__name__}")
                continue
        if res.converged and np.isfinite(res.llf):
            fits.append((res.llf, opt, res))
        else:
            notes.append(f"{opt}: não convergiu")
        notes += [f"{opt}: {w.message}" for w in caught if issubclass(w.category, ConvergenceWarning)]
    if not fits:
        raise RuntimeError(f"Modelo misto não convergiu com nenhum otimizador: {formula} (vc={vc}, reml={reml})")
    llfs = [f[0] for f in fits]
    llf, opt, res = max(fits, key=lambda f: f[0])
    res.otimizador = opt
    res.dif_llf_otimizadores = float(max(llfs) - min(llfs))
    return res, notes


# Estruturas de efeitos aleatórios: principal e sensibilidade (modelo simplificado, só (1|imagem)).
STRUCTURES = {role: spec["vc"] for role, spec in I["mixed"]["random_effects"].items()}
SPEC_LABEL = {role: spec["label"] for role, spec in I["mixed"]["random_effects"].items()}


def _n_params(res) -> int:
    """Efeitos fixos + componentes de variância + variância residual."""
    return res.k_fe + res.k_re2 + res.k_vc + 1


def _wald_interaction(res):
    """Wald conjunto dos termos de interação, com matriz de restrições explícita."""
    names = list(res.model.exog_names)
    inter = [i for i, n in enumerate(names) if ":" in n]
    R = np.zeros((len(inter), len(res.params)))
    for row, i in enumerate(inter):
        R[row, i] = 1
    return res.wald_test(R, scalar=True)


def _term(m: str) -> str:
    return f"C(method)[T.{m}]"


def _contrasts(res, label: str, metric: str) -> list[dict]:
    """Contrastes método − bicúbica em cada escala e a mudança entre escalas (interação)."""
    names = list(res.model.exog_names)
    rows = []
    hi_scale = SCALES[-1]
    for m in I["h3_methods"]:
        # Codificação de tratamento (referências: bicúbica e a menor escala):
        #   contraste na escala baixa = β_método; na escala alta = β_método + β_método:escala;
        #   interação (mudança do contraste entre escalas) = β_método:escala.
        for what in ["x" + str(SCALES[0]), "x" + str(hi_scale), "interacao"]:
            L = np.zeros((1, len(names)))
            if what != "interacao":
                L[0, names.index(_term(m))] = 1
            if what != "x" + str(SCALES[0]):
                L[0, names.index(f"{_term(m)}:C(scale)[T.{hi_scale}]")] = 1
            t = res.t_test(L)
            rows.append({"metric": metric, "modelo": label, "method": m, "efeito": what,
                         "estimativa": float(np.squeeze(t.effect)), "ep": float(np.squeeze(t.sd)),
                         "z": float(np.squeeze(t.tvalue)), "p": float(np.squeeze(t.pvalue))})
    return rows


def _residual_plot(res, metric: str, response: str, role: str) -> None:
    resid, fitted = res.resid, res.fittedvalues
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    axes[0].scatter(osm, osr, s=6, color=COLORS["edsr"], linewidths=0)
    axes[0].plot(osm, slope * osm + intercept, color="#52514e", linewidth=1)
    axes[0].set_xlabel("quantis teóricos"); axes[0].set_ylabel("resíduos")
    axes[0].set_title("QQ-plot normal dos resíduos")
    axes[1].scatter(fitted, resid, s=6, color=COLORS["edsr"], alpha=0.5, linewidths=0)
    axes[1].axhline(0, color="#52514e", linewidth=1, linestyle="--")
    axes[1].set_xlabel("valores ajustados"); axes[1].set_ylabel("resíduos")
    axes[1].set_title("Resíduos vs. ajustados")
    fig.suptitle(f"Modelo misto {role} {SPEC_LABEL[role]} — {METRICS[metric]['label']}"
                 + (" (log)" if response == "log" else ""))
    fig.tight_layout()
    plot_style.save(fig, f"residuos_{metric}" + ("_log" if response == "log" else "") + f"_{role}")


def mixed_models(df: pd.DataFrame):
    """H3 por modelo misto.

    - Teste da interação: LRT com ajustes por ML (com vs. sem interação).
    - Contrastes e EPs: ajuste por REML do modelo completo.
    - Comparação das estruturas aleatórias: ajustes por REML (mesmos efeitos fixos), AIC REML
      e LRT na fronteira.
    """
    full_f = "y ~ C(method) * C(scale)"
    add_f = "y ~ C(method) + C(scale)"
    buf = io.StringIO()
    rows, lrt_rows, re_rows = [], [], []
    for metric, meta in METRICS.items():
        data = _prepare(df, metric)
        print(f"\n{'=' * 100}\n{meta['label']}\n{'=' * 100}", file=buf)
        variants = [("original", data)]
        if metric == "lpips":
            if (data["y"] <= 0).any():
                raise RuntimeError("LPIPS <= 0: log indefinido")
            variants.append(("log", data.assign(y=np.log(data["y"]))))
        skew_by = {}
        for response, d in variants:
            reml_fits = {}
            for role, vc in STRUCTURES.items():
                spec = SPEC_LABEL[role]
                full_ml, c1 = _fit(d, full_f, vc, reml=False)
                add_ml, c2 = _fit(d, add_f, vc, reml=False)
                full_reml, c3 = _fit(d, full_f, vc, reml=True)
                reml_fits[role] = full_reml
                lr = 2 * (full_ml.llf - add_ml.llf)
                gl = full_ml.k_fe - add_ml.k_fe
                p = float(stats.chi2.sf(lr, gl))
                r = full_reml.resid
                skew_by[(response, role)] = float(stats.skew(r))
                lrt_rows.append({"metric": metric, "resposta": response, "papel": role,
                                 "efeitos_aleatorios": spec, "LR_ML": lr, "gl": gl, "p": p,
                                 "assimetria_residuos_REML": float(stats.skew(r)),
                                 "curtose_residuos_REML": float(stats.kurtosis(r))})
                print(f"\n--- resposta: {response} | modelo {role}: {spec} | ajuste REML ---", file=buf)
                print(full_reml.summary(), file=buf)
                print(f"LRT da interação método × escala (ajustes ML): LR = {lr:.3f}, gl = {gl}, p = {p:.3g}",
                      file=buf)
                print(f"Resíduos (REML): assimetria = {stats.skew(r):.3f}, curtose em excesso = "
                      f"{stats.kurtosis(r):.3f}", file=buf)
                for nm_, ft in [("ML completo", full_ml), ("ML aditivo", add_ml), ("REML completo", full_reml)]:
                    print(f"Ajuste {nm_}: otimizador {ft.otimizador}; diferença máx. de logLik entre "
                          f"otimizadores convergidos = {ft.dif_llf_otimizadores:.3g}", file=buf)
                if c1 or c2 or c3:
                    print(f"Notas dos otimizadores: {c1 + c2 + c3}", file=buf)
                rows += _contrasts(full_reml, f"{response}|{role}", metric)
                _residual_plot(full_reml, metric, response, role)

            # Comparação das estruturas aleatórias (REML, mesmos efeitos fixos).
            p_, s_ = reml_fits["principal"], reml_fits["sensibilidade"]
            lr_re = 2 * (p_.llf - s_.llf)
            p_re = 0.5 * float(stats.chi2.sf(lr_re, 1))      # fronteira: ½χ²₀ + ½χ²₁
            aic_p = -2 * p_.llf + 2 * _n_params(p_)
            aic_s = -2 * s_.llf + 2 * _n_params(s_)
            re_rows.append({"metric": metric, "resposta": response, "ajuste": "REML",
                            "logLik_REML_sensibilidade": s_.llf, "logLik_REML_principal": p_.llf,
                            "LR_REML": lr_re, "p_fronteira": p_re,
                            "AIC_REML_sensibilidade": aic_s, "AIC_REML_principal": aic_p,
                            "delta_AIC": aic_p - aic_s,
                            "var_imagem_principal": float(p_.cov_re.iloc[0, 0]),
                            "var_imagem:escala_principal": float(p_.vcomp[0]),
                            "var_residual_principal": float(p_.scale),
                            "var_imagem_sensibilidade": float(s_.cov_re.iloc[0, 0]),
                            "var_residual_sensibilidade": float(s_.scale)})
            print(f"\nComparação das estruturas aleatórias ({response}, REML): LR = {lr_re:.2f}, "
                  f"p (fronteira) = {p_re:.3g}; AIC_REML {aic_s:.1f} (sensibilidade) -> {aic_p:.1f} (principal)",
                  file=buf)
        if metric == "lpips":
            thr = I["mixed"]["lpips_log_skew_threshold"]
            so, sl = skew_by[("original", "principal")], skew_by[("log", "principal")]
            verdict = ("log não corrige a assimetria; mantém-se a escala original" if abs(sl) >= abs(so)
                       else "log reduz a assimetria")
            print(f"\nLPIPS — critério de transformação no modelo principal: |assimetria| original = {abs(so):.2f} "
                  f"({'>' if abs(so) > thr else '<='} {thr}); com log = {abs(sl):.2f}. Conclusão: {verdict}.",
                  file=buf)
    contr = pd.DataFrame(rows)
    # Holm por métrica × modelo × efeito (6 métodos por família).
    contr["p_holm"] = np.nan
    for _, idx in contr.groupby(["metric", "modelo", "efeito"]).groups.items():
        contr.loc[idx, "p_holm"] = holm(contr.loc[idx, "p"])
    lrt = pd.DataFrame(lrt_rows)
    re_cmp = pd.DataFrame(re_rows)
    print("\n\nResumo dos testes de razão de verossimilhança da interação (ML):\n", file=buf)
    print(lrt.to_string(index=False), file=buf)
    print("\nComparação das estruturas de efeitos aleatórios (REML):\n", file=buf)
    print(re_cmp.to_string(index=False), file=buf)
    return contr, lrt, re_cmp, buf.getvalue()


# ------------------------------------------------------- robustez

def art_align(data: pd.DataFrame) -> pd.Series:
    """Alinhamento ART (Wobbrock et al., 2011) para a interação método × escala."""
    cell = data.groupby(["method", "scale"], observed=True)["y"].transform("mean")
    m_a = data.groupby("method", observed=True)["y"].transform("mean")
    m_b = data.groupby("scale", observed=True)["y"].transform("mean")
    grand = data["y"].mean()
    return (data["y"] - cell) + (cell - m_a - m_b + grand)


def art_and_rank_models(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    buf = io.StringIO()
    rows, contr_rows = [], []
    print("Robustez: ART implementada em Python.\n"
          "1) ART: resposta alinhada para a interação (Wobbrock et al., 2011), postos médios sobre\n"
          "   os 1.400 valores, mesmo modelo misto do principal, (1|imagem) + (1|imagem:escala), e,\n"
          "   como sensibilidade, só (1|imagem); só o termo de interação é interpretado. Teste: LRT\n"
          "   (ajustes ML) com vs. sem interação e Wald conjunto; contrastes do ajuste REML.\n"
          "2) Postos dentro de cada imagem × escala (1 = pior, 7 = melhor, no sentido da métrica).\n"
          "   O modelo misto é degenerado aqui (variância de imagem = 0), então usa-se OLS com EP\n"
          "   robusto agrupado por imagem; o teste principal é o Wald conjunto (o LRT do OLS supõe\n"
          "   independência e é só informativo).\n"
          "3) Contrastes não paramétricos de interação: para cada método, Wilcoxon pareado de\n"
          "   d_i = (método − bicúbica)_4× − (método − bicúbica)_2× por imagem, Holm por métrica.\n",
          file=buf)
    for metric, meta in METRICS.items():
        data = _prepare(df, metric)
        print(f"\n{'=' * 100}\n{meta['label']}\n{'=' * 100}", file=buf)
        sign = 1 if meta["higher_is_better"] else -1
        art = stats.rankdata(art_align(data))
        within = data.assign(v=sign * data["y"]).groupby(
            ["image", "scale"], observed=True)["v"].rank(method="average").to_numpy()
        responses = [(f"ART|{role}", art, vc) for role, vc in STRUCTURES.items()]
        responses.append(("postos dentro de imagem×escala", within, None))
        for label, yr, vc in responses:
            d = data.assign(y=yr)
            contr_fit = None
            if vc is None:
                # Postos dentro de cada imagem × escala somam sempre o mesmo valor: a variância do
                # intercepto aleatório de imagem é exatamente 0 e o modelo misto é degenerado
                # (Hessiana singular). Equivale a OLS; usa EP robusto agrupado por imagem.
                groups = d["image"].astype("category").cat.codes
                fit = lambda f: smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": groups})  # noqa: E731
                full, add = fit("y ~ C(method) * C(scale)"), fit("y ~ C(method) + C(scale)")
            else:
                full, _ = _fit(d, "y ~ C(method) * C(scale)", vc, reml=False)
                add, _ = _fit(d, "y ~ C(method) + C(scale)", vc, reml=False)
                contr_fit, _ = _fit(d, "y ~ C(method) * C(scale)", vc, reml=True)
            lr = 2 * (full.llf - add.llf)
            gl = len(full.model.exog_names) - len(add.model.exog_names)
            p = float(stats.chi2.sf(lr, gl))
            wald = _wald_interaction(full)
            rows.append({"metric": metric, "procedimento": label, "LR": lr, "gl": gl, "p_LRT": p,
                         "wald_chi2": float(wald.statistic), "p_wald": float(wald.pvalue)})
            print(f"\n--- {label} ---", file=buf)
            print(full.summary(), file=buf)
            if contr_fit is not None:
                for nm_, ft in [("ML completo", full), ("ML aditivo", add), ("REML completo", contr_fit)]:
                    print(f"Ajuste {nm_}: otimizador {ft.otimizador}; diferença máx. de logLik entre "
                          f"otimizadores convergidos = {ft.dif_llf_otimizadores:.3g}", file=buf)
            print(f"Interação: LRT LR = {lr:.3f}, gl = {gl}, p = {p:.3g}; "
                  f"Wald χ² = {float(wald.statistic):.3f}, p = {float(wald.pvalue):.3g}", file=buf)
            for c in _contrasts(contr_fit if contr_fit is not None else full, label, metric):
                if c["efeito"] == "interacao":
                    contr_rows.append(c)
        # Wilcoxon correspondente a cada contraste: (m − bic) em cada escala e a diferença das
        # diferenças (m − bic)_4× − (m − bic)_2× por imagem. Estimativa = mediana.
        ws = {s: wide(df, metric, s) for s in SCALES}
        for m in I["h3_methods"]:
            diffs = {f"x{s}": (ws[s][m] - ws[s][REF]).to_numpy() for s in SCALES}
            diffs["interacao"] = diffs[f"x{SCALES[-1]}"] - diffs[f"x{SCALES[0]}"]
            for what, dd in diffs.items():
                w = wilcoxon(dd)
                contr_rows.append({"metric": metric, "modelo": "Wilcoxon", "method": m, "efeito": what,
                                   "estimativa": float(np.median(dd)), "ep": np.nan, "z": w["z"], "p": w["p"]})
    contr = pd.DataFrame(contr_rows)
    for _, idx in contr.groupby(["metric", "modelo", "efeito"]).groups.items():
        contr.loc[idx, "p_holm"] = holm(contr.loc[idx, "p"])
    summary = pd.DataFrame(rows)
    print("\n\nResumo da interação (robustez):\n", file=buf)
    print(summary.to_string(index=False), file=buf)
    print("\nContrastes de interação por método (Holm por métrica × procedimento):\n", file=buf)
    print(contr.to_string(index=False), file=buf)
    return contr, summary, buf.getvalue()


# ------------------------------------------------------- tabela da H3

def _conclusion(est: float, p_holm: float, efeito: str, metric: str) -> str:
    if not p_holm < ALPHA:
        return "n.s."
    adv = est if METRICS[metric]["higher_is_better"] else -est   # > 0 = a favor do método
    if efeito == "interacao":
        return "vantagem aumenta" if adv > 0 else "vantagem diminui"
    return "método melhor" if adv > 0 else "bicúbica melhor"


def h3_table(h3: pd.DataFrame) -> pd.DataFrame:
    """Contrastes do modelo principal e do de sensibilidade lado a lado, com o Wilcoxon."""
    keys = ["metric", "method", "efeito"]
    pick = lambda modelo, cols: h3[h3["modelo"] == modelo].set_index(keys)[cols]  # noqa: E731
    t = pick("original|principal", ["estimativa", "ep", "p_holm"]).rename(
        columns={"ep": "ep_principal", "p_holm": "p_holm_principal"})
    t = t.join(pick("original|sensibilidade", ["ep", "p_holm"]).rename(
        columns={"ep": "ep_sensibilidade", "p_holm": "p_holm_sensibilidade"}))
    t = t.join(pick("Wilcoxon", ["estimativa", "p_holm"]).rename(
        columns={"estimativa": "wilcoxon_mediana", "p_holm": "p_holm_wilcoxon"}))
    t = t.reset_index()
    for src, est in [("principal", "estimativa"), ("sensibilidade", "estimativa"), ("wilcoxon", "wilcoxon_mediana")]:
        t[f"conclusao_{src}"] = [_conclusion(r[est], r[f"p_holm_{src}"], r["efeito"], r["metric"])
                                 for _, r in t.iterrows()]
    t["difere_principal_sensibilidade"] = t["conclusao_principal"] != t["conclusao_sensibilidade"]
    t["difere_principal_wilcoxon"] = t["conclusao_principal"] != t["conclusao_wilcoxon"]
    order = {m: i for i, m in enumerate(METRICS)}
    t = t.sort_values(["metric", "method", "efeito"], key=lambda c: c.map(order) if c.name == "metric" else c)
    t.to_csv(resolve(O["h3_table_csv"]), index=False)

    # Versão Markdown da tabela.
    fmt = lambda x: f"{x:.3g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.1e}"  # noqa: E731
    lines = ["| Métrica | Método | Efeito | Estimativa | EP principal | p Holm principal | EP simplificado | "
             "p Holm simplificado | Wilcoxon (mediana; p Holm) | Conclusão principal | Diverge |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in t.iterrows():
        flags = []
        if r["difere_principal_sensibilidade"]:
            flags.append(f"simplificado: {r['conclusao_sensibilidade']}")
        if r["difere_principal_wilcoxon"]:
            flags.append(f"Wilcoxon: {r['conclusao_wilcoxon']}")
        lines.append(f"| {METRICS[r['metric']]['label']} | {LABELS[r['method']]} | {r['efeito']} | "
                     f"{fmt(r['estimativa'])} | {fmt(r['ep_principal'])} | {fmt(r['p_holm_principal'])} | "
                     f"{fmt(r['ep_sensibilidade'])} | {fmt(r['p_holm_sensibilidade'])} | "
                     f"{fmt(r['wilcoxon_mediana'])}; {fmt(r['p_holm_wilcoxon'])} | {r['conclusao_principal']} | "
                     f"{'**' + '; '.join(flags) + '**' if flags else '—'} |")
    resolve(O["h3_table_md"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return t


# --------------------------------------------------------------------- poder

def power_analysis(h2: pd.DataFrame) -> str:
    P = I["power"]
    alpha_w = ALPHA / P["n_tests_worst_case"]
    tp = TTestPower()
    dz = tp.solve_power(effect_size=None, nobs=P["nobs"], alpha=alpha_w, power=P["power"],
                        alternative="two-sided")
    n_eff = P["nobs"] * P["wilcoxon_are"]
    dz_w = tp.solve_power(effect_size=None, nobs=n_eff, alpha=alpha_w, power=P["power"],
                          alternative="two-sided")
    lines = [
        "Análise de sensibilidade do poder (item 5.5)",
        "=" * 60,
        f"Desenho: teste pareado bilateral, n = {P['nobs']} pares.",
        f"α corrigido no pior caso de Holm: {ALPHA} / {P['n_tests_worst_case']} = {alpha_w:.5f}.",
        f"Poder alvo: {P['power']:.0%}.",
        "",
        f"Teste t pareado: menor efeito detectável dz = {dz:.3f}.",
        f"Wilcoxon (eficiência relativa assintótica ≈ {P['wilcoxon_are']} em relação ao t, sob normalidade):",
        f"  n efetivo ≈ {n_eff:.1f} -> dz mínimo ≈ {dz_w:.3f}.",
        "  (Para distribuições de caudas pesadas, a ERA do Wilcoxon pode ser > 1; o valor acima é",
        "   o caso normal, conservador para o Wilcoxon.)",
        "",
        "Comparação com os efeitos observados na H2 (dz = média da diferença / DP da diferença):",
    ]
    for r in h2.itertuples():
        flag = "acima" if abs(r.dz_observado) >= dz_w else "ABAIXO"
        lines.append(f"  {r.metric:7s} {r.scale}x {r.contraste:18s} dz = {r.dz_observado:+.3f} ({flag} do mínimo detectável)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- main

def main() -> None:
    plot_style.apply()
    df = load()
    rng = np.random.default_rng(CFG["seed"])

    fr = h1_friedman(df)
    ph = posthoc(df, rng)
    h2 = h2_contrasts(df, rng)

    contr, lrt, re_cmp, mixed_txt = mixed_models(df)
    art_contr, art_sum, art_txt = art_and_rank_models(df)
    h3 = pd.concat([contr, art_contr], ignore_index=True)
    h3.to_csv(resolve(O["h3_csv"]), index=False)
    table = h3_table(h3)
    re_cmp.to_csv(resolve(O["random_structure_csv"]), index=False)
    lrt.to_csv(resolve(O["lrt_csv"]), index=False)
    resolve(O["mixed_txt"]).write_text(mixed_txt, encoding="utf-8")
    resolve(O["art_txt"]).write_text(art_txt, encoding="utf-8")
    resolve(O["power_txt"]).write_text(power_analysis(h2), encoding="utf-8")

    with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 250,
                           "display.max_rows", 500):
        print(fr.to_string(index=False))
        print(h2[["metric", "scale", "contraste", "mediana_dif", "ic95_low", "ic95_high", "rank_biserial",
                  "p_holm", "pct_modelo_melhor", "direcao"]].to_string(index=False))
        print(lrt.to_string(index=False))
        print(re_cmp.to_string(index=False))
        print(art_sum.to_string(index=False))
        print(table[table["difere_principal_sensibilidade"] | table["difere_principal_wilcoxon"]][
            ["metric", "method", "efeito", "estimativa", "ep_principal", "ep_sensibilidade",
             "conclusao_principal", "conclusao_sensibilidade", "conclusao_wilcoxon"]].to_string(index=False))
        print(f"Post hoc: {int(ph['rejeita_H0'].sum())}/{len(ph)} pares significativos após Holm")
    print(resolve(O["power_txt"]).read_text())


if __name__ == "__main__":
    main()
