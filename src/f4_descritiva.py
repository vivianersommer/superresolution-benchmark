"""Fase 4 — estatística descritiva, figuras e forma das diferenças pareadas.

Entrada: results/metrics_long.csv (+ results/runtime_long.csv para o gráfico de custo).
Saídas:
  results/descritiva.csv               n, média, DP, mediana, Q1, Q3, IQR, mín., máx.
  results/medianas_ic.csv              mediana com IC 95% bootstrap por método × escala × métrica
  results/normalidade_diferencas.csv   Shapiro-Wilk das diferenças (método − bicúbica)
  results/figs/boxplot_<métrica>.*     boxplots por método, um painel por escala
  results/figs/dif_pareada_<métrica>.* diferença pareada em relação à bicúbica, por imagem
  results/figs/interacao_<métrica>.*   mediana (IC 95%) por método e escala
  results/figs/qq_<métrica>.*          QQ-plots das diferenças pareadas
  results/figs/qualidade_tempo.*       PSNR-Y mediano vs. tempo mediano 

Uso: .venv/bin/python src/f4_descritiva.py
"""
import numpy as np
import pandas as pd
from scipy import stats

import plot_style
from common import load_config, resolve
from plot_style import COLORS, LABELS, METHODS, plt

CFG = load_config()
S = CFG["statistics"]
D = CFG["descritiva"]
METRICS = S["metrics"]
REF = S["reference_method"]
SCALES = CFG["scales"]


def load() -> pd.DataFrame:
    df = pd.read_csv(resolve(CFG["metricas"]["metrics_csv"]))
    if df[list(METRICS)].isna().any().any():
        raise RuntimeError("metrics_long com valores ausentes")
    return df


def wide(df: pd.DataFrame, metric: str, scale: int) -> pd.DataFrame:
    """Imagens nas linhas, métodos nas colunas (blocos pareados)."""
    w = df[df["scale"] == scale].pivot(index="image", columns="method", values=metric)[METHODS]
    if w.isna().any().any():
        raise RuntimeError(f"Pareamento incompleto em {metric} x{scale}")
    return w


# ------------------------------------------------------------------ tabelas

def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for s in SCALES:
            for m in METHODS:
                x = df.loc[(df["scale"] == s) & (df["method"] == m), metric]
                q1, med, q3 = x.quantile([0.25, 0.5, 0.75])  # quantis tipo 7 (interpolação linear)
                rows.append({"metric": metric, "scale": s, "method": m,
                             "family": "dl" if m in CFG["methods"]["dl"] else "classic",
                             "n": len(x), "mean": x.mean(), "sd": x.std(ddof=1), "median": med,
                             "q1": q1, "q3": q3, "iqr": q3 - q1, "min": x.min(), "max": x.max()})
    out = pd.DataFrame(rows)
    out.to_csv(resolve(D["table_csv"]), index=False)
    return out


def bootstrap_median_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(S["n_boot"], len(x)))
    meds = np.median(x[idx], axis=1)
    a = (1 - S["ci_level"]) / 2
    return float(np.quantile(meds, a)), float(np.quantile(meds, 1 - a))


def median_cis(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(CFG["seed"])
    rows = []
    for metric in METRICS:
        for s in SCALES:
            for m in METHODS:
                x = df.loc[(df["scale"] == s) & (df["method"] == m), metric].to_numpy()
                lo, hi = bootstrap_median_ci(x, rng)
                rows.append({"metric": metric, "scale": s, "method": m, "median": np.median(x),
                             "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows)
    out.to_csv(resolve(D["median_ci_csv"]), index=False)
    return out


def normality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for s in SCALES:
            w = wide(df, metric, s)
            for m in METHODS:
                if m == REF:
                    continue
                d = (w[m] - w[REF]).to_numpy()
                W, p = stats.shapiro(d)
                rows.append({"metric": metric, "scale": s, "method": m, "contraste": f"{m} - {REF}",
                             "n": len(d), "shapiro_W": W, "shapiro_p": p,
                             "assimetria": stats.skew(d), "curtose_excesso": stats.kurtosis(d),
                             "normal_5pct": p >= S["alpha"]})
    out = pd.DataFrame(rows)
    out.to_csv(resolve(D["normality_csv"]), index=False)
    return out


# ------------------------------------------------------------------ figuras

def _better_note(metric: str) -> str:
    return "maior = melhor" if METRICS[metric]["higher_is_better"] else "menor = melhor"


def fig_boxplots(df: pd.DataFrame) -> None:
    for metric, meta in METRICS.items():
        fig, axes = plt.subplots(1, len(SCALES), figsize=(7.2, 3.2), sharey=False)
        for ax, s in zip(axes, SCALES):
            w = wide(df, metric, s)
            bp = ax.boxplot([w[m] for m in METHODS], patch_artist=True, widths=0.6,
                            medianprops={"color": "#0b0b0b", "linewidth": 1.5},
                            whiskerprops={"color": "#52514e", "linewidth": 1},
                            capprops={"color": "#52514e", "linewidth": 1},
                            flierprops={"marker": "o", "markersize": 3, "markeredgewidth": 0.5,
                                        "markeredgecolor": "#52514e", "markerfacecolor": "none"})
            for patch, m in zip(bp["boxes"], METHODS):
                patch.set(facecolor=COLORS[m], edgecolor="white", linewidth=1, alpha=0.9)
            ax.set_xticks(range(1, len(METHODS) + 1), [LABELS[m] for m in METHODS], rotation=45, ha="right")
            ax.set_title(f"{s}×")
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel(f"{meta['label']} ({_better_note(metric)})")
        fig.suptitle(f"{meta['label']} por método (n = 100 imagens por caixa)")
        fig.tight_layout()
        plot_style.save(fig, f"boxplot_{metric}")


def fig_paired_differences(df: pd.DataFrame) -> None:
    rng = np.random.default_rng(CFG["seed"])
    others = [m for m in METHODS if m != REF]
    for metric, meta in METRICS.items():
        fig, axes = plt.subplots(1, len(SCALES), figsize=(7.2, 3.2))
        for ax, s in zip(axes, SCALES):
            w = wide(df, metric, s)
            for i, m in enumerate(others):
                d = (w[m] - w[REF]).to_numpy()
                jitter = rng.uniform(-0.18, 0.18, len(d))
                ax.scatter(i + jitter, d, s=9, color=COLORS[m], alpha=0.55, linewidths=0)
                ax.hlines(np.median(d), i - 0.3, i + 0.3, color="#0b0b0b", linewidth=2)
            ax.axhline(0, color="#52514e", linewidth=1, linestyle="--")
            ax.set_xticks(range(len(others)), [LABELS[m] for m in others], rotation=45, ha="right")
            ax.set_title(f"{s}×")
            ax.grid(axis="x", visible=False)
        sign = ">" if meta["higher_is_better"] else "<"
        axes[0].set_ylabel(f"Δ {meta['label']} (método − bicúbica)")
        fig.suptitle(f"Diferença pareada em relação à bicúbica, por imagem "
                     f"(Δ {sign} 0 = melhor que a bicúbica; traço = mediana)")
        fig.tight_layout()
        plot_style.save(fig, f"dif_pareada_{metric}")


def fig_interaction(ci: pd.DataFrame) -> None:
    for metric, meta in METRICS.items():
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        sub = ci[ci["metric"] == metric]
        offsets = np.linspace(-0.12, 0.12, len(METHODS))
        for off, m in zip(offsets, METHODS):
            r = sub[sub["method"] == m].sort_values("scale")
            x = np.arange(len(SCALES)) + off
            ax.errorbar(x, r["median"], yerr=[r["median"] - r["ci_low"], r["ci_high"] - r["median"]],
                        color=COLORS[m], marker="o", markersize=5, capsize=2, linewidth=2,
                        elinewidth=1, label=LABELS[m])
        ax.set_xticks(range(len(SCALES)), [f"{s}×" for s in SCALES])
        ax.set_xlim(-0.4, len(SCALES) - 0.6)
        ax.set_xlabel("Fator de escala")
        ax.set_ylabel(f"Mediana de {meta['label']} ({_better_note(metric)})")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        ax.set_title("Mediana e IC 95% bootstrap por método e escala")
        fig.tight_layout()
        plot_style.save(fig, f"interacao_{metric}")


def fig_qq(df: pd.DataFrame) -> None:
    others = [m for m in METHODS if m != REF]
    for metric, meta in METRICS.items():
        fig, axes = plt.subplots(len(SCALES), len(others), figsize=(10, 3.6), sharex=True)
        for i, s in enumerate(SCALES):
            w = wide(df, metric, s)
            for j, m in enumerate(others):
                ax = axes[i, j]
                d = (w[m] - w[REF]).to_numpy()
                (osm, osr), (slope, intercept, _) = stats.probplot(d, dist="norm")
                ax.scatter(osm, osr, s=8, color=COLORS[m], linewidths=0)
                ax.plot(osm, slope * osm + intercept, color="#52514e", linewidth=1)
                ax.tick_params(labelsize=7)
                if i == 0:
                    ax.set_title(LABELS[m], fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"{s}×\nquantis amostrais", fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("quantis teóricos", fontsize=8)
        fig.suptitle(f"QQ-plot normal das diferenças pareadas (método − bicúbica): {meta['label']}")
        fig.tight_layout()
        plot_style.save(fig, f"qq_{metric}")


LABEL_OFFSETS = {(2, "espcn"): ((5, -12), "left"), (4, "espcn"): ((-6, -12), "right")}


def fig_quality_time(df: pd.DataFrame) -> None:
    rt = pd.read_csv(resolve(CFG["reconstrucao"]["runtime_csv"]))
    q = df.groupby(["scale", "method"])["psnr_y"].median().rename("psnr")
    t = rt.groupby(["scale", "method"])["runtime_ms"].median().rename("ms")
    j = pd.concat([q, t], axis=1).reset_index()
    fig, axes = plt.subplots(1, len(SCALES), figsize=(7.2, 3.2))
    for ax, s in zip(axes, SCALES):
        r = j[j["scale"] == s]
        for _, row in r.iterrows():
            ax.scatter(row["ms"], row["psnr"], s=40, color=COLORS[row["method"]],
                       edgecolors="white", linewidths=1.5, zorder=3)
            # ESPCN fica colado a outros pontos: rótulo abaixo (à direita em 2x, à esquerda em 4x).
            xy, ha = LABEL_OFFSETS.get((s, row["method"]), ((5, 4), "left"))
            ax.annotate(LABELS[row["method"]], (row["ms"], row["psnr"]), textcoords="offset points",
                        xytext=xy, ha=ha, fontsize=7, color="#52514e")
        ax.set_xscale("log")
        ax.set_title(f"{s}×")
        ax.set_xlabel("Tempo mediano por imagem (ms, CPU, escala log)")
    axes[0].set_ylabel("PSNR-Y mediano (dB)")
    fig.suptitle("Qualidade vs. tempo de reconstrução")
    fig.tight_layout()
    plot_style.save(fig, "qualidade_tempo")


def main() -> None:
    plot_style.apply()
    df = load()
    desc = descriptive_table(df)
    ci = median_cis(df)
    norm = normality(df)
    fig_boxplots(df)
    fig_paired_differences(df)
    fig_interaction(ci)
    fig_qq(df)
    fig_quality_time(df)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 220):
        print(desc[["metric", "scale", "method", "mean", "sd", "median", "iqr"]].to_string(index=False))
        print(norm[["metric", "scale", "method", "shapiro_W", "shapiro_p", "assimetria"]].to_string(index=False))
        print(f"Diferenças compatíveis com normalidade (p >= {S['alpha']}): "
              f"{int(norm['normal_5pct'].sum())}/{len(norm)}")


if __name__ == "__main__":
    main()
