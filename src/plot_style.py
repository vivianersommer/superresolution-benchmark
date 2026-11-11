"""Estilo comum das figuras (mesma paleta e fonte em todas as fases)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import load_config, resolve  # noqa: E402

CFG = load_config()
F = CFG["figuras"]
METHODS = CFG["methods"]["dl"] + CFG["methods"]["classic"]
COLORS = F["colors"]
LABELS = F["method_labels"]


def apply() -> None:
    plt.rcParams.update({
        "font.family": F["font_family"],
        "font.size": F["font_size"],
        "axes.edgecolor": F["text_secondary"],
        "axes.labelcolor": F["text_primary"],
        "axes.titlesize": F["font_size"] + 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": F["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": F["text_secondary"],
        "ytick.color": F["text_secondary"],
        "text.color": F["text_primary"],
        "legend.frameon": False,
        "lines.linewidth": 2,
        "savefig.bbox": "tight",
    })


def save(fig, name: str, subdir: str = "results/figs") -> list:
    out_dir = resolve(subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in F["formats"]:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=F["dpi"])
        paths.append(p)
    plt.close(fig)
    return paths
