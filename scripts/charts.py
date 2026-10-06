"""Render README charts from results/*.json into docs/.

Usage: uv run scripts/charts.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DOCS = ROOT / "docs"

SURFACE = "#fcfcfb"
INK = "#1d1c1a"
INK_2 = "#52514e"
GRID = "#e4e2dd"
TABPFN = "#2a78d6"   # categorical slot 1
BASELINE = "#eb6834"  # categorical slot 2
MUTED = "#a9a7a1"

LABELS = {"lgbm": "LightGBM", "lgbm_text": "LightGBM + TF-IDF text",
          "plus": "TabPFN-3.5 Plus", "thinking": "TabPFN-3.5 Thinking"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig(w, h):
    f, ax = plt.subplots(figsize=(w, h), dpi=200)
    f.patch.set_facecolor(SURFACE)
    style(ax)
    return f, ax


def benchmark():
    m = json.loads((RES / "metrics.json").read_text())
    order = [k for k in ("lgbm", "lgbm_text", "plus", "thinking") if k in m]
    f, axes = plt.subplots(1, 2, figsize=(9, 2.6), dpi=200)
    f.patch.set_facecolor(SURFACE)
    for ax, metric, title, better in [(axes[0], "roc_auc", "ROC AUC", "higher is better"),
                                      (axes[1], "log_loss", "Log loss", "lower is better")]:
        style(ax)
        vals = [m[k][metric] for k in order]
        colors = [TABPFN if k in ("plus", "thinking") else BASELINE for k in order]
        y = range(len(order))[::-1]
        # Dots, not bars: the axis doesn't start at zero, so bar length would mislead.
        lo = min(vals) - (max(vals) - min(vals)) * 1.5 - 0.005
        ax.hlines(list(y), lo, vals, color=GRID, linewidth=1)
        ax.scatter(vals, list(y), color=colors, s=70, zorder=3)
        ax.set_xlim(lo, max(vals) + (max(vals) - lo) * 0.18)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f"   {v:.3f}", va="center", color=INK, fontsize=8)
        ax.set_yticks(list(y))
        ax.set_yticklabels([LABELS[k] for k in order] if ax is axes[0] else [], color=INK, fontsize=8)
        ax.set_title(f"{title}  ({better})", loc="left", color=INK, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    f.tight_layout()
    f.savefig(DOCS / "benchmark.png", facecolor=SURFACE)


def coldstart():
    c = json.loads((RES / "coldstart.json").read_text())
    shots = [int(k) for k in c["by_shots"]]
    f, ax = fig(6.5, 3.2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for key, color, label in [("tabpfn_plus", TABPFN, "TabPFN-3.5 Plus"), ("lgbm", BASELINE, "LightGBM")]:
        vals = [c["by_shots"][str(n)][key]["mean_repo_auc"] for n in shots]
        x = range(len(shots))
        ax.plot(list(x), vals, color=color, linewidth=2, marker="o", markersize=5)
        ax.text(len(shots) - 1 + 0.08, vals[-1], label, color=INK, fontsize=8, va="center")
    ax.set_xticks(range(len(shots)))
    ax.set_xticklabels([str(n) for n in shots], fontsize=8)
    ax.set_xlim(-0.2, len(shots) - 1 + 1.3)
    ax.set_xlabel("PRs from the new repo in context / training", color=INK_2, fontsize=8)
    ax.set_ylabel("Mean per-repo ROC AUC", color=INK_2, fontsize=8)
    ax.set_title(f"Cold start on {len(c['heldout'])} repos never seen in training",
                 loc="left", color=INK, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    f.tight_layout()
    f.savefig(DOCS / "coldstart.png", facecolor=SURFACE)


def calibration():
    st = json.loads((RES / "stats.json").read_text())["models"]
    f, ax = fig(5.2, 4.4)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    series = [("plus", TABPFN, "TabPFN-3.5 Plus"), ("lgbm_text", BASELINE, "LightGBM + TF-IDF text")]
    for key, color, label in series:
        if key not in st:
            continue
        b = st[key]["calibration"]
        x, y = [r["predicted"] for r in b], [r["observed"] for r in b]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5, zorder=3)
    # Direct labels with the ECE, placed in the empty upper-left area to avoid the lines.
    for i, (key, color, label) in enumerate(s for s in series if s[0] in st):
        yy = 0.95 - i * 0.07
        ax.text(0.09, yy, f"{label}  (ECE {st[key]['ece']:.3f})", color=INK, fontsize=8,
                va="center", transform=ax.transAxes)
        ax.plot([0.03, 0.07], [yy, yy], color=color, linewidth=2, marker="o", markersize=5,
                markevery=[1], transform=ax.transAxes)
    yy = 0.95 - len([s for s in series if s[0] in st]) * 0.07
    ax.plot([0.03, 0.07], [yy, yy], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), transform=ax.transAxes)
    ax.text(0.09, yy, "Perfect calibration", color=INK_2, fontsize=8, va="center", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted chance of merge (10 equal-size groups)", color=INK_2, fontsize=8)
    ax.set_ylabel("Share that actually merged", color=INK_2, fontsize=8)
    ax.set_title("Calibration on 5,000 held-out PRs", loc="left", color=INK, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    f.tight_layout()
    f.savefig(DOCS / "calibration.png", facecolor=SURFACE)


def main():
    DOCS.mkdir(exist_ok=True)
    if (RES / "metrics.json").exists():
        benchmark()
    if (RES / "coldstart.json").exists():
        coldstart()
    if (RES / "stats.json").exists():
        calibration()
    print(sorted(p.name for p in DOCS.glob("*.png")))


if __name__ == "__main__":
    main()
