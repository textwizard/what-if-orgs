"""Figure of the reference organizations: drop rates against the record, theory mix per firm, validation, coherence.

Reads the outputs of whatif/record.py and whatif/weights.py for one model (default config record.models[0]) and draws
four panels in the journal style of the explorer (Helvetica, black rules, panel letters):
  a  drop rate per product-year in each window: the record and the agent under each condition, with 95% intervals
     from resampling firms
  b  theory mix of each firm in the exploration window, one stacked bar per firm, sorted by the dominant theory
  c  validation on the holdout window: log loss of each predictor (lower is better), base rate to firm weights
  d  coherence: firm weights against the emphases of the theory of value in the context year, Spearman rho

Colours: no theory grey, behavioral theory violet, absorptive capacity green (as in the explorer), escalation gold and
real options plum, chosen so that every pair of the five passes the colour-vision checks (OKLab Delta E >= 10 under
protan and deutan simulation, >= 16 unsimulated); the record is black. The gold is light against white and therefore
always carries a text label.

Run from whatif-organizations with PYTHONPATH set: python3 ../09_whatif_pilot/code/54_reference_figure.py [model] [framing]
Output: 09_whatif_pilot/analysis/figures/reference_organizations_<model>_<framing>.{pdf,png,svg}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from whatif import record
from whatif.config import CONFIG, ROOT

MODEL = sys.argv[1] if len(sys.argv) > 1 else CONFIG["record"]["models"][0]
FRAMING = sys.argv[2] if len(sys.argv) > 2 else CONFIG["record"]["framings"][0]
REC = ROOT / "data" / "closed" / "record" / MODEL
TAG = f"_{FRAMING}"
THEORIES = [f"{t}{TAG}" for t in CONFIG["record"]["theories"]]


def base(c):
    """Theory of a condition name such as btf_F."""
    return c.split("_")[0]
FIG = Path(__file__).resolve().parent.parent / "analysis" / "figures"
COL = {"none": "#7d8290", "btf": "#5a3fb0", "ac": "#2b8a3e", "esc": "#c9971a", "ro": "#8c2a5b", "record": "#111111"}
NAME = {"none": "no theory", "btf": "behavioral theory", "ac": "absorptive capacity", "esc": "escalation",
        "ro": "real options", "record": "record", "base rate": "base rate", "common": "common weights",
        "regime": "regime weights", "firm": "firm weights"}
EMPH = {"btf": "performance", "ac": "knowledge", "esc": "commitment", "ro": "experimentation"}
TXT, AXIS, HAIR = "#111111", "#777777", "#dcdcdc"
plt.rcParams.update({"font.family": "Helvetica", "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 6.5,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS, "axes.labelcolor": TXT,
                     "text.color": TXT})
B = CONFIG["analysis"]["bootstrap"]
rng = np.random.default_rng(CONFIG["seed"])


def clean(ax, left=True, bottom=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=2, pad=2)


def letter(ax, s):
    ax.text(-0.08, 1.06, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def firm_interval(df, col):
    """Mean of a column with a 95% percentile interval from resampling firms."""
    groups = [g[col].values for _, g in df.groupby("slug")]
    means = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        means.append(np.concatenate([groups[i] for i in pick]).mean())
    return df[col].mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def panel_a(ax, d):
    conditions = [c for c in ["none"] + THEORIES if c in set(d.condition)]
    windows = [w for w in ("exploration", "gap", "holdout") if w in set(d.window)]
    x0 = 0
    ticks, labels = [], []
    for w in windows:
        dw = d[d.window == w]
        rec = dw[dw.condition == conditions[0]]
        series = [("record", rec, "recorded_drop")] + [(c, dw[dw.condition == c], "p_drop") for c in conditions]
        for k, (name, df, col) in enumerate(series):
            if not len(df):
                continue
            m, lo, hi = firm_interval(df.assign(**{col: df[col].astype(float)}), col)
            x = x0 + k
            ax.plot([x, x], [lo, hi], color=COL[base(name)], lw=0.8, solid_capstyle="butt")
            ax.scatter([x], [m], s=14, color=COL[base(name)], zorder=3, edgecolor="white", linewidth=0.5)
            if name == "record":
                ax.text(x, hi + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=5.5)
        ticks.append(x0 + (len(series) - 1) / 2)
        labels.append({"exploration": "products of 2018–19\ndecisions 2019",
                       "gap": "products of 2018–19\ndecided 2022, across the gap",
                       "holdout": "products of 2023–24\ndecisions 2024 and 2025"}[w])
        x0 += len(series) + 1.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("share of product-years dropped")
    ax.set_ylim(0, max(0.05, ax.get_ylim()[1]))
    ax.yaxis.grid(True, color=HAIR, lw=0.4)
    ax.set_axisbelow(True)
    clean(ax, bottom=False)
    for name in ["record"] + conditions:
        ax.scatter([], [], s=14, color=COL[base(name)], label=NAME[base(name)])
    ax.legend(frameon=False, ncol=3, fontsize=5.8, loc="upper left", bbox_to_anchor=(0, 1.02), handletextpad=0.3,
              columnspacing=0.8)


def panel_b(ax, firm):
    f = firm[firm.window == "exploration"].copy()
    C = [c for c in ["none"] + THEORIES if c in f.columns]
    f[C] = f[C].astype(float)
    f["dominant"] = f[C].idxmax(axis=1)
    f["dom_w"] = f[C].max(axis=1)
    order = {c: i for i, c in enumerate(C)}
    f = f.sort_values(["dominant", "dom_w"], key=lambda s: s.map(order) if s.name == "dominant" else -s)
    y = np.arange(len(f))
    left = np.zeros(len(f))
    for c in C:
        ax.barh(y, f[c].values, left=left, height=0.8, color=COL[base(c)], edgecolor="white", linewidth=0.6,
                label=NAME[base(c)])
        left += f[c].values
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("weight in the firm's mix (exploration window)")
    ax.set_ylabel(f"{len(f)} firms, sorted by dominant theory")
    ax.invert_yaxis()
    clean(ax, left=False)
    ax.legend(frameon=False, ncol=5, fontsize=5.8, loc="lower left", bbox_to_anchor=(0, 1.0), handletextpad=0.3,
              columnspacing=0.8, handlelength=1.0)


def panel_c(ax, val):
    order = [k for k in ["base rate", "none"] + THEORIES + ["common", "regime", "firm"] if k in val.index]
    v = val.loc[order]
    y = np.arange(len(v))
    colors = [COL.get(base(k), "#111111") for k in order]
    ax.hlines(y, 0, v.logloss.values, color=HAIR, lw=0.6)
    ax.scatter(v.logloss.values, y, s=16, color=colors, zorder=3, edgecolor="white", linewidth=0.5)
    for yi, (k, r) in zip(y, v.iterrows()):
        ax.text(r.logloss + 0.004, yi, f"{r.logloss:.3f}" + (f"   AUC {r.auc:.2f}" if pd.notna(r.auc) else ""),
                va="center", fontsize=5.5)
    ax.set_yticks(y)
    ax.set_yticklabels([NAME.get(base(k), k) for k in order])
    ax.invert_yaxis()
    ax.set_xlabel("log loss on the holdout window (lower is better)")
    lo = max(0, v.logloss.min() - 0.03)
    ax.set_xlim(lo, v.logloss.max() + 0.08)
    clean(ax)


def panel_d(fig, gs, firm, values):
    years = CONFIG["simulation"]["context_years"]["exploration"][0]
    f = firm[firm.window == "exploration"].merge(values[values.year == years], on="slug")
    axes = gs.subgridspec(2, 2, wspace=0.55, hspace=0.7)
    first = None
    for k, (t, emph) in enumerate(EMPH.items()):
        theory = f"{t}{TAG}"
        ax = fig.add_subplot(axes[k // 2, k % 2])
        first = first or ax
        if theory not in f.columns or emph not in f.columns:
            ax.axis("off")
            continue
        x, y = f[emph].astype(float), f[theory].astype(float)
        points = x.max() > 3                                  # relative coding in points of 100, else scores 0 to 3
        jitter = rng.uniform(-0.12, 0.12, len(x)) * (8 if points else 1)
        ax.scatter(x + jitter, y, s=10, color=COL[t], alpha=0.75, edgecolor="white", linewidth=0.4)
        from scipy.stats import spearmanr
        rho, p = spearmanr(x, y) if len(x) >= 5 else (np.nan, np.nan)
        ax.text(0.02, 0.97, f"{NAME[t]}\nvs {emph}\nrho {rho:.2f}" + (f", p {p:.2f}" if pd.notna(p) else ""),
                transform=ax.transAxes, va="top", fontsize=5.5)
        ax.set_xticks([0, 25, 50, 75, 100] if points else [0, 1, 2, 3])
        ax.set_xlim((-6, 106) if points else (-0.4, 3.4))
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1])
        if k % 2 == 0:
            ax.set_ylabel("firm weight")
        if k // 2 == 1:
            ax.set_xlabel("emphasis on the website (points of 100)" if points else "emphasis on the website (0–3)")
        clean(ax)
    return first


def main():
    d = pd.read_csv(REC / "decisions.csv")
    d = record.label(d[d.p_drop.notna()])
    firm = pd.read_csv(REC / f"weights_firm{TAG}.csv")
    val = pd.read_csv(REC / f"validation{TAG}.csv", index_col=0)
    values = []
    for f in (ROOT / "data" / "closed" / "funnel").glob("value_*.csv"):
        if f.name in ("value_opus.csv", "value_gpt54.csv"):           # relative coding of the context years
            values.append(pd.read_csv(f))
    values = pd.concat(values).groupby(["slug", "year"])[list(EMPH.values())].mean().reset_index()

    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1.05], wspace=0.35, hspace=0.5)
    ax_a = fig.add_subplot(gs[0, 0]); panel_a(ax_a, d); letter(ax_a, "a")
    ax_b = fig.add_subplot(gs[0, 1]); panel_b(ax_b, firm); letter(ax_b, "b")
    ax_c = fig.add_subplot(gs[1, 0]); panel_c(ax_c, val); letter(ax_c, "c")
    ax_d = panel_d(fig, gs[1, 1], firm, values); letter(ax_d, "d")
    fig.suptitle(f"Reference organizations of {firm.slug.nunique()} firms, {MODEL}, framing {FRAMING}", x=0.02,
                 ha="left", fontsize=8, fontweight="bold", y=0.995)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"reference_organizations_{MODEL}{TAG}.{ext}", dpi=300, bbox_inches="tight")
    print("saved", FIG / f"reference_organizations_{MODEL}{TAG}.pdf")


if __name__ == "__main__":
    main()
