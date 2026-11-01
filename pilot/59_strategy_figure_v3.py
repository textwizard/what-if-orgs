"""Figure 2 of the proposal and the platform diagram of the artifact: the empirical strategy as run.

Three panels in the style of the journal figures (Helvetica, panel letters, black rules, measured in navy, simulated
in violet, checks in grey): (a) the record, measured from archived websites, (b) the reference organization simulated
under theories and its what-if variants, (c) weights fitted on the exploration window, validation on the holdout
window, coherence with the stated theory of value. Values are illustrative; the artifact carries the data.

Run from 09_whatif_pilot: python3 code/59_strategy_figure_v3.py
Outputs: analysis/figures/empirical_strategy_v3.{pdf,svg,png}, explorer/strategy_v3.svg (for the page)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent.parent
OBS, SIM, NONE, INK, MUTE, HAIR, RED = "#26406e", "#5a3fb0", "#7d8290", "#111111", "#666666", "#dcdcdc", "#c8322b"
ESC, RO, AC = "#c9971a", "#8c2a5b", "#2b8a3e"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 6.5, "pdf.fonttype": 42, "svg.fonttype": "none"})

fig = plt.figure(figsize=(7.2, 2.55))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 720); ax.set_ylim(92, 328); ax.axis("off")


def box(x, y, w, h, color=INK, lw=0.7, fill="none", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=3", fc=fill, ec=color, lw=lw, ls=ls))


def text(x, y, s, **k):
    ax.text(x, y, s, **{"va": "center", "ha": "left", "color": INK, "fontsize": 6.5, **k})


def arrow(x1, y1, x2, y2, color=MUTE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=color, shrinkA=2, shrinkB=2))


# panel letters and headings
for x, letter, head in ((8, "a", "What did the firm do?  The record"), (258, "b", "What would it do under each theory?"), (500, "c", "Which theory explains it?")):
    text(x, 318, letter, fontweight="bold", fontsize=9); text(x + 14, 318, head, fontweight="bold", fontsize=7.5)

# a: the record from archived websites
box(8, 150, 232, 155, color=HAIR)
text(16, 294, "Archived websites, 2017 to 2026", color=OBS, fontweight="bold")
text(16, 283, "Common Crawl pointers; pages coded by a language model", color=MUTE, fontsize=5.8)
text(16, 274, "with a verbatim quote behind every statement", color=MUTE, fontsize=5.8)
years = [2017, 2018, 2019, 2022, 2023, 2024, 2025, 2026]
x0, cw = 92, 17
for i, y in enumerate(years):
    text(x0 + i * cw + cw / 2, 262, str(y)[2:], ha="center", fontsize=5.5, color=MUTE)
ax.plot([x0 + 3 * cw - 1] * 2, [196, 258], color=HAIR, lw=0.6, ls=(0, (2, 2)))
rows = [("strategy, situation", [1, 0, 0, 1, 0, 0, 0, 0], "ctx"), ("product A", [0, 3, 3, 2, 2, 1, 1, 1], "p"),
        ("product B", [0, 0, 2, 2, 0, -1, -1, -1], "p"), ("product C", [0, 0, 0, 0, 2, 2, 3, 3], "p")]
for r, (lab, lv, kind) in enumerate(rows):
    yy = 246 - r * 15
    text(16, yy, lab, fontsize=5.8)
    for i, v in enumerate(lv):
        if kind == "ctx":
            if v: ax.add_patch(Rectangle((x0 + i * cw + 2, yy - 5), cw - 4, 10, fc=OBS, ec="none", alpha=0.9))
        elif v > 0:
            ax.add_patch(Rectangle((x0 + i * cw + 2, yy - 5), cw - 4, 10, fc=OBS, ec="none", alpha=[0, 0.3, 0.6, 1][v]))
        elif v == -1 and lv[i - 1] != -1:
            text(x0 + i * cw + cw / 2, yy, "×", ha="center", color=RED, fontweight="bold", fontsize=8)
text(16, 184, "shades: home page, page of its own, mentioned; × taken off the market", fontsize=5.3, color=MUTE)
text(16, 174, "stated results, problems, investments with quotes; theory of value", fontsize=5.3, color=MUTE)
text(16, 162, "exploration 2018–19 (context 2017) · holdout 2023–24 (context 2022)", fontsize=5.3, color=OBS)
# a lower: measurement box
box(8, 100, 232, 42, color=HAIR)
text(16, 132, "Anchors measured from the website", fontweight="bold", color=OBS)
text(16, 121, "stated strategy, situation of the firm and industry, NACE division,", fontsize=5.8, color=MUTE)
text(16, 111, "relatedness of each new product to the established offer", fontsize=5.8, color=MUTE)

# b: the reference organization
box(258, 100, 232, 205, color=HAIR)
box(268, 228, 96, 70, color=NONE); text(273, 290, "Language model", fontweight="bold", fontsize=6.2); text(273, 278, "general reasoning", fontsize=5.4, color=MUTE); text(273, 268, "open-weight and", fontsize=5.4, color=MUTE); text(273, 258, "commercial models", fontsize=5.4, color=MUTE); text(273, 246, "memory of its own", fontsize=5.4, color=MUTE); text(273, 236, "decisions", fontsize=5.4, color=MUTE)
box(372, 228, 108, 70, color=SIM); text(377, 290, "Theory pool", fontweight="bold", fontsize=6.2, color=SIM)
for k, (t, c) in enumerate((("no theory", NONE), ("behavioral theory", SIM), ("absorptive capacity", AC), ("escalation", ESC), ("real options", RO))):
    text(377, 278 - k * 9.8, t, fontsize=5.5, color=c)
text(268, 219, "Reference organization: reads the firm's own record year by year", fontsize=5.6, fontweight="bold")
text(268, 210, "and gives the probability that each product leaves the market", fontsize=5.6, fontweight="bold")
arrow(241, 125, 262, 160, color=OBS); text(242, 112, "anchors", fontsize=5.0, color=OBS)
arrow(241, 230, 262, 200, color=OBS); text(243, 237, "record", fontsize=5.0, color=OBS)
for k, (t, c) in enumerate((("no theory", NONE), ("behavioral theory", SIM), ("absorptive capacity", AC), ("escalation", ESC), ("real options", RO))):
    yy = 184 - k * 12
    ax.plot([300, 300], [yy - 3, yy + 3], color=c, lw=2.5, solid_capstyle="butt")
    text(305, yy, t, fontsize=5.6, color=c)
    ax.plot([385, 385 + [40, 50, 48, 36, 46][k]], [yy, yy], color=c, lw=3, solid_capstyle="butt", alpha=0.85)
text(385, 197, "share of product-years dropped", fontsize=5.0, color=MUTE)
ax.plot([385 + 38, 385 + 38], [130, 193], color=INK, lw=0.8); text(385 + 40, 127, "record", fontsize=5.0, color=INK)
text(268, 114, "What-if: the same firm with another theory, without its anchors,", fontsize=5.4, color=MUTE)
text(268, 105, "or with its own earlier decisions as its past", fontsize=5.4, color=MUTE)

# c: weights, validation, coherence
box(500, 100, 212, 205, color=HAIR)
text(508, 294, "Weights", fontweight="bold")
text(508, 284, "mixture of the conditions that fits the", fontsize=5.6, color=MUTE); text(508, 275, "2019 decisions; common, by regime, by firm", fontsize=5.6, color=MUTE)
xw = 508
for k, (c, w) in enumerate(((NONE, 0.64), (SIM, 0.0), (AC, 0.0), (ESC, 0.02), (RO, 0.34))):
    ax.add_patch(Rectangle((xw, 258), w * 190, 9, fc=c, ec="white", lw=0.5)); xw += w * 190
text(508, 248, "Validation on the holdout window", fontweight="bold")
text(508, 238, "level and ranking against the record", fontsize=5.6, color=MUTE); text(508, 229, "and against the ceiling from the cues", fontsize=5.6, color=MUTE)
ax.add_patch(Rectangle((508, 160), 190, 60, fc="white", ec=HAIR, lw=0.6))
import numpy as np
rng = np.random.default_rng(3)
xs = rng.uniform(515, 690, 18); ys = 165 + (xs - 515) * 0.22 + rng.normal(0, 7, 18)
ax.scatter(xs, ys, s=7, color=SIM, alpha=0.8, lw=0); ax.plot([515, 690], [168, 205], color=INK, lw=0.7)
text(600, 154, "simulated against recorded", fontsize=5.2, color=MUTE, ha="center")
text(508, 142, "Own history", fontweight="bold")
text(508, 132, "the agent with its own earlier decisions as its past:", fontsize=5.6, color=MUTE); text(508, 123, "does an early doubt harden into a withdrawal?", fontsize=5.6, color=MUTE)
text(508, 108, "Checks: coherence, re-identification, look-ahead", fontsize=5.4, color=MUTE)
arrow(490, 200, 500, 200)

out = HERE / "analysis" / "figures"
for ext in ("pdf", "svg", "png"):
    fig.savefig(out / f"empirical_strategy_v3.{ext}", dpi=300, bbox_inches="tight")
fig.savefig(HERE / "explorer" / "strategy_v3.svg", bbox_inches="tight")
print("saved", out / "empirical_strategy_v3.pdf")
