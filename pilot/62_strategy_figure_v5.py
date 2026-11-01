"""Empirical strategy figure, v5, drawn from scratch to Nature's artwork rules (183 mm wide, under 170 mm high, Helvetica,
8 pt bold lowercase panel letters, all other text 5 to 7 pt, no coloured text, colour only in marks).

a  The architecture in one strip: six components in three stages (measure, simulate, compare), each a symbol with a name
   and a short line, connected by arrows.
b–g  What the data look like at each component, one panel each, in a 3 x 2 grid in the same order. All values illustrative.

Run from 09_whatif_pilot: python3 code/62_strategy_figure_v5.py
Outputs: analysis/figures/architecture.{pdf,svg,png}, explorer/architecture.svg, submission figures folder (architecture.pdf)
"""
import random
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent.parent
OBS, INK, MUTE, FAINT, HAIR, RED, PANEL, PAPER = "#1f3a68", "#111111", "#555b66", "#8b93a1", "#d7dae0", "#c8322b", "#f4f5f7", "#ffffff"
THEORY = {"none": "#7d8290", "btf": "#5a3fb0", "ac": "#2b8a3e", "esc": "#c9971a", "ro": "#8c2a5b"}
TH = list(THEORY.values())
plt.rcParams.update({"font.family": "Helvetica", "font.size": 6, "pdf.fonttype": 42, "svg.fonttype": "none"})

W, H = 720, 596
fig = plt.figure(figsize=(W / 100, H / 100))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def text(x, y, s, **k):
    ax.text(x, y, s, **{"va": "center", "ha": "left", "color": INK, "fontsize": 5.5, **k})


def lines_(x, y, ls, dy=7, **k):
    for i, l in enumerate(ls):
        text(x, y - i * dy, l, **k)


def rbox(x, y, w, h, fc=PAPER, ec=INK, lw=0.6, r=3, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(x1, y1, x2, y2, color=INK, lw=0.7, z=4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=5.5, lw=lw, color=color, shrinkA=1, shrinkB=1, zorder=z))


def page(x, y, w, h, lines=3, z=3, lw=0.55):
    ax.add_patch(Rectangle((x, y), w, h, fc=PAPER, ec=INK, lw=lw, zorder=z))
    for i in range(lines):
        ax.plot([x + w * 0.2, x + w * 0.8], [y + h - (i + 1) * h / (lines + 1)] * 2, color=HAIR, lw=0.8, zorder=z + 1)


def agent(x, y, w, h, theory=THEORY["btf"], z=3, lw=0.6, anchors=True):
    """the what-if organization: a body with three slots, anchors (navy), theory text (coloured), memory (loop)"""
    rbox(x, y, w, h, fc=PAPER, ec=INK, lw=lw, r=min(3, h * 0.18), z=z)
    if anchors:
        ax.add_patch(Rectangle((x + w * 0.12, y + h * 0.55), w * 0.24, h * 0.3, fc=OBS, ec="none", zorder=z + 1))
    else:
        ax.add_patch(Rectangle((x + w * 0.12, y + h * 0.55), w * 0.24, h * 0.3, fc=PAPER, ec=OBS, lw=0.5, ls=(0, (1, 1)), zorder=z + 1))
    ax.add_patch(Rectangle((x + w * 0.41, y + h * 0.55), w * 0.24, h * 0.3, fc=theory, ec="none", zorder=z + 1))
    cx, cy, rr = x + w * 0.8, y + h * 0.7, h * 0.13
    ax.add_patch(Circle((cx, cy), rr, fc="none", ec=MUTE, lw=lw, zorder=z + 1))
    ax.add_patch(Polygon([[cx + rr, cy + 1.2], [cx + rr + 1.6, cy - 1], [cx + rr - 1.6, cy - 1]], fc=MUTE, ec="none", zorder=z + 2))
    ax.plot([x + w * 0.12, x + w * 0.88], [y + h * 0.36] * 2, color=HAIR, lw=1, zorder=z + 1)
    ax.plot([x + w * 0.12, x + w * 0.62], [y + h * 0.22] * 2, color=HAIR, lw=1, zorder=z + 1)


def letter(x, y, l, title):
    text(x, y, l, fontweight="bold", fontsize=8)
    text(x + 11, y, title, fontweight="bold", fontsize=7)


# =====================================================================================================================
# a  the architecture: six components in three stages
# =====================================================================================================================
letter(8, 586, "a", "Architecture")
BW, BG, BX0, BY, BH = 104, 14.4, 13, 468, 90
boxes = [
    ("Archive", ["public snapshots of each", "firm's website, several a", "year, 2017 to 2026"]),
    ("Measurement", ["a language model codes each", "firm-year under protocols;", "every statement keeps a quote"]),
    ("What-if organization", ["an agent built from the firm:", "anchors, a theory text, memory", "of its own decisions"]),
    ("Simulate", ["the same firm and record,", "decided year by year under no", "theory and each theory"]),
    ("Calibrate, validate", ["a mix of the conditions fitted", "before the pandemic, tested", "after it against the record"]),
    ("What-if experiments", ["another theory, no anchors, or", "its own history: outcomes the", "record cannot show"]),
]
bx = [BX0 + i * (BW + BG) for i in range(6)]
for i, ((name, desc), x) in enumerate(zip(boxes, bx)):
    rbox(x, BY, BW, BH, fc=PANEL, ec="none", r=4, z=1)
    text(x + 7, BY + 36, name, fontweight="bold", fontsize=6)
    lines_(x + 7, BY + 25, desc, dy=6.8, color=MUTE, fontsize=5)
    if i < 5:
        arrow(x + BW + 1, BY + BH * 0.62, x + BW + BG - 1, BY + BH * 0.62, color=INK, lw=0.8)
for (a_, b_, lab) in ((0, 1, "Measure"), (2, 3, "Simulate"), (4, 5, "Compare")):
    ax.plot([bx[a_] + 2, bx[b_] + BW - 2], [BY + BH + 9] * 2, color=INK, lw=0.6)
    text((bx[a_] + bx[b_] + BW) / 2, BY + BH + 15, lab, ha="center", fontweight="bold", fontsize=6)
GY = BY + 66                                                   # glyph centre line
# glyphs
x = bx[0] + BW / 2
for k in range(3):
    page(x - 14 + k * 6, GY - 11 + k * 4, 18, 22, lines=3, z=3 + k)
x = bx[1] + BW / 2
page(x - 26, GY - 12, 18, 24, lines=4, z=3); ax.add_patch(Rectangle((x - 23, GY - 1), 12, 3.2, fc=OBS, alpha=0.3, ec="none", zorder=5))
rbox(x - 2, GY - 5, 22, 12, fc=PAPER, ec=OBS, lw=0.6, r=2.5, z=4); text(x + 9, GY + 1, "LM", ha="center", fontsize=5.5, fontweight="bold")
x = bx[2] + BW / 2
agent(x - 20, GY - 12, 40, 24)
x = bx[3] + BW / 2
for k, c in enumerate(TH):
    agent(x - 29 + k * 13, GY - 9 - abs(k - 2) * 1.5, 12, 17, theory=c, lw=0.45)
x = bx[4] + BW / 2
acc = 0
for wv, c in zip([0.45, 0.25, 0.12, 0.08, 0.10], TH):
    ax.add_patch(Rectangle((x - 28 + acc * 56, GY + 3), wv * 56 - 0.5, 7, fc=c, ec="none", zorder=3)); acc += wv
ax.plot([x - 28, x + 14], [GY - 5] * 2, color=OBS, lw=1.6, zorder=3); text(x + 17, GY - 5, "record", fontsize=5)
ax.plot([x - 28, x + 14], [GY - 11] * 2, color=MUTE, lw=1.6, ls=(0, (2, 1.4)), zorder=3); text(x + 17, GY - 11, "mix", fontsize=5)
x = bx[5] + BW / 2
agent(x - 36, GY - 7, 20, 14, lw=0.5)
for dy, th, anc in ((10, THEORY["ac"], True), (0, THEORY["btf"], False), (-10, THEORY["btf"], True)):
    agent(x + 4, GY - 5 + dy, 16, 11, theory=th, lw=0.45, anchors=anc)
    arrow(x - 15, GY, x + 3, GY + dy, color=MUTE, lw=0.5)
ax.add_patch(Rectangle((x + 4 + 16 * 0.12, GY - 15 + 11 * 0.55), 16 * 0.24, 11 * 0.3, fc=PANEL, ec=OBS, lw=0.5, ls=(0, (1, 1)), zorder=6))

# =====================================================================================================================
# b–g  the data at each component, 3 x 2
# =====================================================================================================================
PW, PH, PGX, PGY, PX0 = 226, 196, 14, 18, 7
ptop = [BY - 22, BY - 22 - PH - PGY]                         # top edge of the two panel rows
pleft = [PX0 + i * (PW + PGX) for i in range(3)]
cells = {}
for i, l in enumerate("bcdefg"):
    r_, c_ = divmod(i, 3); cells[l] = (pleft[c_], ptop[r_] - PH)
    rbox(pleft[c_], ptop[r_] - PH, PW, PH, fc=PAPER, ec=HAIR, lw=0.5, r=3, z=1)
text(W - 8, BY - 13, "b to g: what the data look like at each component; values illustrative", ha="right", color=MUTE, fontsize=5)
IN = 9

# ---- b archive ----
px, py = cells["b"]; px += IN; top = py + PH - 10
letter(px, top, "b", "Archive")
text(px, top - 14, "captures per year and page role", color=MUTE, fontsize=5.5)
years = ["17", "18", "19", "", "22", "23", "24", "25", "26"]
XC = lambda i: px + 46 + i * 18
for i, yv in enumerate(years):
    text(XC(i), top - 27, yv, ha="center", color=MUTE, fontsize=5)
random.seed(3)
for r_, role in enumerate(["home", "products", "news", "about", "careers"]):
    yy = top - 40 - r_ * 13
    text(px, yy, role, color=MUTE, fontsize=5.5)
    for i, yv in enumerate(years):
        if yv == "":
            continue
        for q in range(random.choice([1, 2, 2, 3])):
            ax.add_patch(Circle((XC(i) - 3 + q * 3, yy + 1.3 - q * 1.6), 1.2, fc=OBS, ec="none", alpha=0.85, zorder=3))
ax.add_patch(Rectangle((XC(3) - 6, top - 96), 12, 66, fc=PANEL, ec="none", zorder=2)); text(XC(3), top - 101, "pandemic", ha="center", color=FAINT, fontsize=5)
lines_(px, top - 122, ["every page carries a Common Crawl pointer", "(file, offset, length, digest): anyone can", "fetch the same page and repeat the coding"], dy=7, color=MUTE, fontsize=5.3)
lines_(px, top - 150, ["per firm and year: 15 to 50 pages read,", "up to 45,000 characters, by page role"], dy=7, color=MUTE, fontsize=5.3)

# ---- c measurement ----
px, py = cells["c"]; px += IN; top = py + PH - 10
letter(px, top, "c", "Measurement")
text(px, top - 14, "anchors of the firm", color=MUTE, fontsize=5.5)
rbox(px, top - 62, PW - 2 * IN, 42, fc=PAPER, ec=HAIR, lw=0.5, r=2, z=2)
for i, (k, v) in enumerate((("offers", "what it sells"), ("customers", "to whom"), ("why it wins", "its theory of value"), ("situation", "industry, size, projects"))):
    text(px + 4, top - 27 - i * 10, k, fontsize=5.5, fontweight="bold"); text(px + 48, top - 27 - i * 10, v, color=MUTE, fontsize=5.5)
text(px, top - 74, "record of its new products", color=MUTE, fontsize=5.5)
yrs = ["18", "19", "22", "23", "24", "25", "26"]
XR = lambda i: px + 56 + i * 14
for i, yv in enumerate(yrs):
    text(XR(i), top - 86, yv, ha="center", color=MUTE, fontsize=5)
rows = [("product A", [3, 3, 2, 2, 1, 1, 1], None, [1]), ("product B", [0, 2, 2, -1, 0, 0, 0], 3, []), ("product C", [0, 0, 0, 2, 3, 3, 3], None, [4])]
shades = {1: 0.38, 2: 0.66, 3: 1.0}
for r_, (lab, lv, drop, res) in enumerate(rows):
    yy = top - 99 - r_ * 13
    text(px, yy, lab, color=MUTE, fontsize=5.5)
    for i, v in enumerate(lv):
        if drop is not None and i > drop:
            continue
        if v == -1:
            text(XR(i), yy, "×", ha="center", color=RED, fontsize=7.5, fontweight="bold")
        elif v > 0:
            ax.add_patch(Rectangle((XR(i) - 5.5, yy - 3.8), 11, 7.6, fc=OBS, ec="none", alpha=shades[v], zorder=3))
        if i in res:
            ax.add_patch(Circle((XR(i) + 3.5, yy + 2.8), 1.7, fc=PAPER, ec=INK, lw=0.5, zorder=5))
for k, (lab, op) in enumerate((("home page", 1.0), ("own page", 0.66), ("mentioned", 0.38))):
    ax.add_patch(Rectangle((px + k * 62, top - 146), 8, 6, fc=OBS, ec="none", alpha=op, zorder=3)); text(px + 11 + k * 62, top - 143, lab, color=MUTE, fontsize=5)
text(px, top - 155, "×", color=RED, fontsize=7, fontweight="bold"); text(px + 8, top - 155, "taken off the market", color=MUTE, fontsize=5)
ax.add_patch(Circle((px + 92, top - 155), 1.7, fc=PAPER, ec=INK, lw=0.5)); text(px + 97, top - 155, "a stated result, with its quote", color=MUTE, fontsize=5)
text(px, top - 168, "“… now in use at forty sites” — verbatim, or it is not kept", color=MUTE, fontsize=5, style="italic")

# ---- d what-if organization ----
px, py = cells["d"]; px += IN; top = py + PH - 10
letter(px, top, "d", "What-if organization")
text(px, top - 14, "what the agent reads in year t", color=MUTE, fontsize=5.5)
rbox(px, top - 100, PW - 2 * IN, 80, fc=PAPER, ec=HAIR, lw=0.5, r=2, z=2)
blocks = [("the firm", "stated strategy and situation, from its website"), ("the theory", "what management weighs when results come in"),
          ("the record to t−1", "A: own page, result stated · B: mentioned only"), ("its own memory", "earlier decisions and reasons (multi-period)")]
for i, (k, l) in enumerate(blocks):
    yy = top - 28 - i * 19
    text(px + 4, yy, k, fontsize=5.5, fontweight="bold"); text(px + 4, yy - 7, l, color=MUTE, fontsize=5)
arrow(px + (PW - 2 * IN) / 2, top - 103, px + (PW - 2 * IN) / 2, top - 112, color=INK)
text(px, top - 120, "what it answers", color=MUTE, fontsize=5.5)
rbox(px, top - 172, PW - 2 * IN, 46, fc=PAPER, ec=HAIR, lw=0.5, r=2, z=2)
text(px + 4, top - 134, "A   keep    p = 0.15", fontsize=5.5); text(px + 4, top - 144, "B   drop    p = 0.60", fontsize=5.5)
lines_(px + 4, top - 155, ["“B falls short of aspiration and ties up resources", "the core line needs …”    p: probability of withdrawal"], dy=7, color=MUTE, fontsize=5, style="italic")

# ---- e replay ----
px, py = cells["e"]; px += IN; top = py + PH - 10
letter(px, top, "e", "Simulate under theories")
text(px, top - 14, "probability of withdrawal in year t, five copies of one firm", color=MUTE, fontsize=5.5)
XM = lambda j: px + 58 + j * 20
for j, hd in enumerate(["none", "BTF", "AC", "ESC", "RO"]):
    ax.add_patch(Rectangle((XM(j) - 8, top - 24), 16, 2.4, fc=TH[j], ec="none", zorder=3)); text(XM(j), top - 31, hd, ha="center", fontsize=5.5, fontweight="bold")
text(px + 180, top - 31, "record", ha="center", fontsize=5.5, fontweight="bold")
mat = [("product A", [0.15, 0.20, 0.12, 0.10, 0.18], False), ("product B", [0.35, 0.60, 0.30, 0.20, 0.45], True),
       ("product C", [0.10, 0.12, 0.25, 0.08, 0.10], False), ("product D", [0.25, 0.30, 0.40, 0.15, 0.20], True)]
for r_, (lab, ps, rec) in enumerate(mat):
    yy = top - 45 - r_ * 16
    text(px, yy, lab, color=MUTE, fontsize=5.5)
    for j, pv in enumerate(ps):
        ax.add_patch(Rectangle((XM(j) - 8, yy - 6.5), 16, 13, fc=TH[j], alpha=0.12 + pv * 1.3, ec="none", zorder=3)); text(XM(j), yy, f"{int(pv * 100)}", ha="center", fontsize=5.5)
    text(px + 180, yy, "×" if rec else "kept", ha="center", color=RED if rec else MUTE, fontsize=7.5 if rec else 5.5, fontweight="bold" if rec else "normal")
lines_(px, top - 122, ["one row per product and decision year; the agent", "never sees the year it decides about", "", "every cell keeps the reason the agent gave,", "so a theory can be read at work"], dy=7, color=MUTE, fontsize=5.3)
lines_(px, top - 163, ["BTF behavioral theory · AC absorptive capacity", "ESC escalation · RO real options"], dy=7, color=FAINT, fontsize=5)

# ---- f calibrate, validate ----
px, py = cells["f"]; px += IN; top = py + PH - 10
letter(px, top, "f", "Calibration, validation")
text(px, top - 14, "fit the mix: products of 2018, decided 2019", color=MUTE, fontsize=5.5)
for lvl, (lab, ws) in enumerate((("all firms", [0.45, 0.25, 0.12, 0.08, 0.10]), ("by regime", [0.35, 0.30, 0.20, 0.05, 0.10]), ("per firm", [0.55, 0.10, 0.10, 0.15, 0.10]))):
    yy = top - 28 - lvl * 12
    text(px, yy, lab, color=MUTE, fontsize=5.5)
    acc = 0
    for wv, c in zip(ws, TH):
        ax.add_patch(Rectangle((px + 44 + acc * 150, yy - 4), wv * 150 - 0.5, 8, fc=c, ec="none", zorder=3)); acc += wv
text(px, top - 64, "regime and firm weights shrink toward the common mix", color=FAINT, fontsize=5)
text(px, top - 80, "test it: products of 2023–24, decided 2024–25", color=MUTE, fontsize=5.5)
gx0, gx1 = px + 44, px + 194
for t_, lab in ((0, "0%"), (0.5, "15%"), (1, "30%")):
    xx = gx0 + t_ * (gx1 - gx0); ax.plot([xx, xx], [top - 142, top - 88], color=HAIR, lw=0.5, zorder=2); text(xx, top - 148, lab, ha="center", color=FAINT, fontsize=5)
for r_, (lab, rec, pred) in enumerate((("regime 1", 0.42, 0.38), ("regime 2", 0.6, 0.5), ("regime 3", 0.3, 0.42), ("regime 4", 0.5, 0.55))):
    yy = top - 96 - r_ * 13
    text(px, yy, lab, color=MUTE, fontsize=5.5)
    ax.plot([gx0 + rec * (gx1 - gx0), gx0 + pred * (gx1 - gx0)], [yy, yy], color=HAIR, lw=2, zorder=2)
    ax.add_patch(Circle((gx0 + rec * (gx1 - gx0), yy), 2.3, fc=OBS, ec="none", zorder=4)); ax.add_patch(Circle((gx0 + pred * (gx1 - gx0), yy), 2.3, fc=PAPER, ec=MUTE, lw=0.7, zorder=4))
ax.add_patch(Circle((px + 3, top - 160), 2.2, fc=OBS, ec="none")); text(px + 8, top - 160, "recorded share withdrawn", color=MUTE, fontsize=5)
ax.add_patch(Circle((px + 96, top - 160), 2.2, fc=PAPER, ec=MUTE, lw=0.7)); text(px + 101, top - 160, "predicted by the fitted mix", color=MUTE, fontsize=5)
text(px, top - 170, "scored by level, log loss and ranking against the record", color=FAINT, fontsize=5)

# ---- g what-if experiments ----
px, py = cells["g"]; px += IN; top = py + PH - 10
letter(px, top, "g", "What-if experiments")
text(px, top - 14, "the same firm, simulated otherwise", color=MUTE, fontsize=5.5)
agent(px + 2, top - 46, 30, 18, lw=0.5)
for k, (lab, th, anc, sub) in enumerate((("another theory", THEORY["ac"], True, "swap the text, keep the firm"), ("no anchors", THEORY["btf"], False, "the firm reduced to industry, size, age"), ("own history", THEORY["btf"], True, "its own first-year decisions as its past"))):
    yy = top - 27 - k * 17
    agent(px + 52, yy - 6, 16, 12, theory=th, lw=0.45, anchors=anc)
    arrow(px + 33, top - 37, px + 51, yy, color=MUTE, lw=0.5)
    text(px + 73, yy + 2, lab, fontsize=5.5, fontweight="bold"); text(px + 73, yy - 5, sub, color=MUTE, fontsize=5)
ax.add_patch(Rectangle((px + 52 + 16 * 0.12, top - 27 - 34 - 6 + 12 * 0.55), 16 * 0.24, 12 * 0.3, fc=PANEL, ec=OBS, lw=0.5, ls=(0, (1, 1)), zorder=6))
text(px, top - 86, "own history: share of the first-year cohort still on the market", color=MUTE, fontsize=5.5)
gx0, gx1, gy0, gy1 = px + 20, px + 108, top - 160, top - 96
ax.plot([gx0, gx0], [gy0, gy1], color=FAINT, lw=0.5); ax.plot([gx0, gx1], [gy0, gy0], color=FAINT, lw=0.5)
text(gx0 - 3, gy1, "100%", ha="right", color=FAINT, fontsize=5); text(gx0 - 3, gy0, "50%", ha="right", color=FAINT, fontsize=5)
sc = lambda v: (v - 0.5) / 0.5
for t_, lab in enumerate(("launch", "year 1", "year 2", "later")):
    text(gx0 + t_ * (gx1 - gx0) / 3, gy0 - 6, lab, ha="center", color=FAINT, fontsize=5)
ax.plot([gx0 + t_ * (gx1 - gx0) / 3 for t_ in range(4)], [gy0 + sc(q) * (gy1 - gy0) for q in [1, 0.88, 0.80, 0.76]], color=OBS, lw=1.4, zorder=4)
curves = (("esc", [1, 0.93, 0.90]), ("ac", [1, 0.86, 0.78]), ("ro", [1, 0.90, 0.66]), ("btf", [1, 0.78, 0.62]))
for k, v in curves:
    ax.plot([gx0 + t_ * (gx1 - gx0) / 3 for t_ in range(3)], [gy0 + sc(q) * (gy1 - gy0) for q in v], color=THEORY[k], lw=1.1, zorder=3)
    ax.add_patch(Circle((gx0 + 2 * (gx1 - gx0) / 3, gy0 + sc(v[2]) * (gy1 - gy0)), 1.6, fc=THEORY[k], ec="none", zorder=5))
for q, lab, col in ((0.91, "ESC", THEORY["esc"]), (0.81, "AC", THEORY["ac"]), (0.74, "record", OBS), (0.66, "RO", THEORY["ro"]), (0.59, "BTF", THEORY["btf"])):
    text(gx1 + 3, gy0 + sc(q) * (gy1 - gy0), lab, fontsize=5, color=col)
lines_(px + 138, top - 100, ["the agent decides year 1,", "then carries its own", "decisions into year 2;", "the cohort is compared", "with the record", "", "escalation persists,", "the behavioral theory", "withdraws on weak results"], dy=7, color=MUTE, fontsize=4.8)

out = HERE / "analysis" / "figures"; out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "svg", "png"):
    fig.savefig(out / f"architecture.{ext}", dpi=300 if ext == "png" else None)
shutil.copy(out / "architecture.svg", HERE / "explorer" / "architecture.svg")
prop = HERE.parent / "org_sci" / "org_sci_sub" / "What_If_Organizations" / "figures"
if prop.exists():
    shutil.copy(out / "architecture.pdf", prop / "architecture.pdf")
print("written", out / "architecture.*", "and explorer/architecture.svg")
