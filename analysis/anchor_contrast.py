"""What the web anchors contribute: the same agents with and without the firm's measured context.

For one model, compares the conditions with the web anchors (stated strategy, situation of the firm and industry,
relatedness of the products) with the same conditions where the firm is reduced to industry, size, and age (suffix
-na in whatif/record.py): the drop level against the record per window, the discrimination (AUC), the log loss, the
agreement of the firm-level drop shares with the record, and the attention to the product attributes the record shows.

Run from whatif-organizations with PYTHONPATH set: PYTHONPATH=. python analysis/anchor_contrast.py <model>
Output: printed tables and data/closed/record/<model>/anchor_contrast.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from whatif import record
from whatif.config import CONFIG, ROOT

import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("diag", Path(__file__).with_name("diagnostics.py"))
diag = importlib.util.module_from_spec(spec)
_argv, sys.argv = sys.argv, ["x", sys.argv[1]]
spec.loader.exec_module(diag)
sys.argv = _argv
MODEL = sys.argv[1]
REC = ROOT / "data" / "closed" / "record" / MODEL


def auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, bool)
    r = rankdata(p)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def logloss(p, y, eps=0.02):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    d = pd.read_csv(REC / "decisions.csv")
    d = record.label(d[d.p_drop.notna()].copy())
    if "anchors" not in d.columns or (d.anchors == "none").sum() == 0:
        print("no unanchored conditions for", MODEL)
        return
    d = diag.features(d)
    base = ["none"] + [f"{t}_F" for t in CONFIG["record"]["theories"]]
    rows = []
    for c in base:
        for anchors, cond in (("web", c), ("none", f"{c}-na"), ("own removed", f"{c}-oa")):   # -oa: the theory without its own anchors
            g = d[d.condition == cond]
            if not len(g):
                continue
            r = {"condition": c, "anchors": anchors, "n": len(g), "mean_p": g.p_drop.mean(),
                 "recorded": g.recorded_drop.mean(), "auc": auc(g.p_drop, g.recorded_drop),
                 "logloss": logloss(g.p_drop, g.recorded_drop)}
            for w, gw in g.groupby("window"):
                r[f"mean_p_{w}"] = gw.p_drop.mean()
                r[f"auc_{w}"] = auc(gw.p_drop, gw.recorded_drop)
                f = gw.groupby("slug").agg(p=("p_drop", "mean"), rec=("recorded_drop", "mean"), n=("p_drop", "size"))
                f = f[f.n >= 3]
                r[f"firm_rho_{w}"] = spearmanr(f.p, f.rec)[0]
            for cue, key in (("relatedness", "distant"), ("last_level", 1), ("result_before", True), ("age", 3)):
                r[f"p_{cue}_{key}"] = g[g[cue] == key].p_drop.mean()
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(REC / "anchor_contrast.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{MODEL}: anchored against unanchored agents, {d.slug.nunique()} firms\n")
    print(t[["condition", "anchors", "n", "recorded", "mean_p", "mean_p_exploration", "mean_p_holdout", "auc", "auc_holdout", "logloss",
             "firm_rho_exploration", "firm_rho_holdout"]].round(3).to_string(index=False))
    print("\nAttention to product attributes, mean probability: distant products, products mentioned only, products with a stated result, "
          "products three years old")
    print(t[["condition", "anchors", "p_relatedness_distant", "p_last_level_1", "p_result_before_True", "p_age_3"]].round(3).to_string(index=False))
    w, n, o = t[t.anchors == "web"], t[t.anchors == "none"], t[t.anchors == "own removed"]
    print(f"\nMean over conditions: with anchors p {w.mean_p.mean():.3f}, AUC {w.auc.mean():.3f}, log loss {w.logloss.mean():.3f}; "
          f"without anchors p {n.mean_p.mean():.3f}, AUC {n.auc.mean():.3f}, log loss {n.logloss.mean():.3f}; recorded {t.recorded.iloc[0]:.3f}")
    if len(o):
        print(f"Theories without their own anchors (fair test, {len(o)} conditions): p {o.mean_p.mean():.3f}, AUC {o.auc.mean():.3f}, log loss {o.logloss.mean():.3f}; "
              f"the same theories with all anchors: p {w[w.condition != 'none'].mean_p.mean():.3f}, AUC {w[w.condition != 'none'].auc.mean():.3f}")
    # agreement between anchored and unanchored judgments of the same product-year
    wide = d[d.condition.isin(base + [f"{c}-na" for c in base] + [f"{c}-oa" for c in base])].pivot_table(index=["slug", "window", "year", "product"], columns="condition", values="p_drop")
    print("\nCorrelation of the anchored and unanchored probability on the same product-year: "
          + ", ".join(f"{c} {wide[c].corr(wide[f'{c}-na']):.2f}" for c in base if f"{c}-na" in wide.columns))
    if any(f"{c}-oa" in wide.columns for c in base):
        print("Correlation of the probability with and without the theory's own anchors: "
              + ", ".join(f"{c} {wide[c].corr(wide[f'{c}-oa']):.2f}" for c in base if f"{c}-oa" in wide.columns))


if __name__ == "__main__":
    main()
