"""Diagnostics of the simulation on the record: ranking per condition, mean probability by product attribute and condition,
firm-level agreement, and the regression on the product record that bounds how predictable the recorded withdrawals are.

Run from the repository root: PYTHONPATH=. python analysis/diagnostics.py <model>
Output: printed tables and data/closed/record/<model>/diagnostics.txt
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

from whatif import record
from whatif.config import CONFIG, ROOT

MODEL = sys.argv[1]
REC = ROOT / "data" / "closed" / "record" / MODEL
CD = ROOT / "data" / "closed" / CONFIG["record"]["coding_dir"]


def auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, bool)
    r = rankdata(p)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def features(d):
    """Last prominence, stated result before the decision, relatedness, and age, per decision row."""
    p = pd.read_csv(ROOT / "data" / "closed" / "products.csv")
    p["key"] = p.slug + "|" + p.generic + "|" + p.entry.astype(str)
    d["key"] = d.slug + "|" + d.generic + "|" + d.entry.astype(str)
    lv = {r.key: json.loads(r.levels) for r in p.drop_duplicates("key").itertuples()}

    def last_level(r):
        L = lv.get(r.key, {})
        ys = [int(y) for y in L if r.entry <= int(y) < r.year]
        return L[str(max(ys))] if ys else np.nan
    d["last_level"] = d.apply(last_level, axis=1)
    ev = {}
    for s in d.slug.unique():
        c = json.loads((CD / f"{s}.json").read_text())
        names = {n: r.key for r in p[p.slug == s].itertuples() for n in r.names.split("|")}
        for y in c["years"]:
            for e in y.get("evidence", []):
                k = names.get((e.get("product") or "").strip())
                if k and e.get("type") == "result" and (e.get("strength") or 0) >= 2:
                    ev.setdefault(k, []).append(y["year"])
    d["result_before"] = d.apply(lambda r: any(r.entry <= y < r.year for y in ev.get(r.key, [])), axis=1)
    rel = {}
    for s in d.slug.unique():
        c = json.loads((ROOT / "data" / "closed" / "context" / f"{s}.json").read_text())["windows"]
        for w, v in c.items():
            for k, r in enumerate(v["relatedness"] or []):
                rel[(s, w, "ABCDEFGHIJ"[k])] = r
    d["relatedness"] = [rel.get((s, w, q)) for s, w, q in zip(d.slug, d.window, d["product"])]
    d["age"] = d.year - d.entry
    return d


def main():
    out = []
    pr = lambda *a: (print(*a), out.append(" ".join(str(x) for x in a)))
    d = pd.read_csv(REC / "decisions.csv")
    d = record.label(d[d.p_drop.notna()].copy())
    d = features(d)
    pr(f"{MODEL}: {len(d)} decisions, {d.slug.nunique()} firms, recorded drop {d.recorded_drop.mean():.3f}")
    pr("\nDiscrimination (AUC, tie-corrected) and mean probability by condition, all product-years:")
    for c, g in d.groupby("condition"):
        pr(f"  {c:6s} auc {auc(g.p_drop, g.recorded_drop):.3f}  mean p {g.p_drop.mean():.3f}  drops {(g.decision == 'drop').mean():.3f}")
    pr("\nMean probability by product attribute and condition (record in the last column):")
    conds = ["none"] + [f"{t}_{f}" for f in CONFIG["record"]["framings"] for t in CONFIG["record"]["theories"]]
    conds = [c for c in conds if c in set(d.condition)]
    for col in ("last_level", "result_before", "relatedness", "age"):
        t = d.pivot_table(index=col, columns="condition", values="p_drop", aggfunc="mean")[conds]
        t["record"] = d[d.condition == "none"].groupby(col).recorded_drop.mean()
        t["n"] = d[d.condition == "none"].groupby(col).size()
        pr(f"\n{col}\n" + t.round(2).to_string())
    pr("\nFirm level: Spearman of the firm's mean probability with its recorded drop share, per window (n firms):")
    for w, g in d.groupby("window"):
        line = []
        for c in conds:
            f = g[g.condition == c].groupby("slug").agg(p=("p_drop", "mean"), rec=("recorded_drop", "mean"), n=("p_drop", "size"))
            f = f[f.n >= 3]
            rho, pv = spearmanr(f.p, f.rec)
            line.append(f"{c} {rho:+.2f}")
        pr(f"  {w} ({len(f)}): " + ", ".join(line))
    pr("\nRegression on the product record, fitted on the 2019 decisions, "
       "tested on holdout:")
    X = pd.get_dummies(d[d.condition == "none"][["last_level", "result_before", "relatedness", "age", "window", "recorded_drop", "slug"]]
                       .assign(last_level=lambda x: x.last_level.astype(str), age=lambda x: x.age.clip(upper=4).astype(str)),
                       columns=["last_level", "relatedness", "age"], dtype=float)
    cols = [c for c in X.columns if c not in ("window", "recorded_drop", "slug")]
    X["result_before"] = X.result_before.astype(float)
    cal, val = X[X.window == "exploration"], X[X.window == "holdout"]

    def fit(A, y):
        A = np.column_stack([np.ones(len(A)), A])
        f = lambda b: -np.mean(y * np.log(1 / (1 + np.exp(-A @ b)) + 1e-9) + (1 - y) * np.log(1 - 1 / (1 + np.exp(-A @ b)) + 1e-9)) + 0.01 * np.sum(b[1:] ** 2)
        return minimize(f, np.zeros(A.shape[1]), method="L-BFGS-B").x
    b = fit(cal[cols].values, cal.recorded_drop.astype(float).values)
    pv = 1 / (1 + np.exp(-np.column_stack([np.ones(len(val)), val[cols].values]) @ b))
    pr(f"  held-out AUC of the regression {auc(pv, val.recorded_drop):.3f}; in-sample AUC on the 2019 decisions "
       f"{auc(1 / (1 + np.exp(-np.column_stack([np.ones(len(cal)), cal[cols].values]) @ b)), cal.recorded_drop):.3f}")
    (REC / "diagnostics.txt").write_text("\n".join(out))


if __name__ == "__main__":
    main()
