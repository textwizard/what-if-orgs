"""Firm-level abandonment by technological regime: the record, the agents under each condition, and the CIS benchmark.

A firm abandons in a window if at least one of its new products is dropped in the decision years of the window
(the record), or, for an agent, with probability 1 - prod(1 - p) over its product-years. The CIS benchmark is the
share of enterprises with abandoned or suspended innovation activities in the firm's NACE division (web-coded
division, Germany, CIS 2018, inn_cis11_inact), averaged over the firms of a regime. Weights calibrated on the CIS
shares by regime in the exploration window are validated at firm and regime level in the holdout window.

Run from whatif-organizations with PYTHONPATH set: python3 ../09_whatif_pilot/code/56_regime_cis.py <model> [framing]
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

from whatif import division
from whatif import record
from whatif.config import CONFIG, ROOT

MODEL = sys.argv[1]
FRAMING = sys.argv[2] if len(sys.argv) > 2 else CONFIG["record"]["framings"][0]
REC = ROOT / "data" / "closed" / "record" / MODEL
C = ["none"] + [f"{t}_{FRAMING}" for t in CONFIG["record"]["theories"]]


def auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, bool)
    r = rankdata(p)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def cis_abandon():
    """Share of enterprises with abandoned or suspended innovation activities by NACE key, CIS 2018, all sizes."""
    j = json.load(open(ROOT / "data" / "cache" / "eurostat" / "inn_cis11_inact.json"))
    dims, size = j["id"], j["size"]
    idx = {d: j["dimension"][d]["category"]["index"] for d in dims}
    out = {}
    for code in idx["nace_r2"]:
        pos = {"freq": "A", "enterpr": "ABD", "sizeclas": "TOTAL", "nace_r2": code, "unit": "PC", "geo": "DE", "time": "2018"}
        flat = 0
        for d, n in zip(dims, size):
            flat = flat * n + idx[d][pos[d]]
        if str(flat) in j["value"]:
            out[code] = j["value"][str(flat)]
    return out


def main():
    d = pd.read_csv(REC / "decisions.csv")
    d = record.label(d[d.p_drop.notna() & d.condition.isin(C)])
    regime = {n: r for r, codes in CONFIG["regimes"].items() for n in codes}
    ctx = CONFIG["simulation"]["context_years"]
    cis = cis_abandon()
    rows = []
    for (s, w), g in d.groupby(["slug", "window"]):
        div = division.lookup(s, ctx[record.window_of(w)][-1])
        r = {"slug": s, "window": w, "regime": regime.get(int(div["division"]), "other"),
             "cis": cis.get(div["key"], cis.get(div["key"][0])), "n": g[g.condition == "none"].shape[0],
             "record": float(g[g.condition == "none"].recorded_drop.any())}
        for c in C:
            r[c] = 1 - np.prod(1 - g[g.condition == c].p_drop.values)
        rows.append(r)
    f = pd.DataFrame(rows)
    print(f"{MODEL}, framing {FRAMING}: {f.slug.nunique()} firms; CIS 2018 abandonment share known for "
          f"{f.cis.notna().mean():.0%} of firm-windows\n")
    for w, g in f.groupby("window"):
        t = g.groupby("regime").agg(firms=("slug", "size"), record=("record", "mean"), cis=("cis", "mean"),
                                   **{c: (c, "mean") for c in C})
        t.loc["all"] = [len(g), g.record.mean(), g.cis.mean()] + [g[c].mean() for c in C]
        print(f"== {w}: share of firms dropping at least one new product, by regime (CIS: share of enterprises "
              f"with abandoned innovation activities in the firm's division, in percent/100)")
        t["cis"] = t["cis"] / 100
        print(t.round(2).to_string())
        print("   firm-level AUC of the agents' any-drop probability against the record: "
              + ", ".join(f"{c} {auc(g[c], g.record):.2f}" for c in C))
        rho, p = spearmanr(g.cis, g.record)
        print(f"   Spearman of the division's CIS share with the firm's record across firms: {rho:+.2f} (p {p:.2f})\n")
    # calibration on CIS by regime in exploration, validation in holdout
    cal, val = f[f.window == "exploration"], f[f.window == "holdout"]
    P = cal[C].values
    target = cal.cis.values / 100

    def loss(z):
        w = np.exp(z - z.max()); w /= w.sum()
        return np.mean((P @ w - target) ** 2)
    z = minimize(loss, np.zeros(len(C)), method="L-BFGS-B").x
    w = np.exp(z - z.max()); w /= w.sum()
    print("Weights calibrated on the CIS shares (exploration):", ", ".join(f"{c} {x:.2f}" for c, x in zip(C, w)))
    pv = val[C].values @ w
    print(f"Holdout, firm level: AUC {auc(pv, val.record):.2f}, predicted share of firms abandoning {pv.mean():.2f} "
          f"against recorded {val.record.mean():.2f}")
    t = val.assign(pred=pv).groupby("regime").agg(firms=("slug", "size"), record=("record", "mean"), pred=("pred", "mean"))
    print("Holdout, regime level:\n" + t.round(2).to_string())
    rho, p = spearmanr(t.record, t.pred)
    print(f"Spearman across regimes {rho:+.2f} (p {p:.2f}, {len(t)} regimes)")
    f.to_csv(REC / f"firm_abandonment_{FRAMING}.csv", index=False)


if __name__ == "__main__":
    main()
