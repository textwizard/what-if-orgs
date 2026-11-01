"""Mixing weights of the theories per firm, their validation after the pandemic, and their coherence with the website.

The decisions of the what-if organization under each condition (record.py) give, for every product-year, the probability
that the firm drops the product under no theory and under each theory. A reference organization mixes these
probabilities with weights on the simplex, p = sum_c w_c p_c. Weights are fitted on the exploration window by minimizing
the log loss against the recorded drops, at three levels: common to all firms, per technological regime, and per firm,
shrunk toward the common weights with a penalty lambda * n_i * ||w_i - w||^2, since two to six products per firm would
otherwise fit noise. Validation applies the exploration weights to the holdout window and compares log loss, Brier
score, AUC, and hit rate with the agent without theory, each theory alone, and the base rate. Coherence relates the firm
weights to the theory of value coded from the website (protocols/value.md) in the context year of each window: the weight
of the behavioral theory to the emphasis on performance, absorptive capacity to knowledge, escalation to commitment, and
real options to experimentation, and the change of the weights between the windows to the change of the emphases.

Run: python -m whatif.weights <model>
Outputs in data/closed/record/<model>/: weights_common.csv, weights_regime.csv, weights_firm.csv, validation.csv,
coherence.csv, and the printed tables.
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

from whatif import record
from whatif.config import CONFIG, ROOT

OUT = ROOT / "data" / "closed" / "record"
EPS = 0.02
LAMBDA = 1.0
PAIRS = {"btf": "performance", "ac": "knowledge", "esc": "commitment", "ro": "experimentation"}


def table(model):
    """One row per product-year with the drop probability under each condition and the recorded drop."""
    d = pd.read_csv(OUT / model / "decisions.csv")
    d = record.label(d[d.p_drop.notna()])
    wide = d.pivot_table(index=["slug", "window", "year", "product", "entry"], columns="condition", values="p_drop")
    rec = d.groupby(["slug", "window", "year", "product", "entry"]).recorded_drop.first()
    wide = wide.join(rec).dropna().reset_index()
    sample = pd.read_csv(ROOT / "data" / "closed" / "sample.csv")
    sample["slug"] = sample.domain.str.replace(".", "-", regex=False)
    nace2 = dict(zip(sample.slug, sample.nace4.astype(str).str.zfill(4).str[:2].astype(int)))
    regime = {n: r for r, codes in CONFIG["regimes"].items() for n in codes}
    if CONFIG["coding"].get("division_source") == "web":              # division read from the website of the window
        from whatif import division
        ctx = CONFIG["simulation"]["context_years"]
        wide["regime"] = [regime.get(int(division.lookup(s, ctx[record.window_of(w)][-1])["division"]), "other")
                          for s, w in zip(wide.slug, wide.window)]
    else:
        wide["regime"] = wide.slug.map(lambda s: regime.get(nace2.get(s), "other"))
    return wide


def conditions(wide, framing):
    """No theory and the theories in one framing, as the columns of the table."""
    return ["none"] + [f"{t}_{framing}" for t in CONFIG["record"]["theories"] if f"{t}_{framing}" in wide.columns]


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit(P, y, prior=None, lam=0.0):
    """Weights on the simplex that minimize the log loss of the mixture, with an optional pull toward prior."""
    n = P.shape[1]

    def f(z):
        w = softmax(z)
        loss = logloss(P @ w, y)
        return loss + (lam * np.sum((w - prior) ** 2) if prior is not None else 0.0)
    best = None
    for start in [np.zeros(n)] + [np.random.default_rng(k).normal(size=n) for k in range(4)]:
        r = minimize(f, start, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return softmax(best.x)


def auc(p, y):
    """Area under the ROC curve with average ranks for ties, which are frequent since confidences repeat."""
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    ranks = rankdata(p)
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def metrics(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return {"n": len(y), "recorded": y.mean(), "predicted": p.mean(), "logloss": logloss(p, y),
            "brier": np.mean((p - y) ** 2), "auc": auc(p, y), "hit": np.mean((p > 0.5) == (y == 1))}


def main(model):
    wide = table(model)
    print(f"{len(wide)} product-years of {wide.slug.nunique()} firms")
    for framing in CONFIG["record"]["framings"]:
        print(f"\n===== framing {framing} =====")
        run(model, wide, framing)


def run(model, wide, framing):
    C = conditions(wide, framing)
    tag = f"_{framing}"
    cal, val = wide[wide.window == "exploration"], wide[wide.window == "holdout"]
    print(f"exploration {len(cal)}, holdout {len(val)}; conditions {C}")
    y_cal, y_val = cal.recorded_drop.astype(float).values, val.recorded_drop.astype(float).values
    P_cal, P_val = cal[C].values, val[C].values

    common = fit(P_cal, y_cal)
    pd.DataFrame([dict(zip(C, common))]).to_csv(OUT / model / f"weights_common{tag}.csv", index=False)
    print("\nCommon weights:", ", ".join(f"{c} {w:.2f}" for c, w in zip(C, common)))

    rows = []
    for r, g in cal.groupby("regime"):
        w = fit(g[C].values, g.recorded_drop.astype(float).values, prior=common, lam=LAMBDA) if len(g) >= 10 else common
        rows.append({"regime": r, "n": len(g), **dict(zip(C, w))})
    regime = pd.DataFrame(rows).set_index("regime")
    regime.to_csv(OUT / model / f"weights_regime{tag}.csv")
    print("\nWeights by regime:\n" + regime.round(2).to_string())

    firm = {}
    for window, g in wide.groupby("window"):
        for s, gg in g.groupby("slug"):
            w = fit(gg[C].values, gg.recorded_drop.astype(float).values, prior=common, lam=LAMBDA * len(gg))
            firm[(s, window)] = {"n": len(gg), **dict(zip(C, w))}
    firm = pd.DataFrame(firm).T.rename_axis(["slug", "window"]).reset_index()
    firm.to_csv(OUT / model / f"weights_firm{tag}.csv", index=False)
    f_cal = firm[firm.window == "exploration"].set_index("slug")
    print("\nFirm weights (exploration), mean and spread:\n" + f_cal[C].astype(float).describe().loc[["mean", "std", "min", "max"]].round(2).to_string())

    preds = {"base rate": np.full(len(val), y_cal.mean()), "none": P_val[:, 0]}
    preds.update({c: P_val[:, k] for k, c in enumerate(C) if c != "none"})
    preds["common"] = P_val @ common
    preds["regime"] = np.array([P_val[i] @ regime.loc[r, C].values.astype(float) if r in regime.index else P_val[i] @ common
                                for i, r in enumerate(val.regime)])
    preds["firm"] = np.array([P_val[i] @ f_cal.loc[s, C].values.astype(float) if s in f_cal.index else P_val[i] @ common
                              for i, s in enumerate(val.slug)])
    validation = pd.DataFrame({k: metrics(p, y_val) for k, p in preds.items()}).T
    validation.to_csv(OUT / model / f"validation{tag}.csv")
    print("\nValidation on the holdout window (lower log loss and Brier, higher AUC and hit rate are better):\n"
          + validation.round(3).to_string())

    values = []
    for coder in ("opus", "gpt54"):
        f = ROOT / "data" / "closed" / "funnel" / f"value_{coder}.csv"
        if f.exists():
            values.append(pd.read_csv(f)[["slug", "year"] + list(PAIRS.values())].assign(coder=coder))
    rows = []
    if values:
        v = pd.concat(values).groupby(["slug", "year"])[list(PAIRS.values())].mean().reset_index()
        years = {w: CONFIG["simulation"]["context_years"][w][0] for w in ("exploration", "holdout")}
        for window, year in years.items():
            m = firm[firm.window == window].merge(v[v.year == year], on="slug")
            for t, emphasis in PAIRS.items():
                theory = f"{t}_{framing}"
                if theory in m.columns and len(m) >= 5:
                    rho, p = spearmanr(m[theory].astype(float), m[emphasis])
                    rows.append({"window": window, "theory": theory, "emphasis": emphasis, "n": len(m), "rho": rho, "p": p})
        a = firm[firm.window == "exploration"].merge(v[v.year == years["exploration"]], on="slug")
        b = firm[firm.window == "holdout"].merge(v[v.year == years["holdout"]], on="slug")
        m = a.merge(b, on="slug", suffixes=("_a", "_b"))
        for t, emphasis in PAIRS.items():
            theory = f"{t}_{framing}"
            if f"{theory}_a" in m.columns and len(m) >= 5:
                rho, p = spearmanr(m[f"{theory}_b"].astype(float) - m[f"{theory}_a"].astype(float),
                                   m[f"{emphasis}_b"] - m[f"{emphasis}_a"])
                rows.append({"window": "change", "theory": theory, "emphasis": emphasis, "n": len(m), "rho": rho, "p": p})
    coherence = pd.DataFrame(rows)
    coherence.to_csv(OUT / model / f"coherence{tag}.csv", index=False)
    if len(coherence):
        print("\nCoherence of firm weights with the theory of value on the website (Spearman):\n"
              + coherence.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CONFIG["record"]["models"][0])
