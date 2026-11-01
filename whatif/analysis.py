"""Responses of the what-if organizations to early results, and the comparison with the recorded decisions of firms.

For each condition, the hot stove outcome is the share of products with high potential that are off the market at
the end of period 4 after an early failure, and the escalation outcome is the share of products with low potential
that are still on the market after an early success. The drop rates after an early failure and after an early success
are compared with the drop rates of real innovations without and with a stated success in their first year, within
the same window. Intervals are percentile intervals from resampling firms. As a robustness check, all outcomes are
repeated without the firms that the model named in the re-identification test (probes.py).

Outputs: data/closed/analysis.csv (one row per sample, condition, and outcome), printed as a table.
"""
import json

import numpy as np
import pandas as pd

from whatif.config import CONFIG, ROOT

RUNS = ROOT / "data" / "closed" / "runs"
B = CONFIG["analysis"]["bootstrap"]


def products():
    rows = []
    for f in RUNS.glob("*/*.json"):
        r = json.loads(f.read_text())
        final = r["periods"][-1]["held"]
        labels = "ABCDEFGHIJ"
        for k in range(len(r["products"])):
            missing = sum(p["decisions"].get(labels[k], {}).get("missing", False) for p in r["periods"])
            rows.append({"condition": f.parent.name, "slug": r["slug"], "window": r["window"], "high": r["high"][k],
                         "failure": r["early"][k] == "failure", "dropped": not final[k], "missing": missing})
    return pd.DataFrame(rows)


def records():
    """Real innovations whose fate is observed within the window, with their stated success and drop."""
    p = pd.read_csv(ROOT / "data" / "closed" / "products.csv")
    rows = []
    for window, (a, _) in CONFIG["simulation"]["windows"].items():
        end = CONFIG["analysis"]["record_end"][window]            # last year in which a drop is observed
        q = p[p.innovation & p.entry.between(a, end - 1)]
        for _, r in q.iterrows():
            rows.append({"condition": "real firms", "slug": r.slug, "window": window, "failure": not r.success,
                         "dropped": pd.notna(r["drop"]) and r["drop"] <= end})
    return pd.DataFrame(rows)


def interval(df, stat, rng):
    firms = df.slug.unique()
    if len(firms) < 2:
        return np.nan, np.nan
    groups = {s: g for s, g in df.groupby("slug")}
    vals = [stat(pd.concat([groups[s] for s in rng.choice(firms, len(firms))])) for _ in range(B)]
    return np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)


def main():
    rng = np.random.default_rng(CONFIG["seed"])
    sim, real = products(), records()
    outcomes = {
        "hot stove": (lambda d: d[d.high & d.failure].dropped.mean()),
        "escalation": (lambda d: 1 - d[~d.high & ~d.failure].dropped.mean()),
        "drop after early failure": (lambda d: d[d.failure].dropped.mean()),
        "drop after early success": (lambda d: d[~d.failure].dropped.mean()),
    }
    probes = pd.read_csv(ROOT / "data" / "closed" / "reidentification.csv")
    named = set(probes[probes.identified].slug)
    rows = []
    for sample, keep in (("all firms", lambda d: d), ("not re-identified", lambda d: d[~d.slug.isin(named)])):
        for cond, d in list(sim.groupby("condition")) + [("real firms", real)]:
            d = keep(d)
            for name, stat in outcomes.items():
                if cond == "real firms" and name in ("hot stove", "escalation"):
                    continue
                lo, hi = interval(d, stat, rng)
                rows.append({"sample": sample, "condition": cond, "outcome": name, "value": stat(d), "low": lo,
                             "high": hi, "products": len(d), "firms": d.slug.nunique()})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "closed" / "analysis.csv", index=False)
    for sample, o in out.groupby("sample"):
        print(f"\n{sample}, {o.firms.max()} firms")
        print(o.pivot(index="condition", columns="outcome", values="value").round(2).to_string())
    if len(sim):
        print("missing decisions:", int(sim.groupby("condition").missing.sum().sum()))


if __name__ == "__main__":
    main()
