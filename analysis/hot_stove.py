"""Hot stove analysis of the own-history run (whatif/memory.py) against the record run (whatif/record.py), per model.

The hot stove effect (Denrell and March 2001) concerns the same alternative: an early negative reading stops the sampling
of that alternative, so that an underestimate is never corrected. Here the alternative is a product. Reads
data/closed/record/<model>/memory.csv (second-year decisions with the agent's own first-year decisions as its past, ten
draws from its own probabilities and the record-as-memory control) and decisions.csv (the record branch, in which the
same products are decided without a memory). Writes hotstove.csv (one row per condition), hotstove_firms.csv (cohort
survival per firm-window and condition), and hotstove.txt.

Three readings per condition of the main framing, on the products the agent kept in the first year:
  persistence   second-year probability of withdrawal with the agent's own first-year ruling as its past, against the
                record branch (no memory) on the same products the firm still had, and against the firm's own
                second-year withdrawal of these products
  hardening     the extra second-year probability on products the agent itself doubted in the first year (own
                probability in the top third of the condition) over the others, with its own memory against the record
                branch; the excess is the same-product hot stove effect at the level of judgment; 95% interval from
                resampling firms
  timing        share of the first-year cohort withdrawn in the first year and in the second, agent (mean over draws)
                against the record, per firm-window with equal weights
The control branch (the firm's actual decisions presented as the agent's own) is reported as the framing check.

Run from the repository root: PYTHONPATH=. python analysis/hot_stove.py <model>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whatif.config import CONFIG, ROOT

MODEL = sys.argv[1] if len(sys.argv) > 1 else CONFIG["record"]["models"][0]
REC = ROOT / "data" / "closed" / "record" / MODEL
rng = np.random.default_rng(11)
KEY = ["slug", "window", "year", "product"]


def firm_ci(df, fn, B=400):
    """95% interval of a statistic by resampling firms."""
    firms = df.slug.unique()
    parts = {s: g for s, g in df.groupby("slug")}
    vals = []
    for _ in range(B):
        pick = rng.choice(firms, len(firms), replace=True)
        v = fn(pd.concat([parts[s] for s in pick], ignore_index=True))
        if v == v:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def doubt_excess(g, cut):
    """Extra second-year probability on doubted products, own memory minus the record branch, on the same live products."""
    live = g[g.record_state == "live"]
    hi, lo = live[live.own_p1 >= cut], live[live.own_p1 < cut]
    if not len(hi) or not len(lo):
        return np.nan
    return float((hi.p_drop.mean() - lo.p_drop.mean()) - (hi.p_record.mean() - lo.p_record.mean()))


def main():
    m = pd.read_csv(REC / "memory.csv")
    m = m[m.p_drop.notna()].copy()
    d = pd.read_csv(REC / "decisions.csv")
    d = d[d.p_drop.notna()].copy()
    prod = pd.read_csv(ROOT / "data" / "closed" / "products.csv")[["slug", "generic", "entry", "drop"]].drop_duplicates(["slug", "generic", "entry"])
    years = CONFIG["record"]["years"]
    out, lines, firm_rows = [], [], []
    pr = lambda *a: (print(*a), lines.append(" ".join(str(x) for x in a)))
    pr(f"{MODEL}: {len(m)} own-history rows, {m.groupby(['slug', 'window']).ngroups} firm-windows, {m.slug.nunique()} firms, "
       f"branches {sorted(m.branch.unique(), key=lambda b: (b != 'record-as-memory', len(b), b))}")
    rec = d[d.condition.isin(m.condition.unique())][KEY + ["condition", "p_drop"]].rename(columns={"p_drop": "p_record"})
    draws = m[m.branch.str.startswith("draw") & m.decided_year1].copy()
    ctrl = m[(m.branch == "record-as-memory") & m.decided_year1].copy()
    for c in sorted(m.condition.unique(), key=lambda c: (c == "none", c)):
        g = draws[draws.condition == c].merge(rec[rec.condition == c], on=KEY, how="left")
        live = g[g.record_state == "live"]
        cc = ctrl[ctrl.condition == c].merge(rec[rec.condition == c], on=KEY, how="left")
        cl = cc[cc.record_state == "live"]
        row = {"model": MODEL, "condition": c, "theory": c.split("_")[0] if "_" in c else "none",
               "n_kept_rows": len(g), "n_live_rows": len(live), "n_firm_windows": g.groupby(["slug", "window"]).ngroups}
        # persistence
        row["p2_own_all"] = float(g.p_drop.mean())
        row["p2_own"] = float(live.p_drop.mean()) if len(live) else np.nan
        row["p2_record"] = float(live.p_record.mean()) if len(live) else np.nan
        row["persist"] = row["p2_own"] - row["p2_record"]
        row["persist_lo"], row["persist_hi"] = firm_ci(live, lambda x: float((x.p_drop - x.p_record).mean())) if len(live) else (np.nan, np.nan)
        row["firm_drop2"] = float(live.recorded_drop.astype(float).mean()) if len(live) else np.nan
        row["framing"] = float((cl.p_drop - cl.p_record).mean()) if len(cl) else np.nan
        # hardening
        cut = float(g.own_p1.quantile(2 / 3))
        row["doubt_cut"] = cut
        hi, lo = live[live.own_p1 >= cut], live[live.own_p1 < cut]
        row["doubt_own"] = float(hi.p_drop.mean() - lo.p_drop.mean()) if len(hi) and len(lo) else np.nan
        row["doubt_record"] = float(hi.p_record.mean() - lo.p_record.mean()) if len(hi) and len(lo) else np.nan
        row["doubt_excess"] = doubt_excess(g, cut)
        row["doubt_lo"], row["doubt_hi"] = firm_ci(g, lambda x: doubt_excess(x, cut))
        # timing and survival, per firm-window
        surv = []
        for (s, w, b), gb in g.groupby(["slug", "window", "branch"]):
            n = int(gb.n_decided_year1.iloc[0])
            surv.append({"slug": s, "window": w, "branch": b, "n_cohort": n, "s1": 1 - gb.n_dropped_year1.iloc[0] / n,
                         "s2": float((1 - gb.p_drop).sum() / n)})
        surv = pd.DataFrame(surv).groupby(["slug", "window"]).agg(n_cohort=("n_cohort", "first"), s1=("s1", "mean"), s2=("s2", "mean"))
        d1 = d[(d.condition == c) & d.year.isin([y[0] for y in years.values()])][["slug", "window", "generic", "entry"]].drop_duplicates()
        d1 = d1.merge(prod, on=["slug", "generic", "entry"], how="left")
        y1 = {w: v[0] for w, v in years.items()}
        y2 = {w: v[-1] for w, v in years.items()}
        d1["r1"] = [(pd.isna(x) or x > y1[w]) for x, w in zip(d1["drop"], d1.window)]
        d1["r2"] = [(pd.isna(x) or x > y2[w]) for x, w in zip(d1["drop"], d1.window)]
        recd = d1.groupby(["slug", "window"]).agg(r1=("r1", "mean"), r2=("r2", "mean"))
        fw = surv.join(recd, how="inner").reset_index()
        fw["condition"] = c
        firm_rows.append(fw.rename(columns={"s2": "agent_survival", "r2": "record_survival"}))
        for k in ("s1", "s2", "r1", "r2"):
            row[k] = float(fw[k].mean())
        row["survival_rho"] = float(fw[["s2", "r2"]].corr(method="spearman").iloc[0, 1]) if len(fw) > 3 else np.nan
        out.append(row)
        pr(f"\n{c}: {row['n_kept_rows']} kept product rows in {row['n_firm_windows']} firm-windows")
        pr(f"  persistence: year-2 withdrawal of products it ruled for, own memory {row['p2_own']:.3f} vs record branch {row['p2_record']:.3f}, "
           f"diff {row['persist']:+.3f} [{row['persist_lo']:+.3f}, {row['persist_hi']:+.3f}]; the firms withdrew {row['firm_drop2']:.3f} of them "
           f"(n {row['n_live_rows']}); framing check {row['framing']:+.3f}")
        pr(f"  hardening: doubted products (own p1 >= {cut:.2f}) over the others, own memory {row['doubt_own']:+.3f} vs record branch "
           f"{row['doubt_record']:+.3f}, excess {row['doubt_excess']:+.3f} [{row['doubt_lo']:+.3f}, {row['doubt_hi']:+.3f}]")
        pr(f"  timing: cohort withdrawn in year 1 agent {1 - row['s1']:.3f} vs record {1 - row['r1']:.3f}; in year 2 agent {row['s1'] - row['s2']:.3f} "
           f"vs record {row['r1'] - row['r2']:.3f}; survival after year 2 agent {row['s2']:.3f} vs record {row['r2']:.3f}, Spearman {row['survival_rho']:.2f}")
    pd.DataFrame(out).to_csv(REC / "hotstove.csv", index=False)
    if firm_rows:
        pd.concat(firm_rows, ignore_index=True).to_csv(REC / "hotstove_firms.csv", index=False)
    (REC / "hotstove.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
