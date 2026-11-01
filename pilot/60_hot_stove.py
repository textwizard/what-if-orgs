"""Hot stove analysis of the memory run (whatif/memory.py) against the record run (whatif/record.py), per model.

Reads data/closed/record/<model>/memory.csv (second-year decisions with the agent's own first-year history, ten draws and
the record-as-memory control) and decisions.csv (the record branch). Writes hotstove.csv (one row per condition and
measure), hotstove_firms.csv (survival per firm, window, condition), and hotstove.txt.

Measures, per condition of the main framing:
  slope        hot stove: within firm-window, the change of the second-year probability of a surviving product when the
               share of the other first-year products the same agent dropped rises from 0 to 1 (OLS with firm-window
               fixed effects on the draws; leave-one-out share; 95% interval from resampling firms)
  contrast     mean second-year probability in draws with at least one own drop minus draws without, within firm-window
  framing      record-as-memory minus the record branch, same products the firm still had: what the memory framing does
  doubt        escalation: products the agent itself doubted in the first year (own probability in the top third of the
               condition) minus the others, in the own-history draws, relative to the same difference in the record branch
  survival     firm level: expected share of the first-year cohort still on the market after the second year under the
               agent's own history (mean over draws) against the recorded share; correlation across firm-windows

Run: python code/60_hot_stove.py <model>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whatif import record
from whatif.config import CONFIG, ROOT

MODEL = sys.argv[1] if len(sys.argv) > 1 else CONFIG["record"]["models"][0]
REC = ROOT / "data" / "closed" / "record" / MODEL
rng = np.random.default_rng(11)


def fe_slope(df, x, y, group):
    """OLS slope of y on x after demeaning both within group."""
    g = df.groupby(group)
    xd, yd = df[x] - g[x].transform("mean"), df[y] - g[y].transform("mean")
    v = float((xd * xd).sum())
    return float((xd * yd).sum() / v) if v > 0 else np.nan


def firm_ci(df, fn, B=400):
    """95% interval of a statistic by resampling firms."""
    firms = df.slug.unique()
    parts = {s: g for s, g in df.groupby("slug")}
    vals = []
    for _ in range(B):
        pick = rng.choice(firms, len(firms), replace=True)
        vals.append(fn(pd.concat([parts[s] for s in pick], ignore_index=True)))
    vals = [v for v in vals if v == v]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def main():
    m = pd.read_csv(REC / "memory.csv")
    m = m[m.p_drop.notna()].copy()
    d = pd.read_csv(REC / "decisions.csv")
    d = d[d.p_drop.notna()].copy()
    prod = pd.read_csv(ROOT / "data" / "closed" / "products.csv")
    out, lines, firm_rows = [], [], []
    pr = lambda *a: (print(*a), lines.append(" ".join(str(x) for x in a)))
    pr(f"{MODEL}: {len(m)} memory rows, {m.slug.nunique()} firms, branches {sorted(m.branch.unique(), key=lambda b: (b != 'record-as-memory', len(b), b))}")
    draws = m[m.branch.str.startswith("draw")].copy()
    ctrl = m[m.branch == "record-as-memory"].copy()
    # the record branch of the same second years, same products
    key = ["slug", "window", "year", "product"]
    rec = d[d.condition.isin(m.condition.unique())][key + ["condition", "p_drop", "recorded_drop"]].rename(columns={"p_drop": "p_record"})
    # leave-one-out share of own drops among the other first-year products
    draws["others_dropped"] = np.where(draws.n_decided_year1 > 1, draws.n_dropped_year1 / (draws.n_decided_year1 - 1).clip(lower=1), np.nan)
    draws["fw"] = draws.slug + "|" + draws.window
    for c in sorted(m.condition.unique(), key=lambda c: (c != "none", c)):
        dc = draws[(draws.condition == c) & draws.decided_year1 & draws.others_dropped.notna()]
        row = {"model": MODEL, "condition": c, "theory": c.split("_")[0] if "_" in c else "none", "n_draw_rows": len(dc)}
        # hot stove slope and contrast
        row["slope"] = fe_slope(dc, "others_dropped", "p_drop", "fw")
        row["slope_lo"], row["slope_hi"] = firm_ci(dc, lambda g: fe_slope(g, "others_dropped", "p_drop", "fw"))
        dc = dc.assign(any_drop=(dc.n_dropped_year1 > 0).astype(float))
        row["contrast"] = fe_slope(dc, "any_drop", "p_drop", "fw")
        row["contrast_lo"], row["contrast_hi"] = firm_ci(dc, lambda g: fe_slope(g, "any_drop", "p_drop", "fw"))
        row["mean_p_no_drop"] = float(dc[dc.n_dropped_year1 == 0].p_drop.mean())
        row["mean_p_any_drop"] = float(dc[dc.n_dropped_year1 > 0].p_drop.mean())
        # framing: record-as-memory against the record branch on products the firm still had
        cc = ctrl[(ctrl.condition == c) & (ctrl.record_state == "live")].merge(rec[rec.condition == c], on=key, suffixes=("", "_r"))
        row["n_control"] = len(cc)
        row["p_record_branch"] = float(cc.p_record.mean()) if len(cc) else np.nan
        row["p_record_as_memory"] = float(cc.p_drop.mean()) if len(cc) else np.nan
        row["framing"] = row["p_record_as_memory"] - row["p_record_branch"]
        row["framing_lo"], row["framing_hi"] = firm_ci(cc, lambda g: float((g.p_drop - g.p_record).mean())) if len(cc) else (np.nan, np.nan)
        # doubt: products the agent doubted in the first year, in the draws (kept by construction) and in the record branch
        dd = draws[(draws.condition == c) & draws.decided_year1 & draws.own_p1.notna()]
        if len(dd):
            cut = dd.own_p1.quantile(2 / 3)
            dd = dd.assign(doubted=(dd.own_p1 >= cut).astype(float))
            row["doubt_cut"] = float(cut)
            row["doubt_draws"] = float(dd[dd.doubted == 1].p_drop.mean() - dd[dd.doubted == 0].p_drop.mean())
            rr = dd[dd.record_state == "live"].drop_duplicates(key).merge(rec[rec.condition == c], on=key)
            row["doubt_record"] = float(rr[rr.doubted == 1].p_record.mean() - rr[rr.doubted == 0].p_record.mean()) if len(rr) else np.nan
            row["doubt"] = row["doubt_draws"] - row["doubt_record"]
        # survival at the firm level
        surv = []
        for (s, w, b), g in draws[draws.condition == c].groupby(["slug", "window", "branch"]):
            dec = g[g.decided_year1]
            if not len(dec):
                continue
            n = int(dec.n_decided_year1.iloc[0])
            kept = len(dec)                                 # decided products still on the market in this history
            surv.append({"slug": s, "window": w, "branch": b, "n_cohort": n, "agent_survival": float((1 - dec.p_drop).sum() / n)})
        surv = pd.DataFrame(surv)
        if len(surv):
            agent = surv.groupby(["slug", "window"]).agent_survival.mean().rename("agent_survival")
            # recorded survival of the same cohort: first-year decided products not dropped up to and including the second year
            y1 = {w: CONFIG["record"]["years"][w][0] for w in CONFIG["record"]["years"]}
            y2 = {w: CONFIG["record"]["years"][w][-1] for w in CONFIG["record"]["years"]}
            d1 = d[(d.condition == c) & d.year.isin(y1.values())][["slug", "window", "generic", "entry"]].drop_duplicates()
            d1 = d1.merge(prod[["slug", "generic", "entry", "drop"]].drop_duplicates(["slug", "generic", "entry"]), on=["slug", "generic", "entry"], how="left")
            d1["survives"] = [(pd.isna(x) or x > y2[w]) for x, w in zip(d1["drop"], d1.window)]
            recorded = d1.groupby(["slug", "window"]).survives.mean().rename("record_survival")
            fw = pd.concat([agent, recorded], axis=1).dropna().reset_index()
            fw["condition"] = c
            firm_rows.append(fw)
            row["n_firm_windows"] = len(fw)
            row["agent_survival"] = float(fw.agent_survival.mean())
            row["record_survival"] = float(fw.record_survival.mean())
            row["survival_rho"] = float(fw[["agent_survival", "record_survival"]].corr(method="spearman").iloc[0, 1]) if len(fw) > 3 else np.nan
        out.append(row)
        pr(f"\n{c}: {row['n_draw_rows']} draw rows")
        pr(f"  hot stove slope {row['slope']:+.3f} [{row['slope_lo']:+.3f}, {row['slope_hi']:+.3f}]; any own drop vs none {row['contrast']:+.3f} "
           f"[{row['contrast_lo']:+.3f}, {row['contrast_hi']:+.3f}] (means {row['mean_p_no_drop']:.3f} -> {row['mean_p_any_drop']:.3f})")
        pr(f"  framing: record-as-memory {row['p_record_as_memory']:.3f} vs record branch {row['p_record_branch']:.3f}, diff {row['framing']:+.3f} "
           f"[{row['framing_lo']:+.3f}, {row['framing_hi']:+.3f}] on {row['n_control']} products")
        if "doubt" in row:
            pr(f"  doubted products (own p1 >= {row['doubt_cut']:.2f}): draws {row['doubt_draws']:+.3f}, record branch {row['doubt_record']:+.3f}, "
               f"excess {row['doubt']:+.3f}")
        if "agent_survival" in row:
            pr(f"  survival of the first-year cohort after the second year: agent {row['agent_survival']:.3f} vs record {row['record_survival']:.3f} "
               f"over {row['n_firm_windows']} firm-windows, Spearman {row['survival_rho']:.2f}")
    pd.DataFrame(out).to_csv(REC / "hotstove.csv", index=False)
    if firm_rows:
        pd.concat(firm_rows, ignore_index=True).to_csv(REC / "hotstove_firms.csv", index=False)
    (REC / "hotstove.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
