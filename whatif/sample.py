"""Draw the main sample and the pilot subsample from the sample frame of Dahlke et al. (2026).

Firms are eligible if Common Crawl holds their website in the exploration, pandemic, and hold-out periods and if
Eurostat reports annual turnover for their industry, which the pandemic test needs to set the shock by industry. Cells
are formed by technological regime, size class, pandemic class of the industry, and tertile of the web-based innovation
probability. The sample size of a cell is proportional to the square root of its size, multiplied by the weights in
config.yaml, so that small cells, industries affected by the pandemic, and likely innovators are represented well.
Within cells, firms are drawn at random, which gives each firm a known inclusion probability. The pilot subsample is
drawn in the same way from sampled firms whose websites are archived in every year.

Outputs: data/closed/sample.csv with all variables, which stays local, and data/public/pilot_firms.csv with the
domains of the pilot firms, which is the only data file released with the repository.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whatif.config import CONFIG, PATHS, ROOT
from whatif.eurostat import DIR as EUROSTAT

S = CONFIG["sample"]


def coverage(domains):
    """Years in which Common Crawl holds the website of each domain, read from the yearly WebAI score files."""
    years = {d: set() for d in domains}
    for y in range(2016, 2026):
        ids = pd.read_parquet(Path(PATHS["coverage"]) / f"de_{y}_predictor_intensity_score.gz", columns=["id"])["id"]
        for d in set(ids.str.lower().str.removeprefix("www.")) & years.keys():
            years[d].add(y)
    return years


def pandemic_classes():
    """Change of annual turnover from 2019 to 2020 by NACE division, from Eurostat short-term statistics.

    Where Eurostat reports a division only through one of its groups, the group stands for the division.
    """
    change = {}
    for f in ("sts_intv_a", "sts_trtu_a", "sts_setu_a"):
        j = json.load(open(EUROSTAT / f"{f}.json"))
        dims, size = j["id"], j["size"]
        index = {d: j["dimension"][d]["category"]["index"] for d in dims}
        want = {"indic_bt": "NETTUR", "s_adj": "NSA", "unit": "I21"}
        for code in sorted(index["nace_r2"], key=len, reverse=True):   # divisions overwrite their groups
            if len(code) not in (3, 4) or not code[1:].isdigit():
                continue
            vals = []
            for year in ("2019", "2020"):
                pos = {**want, "nace_r2": code, "time": year}
                flat = 0
                for d, n in zip(dims, size):
                    flat = flat * n + (index[d][pos[d]] if d in pos else 0)
                vals.append(j["value"].get(str(flat)))
            if None not in vals:
                change[int(code[1:3])] = vals[1] / vals[0] - 1
    t = S["pandemic_threshold"]
    return {n: "hit" if c <= t["hit"] else "boosted" if c >= t["boosted"] else "stable" for n, c in change.items()}


def frame():
    """Eligible firms with the variables that define the cells."""
    p = pd.read_parquet(PATHS["frame"])
    f = p[p.period == "train"][["firm_id", "nace2", "nace4", "Sizeclassification", "inno_innoprob"]].copy()
    f.columns = ["domain", "nace2", "nace4", "size", "innoprob"]
    f["domain"] = f["domain"].str.lower().str.removeprefix("www.")
    f = f.dropna(subset=["innoprob"])
    f = f[~f.nace4.astype(str).isin(S["exclude_nace4"]) & ~f.nace2.astype(str).isin(S["exclude_nace2"])]

    years = coverage(set(f.domain))
    def covered(d):
        ys = years.get(d, set())
        return all(sum(a <= y <= b for y in ys) >= k for a, b, k in S["coverage"].values())
    f = f[f.domain.map(covered)]
    f["covered_years"] = f.domain.map(lambda d: len(years[d]))

    regime = {n: r for r, codes in CONFIG["regimes"].items() for n in codes}
    f["regime"] = f.nace2.astype(int).map(regime).fillna("other")
    f["pandemic"] = f.nace2.astype(int).map(pandemic_classes()).fillna("unknown")
    if S["require_turnover"]:
        f = f[f.pandemic != "unknown"]
    f["innovation"] = pd.qcut(f.innoprob, 3, labels=["low", "mid", "high"]).astype(str)
    f["cell"] = f[["regime", "size", "pandemic", "innovation"]].agg("|".join, axis=1)
    return f.reset_index(drop=True)


def allocate(sizes, weights, n):
    """Cell sample sizes proportional to weight times the square root of cell size, capped at cell size."""
    target = weights * np.sqrt(sizes)
    alloc = pd.Series(0, index=sizes.index)
    open_ = sizes.index
    while n - alloc.sum() > 0 and len(open_):
        share = target[open_] / target[open_].sum() * (n - alloc.sum())
        add = np.minimum(np.floor(share).clip(lower=1), sizes[open_] - alloc[open_])
        alloc[open_] += add.astype(int)
        open_ = alloc.index[alloc < sizes]
        if add.sum() == 0:
            break
    while alloc.sum() > n:                       # remove the surplus from the largest cells
        alloc[alloc.idxmax()] -= 1
    return alloc


def draw(f, n, weights, rng):
    """Simple random sample within cells, with inclusion probabilities."""
    sizes = f.groupby("cell").size()
    alloc = allocate(sizes, weights.reindex(sizes.index), n)
    picked = [g.sample(alloc[c], random_state=rng) for c, g in f.groupby("cell") if alloc[c] > 0]
    out = pd.concat(picked)
    out["inclusion_probability"] = out.cell.map(alloc / sizes)
    return out


def main():
    rng = np.random.RandomState(CONFIG["seed"])
    f = frame()
    cells = f.groupby("cell").first()
    w = (cells.pandemic.map(S["weight_pandemic_class"]) * cells.innovation.map(S["weight_innovation_tertile"]))
    main_sample = draw(f, S["n_main"], w, rng)

    rich = main_sample[main_sample.covered_years >= S["pilot_min_covered_years"]].copy()
    rich["cell"] = rich[["regime", "pandemic"]].agg("|".join, axis=1)
    pilot = draw(rich, S["n_pilot"], pd.Series(1.0, index=rich.cell.unique()), rng)
    main_sample["pilot"] = main_sample.domain.isin(pilot.domain)
    main_sample["pilot_inclusion_probability"] = main_sample.domain.map(
        pilot.set_index("domain").inclusion_probability)

    closed, public = ROOT / "data" / "closed", ROOT / "data" / "public"
    closed.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    main_sample.to_csv(closed / "sample.csv", index=False)
    main_sample.loc[main_sample.pilot, ["domain"]].sort_values("domain").to_csv(public / "pilot_firms.csv", index=False)
    print(f"frame {len(f):,} eligible firms, sample {len(main_sample):,}, pilot {int(main_sample.pilot.sum())}")
    for col in ("regime", "size", "pandemic", "innovation"):
        print(pd.crosstab(main_sample[col], main_sample.pilot, margins=True).to_string(), "\n")


if __name__ == "__main__":
    main()
