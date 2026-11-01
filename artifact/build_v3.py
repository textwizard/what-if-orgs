"""Build the research artifact (v3): one page per component of the proof of concept, from the data of the final sample.

Writes data_v3.json (or data_public.json with --public) for the page template: funnel and counts, the record of every
firm, the simulation on the record (levels, ranking, anchor contrasts), weights and validation, the own-history run
(hot stove measures and cohort survival over time), the theory swap, the firms with their sources, and the methods
(protocols, prompts, models, release). build_v5.py injects the data into template_v5.html; figures are drawn in the page.

Run from whatif-organizations with PYTHONPATH set: python3 ../09_whatif_pilot/explorer/build_v3.py [--public]
--public leaves out everything that comes from the closed frame (its NACE code and size class per firm), so that the
page can be released with the repository; the local build keeps them for the agreement checks.
Outputs: explorer/index_v3.html (local) or explorer/public/index.html (--public)
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from whatif import context, record
from whatif.config import CONFIG, ROOT

HERE = Path(__file__).parent
PUBLIC = "--public" in sys.argv
REC = ROOT / "data" / "closed" / "record"
CD = ROOT / "data" / "closed" / CONFIG["record"]["coding_dir"]
MODELS = {"gpt-5.4": "gpt-5.4", "claude-opus-5-5": "Claude Opus 5.5", "deepseek-v4-pro": "DeepSeek V4 Pro"}
FRAMINGS = {"F": "Considerations", "B": "Conditions"}
THEORIES = {"none": "No theory", "btf": "Behavioral theory", "ac": "Absorptive capacity", "esc": "Escalation",
            "ro": "Real options"}
EMPH = ["performance", "knowledge", "commitment", "experimentation"]
LEVELS = {0: "not mentioned", 1: "mentioned", 2: "own page", 3: "home page"}
rng = np.random.default_rng(CONFIG["seed"])
sys.path.insert(0, str(HERE.parent / "code"))
import importlib.util
spec = importlib.util.spec_from_file_location("diag", HERE.parent / "code" / "55_record_diagnostics.py")
diag = importlib.util.module_from_spec(spec)
sys.argv_backup, sys.argv = sys.argv, ["x", "claude-opus-5-5"]
spec.loader.exec_module(diag)
sys.argv = sys.argv_backup


def auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, bool)
    r = rankdata(p)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def interval(df, col, B=400):
    groups = [g[col].astype(float).values for _, g in df.groupby("slug")]
    means = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        means.append(np.concatenate([groups[i] for i in pick]).mean())
    return [round(float(df[col].astype(float).mean()), 4), round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4)]


def decisions(model):
    f = REC / model / "decisions.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = record.label(d[d.p_drop.notna()].copy())
    return diag.features(d)


def reference(d):
    """Levels per window with intervals, cue tables, and discrimination, for one model."""
    conds = ["none"] + [f"{t}_{f}" for f in FRAMINGS for t in THEORIES if t != "none"]
    conds = [c for c in conds if c in set(d.condition)]
    out = {"conditions": conds, "levels": {}, "cues": {}, "auc": {}, "auc_holdout": {}, "n": int(len(d[d.condition == "none"]))}
    for w, g in d.groupby("window"):
        out["levels"][w] = {"record": interval(g[g.condition == "none"], "recorded_drop")}
        for c in conds:
            out["levels"][w][c] = interval(g[g.condition == c], "p_drop")
    for c in conds:
        out["auc"][c] = auc(d[d.condition == c].p_drop, d[d.condition == c].recorded_drop)
        h = d[(d.condition == c) & (d.window == "holdout")]
        out["auc_holdout"][c] = auc(h.p_drop, h.recorded_drop)
    out["anchors"] = anchor_contrast(d)
    base = d[d.condition == "none"]
    for cue, keys in (("relatedness", ["close", "related", "distant"]), ("last_level", [1, 2, 3]),
                      ("result_before", [False, True]), ("age", [1, 2, 3, 4])):
        out["cues"][cue] = {}
        for k in keys:
            sel = d[d[cue] == k]
            if not len(sel):
                continue
            out["cues"][cue][str(k).lower()] = {"n": int(len(base[base[cue] == k])),
                                        "record": round(float(base[base[cue] == k].recorded_drop.mean()), 3),
                                        **{c: round(float(sel[sel.condition == c].p_drop.mean()), 3) for c in conds}}
    return out


def ceiling(d):
    """AUC of a logistic regression on the cues, fitted on exploration, tested on holdout (the predictability of the
    record from what the agent sees)."""
    from scipy.optimize import minimize
    base = d[d.condition == "none"]
    X = pd.get_dummies(base[["last_level", "result_before", "relatedness", "age", "window", "recorded_drop"]]
                       .assign(last_level=lambda x: x.last_level.astype(str), age=lambda x: x.age.clip(upper=4).astype(str)),
                       columns=["last_level", "relatedness", "age"], dtype=float)
    cols = [c for c in X.columns if c not in ("window", "recorded_drop")]
    X["result_before"] = X.result_before.astype(float)
    cal, val = X[X.window == "exploration"], X[X.window == "holdout"]
    A = np.column_stack([np.ones(len(cal)), cal[cols].values]); y = cal.recorded_drop.astype(float).values
    f = lambda b: -np.mean(y * np.log(1 / (1 + np.exp(-A @ b)) + 1e-9) + (1 - y) * np.log(1 - 1 / (1 + np.exp(-A @ b)) + 1e-9)) + 0.01 * np.sum(b[1:] ** 2)
    b = minimize(f, np.zeros(A.shape[1]), method="L-BFGS-B").x
    pv = 1 / (1 + np.exp(-np.column_stack([np.ones(len(val)), val[cols].values]) @ b))
    return {"holdout": round(auc(pv, val.recorded_drop), 3), "exploration": round(auc(1 / (1 + np.exp(-A @ b)), y), 3)}


def weights(model):
    out = {}
    for fr in FRAMINGS:
        f = REC / model / f"weights_common_{fr}.csv"
        if not f.exists():
            continue
        common = pd.read_csv(f).iloc[0].to_dict()
        regime = pd.read_csv(REC / model / f"weights_regime_{fr}.csv")
        firm = pd.read_csv(REC / model / f"weights_firm_{fr}.csv")
        val = pd.read_csv(REC / model / f"validation_{fr}.csv", index_col=0)
        coh = pd.read_csv(REC / model / f"coherence_{fr}.csv")
        C = [c for c in common]
        fe = firm[firm.window == "exploration"]
        out[fr] = {"conditions": C, "common": {c: round(float(common[c]), 3) for c in C},
                   "regime": [{"regime": r.regime.replace("_", " "), "n": int(r.n), **{c: round(float(getattr(r, c)), 3) for c in C}}
                              for r in regime.itertuples()],
                   "firm_sd": {c: round(float(fe[c].astype(float).std()), 3) for c in C},
                   "firm_mean": {c: round(float(fe[c].astype(float).mean()), 3) for c in C},
                   "validation": [{"predictor": k, **{m: (None if pd.isna(v) else round(float(v), 3)) for m, v in r.items()}}
                                  for k, r in val.iterrows()],
                   "gap": ([{"predictor": k, **{m: (None if pd.isna(v) else round(float(v), 3)) for m, v in r.items()}}
                            for k, r in pd.read_csv(REC / model / f"gap_{fr}.csv", index_col=0).iterrows()]
                           if (REC / model / f"gap_{fr}.csv").exists() else None),
                   "coherence": [{k: (None if pd.isna(v) else (round(float(v), 3) if isinstance(v, float) else v))
                                  for k, v in r.items()} for _, r in coh.iterrows()]}
    return out


def record_page(d):
    """One firm's trajectories, drop rates by prominence and age, lapses, and the ceiling."""
    prod = pd.read_csv(ROOT / "data" / "closed" / "products.csv")
    slugs = [l.strip() for l in open(ROOT / "data" / "closed" / "sample_final_slugs.txt") if l.strip()]
    p = prod[prod.slug.isin(slugs) & prod.innovation]
    base = d[d.condition == "none"]
    by = lambda col: {str(k): {"n": int(len(g)), "drop": round(float(g.recorded_drop.mean()), 3)} for k, g in base.groupby(col)}
    lapses = 0
    for r in p.itertuples():
        lv = {int(y): v for y, v in json.loads(r.levels).items()}
        inf = sorted(json.loads(r.informative))
        seq = [lv[y] for y in inf if y >= r.entry]
        lapses += any(seq[i] == 0 and any(v > 0 for v in seq[i + 1:]) for i in range(len(seq)))
    return {"by_level": by("last_level"), "by_age": by("age"), "n_innovations": int(len(p)),
            "n_dropped": int(p["drop"].notna().sum()), "lapses": int(lapses), "ceiling": ceiling(d),
            "drop_by_window": {w: round(float(g.recorded_drop.mean()), 3) for w, g in base.groupby("window")}}


def firms(ds):
    """Every firm with its sources: profile, situation, division, value emphases, products with trajectories and
    evidence, and the decisions of every model and condition with reasons."""
    prod = pd.read_csv(ROOT / "data" / "closed" / "products.csv")
    slugs = [l.strip() for l in open(ROOT / "data" / "closed" / "sample_final_slugs.txt") if l.strip()]
    sample = pd.read_csv(ROOT / "data" / "closed" / "sample.csv")
    sample["slug"] = sample.domain.str.replace(".", "-", regex=False)
    div = pd.read_csv(ROOT / "data" / "closed" / "division.csv", dtype={"division": str, "division_frame": str})
    values = {}
    for coder in ("opus", "gpt54"):
        f = ROOT / "data" / "closed" / "funnel" / f"value_{coder}.csv"
        if f.exists():
            values[coder] = pd.read_csv(f)
    regime = {n: r for r, codes in CONFIG["regimes"].items() for n in codes}
    out = {}
    for s in slugs:
        coded = json.loads((CD / f"{s}.json").read_text())
        ctx = json.loads((context.OUT / f"{s}.json").read_text())["windows"]
        firm = sample[sample.slug == s].iloc[0]
        dv = div[div.slug == s].sort_values("year")
        d_web = dv.iloc[-1]
        f = {"domain": firm.domain, "division": {"web": d_web.division, "label": d_web.label, "year": int(d_web.year),
                                                 "quote": d_web.quote, "confidence": None if pd.isna(d_web.confidence) else float(d_web.confidence),
                                                 "regime": regime.get(int(d_web.division), "other").replace("_", " ")},
             "windows": {}, "values": {}, "products": [], "decisions": {}}
        if not PUBLIC:
            f["division"]["frame"] = d_web.division_frame
        for coder, v in values.items():
            for r in v[v.slug == s].itertuples():
                f["values"].setdefault(str(r.year), {})[coder] = {e: {"points": float(getattr(r, e)), "quote": getattr(r, f"{e}_quote")} for e in EMPH}
        for w, p in coded.get("profiles", {}).items():
            c = ctx.get(w)
            f["windows"][w] = {"seed_year": p["seed_year"], "strategy": p["strategy"],
                               "size": p["context"].get("size"), "founded": p["context"].get("founded"),
                               "industry": p["context"].get("industry"), "situation": context.render(c) if c else None}
        ev = {}
        names = {}
        pr = prod[(prod.slug == s) & prod.innovation].sort_values("entry")
        for r in pr.itertuples():
            for n in r.names.split("|"):
                names[n] = r.product
        for y in coded["years"]:
            for e in y.get("evidence", []):
                k = names.get((e.get("product") or "").strip())
                if k is not None:
                    ev.setdefault(k, []).append({"year": y["year"], "type": e.get("type"), "strength": e.get("strength"), "quote": e.get("quote")})
        for r in pr.itertuples():
            f["products"].append({"id": int(r.product), "name": r.names.split("|")[0], "generic": r.generic, "entry": int(r.entry),
                                  "drop": None if pd.isna(r.drop) else int(r.drop),
                                  "levels": {int(y): v for y, v in json.loads(r.levels).items()},
                                  "informative": json.loads(r.informative), "evidence": ev.get(int(r.product), [])})
        for m, d in ds.items():
            g = d[d.slug == s]
            for r in g.itertuples():
                f["decisions"].setdefault(m, {}).setdefault(r.condition, {}).setdefault(f"{r.window}|{r.year}", {})[r.product] = \
                    {"generic": r.generic, "entry": int(r.entry), "p": round(float(r.p_drop), 2), "decision": r.decision,
                     "reason": r.reason, "recorded": bool(r.recorded_drop)}
        out[s] = f
    return out


def web_measure(slugs):
    """Pages, characters, roles, and new text per firm and year, from the pages read for the agent."""
    import hashlib
    out = {}
    for s in slugs:
        by = {}
        for line in open(ROOT / "data" / "closed" / "pages" / f"{s}.jsonl", encoding="utf-8"):
            p = json.loads(line)
            by.setdefault(int(p["year"]), []).append(p)
        seen, rows = set(), {}
        for y in sorted(by):
            pages = by[y]
            hashes = {hashlib.sha1(p["text"].encode()).hexdigest() for p in pages}
            rows[y] = {"pages": len(pages), "chars": sum(len(p["text"]) for p in pages),
                       "roles": {r: sum(p["role"] == r for p in pages) for r in ("home", "about", "offer", "news", "rnd", "career", "other") if any(p["role"] == r for p in pages)},
                       "new_share": round(len(hashes - seen) / len(hashes), 2) if hashes else None}
            seen |= hashes
        out[s] = rows
    return out


def anchor_contrast(d):
    """Anchored against unanchored agents, per condition of the main framing: level, AUC, log loss, attention."""
    if "anchors" not in d.columns or (d.anchors == "none").sum() == 0:
        return None
    base = ["none"] + [f"{t}_F" for t in CONFIG["record"]["theories"] if t != "none"]
    out = {}
    for c in base:
        for anchors, cond in (("web", c), ("none", f"{c}-na"), ("own", f"{c}-oa")):
            g = d[d.condition == cond]
            if not len(g):
                continue
            out.setdefault(c, {})[anchors] = {
                "n": int(len(g)), "mean_p": round(float(g.p_drop.mean()), 3), "auc": round(auc(g.p_drop, g.recorded_drop), 3),
                "logloss": round(float(-np.mean(g.recorded_drop * np.log(g.p_drop.clip(0.02, 0.98)) + (1 - g.recorded_drop) * np.log(1 - g.p_drop.clip(0.02, 0.98)))), 3),
                "holdout_p": round(float(g[g.window == "holdout"].p_drop.mean()), 3),
                "auc_holdout": round(auc(g[g.window == "holdout"].p_drop, g[g.window == "holdout"].recorded_drop), 3),
                "distant": round(float(g[g.relatedness == "distant"].p_drop.mean()), 3),
                "result": round(float(g[g.result_before].p_drop.mean()), 3)}
    wide = d[d.condition.isin(base + [f"{c}-na" for c in base])].pivot_table(index=["slug", "window", "year", "product"], columns="condition", values="p_drop")
    out["correlation"] = {c: round(float(wide[c].corr(wide[f"{c}-na"])), 2) for c in base if f"{c}-na" in wide.columns}
    n0 = d[d.condition == "none"]
    out["recorded"] = round(float(n0.recorded_drop.mean()), 3)
    out["recorded_distant"] = round(float(n0[n0.relatedness == "distant"].recorded_drop.mean()), 3)
    out["recorded_result"] = round(float(n0[n0.result_before].recorded_drop.mean()), 3)
    out["n_distant"] = int((n0.relatedness == "distant").sum()); out["n_result"] = int(n0.result_before.sum())
    return out


def swap(ds):
    """The firm-window-year where the theories disagree most, with the products under every condition, per model."""
    out = {}
    for m, d in ds.items():
        g = d[(d.framing.fillna("F").eq("F") | d.condition.eq("none")) & ~d.condition.str.endswith("-na")]
        spread = g.groupby(["slug", "window", "year", "product"]).p_drop.std().groupby(["slug", "window", "year"]).mean()
        n = g[g.condition == "none"].groupby(["slug", "window", "year"]).size()
        best = (spread[n >= 4]).idxmax()
        sel = d[(d.slug == best[0]) & (d.window == best[1]) & (d.year == best[2]) & ~d.condition.str.endswith("-na")]
        out[m] = {"slug": best[0], "window": best[1], "year": int(best[2]),
                  "rows": [{"condition": r.condition, "product": r.product, "generic": r.generic, "entry": int(r.entry), "p": round(float(r.p_drop), 2),
                            "decision": r.decision, "reason": r.reason, "recorded": bool(r.recorded_drop)} for r in sel.itertuples()]}
    return out


def hotstove(model):
    """The hot stove run (whatif/memory.py, code/60_hot_stove.py): per condition the within-firm slope, the framing effect,
    the doubt excess, and the firm-level survival against the record; and the per-firm survival pairs."""
    f = REC / model / "hotstove.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f)
    rows = {r.condition: {k: (None if pd.isna(v) else (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v))
                          for k, v in r._asdict().items() if k not in ("Index", "model", "condition")} for r in t.itertuples()}
    out = {"conditions": list(t.condition), "rows": rows}
    g = REC / model / "hotstove_firms.csv"
    if g.exists():
        fw = pd.read_csv(g)
        out["firms"] = [{"slug": r.slug, "window": r.window, "condition": r.condition, "agent": round(float(r.agent_survival), 3),
                         "record": round(float(r.record_survival), 3), "n": int(r.n_cohort) if "n_cohort" in fw.columns else None} for r in fw.itertuples()]
    m = pd.read_csv(REC / model / "memory.csv") if (REC / model / "memory.csv").exists() else None
    if m is not None:
        m = m[m.p_drop.notna()]
        out["n_rows"] = int(len(m)); out["n_draws"] = int(m.draw.max()); out["firm_windows"] = int(m.groupby(["slug", "window"]).ngroups)
        out["branches"] = sorted(m.branch.unique(), key=lambda b: (b != "record-as-memory", len(b), b))
        out["curves"] = survival_curves(model, m)
    return out


def survival_curves(model, m):
    """Share of the first-year cohort still on the market, over time: the record at every archived year, the agent under
    each condition at the launch year, after its own first-year decisions (mean over draws), and after the second year
    (expected from its probabilities). Per window, equal weight per firm-window; only firm-windows the condition decided."""
    years = CONFIG["record"]["years"]
    prod = pd.read_csv(ROOT / "data" / "closed" / "products.csv")[["slug", "generic", "entry", "drop"]].drop_duplicates(["slug", "generic", "entry"])
    dec = pd.read_csv(REC / model / "decisions.csv")
    draws = m[m.branch.str.startswith("draw") & m.decided_year1].copy()
    out = {}
    for w, (y1, y2) in years.items():
        launch = CONFIG["simulation"]["windows"][w][0]
        ys = [y for y in [2017, 2018, 2019, 2022, 2023, 2024, 2025] if y >= launch]     # a drop in 2026 is not observable
        cw = {}
        for c, g in draws[draws.window == w].groupby("condition"):
            per = []
            for (slug, b), gb in g.groupby(["slug", "branch"]):
                n = int(gb.n_decided_year1.iloc[0])
                per.append({"slug": slug, "s1": 1 - gb.n_dropped_year1.iloc[0] / n, "s2": (1 - gb.p_drop).sum() / n})
            per = pd.DataFrame(per).groupby("slug").mean()
            d1 = dec[(dec.condition == c) & (dec.window == w) & (dec.year == y1) & dec.slug.isin(per.index)][["slug", "generic", "entry"]].drop_duplicates()
            d1 = d1.merge(prod, on=["slug", "generic", "entry"], how="left")
            rec = [float(d1.groupby("slug").apply(lambda gg: ((gg["drop"].isna()) | (gg["drop"] > y)).mean()).mean()) if y >= y1 else 1.0 for y in ys]
            cw[c] = {"agent": [1.0, round(float(per.s1.mean()), 4), round(float(per.s2.mean()), 4)], "record": [round(v, 4) for v in rec], "n_firm_windows": int(len(per))}
        out[w] = {"years": ys, "agent_years": [launch, y1, y2], "conditions": cw}
    return out


def costs():
    log = pd.read_json(ROOT / "data" / "closed" / "llm_log.jsonl", lines=True)
    log = log[log.time >= CONFIG["llm"]["budget_since"]]
    log["step"] = log.purpose.str.split().str[0]
    steps = {"coding": "Coding of products, evidence, and profiles", "context": "Situation of the firm", "value": "Theory of value",
             "division": "NACE division", "aliases": "Product aliases", "relatedness": "Relatedness", "record": "Replay on the record",
             "simulate": "Simulations on assigned results and instrument tests", "phrasing": "Framing tests", "reidentification": "Re-identification test",
             "recall": "Recall test"}
    t = log.groupby("step").agg(calls=("usd", "size"), tokens_in=("tokens_in", "sum"), tokens_out=("tokens_out", "sum"), usd=("usd", "sum"))
    rows = [{"step": steps.get(k, k), "calls": int(r.calls), "tokens_in": int(r.tokens_in), "tokens_out": int(r.tokens_out), "usd": round(float(r.usd), 2)}
            for k, r in t.sort_values("usd", ascending=False).iterrows() if k in steps]
    return {"rows": rows, "total_usd": round(float(log.usd.sum()), 2), "total_calls": int(len(log))}


def main():
    ds = {m: decisions(m) for m in MODELS}
    ds = {m: d for m, d in ds.items() if d is not None}
    main_model = "claude-opus-5-5" if "claude-opus-5-5" in ds else next(iter(ds))
    div = pd.read_csv(ROOT / "data" / "closed" / "division.csv", dtype={"division": str, "division_frame": str})
    prod = pd.read_csv(ROOT / "data" / "closed" / "products.csv")
    slugs = [l.strip() for l in open(ROOT / "data" / "closed" / "sample_final_slugs.txt") if l.strip()]
    pages = sum(1 for s in slugs for _ in open(ROOT / "data" / "closed" / "pages" / f"{s}.jsonl"))
    value_rel = pd.read_csv(ROOT / "data" / "closed" / "funnel" / "value_opus.csv")
    data = {
        "built": date.today().isoformat(), "public": PUBLIC,
        "models": {m: MODELS[m] for m in ds}, "main_model": main_model, "framings": FRAMINGS, "theories": THEORIES,
        "theory_texts": {f"{t}_{fr}": (ROOT / "theories" / "variants" / f"{t}_{fr}.txt").read_text().strip()
                         for fr in FRAMINGS for t in THEORIES if t != "none"},
        "funnel": [{"stage": "Archive screens", "firms": 880 + 1275 + 2584, "note": "candidates of the frame enumerated in Common Crawl"},
                   {"stage": "Eligible by richness", "firms": 296, "note": "15 pages a year, product and news pages in both windows"},
                   {"stage": "Pool coded cheaply", "firms": 252, "note": "gpt-5.4-mini, products and evidence"},
                   {"stage": "Final sample", "firms": len(slugs), "note": "three new products and a stated result per window"}],
        "counts": {"firms": len(slugs), "pages": pages, "products": int(prod[prod.slug.isin(slugs)].shape[0]),
                   "innovations": int(prod[prod.slug.isin(slugs) & prod.innovation].shape[0]),
                   "product_years": int(len(ds[main_model][ds[main_model].condition == "none"])),
                   "decisions": {m: int(len(d)) for m, d in ds.items()},
                   "division_agreement": round(float(div.agree.mean()), 2)},
        "record": record_page(ds[main_model]),
        "reference": {m: reference(d) for m, d in ds.items()},
        "weights": {m: weights(m) for m in ds},
        "value": {"dominant": value_rel[EMPH].idxmax(axis=1).value_counts().to_dict(),
                  "mean": {e: round(float(value_rel[e].mean()), 1) for e in EMPH}, "n": int(len(value_rel))},
        "hotstove": {m: hotstove(m) for m in ds}, "swap": swap(ds), "firms": firms(ds), "web": web_measure(slugs),
        "methods": {"protocols": {p: (ROOT / "protocols" / f"{p}.md").read_text() for p in ("year", "context", "firm", "value", "division")},
                    "system": {c: record.system_prompt(c) for c in ("none", "btf_F", "btf_B")},
                    "prompt_example": next(record.user_prompt(j[5], j[4], j[6], j[1], j[2]) for j in record.jobs([slugs[0]]) if j[3] == "none"),
                    "settings": {"effort": CONFIG["record"]["effort"], "max_output_tokens": CONFIG["record"]["max_output_tokens"],
                                 "years": CONFIG["record"]["years"], "gap_years": CONFIG["record"].get("gap_years", []), "windows": CONFIG["simulation"]["windows"],
                                 "context_years": CONFIG["simulation"]["context_years"]},
                    "models": {m: {"calls": int(len(d) // 1), "conditions": sorted(set(d.condition))} for m, d in ds.items()}}}
    strategy = (HERE / "strategy_v3.svg").read_text() if (HERE / "strategy_v3.svg").exists() else ""
    page = (HERE / "template_v3.html").read_text().replace("/*STRATEGY*/", strategy).replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, default=str))
    out = HERE / "public" / "index.html" if PUBLIC else HERE / "index_v3.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    (HERE / ("data_public.json" if PUBLIC else "data_v3.json")).write_text(json.dumps(data, ensure_ascii=False, default=str))
    print(f"{out}, {len(page) / 1e6:.1f} MB, models {list(ds)}, {len(data['firms'])} firms")


if __name__ == "__main__":
    main()
