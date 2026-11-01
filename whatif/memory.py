"""The hot stove run: the second decision year of each window with the agent's own first-year decisions as history.

record.py reads every decision year against the record, so that product-years are comparable observations. Here the
agent carries its own history instead. For each firm, window, and condition of the main framing, the first-year
decisions of the record run are turned into histories in two ways:
  draw1..drawK        the decision on each product is drawn from the probability the same agent gave in the first year
                      (config record.memory.draws draws, seeded by firm, window, condition, and draw), so that each draw
                      is one history the agent could have made; products it dropped are gone, products it kept stay,
                      including those the firm itself withdrew
  record-as-memory    the firm's actual first-year decisions presented as the agent's own, the control that separates
                      the framing of a memory from the content of a counterfactual history
In the second year the agent sees the record of every product up to the year before its first decision, its own
first-year decisions with their probabilities and reasons, the years in between named as not observed, and the products
launched since with their real record. The record of the decided year itself is withheld for the decided products,
since it would reveal the firm's decision. The recorded outcome of the second year exists only for products the firm
still had; for the others the row carries the state of the record (dropped before), and the comparison with the record
is made at the firm level, as the share of the cohort that survives.

Run from the repository: python -m whatif.memory slugs.csv [--plan] [model ...] [--branches draw1,draw2]
Outputs: data/closed/record/<model>/memory/<branch>/<condition>/<slug>_<window>.json and
         data/closed/record/<model>/memory.csv, one row per branch, condition, firm, window, and product of the second year.
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from whatif import llm, record
from whatif.config import CONFIG
from whatif.simulate import LABELS, describe

R = CONFIG["record"]
M = R.get("memory", {"draws": 10, "conditions": None})
S = CONFIG["simulation"]
OUT = record.OUT
CONDITIONS = M.get("conditions") or ["none"] + [f"{t}_F" for t in R["theories"]]
BRANCHES = ["record-as-memory"] + [f"draw{k}" for k in range(1, M.get("draws", 10) + 1)]


def first_year(model):
    """First-year decisions of the record run per (slug, window, condition): label -> (p_drop, decision, reason, recorded)."""
    d = pd.read_csv(OUT / model / "decisions.csv")
    d = d[d.condition.isin(CONDITIONS)]
    out = {}
    for (slug, window, condition), g in d.groupby(["slug", "window", "condition"]):
        y1 = R["years"][window][0]
        g1 = g[g.year == y1]
        if len(g1):
            out[(slug, window, condition)] = (y1, {r.product: (r.p_drop, r.decision, r.reason, bool(r.recorded_drop)) for r in g1.itertuples()})
    return out


def seed(*parts):
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def own_history(branch, first, slug, window, condition):
    """The agent's first-year decisions in one branch: label -> dropped (bool)."""
    y1, dec = first
    if branch == "record-as-memory":
        return {lab: rec for lab, (p, d, r, rec) in dec.items()}
    rng = np.random.default_rng(seed(slug, window, condition, branch))
    return {lab: bool(rng.random() < (p if p == p else 0.0)) for lab, (p, d, r, rec) in dec.items()}


def jobs(slugs, model, branches=BRANCHES):
    """Second-year jobs with the agent's own history: (branch, slug, window, year2, condition, products, profile, coded, memory)."""
    first = first_year(model)
    base = [j for j in record.jobs(slugs) if j[3] in CONDITIONS and j[2] == R["years"][j[1]][-1]]
    out = []
    for slug, window, year2, condition, products, profile, coded in base:
        if (slug, window, condition) not in first:
            continue                                    # no first decision year for this firm and window
        y1, dec = first[(slug, window, condition)]
        for branch in branches:
            dropped = own_history(branch, first[(slug, window, condition)], slug, window, condition)
            memory, prods = {}, []
            for k, (row, desc) in enumerate(products):
                row = dict(row)
                lab = LABELS[k]
                if lab in dec:                          # decided by the agent in the first year
                    p, d, reason, rec = dec[lab]
                    own = dropped[lab]
                    memory[lab] = {"year": y1, "dropped": own, "p": p, "reason": reason}
                    row["record_state"] = row["state"]  # live, dropped, or absent in the record of the second year
                    row["state"] = "own-dropped" if own else "live"
                    row["live"] = not own
                    row["recorded_drop"] = row["record_state"] == "live" and row["recorded_drop"]
                    row["own_until"] = y1               # the record is shown up to the year before the first decision
                else:
                    row["record_state"] = row["state"]
                    row["own_until"] = None
                prods.append((row, desc))
            if any(r["live"] for r, _ in prods):
                out.append((branch, slug, window, year2, condition, prods, profile, coded, memory))
    return out


def user_prompt(profile, products, coded, window, year, memory):
    a, b = S["windows"][window]
    y1 = min(m["year"] for m in memory.values()) if memory else None
    lines = ["About the firm", describe(profile), "", f"New products launched in {a} to {b}"]
    for k, (row, desc) in enumerate(products):
        lab = LABELS[k]
        if row["state"] == "own-dropped":
            state = f", taken off the market by you in {memory[lab]['year']}"
        else:
            state = {"dropped": f", no longer on the website since {row['drop']}",
                     "absent": f", not on the website in {row['prev']}"}.get(row["state"], "")
        traits = ", ".join(t for t, on in (("a product line with several models", row.get("line")),
                                           ("presented by the firm as a flagship or focus of its direction", row.get("featured"))) if on)
        lines.append(f"Product {lab}: {desc}, launched in {row['entry']}{', ' + traits if traits else ''}{state}")
    lines += ["", f"What the website showed about the products on the market, by year, up to {year - 1}, and your own "
                  f"decisions of {y1}. Quotes are given as written on the website."]
    for k, (row, desc) in enumerate(products):
        if not row["live"]:
            continue
        lab = LABELS[k]
        if row.get("own_until"):
            m = memory[lab]
            h = record.history(row, coded, row["own_until"]) or "  no archived year"
            h += (f"\n  {m['year']}: your decision was to keep the product; you gave a probability of "
                  f"{round(m['p'] * 100) if m['p'] == m['p'] else 'n/a'} that it would be taken off the market"
                  + (f", reasoning: \"{m['reason']}\"" if m.get("reason") else ""))
            between = [y for y in range(m["year"] + 1, year)]
            if between:
                h += f"\n  {between[0]}" + (f" to {between[-1]}" if len(between) > 1 else "") + ": not observed"
            lines += [f"Product {lab}", h]
        else:
            lines += [f"Product {lab}", record.history(row, coded, year) or "  no archived year"]
    if any(r["state"] == "own-dropped" for r, _ in products):
        gone = [LABELS[k] for k, (r, _) in enumerate(products) if r["state"] == "own-dropped"]
        lines += ["", f"In {y1} you took {', '.join('Product ' + g for g in gone)} off the market. At the time you had put the "
                      "probability of a withdrawal within a year at "
                      + ", ".join(f"{round(memory[g]['p'] * 100) if memory[g]['p'] == memory[g]['p'] else 'n/a'} for Product {g}" for g in gone)
                      + "; the resources went to the established business and to new development."]
    live = [LABELS[k] for k, (row, _) in enumerate(products) if row["live"]]
    lines += ["", f"It is now {year}. For each product on the market ({', '.join(live)}), give the probability, from 0 "
                  "to 100, that the firm takes it off the market within the coming year, and state your decision, keep "
                  "or drop. A dropped product leaves the market, and the resources it releases go to the firm's "
                  "established business or new development projects.",
              "", "Response Format",
              '{"decisions": [{"product": "A", "probability_dropped": 0-100, "decision": "keep", '
              '"reason": "one sentence"}]}']
    return "\n".join(lines)


def purpose(job):
    branch, slug, window, year, condition = job[:5]
    return f"memory {branch} {condition} {slug} {window} {year}"


def items(js):
    return [(record.system_prompt(j[4]), user_prompt(j[6], j[5], j[7], j[2], j[3], j[8]), purpose(j)) for j in js]


def decide(job, model):
    branch, slug, window, year, condition, products, profile, coded, memory = job
    a = llm.ask(record.system_prompt(condition), user_prompt(profile, products, coded, window, year, memory), purpose(job),
                model=model, effort=R["effort"], max_out=R["max_output_tokens"])
    found = {str(x.get("product", "")).replace("Product ", "").strip(): x for x in a.get("decisions", [])}
    decided = [LABELS[k] for k, (r, _) in enumerate(products) if LABELS[k] in memory]
    n_dropped = sum(memory[l]["dropped"] for l in decided)
    rows = []
    for k, (row, desc) in enumerate(products):
        if not row["live"]:
            continue
        lab = LABELS[k]
        x = found.get(lab) or {}
        try:
            p_drop = min(100, max(0, float(x["probability_dropped"]))) / 100
        except (KeyError, TypeError, ValueError):
            p_drop = None
        theory = condition.split("_")[0] if "_" in condition else "none"
        m = memory.get(lab)
        rows.append({"model": model, "branch": branch, "draw": int(branch[4:]) if branch.startswith("draw") else 0,
                     "condition": condition, "theory": theory, "slug": slug, "window": window, "year": year,
                     "product": lab, "generic": row["generic"], "entry": row["entry"],
                     "decided_year1": m is not None, "own_p1": m["p"] if m else None,
                     "n_decided_year1": len(decided), "n_dropped_year1": n_dropped,
                     "share_dropped_year1": n_dropped / len(decided) if decided else None,
                     "record_state": row["record_state"], "recorded_drop": bool(row["recorded_drop"]) if row["record_state"] == "live" else None,
                     "decision": str(x.get("decision", "")).lower().strip() or None, "p_drop": p_drop, "reason": x.get("reason")})
    path = OUT / model / "memory" / branch / condition / f"{slug}_{window}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"branch": branch, "slug": slug, "window": window, "year": year, "condition": condition, "model": model,
                                "memory": memory, "decisions": rows}, ensure_ascii=False, indent=1, default=str))
    return rows


def main():
    slugs = [d.replace(".", "-") for d in pd.read_csv(sys.argv[1]).domain]
    models = [m for m in sys.argv[2:] if m in R["models"]] or R["models"]
    branches = BRANCHES
    if "--branches" in sys.argv:
        branches = sys.argv[sys.argv.index("--branches") + 1].split(",")
    for model in models:
        js = jobs(slugs, model, branches)
        its = items(js)
        tokens = sum(len(a + b) for a, b, _ in its) // 3
        n_fw = len({(j[1], j[2]) for j in js})
        print(f"{model}: {len(its)} calls, {n_fw} firm-windows, {len(branches)} branches, {len(CONDITIONS)} conditions; about "
              f"{tokens:,} input tokens, {llm.cost(model, tokens, 0, 2500 * len(its)) / 2:.2f} USD with 2,500 output tokens per call", flush=True)
        if "--plan" in sys.argv:
            continue
        print("  batch: stored %d, failed %d" % llm.batch(its, model, R["effort"], R["max_output_tokens"]), flush=True)
        rows = []
        for j in js:
            try:
                rows += decide(j, model)
            except RuntimeError as e:
                print("  failed", purpose(j), str(e)[:80])
        df = pd.DataFrame(rows)
        f = OUT / model / "memory.csv"
        if f.exists():                                  # branches submitted in parts: keep the rows of other branches
            old = pd.read_csv(f)
            df = pd.concat([old[~old.branch.isin(df.branch.unique())], df], ignore_index=True)
        df.to_csv(f, index=False)
        print(f"  {len(df)} rows in memory.csv; mean probability by branch type: "
              + ", ".join(f"{b} {g.p_drop.mean():.3f}" for b, g in df.groupby(df.branch.str.replace(r'\d+', '', regex=True))), flush=True)
    print(f"spent {llm.spent():.2f} USD")


if __name__ == "__main__":
    main()
