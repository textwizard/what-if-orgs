"""Replay the what-if organizations under each condition and window.

A what-if organization receives the stated strategy and coarsened characteristics of its firm, the situation of its
industry and of the firm before the window (context.py), the innovations the firm introduced in the window under generic
descriptions with their relatedness to the earlier offer, and, in the theory conditions, a theory text in the system
prompt. In each of four periods it learns the results of the products it holds, and of dropped products if forgone
outcomes are shown, and decides for each product. Its decisions and reasons are carried into the next period.

Outputs: data/closed/runs/<condition>/<slug>_<window>_<run>.json, with results, decisions, and reasons per period.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from whatif import context, llm, outcomes, twin
from whatif.coding import OUT as CODING
from whatif.config import CONFIG, ROOT
from whatif.products import OUT as PRODUCTS, innovations

S = CONFIG["simulation"]
RUNS = ROOT / "data" / "closed" / "runs"
THEORIES = {c: (ROOT / "theories" / f"{c}.txt").read_text().strip() for c in ("btf", "ac", "placebo")}
LABELS = "ABCDEFGHIJ"

ROLE = ("You act as the management of a German firm and decide about the new products it has launched. You will be "
        "given a description of the firm, its new products, and their results in each period. Based on this "
        "description{} your general knowledge of firms in the German economy, you decide as the management of "
        "this type of firm would.")
RULES = ["Base your decisions on the description of the firm, the results of its products{}, and your general "
         "knowledge of similar firms in the German economy.",
         "Do not search the web or use any external information.",
         "Respond with a JSON object in the format specified and nothing else."]


def system_prompt(condition):
    """Role, theory text, and rules, in the structure of the prompts of Dahlke et al. (2026)."""
    if condition not in THEORIES:
        rules = ["Reason over the description of the firm, its situation, and the results of its products together.",
                 RULES[0].format("")] + RULES[1:]
        return ROLE.format(" and") + "\n\nImportant rules:\n" + "\n".join(f"- {r}" for r in rules)
    rules = ["Use the theoretical perspective above to guide your reasoning about the products of this firm.",
             "Reason over the theory and the empirical context provided together, and consider what the theory "
             "implies for a firm in the situation described and for products with the results reported.",
             RULES[0].format(", the theory")] + RULES[1:]
    theory = f"\n\nTheoretical Perspective\n{THEORIES[condition]}"
    return (ROLE.format(", the theoretical perspective below, and") + theory + "\n\nImportant rules:\n"
            + "\n".join(f"- {r}" for r in rules))


def age(founded, seed_year):
    if not isinstance(founded, int) or not seed_year:
        return None
    years = seed_year - founded
    return "under 5 years" if years < 5 else "5 to 20 years" if years <= 20 else "more than 20 years"


def describe(profile):
    s, c, seed_year = profile["strategy"], profile["context"], profile["seed_year"]
    facts = [f"Industry: {c['industry']}" if c.get("industry") else None,
             f"Size: {c['size']} firm" if c.get("size") else None,
             "Location: Germany",
             f"Age: {age(c.get('founded'), seed_year)}" if age(c.get("founded"), seed_year) else None]
    lines = [". ".join(f for f in facts if f) + "."]
    for key, label in (("offer", "What the firm offers"), ("customers", "Its customers"),
                       ("why_we_win", "Why it expects to succeed"), ("resources", "Its key resources"),
                       ("direction", "Where it is heading")):
        if s.get(key):
            lines.append(f"{label}: {s[key]}")
    if profile.get("situation"):
        lines += ["", "Situation of the industry and the firm", profile["situation"]]
    return "\n".join(lines)


def user_prompt(profile, products, period, results, memory, forgone, held):
    lines = ["About the firm", describe(profile), "", "New products"]
    lines += [f"Product {LABELS[k]}: {p}" for k, p in enumerate(products)]
    lines += ["", f"This is period {period} of 4. Results this period, in percent of plan, where the plans were set in "
                  "relation to what is expected for such products in the industry:"]
    for k, pct in enumerate(results):
        if held[k] or forgone:
            status = "" if held[k] else " (dropped, result if it were still on the market)"
            lines.append(f"Product {LABELS[k]}: {pct} percent{status}")
    if memory:
        lines += ["", "Earlier periods"] + memory
    options = "keep or drop it" + (", and for a dropped product, whether to reintroduce it or let it stay dropped"
                                   if forgone else "")
    lines += ["", f"For each product on the market, decide whether to {options}. A dropped product leaves the market, "
                  "and the resources it releases go to the firm's established business or new development projects.",
              "", "Response Format",
              '{"decisions": [{"product": "A", "decision": "keep", "reason": "one sentence", "confidence": 0-100}]}']
    return "\n".join(lines)


def replay(slug, window, run, condition, forgone, products, profile):
    d = outcomes.draws(slug, window, run, len(products))
    held, memory, periods = [True] * len(products), [], []
    for t in range(4):
        results = [row[t] for row in d["percent"]]
        if condition == "twin":
            decisions = {LABELS[k]: {"decision": twin.run(d["percent"], forgone)[k][t]} for k in range(len(products))}
        else:
            a = llm.ask(system_prompt(condition), user_prompt(profile, products, t + 1, results, memory, forgone, held),
                        purpose=f"simulate {condition} {slug} {window} {run}")
            decisions = {x.get("product", "").replace("Product ", "").strip(): x for x in a.get("decisions", [])}
        chosen = {}
        for k in range(len(products)):
            visible = held[k] or forgone
            x = decisions.get(LABELS[k]) if visible else None
            x = x or ({"decision": "keep" if held[k] else "stay dropped", "missing": True} if visible
                      else {"decision": "stay dropped"})
            chosen[LABELS[k]] = x
            dec = x.get("decision", "").lower()
            held[k] = dec in ("keep", "reintroduce") if (held[k] or forgone) else False
            if held[k] or forgone or dec == "drop":
                memory.append(f"Period {t + 1}, Product {LABELS[k]}: {results[k]} percent, {dec}"
                              + (f", {x['reason']}" if x.get("reason") else ""))
        periods.append({"results": results, "decisions": chosen, "held": list(held)})
    out = {"slug": slug, "window": window, "run": run, "condition": condition, "forgone": forgone,
           "products": products, "high": d["high"], "early": d["early"], "periods": periods}
    path = RUNS / f"{condition}{'_forgone' if forgone else ''}" / f"{slug}_{window}_{run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def jobs(slugs):
    """One job per firm, window, run, and condition, for firms with innovations in the window."""
    prod = pd.read_csv(PRODUCTS)
    out = []
    for slug in slugs:
        profiles = json.loads((CODING / f"{slug}.json").read_text())["profiles"]
        f = context.OUT / f"{slug}.json"
        ctx = json.loads(f.read_text())["windows"] if f.exists() else {}
        for window in S["windows"]:
            products = innovations(prod, slug, window)
            if not products or window not in profiles:
                continue
            profile = dict(profiles[window])
            if window in ctx and ctx[window]["new"] == products:
                c = ctx[window]
                profile["situation"] = context.render(c)
                products = [p + (f", {context.RELATED[r]}" if r in context.RELATED else "")
                            for p, r in zip(products, c["relatedness"] or [None] * len(products))]
            for condition, n_runs in S["runs"].items():
                for forgone in ([False, True] if condition in S["forgone"] else [False]):
                    out += [(slug, window, r, condition, forgone, products, profile) for r in range(n_runs)]
    return out


if __name__ == "__main__":
    slugs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    js = jobs(slugs)
    calls = 4 * sum(j[3] != "twin" for j in js)
    print(f"{len(js)} simulated runs, {calls} model calls")
    if "--plan" not in sys.argv:
        with ThreadPoolExecutor(S["workers"]) as ex:
            list(ex.map(lambda j: replay(*j), js))
        print(f"done, spent {llm.spent():.2f} USD")
