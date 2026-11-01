"""Decisions of the what-if organizations about the real new products of their firms, year by year, against the record.

For each firm and window (config simulation.windows), the agent receives the firm as described: its stated
strategy and characteristics from the context year, and the situation of its industry and of the firm (context.py). It
receives the innovations of the window with their relatedness to the earlier offer. For each decision year of the window
(config record.years) it sees, for every product that the record shows on the market in that year, what the website
showed in the years before: the launch year, the prominence of the product in each archived year (on the home page, a
page of its own, mentioned, not mentioned), and the statements the firm made about it (results, problems, investments,
launches, each with its quote), but never the year being decided, since that is the outcome. It decides keep or drop
with a reason and a confidence. The recorded decision is whether the product disappears in that year (products.py).

Decision years are read independently of each other given the record, so that every product-year is a comparable
observation. The agent's own earlier decisions enter only in the own-history run (memory.py), not here.
Conditions are no theory and each theory in config record.theories in each framing in config record.framings, F stating
the considerations and B stating the decision. A product that was absent from the website in the archived year before
the decision is not a decision case of that year, since it is in the set only because it returned (a lapse); years
without archived pages are shown as such, so that missing data is not read as silence. The agent gives the probability
that the product is off the market within a year, and its decision. All calls go through the batch interface first,
then the decisions are read from the cache.

Run from the repository: python -m whatif.record slugs.csv [--plan] [model ...]   (a CSV with a column domain; models default to config)
Outputs: data/closed/record/<model>/<condition>/<slug>_<window>_<year>.json and data/closed/record/<model>/decisions.csv,
one row per condition, firm, window, year, and product.
"""
import json
import sys

import pandas as pd

from whatif import context, llm
from whatif.config import CONFIG, ROOT

LABELS = "ABCDEFGHIJ"

ROLE = ("You act as the management of a German firm and decide about the new products it has launched. You will be "
        "given a description of the firm, its new products, and their results in each period. Based on this "
        "description{} your general knowledge of firms in the German economy, you decide as the management of "
        "this type of firm would.")
RULES = ["Base your decisions on the description of the firm, the results of its products{}, and your general "
         "knowledge of similar firms in the German economy.",
         "Do not search the web or use any external information.",
         "Respond with a JSON object in the format specified and nothing else."]


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

from whatif.products import OUT as PRODUCTS

R = CONFIG["record"]
S = CONFIG["simulation"]
OUT = ROOT / "data" / "closed" / "record"
CODING = ROOT / "data" / "closed" / R["coding_dir"]
THEORIES = {f"{t}_{fr}": (ROOT / "theories" / "variants" / f"{t}_{fr}.txt").read_text().strip()
            for fr in R["framings"] for t in R["theories"]}
CONDITIONS = ["none"] + list(THEORIES)
if "none" in R.get("anchors", ["web"]):                       # the same agents without the web-measured anchors
    CONDITIONS += [f"{c}-na" for c in ["none"] + [f"{t}_{fr}" for fr in R["anchors_none_framings"] for t in R["theories"]]]
if "own" in R.get("anchors", ["web"]):                        # each theory without the anchors of its own constructs
    CONDITIONS += [f"{t}_{fr}-oa" for fr in R["anchors_none_framings"] for t in R["theories"]]
# anchors of each theory's constructs (protocols/anchors.md): removed in the -oa condition of that theory
OWN = {"btf": {"results", "situation"}, "ac": {"knowledge", "relatedness"},
       "esc": {"investment", "launch", "featured"}, "ro": {"line", "relatedness", "range"}}


def anchored(condition):
    """Theory condition and whether the firm's web anchors are shown, from a condition name such as btf_F-na."""
    if condition.endswith("-na"):
        return condition[:-3], False
    return (condition[:-3] if condition.endswith("-oa") else condition), True


def removed(condition):
    """The anchors removed in an own-anchors condition, or an empty set."""
    return OWN[condition[:-3].split("_")[0]] if condition.endswith("-oa") else set()


def strip(profile, descs):
    """The firm reduced to industry, size, and age, and the products to their generic descriptions (the product
    traits, line and featured, are dropped with the other anchors in inputs())."""
    bare = {"strategy": {}, "context": {k: v for k, v in profile["context"].items() if k in ("industry", "size", "founded")},
            "seed_year": profile["seed_year"]}
    return bare, [d.split(", ")[0] for d in descs]
PROMINENCE = {3: "shown on the home page", 2: "presented on a page of its own", 1: "mentioned on the website",
              0: "not mentioned on the website"}
TYPES = {"result": "result", "problem": "problem", "launch": "launch", "investment": "investment"}


def system_prompt(condition):
    """Role, theory text, and rules of the system prompt."""
    condition, _ = anchored(condition)
    if condition == "none":
        rules = ["Reason over the description of the firm, its situation, and what its website showed about its "
                 "products together.", RULES[0].format("")] + RULES[1:]
        return ROLE.format(" and") + "\n\nImportant rules:\n" + "\n".join(f"- {r}" for r in rules)
    rules = ["Use the theoretical perspective above to guide your reasoning about the products of this firm.",
             "Reason over the theory and the empirical context provided together, and consider what the theory "
             "implies for a firm in the situation described and for products with the evidence reported.",
             RULES[0].format(", the theory")] + RULES[1:]
    theory = f"\n\nTheoretical Perspective\n{THEORIES[condition]}"
    return (ROLE.format(", the theoretical perspective below, and") + theory + "\n\nImportant rules:\n"
            + "\n".join(f"- {r}" for r in rules))


def innovations(prod, slug, window):
    """Rows of the first innovations of a firm in a window, in the order of products.innovations()."""
    a, b = S["windows"][window]
    p = prod[(prod.slug == slug) & prod.innovation & prod.entry.between(a, b)]
    return p.sort_values("entry", kind="stable").head(S["max_products"]).to_dict("records")


def evidence(coded, names, until, drop=frozenset()):
    """Statements about a product in the coded years before the decision year, by year; statement types in drop
    are left out (results and problems for "results", investments, launches)."""
    skip = {"result", "problem"} if "results" in drop else set()
    skip |= {t for t in ("investment", "launch") if t in drop}
    out = {}
    for y in coded["years"]:
        if y["year"] >= until:
            continue
        for e in y.get("evidence", []):
            if (e.get("product") or "").strip() in names and e.get("type") in TYPES and e.get("type") not in skip:
                out.setdefault(y["year"], []).append(e)
    return out


def history(row, coded, until, drop=frozenset()):
    """What the website showed about one product in the years before the decision year; years without archived
    pages, and years with too few to tell, are named as such."""
    levels = {int(k): v for k, v in json.loads(row["levels"]).items()}
    informative = json.loads(row["informative"])
    names = set(row["names"].split("|"))
    ev = evidence(coded, names, until, drop)
    lines = []
    for y in range(int(row["entry"]), until):
        if y not in levels:
            lines.append(f"  {y}: no archived pages of the website")
            continue
        if y not in informative:
            lines.append(f"  {y}: too few archived pages to tell")
            continue
        line = f"  {y}: {PROMINENCE[levels[y]]}"
        for e in ev.get(y, []):
            line += f"\n    {TYPES[e['type']]} ({'quantified' if e.get('strength') == 3 else 'specific' if e.get('strength') == 2 else 'vague'}): \"{e.get('quote', '').strip()}\""
        lines.append(line)
    return "\n".join(lines)


def user_prompt(profile, products, coded, window, year, drop=frozenset()):
    a, b = S["windows"][window]
    lines = ["About the firm", describe(profile), "", f"New products launched in {a} to {b}"]
    for k, (row, desc) in enumerate(products):
        state = {"dropped": f", no longer on the website since {row['drop']}",
                 "absent": f", not on the website in {row['prev']}"}.get(row["state"], "")
        traits = ", ".join(t for t, on in (("a product line with several models", row.get("line")),
                                           ("presented by the firm as a flagship or focus of its direction", row.get("featured"))) if on)
        lines.append(f"Product {LABELS[k]}: {desc}, launched in {row['entry']}{', ' + traits if traits else ''}{state}")
    lines += ["", f"What the website showed about the products on the market, by year, up to {year - 1}. Quotes are "
                  "given as written on the website."]
    for k, (row, desc) in enumerate(products):
        if row["live"]:
            lines += [f"Product {LABELS[k]}", history(row, coded, year, drop) or "  no archived year"]
    live = [LABELS[k] for k, (row, _) in enumerate(products) if row["live"]]
    lines += ["", f"It is now {year}. For each product on the market ({', '.join(live)}), give the probability, from 0 "
                  "to 100, that the firm takes it off the market within the coming year, and state your decision, keep "
                  "or drop. A dropped product leaves the market, and the resources it releases go to the firm's "
                  "established business or new development projects.",
              "", "Response Format",
              '{"decisions": [{"product": "A", "probability_dropped": 0-100, "decision": "keep", '
              '"reason": "one sentence"}]}']
    return "\n".join(lines)


def traits(coded, names):
    """Whether a product is ever presented as a line and whether it is ever featured, from the coded product listings."""
    line = featured = False
    for y in coded["years"]:
        for p in y.get("products", []):
            if p["name"].strip() in names:
                line |= p.get("scale") == "line"
                featured |= bool(p.get("featured"))
    return line, featured


def window_of(w):
    """The product window of an analysis set: the gap set belongs to the exploration window."""
    return "exploration" if w == "gap" else w


def label(d):
    """Name the sets of an analysis: the exploration decisions of the gap years become the set 'gap', so that the
    exploration window holds the pre-pandemic decisions only and the holdout window the post-pandemic ones."""
    gap = set(R.get("gap_years", []))
    if gap:
        d = d.copy()
        d.loc[(d.window == "exploration") & d.year.isin(gap), "window"] = "gap"
    return d


def jobs(slugs):
    """One job per firm, window, decision year, and condition, with the products on the market in the record."""
    prod = pd.read_csv(PRODUCTS)
    out = []
    for slug in slugs:
        coded = json.loads((CODING / f"{slug}.json").read_text())
        profiles = coded.get("profiles", {})
        f = context.OUT / f"{slug}.json"
        ctx = json.loads(f.read_text())["windows"] if f.exists() else {}
        for window in S["windows"]:
            rows = innovations(prod, slug, window)
            if not rows or window not in profiles:
                continue
            profile = dict(profiles[window])
            descs = [r["generic"] for r in rows]
            if window in ctx and [n.split(" (")[0] for n in ctx[window]["new"]] == descs:
                c = ctx[window]
                profile["situation"] = context.render(c)
                rel = c["relatedness"] or [None] * len(rows)
                descs = [d + (f", {context.RELATED[r]}" if r in context.RELATED else "") for d, r in zip(descs, rel)]
            informative = sorted(json.loads(rows[0]["informative"]))
            for year in R["years"][window]:
                if year not in informative:
                    continue                                # the archive of this year is too thin to show a record
                prev = max(y for y in informative if y < year)
                products = []
                for row, desc in zip(rows, descs):
                    row = dict(row)
                    drop = None if pd.isna(row["drop"]) else int(row["drop"])
                    if row["entry"] >= year:
                        continue                            # launched in or after the decision year
                    present = json.loads(row["levels"]).get(str(prev), 0) > 0
                    row["prev"] = prev
                    row["state"] = ("dropped" if drop is not None and drop < year else
                                    "live" if present else "absent")  # absent: a lapse, decided only by its return
                    row["live"] = row["state"] == "live"
                    row["drop"] = drop
                    if R.get("anchors_product", True):
                        row["line"], row["featured"] = traits(coded, set(row["names"].split("|")))
                    row["recorded_drop"] = drop == year
                    products.append((row, desc))
                if any(r["live"] for r, _ in products):
                    for condition in CONDITIONS:
                        out.append((slug, window, year, condition, products, profile, coded))
    return out


def inputs(job):
    """Profile and products of a job, stripped of the web anchors when the condition asks for it, or of the anchors of
    the theory's own constructs in an own-anchors condition."""
    slug, window, year, condition, products, profile, coded = job
    if not anchored(condition)[1]:
        bare, descs = strip(profile, [d for _, d in products])
        return bare, [({**row, "line": False, "featured": False}, d) for (row, _), d in zip(products, descs)]
    drop = removed(condition)
    if not drop:
        return profile, products
    profile = dict(profile)
    if "situation" in drop or "knowledge" in drop or "range" in drop:
        text = profile.get("situation") or ""
        sentences = [x for x in text.split(". ") if x]
        keep = []
        for x in sentences:
            low = x.lower()
            if "situation" in drop and ("turnover" in low or "hiring" in low or "expanding" in low or "investing" in low or "growing" in low or "difficult" in low or "no information on growth" in low):
                continue
            if "knowledge" in drop and ("research and development" in low or "cooperation" in low or "eu-funded" in low or "patent" in low):
                continue
            if "range" in drop and "range of products" in low:
                continue
            keep.append(x)
        profile["situation"] = ". ".join(keep).strip()
    out = []
    for row, desc in products:
        row = dict(row)
        if "relatedness" in drop:
            desc = desc.split(", ")[0]
        if "featured" in drop:
            row["featured"] = False
        if "line" in drop:
            row["line"] = False
        out.append((row, desc))
    return profile, out


def items(js, model):
    out = []
    for job in js:
        slug, window, year, condition, products, profile, coded = job
        profile, products = inputs(job)
        out.append((system_prompt(condition), user_prompt(profile, products, coded, window, year, removed(condition)),
                    f"record {condition} {slug} {window} {year}"))
    return out


def decide(job, model):
    slug, window, year, condition, products, profile, coded = job
    profile, products = inputs(job)
    a = llm.ask(system_prompt(condition), user_prompt(profile, products, coded, window, year, removed(condition)),
                f"record {condition} {slug} {window} {year}", model=model, effort=R["effort"],
                max_out=R["max_output_tokens"])
    found = {str(x.get("product", "")).replace("Product ", "").strip(): x for x in a.get("decisions", [])}
    rows = []
    for k, (row, desc) in enumerate(products):
        if not row["live"]:
            continue
        x = found.get(LABELS[k]) or {}
        dec = str(x.get("decision", "")).lower().strip()
        try:
            p_drop = min(100, max(0, float(x["probability_dropped"]))) / 100
        except (KeyError, TypeError, ValueError):
            p_drop = None
        base, web = anchored(condition)
        theory, framing = base.split("_") if "_" in base else ("none", None)
        rows.append({"model": model, "condition": condition, "theory": theory, "framing": framing,
                     "anchors": "none" if not web else ("own removed" if condition.endswith("-oa") else "web"), "slug": slug,
                     "window": window, "year": year, "product": LABELS[k], "generic": row["generic"],
                     "entry": row["entry"], "decision": dec or None, "p_drop": p_drop,
                     "recorded_drop": bool(row["recorded_drop"]), "reason": x.get("reason")})
    path = OUT / model / condition / f"{slug}_{window}_{year}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"slug": slug, "window": window, "year": year, "condition": condition, "model": model,
                                "products": [{"label": LABELS[k], **{kk: v for kk, v in r.items()
                                                                     if kk not in ("levels", "informative")}}
                                             for k, (r, _) in enumerate(products)],
                                "decisions": rows}, ensure_ascii=False, indent=1, default=str))
    return rows


def main():
    slugs = [d.replace(".", "-") for d in pd.read_csv(sys.argv[1]).domain]
    js = jobs(slugs)
    by_year = pd.Series([(j[1], j[2]) for j in js if j[3] == "none"]).value_counts().sort_index()
    print(f"{len(js)} calls per model, {len(js) // len(CONDITIONS)} firm-window-years, {len(CONDITIONS)} conditions:")
    print(by_year.to_string())
    models = [m for m in sys.argv[2:] if m in R["models"]] or R["models"]   # optional: model names on the command line
    for model in models:
        its = items(js, model)
        tokens = sum(len(a + b) for a, b, _ in its) // 3
        print(f"  {model}: {len(its)} calls, about {tokens:,} input tokens, {llm.cost(model, tokens, 0, 0) / 2:.2f} "
              f"USD input in batch, {llm.cost(model, tokens, 0, 2500 * len(its)) / 2:.2f} USD with 2,500 output "
              f"tokens per call")
        if "--plan" in sys.argv:
            continue
        print("  batch: stored %d, failed %d" % llm.batch(its, model, R["effort"], R["max_output_tokens"]), flush=True)
        rows = []
        for j in js:
            try:
                rows += decide(j, model)
            except RuntimeError as e:
                print("  failed", j[3], j[0], j[1], j[2], str(e)[:80])
        df = pd.DataFrame(rows)
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / model / "decisions.csv", index=False)
        print(f"  {len(df)} decisions, drop rate recorded {df.recorded_drop.mean():.3f}, mean probability "
              + ", ".join(f"{c} {g.p_drop.mean():.3f}" for c, g in df.groupby('condition')), flush=True)
    print(f"spent {llm.spent():.2f} USD")


if __name__ == "__main__":
    main()
