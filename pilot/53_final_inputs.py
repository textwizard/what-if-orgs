"""Inputs of the reference organizations for the final sample: firm profiles, context, and yearly theory of value.

The final sample (data/sample/companies_final.csv, code/51_select_final.py) is coded once more for what the agent reads
and for the coherence check, all through the batch interfaces of whatif/llm.py and then from the cache:

  division   NACE division of the firm from its website in the context years (whatif/division.py), used for the
             industry label, the industry indicators, and the regime when config coding.division_source is web
  profiles   stated strategy and characteristics per window (protocols/firm.md) from the latest codeable context year,
             with gpt-5.4, written into the coder's codings as "profiles" (data/closed/<coding_dir>/<slug>.json)
  context    signals of the situation of the firm (protocols/context.md, Opus, from the double coding of
             code/50_funnel_coding.py), industry indicators, EU projects, and relatedness of the new products
             (whatif/context.py) -> data/closed/context/<slug>.json
  value      theory of value (protocols/value.md) in every archived year with gpt-5.4, for the coherence check with the
             year before each decision -> data/closed/funnel/value_gpt54_years.csv. The context years 2017 and 2022 are
             also coded by Opus in the double coding (value_opus.csv).

Run from whatif-organizations with PYTHONPATH set to it:
  python3 ../09_whatif_pilot/code/53_final_inputs.py <division|profiles|context|value|all> [slugs file, default
  data/closed/sample_final.csv]
The coder whose products and codings the agent reads is config record.coding_dir (gpt-5.4-mini in the funnel).
"""
import json
import sys

import pandas as pd

from whatif import coding, context, llm
from whatif.config import CONFIG, ROOT

YEARS = (2017, 2018, 2019, 2022, 2023, 2024, 2025, 2026)
CODING = ROOT / "data" / "closed" / CONFIG["record"]["coding_dir"]
PROFILE_MODEL, PROFILE_EFFORT = "gpt-5.4", "medium"
VALUE_MODEL, VALUE_EFFORT = "gpt-5.4", "medium"


def slugs_of(path):
    return [d.replace(".", "-") for d in pd.read_csv(path).domain]


def profiles(slugs):
    """Firm profile per window from the latest codeable context year, batched, then written into the codings."""
    seeds, items = {}, []
    for s in slugs:
        by = coding.years(s)
        coded = {y["year"]: y for y in json.loads((CODING / f"{s}.json").read_text())["years"]}
        for window, years in CONFIG["simulation"]["context_years"].items():
            seed = [y for y in years if y in coded and coded[y].get("codeable")]
            if seed:
                seeds[(s, window)] = seed[-1]
                items.append((coding.PROTOCOL["firm"], coding.dossier(by[seed[-1]]), f"coding firm {s} {window}"))
    print(f"{len(items)} profile calls")
    print("  batch: stored %d, failed %d" % llm.batch(items, PROFILE_MODEL, PROFILE_EFFORT), flush=True)
    n = 0
    for s in slugs:
        f = CODING / f"{s}.json"
        d = json.loads(f.read_text())
        by = coding.years(s)
        d["profiles"] = {}
        for window in CONFIG["simulation"]["context_years"]:
            if (s, window) not in seeds:
                continue
            p = llm.ask(coding.PROTOCOL["firm"], coding.dossier(by[seeds[(s, window)]]), f"coding firm {s} {window}",
                        model=PROFILE_MODEL, effort=PROFILE_EFFORT)
            p.setdefault("context", {})["industry"] = coding.industry(s, seeds[(s, window)])
            d["profiles"][window] = {**p, "seed_year": seeds[(s, window)]}
            n += 1
        f.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    print(f"  {n} profiles written to {CODING}, spent {llm.spent():.2f} USD")


def contexts(slugs):
    """Context of each firm per window, reading the products of the coder in config record.coding_dir."""
    context.CODING = CODING
    for s in slugs:
        try:
            c = context.firm(s)
            print(s, {w: v["relatedness"] for w, v in c["windows"].items()}, flush=True)
        except Exception as e:
            print("  failed", s, type(e).__name__, str(e)[:100], flush=True)
    print(f"  spent {llm.spent():.2f} USD")


def value(slugs):
    """Theory of value in every archived year, batched, then collected into one table."""
    items = [coding.value_call(s, y, p) for s in slugs for y, p in sorted(coding.years(s).items()) if y in YEARS]
    print(f"{len(items)} value calls")
    print("  batch: stored %d, failed %d" % llm.batch(items, VALUE_MODEL, VALUE_EFFORT), flush=True)
    rows = []
    for s in slugs:
        for y, pages in sorted(coding.years(s).items()):
            if y in YEARS:
                try:
                    rows.append({"slug": s, "year": y,
                                 **coding.code_value(s, y, pages, model=VALUE_MODEL, effort=VALUE_EFFORT)})
                except RuntimeError as e:
                    print("  failed", s, y, str(e)[:80])
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "closed" / "funnel" / "value_gpt54_years.csv", index=False)
    print(f"  {len(df)} firm-years, spent {llm.spent():.2f} USD")
    print(df.groupby("year")[["performance", "knowledge", "commitment", "experimentation"]].mean().round(2).to_string())


if __name__ == "__main__":
    step = sys.argv[1]
    slugs = slugs_of(sys.argv[2] if len(sys.argv) > 2 else ROOT / "data" / "closed" / "sample_final.csv")
    from whatif import division as _division
    for name, f in (("division", _division.main), ("profiles", profiles), ("context", contexts), ("value", value)):
        if step in (name, "all"):
            print(name)
            f(slugs)
