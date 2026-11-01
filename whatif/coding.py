"""Code the archived websites with the language model, following the protocols in protocols/.

For every firm-year with pages, the year protocol records whether the year is codeable, the products the firm names, and
the statements about them (results, problems, launches, and investments). Statements are kept only if their quote appears
verbatim in the pages. The value protocol scores the theory of value of the firm in the context years, from the same
page selection as the situation of the firm (whatif/context.py). Any coder can be given, and the codings of a second
coder are written to a directory of their own. For every firm
and window, the firm protocol records the stated strategy and the characteristics, read from the latest codeable context
year of the window in config.yaml, which is the profile from which the what-if organization is built. The industry in
the profile is the label of the NACE division of the firm in the sample frame, which is broader than the activity the
website describes and makes the firm harder to identify.

Outputs: data/closed/coding/<slug>.json with the coded years and the firm profile.
"""
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from whatif import llm
from whatif.config import CONFIG, ROOT

C = CONFIG["coding"]
PAGES = ROOT / "data" / "closed" / "pages"
OUT = ROOT / "data" / "closed" / "coding"
ORDER = {"home": 0, "about": 1, "offer": 2, "news": 3, "rnd": 4, "career": 5, "imprint": 6, "other": 7}
PROTOCOL = {p: (ROOT / "protocols" / f"{p}.md").read_text() for p in ("year", "firm", "value")}
CONTEXT_ORDER = {"imprint": 0, "home": 1, "about": 2, "rnd": 3, "career": 4, "news": 5, "offer": 6, "other": 7}


def industry(slug, year=None):
    """Label of the NACE division of the firm: coded from its website up to the given year (whatif/division.py) when
    config coding.division_source is web, otherwise from its four-digit code in the sample frame."""
    if C.get("division_source") == "web":
        from whatif import division
        return division.lookup(slug, year)["label"]
    nace4 = next(r["nace4"] for r in csv.DictReader(open(ROOT / "data" / "closed" / "sample.csv"))
                 if r["domain"].replace(".", "-") == slug)
    labels = dict(l.rstrip("\n").split("\t", 1) for l in open(ROOT / "data" / "cache" / "eurostat" / "nace_r2.tsv"))
    return next(v for k, v in labels.items() if re.fullmatch(r"[A-U]" + nace4[:2], k))


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def years(slug):
    by = {}
    for line in open(PAGES / f"{slug}.jsonl", encoding="utf-8"):
        p = json.loads(line)
        by.setdefault(int(p["year"]), []).append(p)
    return by


def dossier(pages, order=ORDER, limit=None):
    """Pages in the given role order, each and the whole cut to the limits in config.yaml, then the page paths."""
    parts, total = [], 0
    for p in sorted(pages, key=lambda p: order[p["role"]]):
        body = p["text"][:C["page_chars"]]
        if total + len(body) > (limit or C["year_chars"]):
            break
        parts.append(f"## {p['role']} | {p['title']}\n{body}")
        total += len(body)
    paths = sorted({p["url"].split("://", 1)[-1].split("/", 1)[-1] for p in pages})[:C["paths"]]
    return "\n\n".join(parts) + "\n\n## paths\n" + "\n".join("/" + x for x in paths)


def verbatim(items, text):
    return [x for x in items if norm(x.get("quote") or "") and norm(x["quote"]) in norm(text)]


def year_call(slug, year, pages):
    """System prompt, website text, and purpose of the call that codes one firm-year."""
    return PROTOCOL["year"], dossier(pages), f"coding year {slug} {year}"


def code_year(slug, year, pages, model=None, effort=None):
    """Coded year; an answer cut off at the token limit twice, which happens when the model repeats itself, is asked
    once more at low effort."""
    system, text, purpose = year_call(slug, year, pages)
    try:
        a = llm.ask(system, text, purpose, model=model, effort=effort, max_out=16000)
    except RuntimeError:
        a = llm.ask(system, text, purpose + " low", model=model, effort="low", max_out=16000)
    a["success"] = verbatim(a.get("success", []), text)
    a["evidence"] = verbatim(a.get("evidence", []), text)
    return {"year": year, **a}


def value_call(slug, year, pages):
    """System prompt, website text, and purpose of the call that scores the theory of value of one year."""
    return PROTOCOL["value"], dossier(pages, CONTEXT_ORDER, C["context_chars"]), f"value {slug} {year}"


def code_value(slug, year, pages, model=None, effort=None):
    """Emphases of the theory of value: points out of 100 distributed over the four emphases (protocols/value.md since
    2026-10-31; scores of 0 to 3 in the earlier version, protocols/v3/value.md), each above 0 kept only with a verbatim
    quote, and the points rescaled to 100 after that check."""
    system, text, purpose = value_call(slug, year, pages)
    a = llm.ask(system, text, purpose, model=model, effort=effort)
    out = {}
    for k in ("performance", "knowledge", "commitment", "experimentation"):
        x = a.get(k) or {}
        value = x.get("points", x.get("score")) or 0
        ok = value > 0 and norm(x.get("quote") or "") and norm(x["quote"]) in norm(text)
        out[k], out[f"{k}_quote"] = (float(value), x["quote"]) if ok else (0.0, None)
    total = sum(out[k] for k in ("performance", "knowledge", "commitment", "experimentation"))
    if "points" in str(a) and total > 0:
        for k in ("performance", "knowledge", "commitment", "experimentation"):
            out[k] = round(100 * out[k] / total, 1)
    return out


def code_firm(slug, model=None, effort=None, out=OUT, profile=True, only=None):
    """Coded years of a firm, those in only if given, and with profile the firm profile of the earlier design."""
    by = years(slug)
    coded = {}
    with ThreadPoolExecutor(C["workers"]) as ex:
        for r in ex.map(lambda y: code_year(slug, y, by[y], model, effort), sorted(y for y in by if not only or y in only)):
            coded[r["year"]] = r
    profiles = {}
    for window, context in CONFIG["simulation"]["context_years"].items() if profile else ():
        seed = [y for y in context if y in coded and coded[y].get("codeable")]
        if seed:
            p = llm.ask(PROTOCOL["firm"], dossier(by[seed[-1]]), purpose=f"coding firm {slug} {window}")
            p["context"]["industry"] = industry(slug)
            profiles[window] = {**p, "seed_year": seed[-1]}
    result = {"slug": slug, "profiles": profiles, "years": [coded[y] for y in sorted(coded)]}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def plan(slugs):
    """Calls and approximate input tokens of a coding run, without calling the model."""
    calls = chars = 0
    for s in slugs:
        by = years(s)
        calls += len(by) + len(CONFIG["simulation"]["context_years"])
        chars += sum(len(dossier(p)) + len(PROTOCOL["year"]) for p in by.values())
    return calls, chars // 3


if __name__ == "__main__":
    slugs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    if "--plan" in sys.argv:
        calls, tokens = plan(slugs)
        print(f"{calls} calls, about {tokens:,} input tokens")
    else:
        for s in slugs:
            r = code_firm(s)
            seeds = {w: p["seed_year"] for w, p in r["profiles"].items()}
            print(s, len(r["years"]), "years, profiles from", seeds, f"spent {llm.spent():.2f} USD")
