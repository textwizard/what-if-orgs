"""Characteristics of firms and their industries that the theories refer to, measured before each window.

The industry is described in the year before the window by the change of its turnover over two years (Eurostat
short-term statistics), which for the hold-out compares with the level before the pandemic, and by how common in-house
R&D, patent applications, and research cooperation with universities are among its innovative firms (Community
Innovation Survey 2014 for the exploration window and 2020 for the hold-out), each placed in the lower, middle, or upper
third of the industries in the survey. The firm is described by signals coded from its archived website in the context
years of config.yaml (protocols/context.md), two years because the early archive is thin. Each year is coded from up to
config.coding.context_chars characters, with the imprint, home, about, research, career, and news pages before the
offer, a signal counts if it is found in either year, and it is kept only with specific evidence (strength 2 or 3 of
protocols/context.md) and a quote that appears verbatim in the pages. The coder is config.coding.context_model. Signals
the website does not mention are left out of the description, since silence on a website is no evidence that the firm
lacks them. The firm is further described by its participation in EU research projects that started before the window
(CORDIS), by the number of products it offered in the context years, and by the relatedness of each new product to that
offer (protocols/relatedness.md).

Outputs: data/closed/context/<slug>.json
"""
import csv
import json
import re
import sys
from collections import defaultdict
from functools import cache

import numpy as np
import pandas as pd

from whatif import eurostat, llm
from whatif.coding import C, OUT as CODING, dossier, norm, years
from whatif.config import CONFIG, PATHS, ROOT
from whatif.products import OUT as PRODUCTS, innovations

OUT = ROOT / "data" / "closed" / "context"
PROTOCOL = {p: (ROOT / "protocols" / f"{p}.md").read_text() for p in ("context", "relatedness")}
SIGNALS = ("hiring", "expansion", "investment", "growth", "difficulty", "rd", "cooperation")
ORDER = {"imprint": 0, "home": 1, "about": 2, "rnd": 3, "career": 4, "news": 5, "offer": 6, "other": 7}
LETTERS = "ABCDEFGHIJ"
CIS = {9: {"rd": ("inn_cis9_exp", dict(sizeclas="TOTAL", type_inn="INNOACT", indic_in="RRDIN_YES", unit="PC")),
           "patents": ("inn_cis9_ipr", dict(sizeclas="TOTAL", type_inn="INNO", indic_in="PROPAT", unit="PC")),
           "universities": ("inn_cis9_coop", dict(sizeclas="TOTAL", type_inn="INNOACT", indic_in="C06", unit="PC"))},
       12: {"rd": ("inn_cis12_exp", dict(indic_inn="ENT", expend="RND_IH", sizeclas="TOTAL", unit="PC")),
            "patents": ("inn_cis12_ipr", dict(sizeclas="TOTAL", enterpr="INN", ipr_trn="PAT", unit="PC")),
            "universities": ("inn_cis12_coop", dict(locpart="TOTAL", coop_ptn="UNIV", sizeclas="TOTAL", unit="PC"))}}
LEGAL = re.compile(r"\b(gmbh|mbh|co|kg|ag|se|ohg|ug|ek|e\.k|haftungsbeschränkt|holding|und|&)\b")


def covering(code, codes):
    """The narrowest NACE code of a table that contains a division, such as C25 or C13-C15 for C14."""
    if code in codes:
        return code
    ranges = [c for c in codes if re.fullmatch(r"[A-U]\d\d-[A-U]\d\d", c)
              and c[0] == code[0] and int(c[1:3]) <= int(code[1:]) <= int(c[5:7])]
    return min(ranges, key=lambda c: int(c[5:7]) - int(c[1:3]), default=None)


@cache
def thirds(wave, key):
    """Boundaries of the lower and upper third of a survey share across the divisions in the survey."""
    table, pos = CIS[wave][key]
    vals = [eurostat.value(table, nace_r2=c, **pos) for c in eurostat.codes(table) if re.fullmatch(r"[A-U]\d\d", c)]
    return np.nanquantile([v for v in vals if v is not None], [1 / 3, 2 / 3])


def industry(code, year):
    """Turnover change and survey shares of the industry of a firm, as bands."""
    out = {}
    for table in ("sts_intv_a", "sts_trtu_a", "sts_setu_a"):
        pos = dict(indic_bt="NETTUR", s_adj="NSA", unit="I21", nace_r2=code)
        if code in eurostat.codes(table):
            a, b = (eurostat.value(table, time=str(y), **pos) for y in (year - 2, year))
            if a and b:
                out["turnover"] = "fell" if b / a < .97 else "grew" if b / a > 1.03 else "stayed about level"
    wave = 9 if year < 2018 else 12
    for key, (table, pos) in CIS[wave].items():
        c = covering(code, eurostat.codes(table))
        v = eurostat.value(table, nace_r2=c, **pos) if c else None
        if v is not None:
            lo, hi = thirds(wave, key)
            out[key] = "less" if v < lo else "more" if v > hi else "about as"
    return out


@cache
def cordis():
    """EU research projects by the website domain and by the simplified name of each participating organisation."""
    start = {}
    by = defaultdict(set)
    for prog in ("fp7", "h2020", "HORIZON"):
        for r in csv.DictReader(open(f"{PATHS['cordis']}/{prog}/project.csv", encoding="utf-8"), delimiter=";"):
            start[r["id"]] = int(r["startDate"][:4]) if r["startDate"][:4].isdigit() else None
        for r in csv.DictReader(open(f"{PATHS['cordis']}/{prog}/organization.csv", encoding="utf-8"), delimiter=";"):
            if r["country"] != "DE":
                continue
            domain = (r["organizationURL"] or "").lower().split("://")[-1].split("/")[0].removeprefix("www.")
            for key in (domain, simple(r["name"])):
                if key:
                    by[key].add((r["projectID"], start.get(r["projectID"])))
    return by


def simple(name):
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]", " ", LEGAL.sub(" ", (name or "").lower()))).strip()


def projects(domain, legal_name, year):
    found = cordis().get(domain, set()) | (cordis().get(simple(legal_name), set()) if legal_name else set())
    return sum(1 for _, y in found if y and y <= year)


def call(slug, year, pages):
    """System prompt, website text, and purpose of the call that codes the signals of one year."""
    return PROTOCOL["context"], dossier(pages, ORDER, C["context_chars"]), f"context {slug} {year}"


def signals(slug, year, pages, model=None, effort=None):
    """Signals coded from the website of one year, each kept only with a verbatim quote and at least specific
    evidence (strength 2)."""
    system, text, purpose = call(slug, year, pages)
    a = llm.ask(system, text, purpose, model=model or C["context_model"], effort=effort or C["context_effort"])
    employees = a.get("employees")
    out = {"legal_name": a.get("legal_name"),
           "employees": employees.get("value") if isinstance(employees, dict) else employees}
    for s in SIGNALS:
        x = a.get(s) or {}
        out[s] = (bool(x.get("value")) and (x.get("strength") or 0) >= 2 and bool(x.get("quote"))
                  and norm(x["quote"]) in norm(text))
        out[f"{s}_quote"] = x.get("quote") if out[s] else None
    return out


def relatedness(slug, offer, new):
    listing = "Products offered in the year before\n" + "\n".join(f"- {p}" for p in offer) + "\n\nNew products\n" + \
              "\n".join(f"Product {LETTERS[k]}: {p}" for k, p in enumerate(new))
    a = llm.ask(PROTOCOL["relatedness"], listing, purpose=f"relatedness {slug}")
    found = {x.get("label"): x.get("relatedness") for x in a.get("products", [])}
    return [found.get(LETTERS[k]) for k in range(len(new))]


def firm(slug):
    sample = pd.read_csv(ROOT / "data" / "closed" / "sample.csv")
    row = sample[sample.domain.str.replace(".", "-", regex=False) == slug].iloc[0]
    code = next(k for k in (l.split("\t")[0] for l in open(eurostat.DIR / "nace_r2.tsv"))
                if re.fullmatch(r"[A-U]" + str(row.nace4).zfill(4)[:2], k))
    coded = {y["year"]: y for y in json.loads((CODING / f"{slug}.json").read_text())["years"]}
    by, prod = years(slug), pd.read_csv(PRODUCTS)
    out = {"slug": slug, "windows": {}}
    for window, context in CONFIG["simulation"]["context_years"].items():
        found = [signals(slug, y, by[y]) for y in context if y in by]
        if not found:
            continue
        s = {k: next((f[k] for f in reversed(found) if f[k]), None) for k in found[0]}
        s.update({k: any(f[k] for f in found) for k in SIGNALS})
        year = CONFIG["simulation"]["windows"][window][0] - 1
        if CONFIG["coding"].get("division_source") == "web":          # division read from the website of the window
            from whatif import division
            code = division.lookup(slug, context[-1])["key"]
        offer = sorted({p["generic"] for y in context for p in coded.get(y, {}).get("products", [])
                        if p.get("generic")})
        new = innovations(prod, slug, window)
        out["windows"][window] = {"years": context, "industry": industry(code, year), **s,
                                  "eu_projects": projects(row.domain, s["legal_name"], year),
                                  "products_offered": len(offer), "new": new,
                                  "relatedness": relatedness(slug, offer, new) if offer and new else None}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{slug}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


BAND = {"less": "less common", "about as": "about equally common", "more": "more common"}
RELATED = {"close": "close to the firm's established offer", "related": "building on the firm's established offer",
           "distant": "distant from the firm's established offer"}


def render(c):
    """The situation of the industry and the firm in the words given to the what-if organization."""
    ind = c["industry"]
    lines = []
    if "turnover" in ind:
        lines.append(f"Over the past two years, turnover in this industry {ind['turnover']}.")
    shares = [f"{label} {BAND[ind[k]]}" for k, label in (("rd", "in-house research and development is"),
                                                         ("patents", "patent applications are"),
                                                         ("universities", "cooperation with universities is"))
              if k in ind]
    if shares:
        text = ", ".join(shares[:-1]) + (", and " if len(shares) > 1 else "") + shares[-1]
        lines.append(f"Compared with most other industries in Germany, among innovative firms in this industry {text}.")
    growth = [w for k, w in (("hiring", "hiring"), ("expansion", "expanding"), ("investment", "investing"),
                             ("growth", "growing"))
              if c.get(k)]
    if c.get("difficulty"):
        lines.append("The firm reports difficulties such as falling sales, job cuts, or closures.")
    if growth:
        lines.append("The firm reports that it is " + " and ".join([", ".join(growth[:-1]), growth[-1]] if
                                                                   len(growth) > 1 else growth) + ".")
    research = (("rd", "research and development of its own"),
                ("cooperation", "research cooperation with universities or research institutes"))
    known = [w for k, w in research if c.get(k)]
    if known:
        lines.append("The firm reports " + " and ".join(known) + ".")
    if c.get("eu_projects"):
        lines.append("It has taken part in EU-funded research projects.")
    if not growth and not c.get("difficulty") and not known:
        lines.append("The website gives no information on growth, difficulties, or research.")
    n = c.get("products_offered")
    if n:
        lines.append(f"It offered a {'narrow' if n <= 3 else 'broad' if n > 12 else 'medium'} range of products.")
    return " ".join(lines)


if __name__ == "__main__":
    slugs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    print(f"{len(slugs)} firms, about {6 * len(slugs)} model calls")
    if "--batch" in sys.argv:                                  # signals of all firms in one batch at half the price
        items = [call(s, y, pages) for s in slugs for y, pages in years(s).items()
                 if y in {y for c in CONFIG["simulation"]["context_years"].values() for y in c}]
        print("batch: stored %d, failed %d" % llm.batch(items, C["context_model"], C["context_effort"]))
    if "--plan" not in sys.argv:
        for s in slugs:
            firm(s)
            print(s, f"spent {llm.spent():.2f} USD")
