"""Trajectories of the products of each firm on its archived website, and the innovations among them.

Names that refer to the same product are grouped by the language model (protocols/aliases.md). For each product and
year, prominence is 3 if a name appears on the home page, 2 if it appears in the title or path of a page of its own or
in the path of any page archived that year, 1 if it appears elsewhere in the pages or in the coded product list, and 0
otherwise. A year is informative if the archive holds at least half the firm's median number of pages and at least
config.products.min_pages pages, and the coder found it codeable. A product is an innovation if it first appears after
the firm's first observed year and the year before its entry is informative, so that its absence before entry is
documented. A product is dropped in year t if it was named in the previous informative year and is absent in t and in
every later informative year, and t is not the last informative year. Models of a collection that the firm replaces
regularly, such as a seasonal range, are marked as seasonal by the coder and do not count as innovations. A stated
success counts if the firm reports it in the year of entry or the year after, as a result of strength 2 or 3 in the
coding of protocols/year.md (or as a success in codings of its first version).

Outputs: data/closed/products.csv, one row per product with entry, innovation status, success, drop year, and the
prominence by year.
"""
import json
import re
import statistics
import sys
from collections import defaultdict

import pandas as pd

from whatif import llm
from whatif.coding import OUT as CODING, years as page_years
from whatif.config import CONFIG, PATHS, ROOT

P = CONFIG["products"]
OUT = ROOT / "data" / "closed" / "products.csv"
ALIASES = (ROOT / "protocols" / "aliases.md").read_text()


def names_of(coded):
    names = {}
    for y in coded["years"]:
        for p in y.get("products", []):
            n = p["name"].strip()
            if len(n) >= P["min_name_chars"]:
                v = names.setdefault(n, {"years": [], "generic": p.get("generic", ""), "seasonal": False})
                v["years"].append(y["year"])
                v["seasonal"] |= bool(p.get("seasonal"))
    return names


def alias_call(coded):
    """System prompt, listing of names, and purpose of the call that groups the names of a firm, or None."""
    names = names_of(coded)
    if not names:
        return None
    listing = "\n".join(f"{n} | {sorted(set(v['years']))} | {v['generic']}" for n, v in sorted(names.items()))
    return ALIASES, listing, f"aliases {coded['slug']}"


def groups(coded):
    """Product groups of a firm, from the names coded in all years."""
    names, c = names_of(coded), alias_call(coded)
    if not c:
        return []
    answer = llm.ask(*c, effort=P["alias_effort"], max_out=P["alias_max_tokens"])
    seen, out = set(), []
    for g in answer.get("groups", []):
        members = [n for n in g["names"] if n in names and n not in seen]
        seen.update(members)
        if members:
            out.append({"names": members, "generic": g.get("generic", "")})
    out += [{"names": [n], "generic": names[n]["generic"]} for n in names if n not in seen]
    for g in out:
        g["seasonal"] = any(names[n]["seasonal"] for n in g["names"])
    return out


def alnum(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


_paths = {}


def load(slugs):
    """Read the paths of many firms in one pass over the capture files, which are large."""
    want = set(slugs) - set(_paths)
    for s in want:
        _paths[s] = defaultdict(set)
    for f in PATHS["captures"]:
        for chunk in pd.read_csv(f, usecols=["slug", "year", "url"], dtype=str, chunksize=2_000_000):
            for slug, year, url in chunk[chunk.slug.isin(want)].itertuples(index=False):
                _paths[slug][int(year)].add(alnum(url.split("://", 1)[-1].split("/", 1)[-1]))


def inventory(slug):
    """Paths of all pages of a firm archived in each year, letters and digits only, for matching product names."""
    if slug not in _paths:
        load([slug])
    return _paths[slug]


def level(names, pages, listed, paths=()):
    """Prominence of a product in one year."""
    pattern = re.compile("|".join(r"(?<!\w)" + re.escape(n.lower()) + r"(?!\w)" for n in names))
    if any(p["role"] == "home" and pattern.search(p["text"].lower()) for p in pages):
        return 3
    keys = [alnum(n) for n in names if len(alnum(n)) >= P["min_name_chars"] + 1]
    if any(pattern.search((p["title"] + " " + p["url"].replace("-", " ")).lower()) for p in pages) or \
            any(k in path for k in keys for path in paths):
        return 2
    if listed or any(pattern.search(p["text"].lower()) for p in pages):
        return 1
    return 0


def firm(slug, coding=CODING):
    coded = json.loads((coding / f"{slug}.json").read_text())
    by = page_years(slug)
    codeable = {y["year"] for y in coded["years"] if y.get("codeable")}
    median = statistics.median(len(v) for v in by.values())
    informative = sorted(y for y, v in by.items() if y in codeable and len(v) >= max(P["min_pages"], median / 2))
    listed = {y["year"]: {p["name"].strip() for p in y.get("products", [])} for y in coded["years"]}
    success = [(s["product"], y["year"]) for y in coded["years"] for s in y.get("success", [])
               + [e for e in y.get("evidence", []) if e.get("type") == "result" and (e.get("strength") or 0) >= 2]]
    paths = inventory(slug)
    rows = []
    for i, g in enumerate(groups(coded)):
        lv = {y: level(g["names"], by[y], bool(listed.get(y, set()) & set(g["names"])), paths.get(y, ()))
              for y in sorted(by)}
        present = [y for y in informative if lv[y] > 0]
        if not present:
            continue
        entry = min(y for y in lv if lv[y] > 0)
        first_obs = informative[0]
        innovation = entry > first_obs and (entry - 1) in informative and not g["seasonal"]
        later = [y for y in informative if y > entry]
        drop = next((t for k, t in enumerate(later[:-1])
                     if lv[t] == 0 and all(lv[u] == 0 for u in later[k:])), None)
        succ = any(name in g["names"] and entry <= y <= entry + 1 for name, y in success)
        rows.append({"slug": slug, "product": i, "generic": g["generic"], "names": "|".join(g["names"]),
                     "entry": entry, "innovation": innovation, "seasonal": g["seasonal"], "success": succ, "drop": drop,
                     "levels": json.dumps(lv), "informative": json.dumps(informative)})
    return rows


def innovations(prod, slug, window):
    """Generic names of the first innovations of a firm in a window, numbered where two share a description."""
    a, b = CONFIG["simulation"]["windows"][window]
    p = prod[(prod.slug == slug) & prod.innovation & prod.entry.between(a, b)]
    names = p.sort_values("entry").generic.head(CONFIG["simulation"]["max_products"]).tolist()
    seen = defaultdict(int)
    out = []
    for n in names:
        seen[n] += 1
        out.append(f"{n} ({seen[n]})" if names.count(n) > 1 else n)
    return out


if __name__ == "__main__":
    slugs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    rows = [r for s in slugs for r in firm(s)]
    pd.DataFrame(rows).to_csv(OUT, index=False)
    df = pd.DataFrame(rows)
    print(f"{len(df)} products, {int(df.innovation.sum())} innovations in {df[df.innovation].slug.nunique()} firms, "
          f"{int(df[df.innovation]['drop'].notna().sum())} innovations dropped, spent {llm.spent():.2f} USD")
