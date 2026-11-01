"""NACE division of each firm, inferred from its archived website instead of taken from the sample frame.

The sample frame carries the division of each firm from a proprietary company database, which cannot be released. For
the firms of the study, the division is therefore read from the website of each context year with protocols/division.md:
the model chooses the two-digit NACE Rev. 2 division of the activity from which the firm earns most of its revenue and
gives a quote. The division is used downstream for the industry label in the firm description, the industry indicators
of the context, and the technological regime (config regimes), when config coding.division_source is web. The agreement
with the division of the frame is reported as a validation and kept in the output.

Run from the repository: python -m whatif.division firms.csv   (a CSV with a column domain)
Output: data/closed/division.csv, one row per firm and context year.
"""
import csv
import re
import sys

import pandas as pd

from whatif import llm
from whatif.coding import C, CONTEXT_ORDER, dossier, norm, years
from whatif.config import CONFIG, ROOT

PROTOCOL = (ROOT / "protocols" / "division.md").read_text()
OUT = ROOT / "data" / "closed" / "division.csv"
_table = None


def divisions():
    """Two-digit division -> (Eurostat key such as C25, label)."""
    out = {}
    for line in open(ROOT / "data" / "cache" / "eurostat" / "nace_r2.tsv"):
        k, v = line.rstrip("\n").split("\t", 1)
        if len(k) == 3 and k[0].isalpha() and k[1:].isdigit():
            out[k[1:]] = (k, v)
    return out


def call(slug, year, pages):
    return PROTOCOL, dossier(pages, CONTEXT_ORDER, C["context_chars"]), f"division {slug} {year}"


def code(slug, year, pages, model=None, effort=None):
    system, text, purpose = call(slug, year, pages)
    a = llm.ask(system, text, purpose, model=model or C["division_model"], effort=effort or C["division_effort"])
    d = re.sub(r"\D", "", str(a.get("division", "")))[:2].zfill(2)
    known = divisions()
    quote = a.get("quote") or ""
    return {"slug": slug, "year": year, "division": d if d in known else None,
            "label": known[d][1] if d in known else None, "confidence": a.get("confidence"),
            "quote": quote, "quote_verbatim": bool(norm(quote)) and norm(quote) in norm(text)}


def lookup(slug, year=None):
    """Division of a firm as coded from its website: the latest context year up to year, or the latest available.
    Returns a dict with division, key, and label, or raises if the firm has not been coded."""
    global _table
    if _table is None:
        _table = pd.read_csv(OUT, dtype={"division": str})
    rows = _table[(_table.slug == slug) & _table.division.notna()].sort_values("year")
    if year is not None and (rows.year <= year).any():
        rows = rows[rows.year <= year]
    if not len(rows):
        raise KeyError(f"no division coded for {slug}")
    d = rows.iloc[-1].division.zfill(2)
    key, label = divisions()[d]
    return {"division": d, "key": key, "label": label, "year": int(rows.iloc[-1].year)}


def main(slugs):
    context_years = sorted({y for c in CONFIG["simulation"]["context_years"].values() for y in c})
    items = [call(s, y, p) for s in slugs for y, p in sorted(years(s).items()) if y in context_years]
    print(f"{len(items)} division calls")
    print("  batch: stored %d, failed %d" % llm.batch(items, C["division_model"], C["division_effort"]), flush=True)
    rows = []
    for s in slugs:
        for y, pages in sorted(years(s).items()):
            if y in context_years:
                try:
                    rows.append(code(s, y, pages))
                except RuntimeError as e:
                    print("  failed", s, y, str(e)[:80])
    df = pd.DataFrame(rows)
    frame = {r["domain"].replace(".", "-"): str(r["nace4"]).zfill(4)[:2]
             for r in csv.DictReader(open(ROOT / "data" / "closed" / "sample.csv"))}
    df["division_frame"] = df.slug.map(frame)
    df["agree"] = df.division == df.division_frame
    df.to_csv(OUT, index=False)
    firm_level = df.groupby("slug").agg(web=("division", lambda x: x.mode().iloc[0] if len(x.mode()) else None),
                                        frame=("division_frame", "first"))
    print(f"  {len(df)} firm-years, agreement with the frame {df.agree.mean():.0%} by firm-year, "
          f"{(firm_level.web == firm_level.frame).mean():.0%} by firm; divisions differ between the two context years "
          f"in {sum(g.division.nunique() > 1 for _, g in df.groupby('slug'))} firms; spent {llm.spent():.2f} USD")


if __name__ == "__main__":
    main([d.replace(".", "-") for d in pd.read_csv(sys.argv[1]).domain])
