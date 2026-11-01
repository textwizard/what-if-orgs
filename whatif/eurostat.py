"""Download the Eurostat tables for Germany that the design uses, and read values from them.

Annual turnover from short-term business statistics classifies industries by their exposure to the pandemic and
describes the recent performance of an industry. The Community Innovation Surveys of 2014 and 2020 describe how
common in-house R&D, patent applications, and cooperation with universities are in an industry. The labels of the
NACE Rev. 2 classification describe the industry of a firm. Tables are stored in data/cache/eurostat.
"""
import json
import urllib.request

from whatif.config import ROOT

API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{}?geo=DE&format=JSON&lang=EN"
TABLES = {"sts_intv_a": "&sinceTimePeriod=2013", "sts_trtu_a": "&sinceTimePeriod=2013",   # industry, trade,
          "sts_setu_a": "&sinceTimePeriod=2013",                                           # and services
          **{f"inn_cis{w}_{t}": "" for w in (9, 12) for t in ("exp", "ipr", "coop")}}
NACE = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/codelist/ESTAT/NACE_R2?format=TSV&lang=en"
DIR = ROOT / "data" / "cache" / "eurostat"


def value(table, **pos):
    """One value of a table, with every dimension of more than one category fixed in pos."""
    j = json.load(open(DIR / f"{table}.json"))
    flat = 0
    for d, n in zip(j["id"], j["size"]):
        index = j["dimension"][d]["category"]["index"]
        flat = flat * n + (index[pos[d]] if d in pos else 0)
    return j["value"].get(str(flat))


def codes(table):
    """NACE codes reported in a table."""
    return list(json.load(open(DIR / f"{table}.json"))["dimension"]["nace_r2"]["category"]["index"])


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    for t, since in TABLES.items():
        urllib.request.urlretrieve(API.format(t) + since, DIR / f"{t}.json")
        print("downloaded", t)
    urllib.request.urlretrieve(NACE, DIR / "nace_r2.tsv")


if __name__ == "__main__":
    main()
