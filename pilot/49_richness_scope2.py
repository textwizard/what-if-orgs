"""Richness of the archives of the candidates of the second screen, measured as for the first.

For each candidate the script counts the archived pages per year and, from the roles of whatif/web.py, the page types
that carry signals of the firm (about, career, research, news) in 2017 and 2022, the product pages in 2017 to 2019 and
in 2022 to 2025, and the news pages in 2018 and 2019 and in 2023 to 2025. The captures are read through DuckDB with the
role function registered as a user function, since the file of the third screen (8.7 GB, 28 million rows) does not fit
in memory (2026-10-31; the pandas version is kept in the history).

Inputs: data/<screen>/captures.csv, data/<screen>/enum_summary.csv, for the screen given as argument (enum_scope2 by
default)
Outputs: data/sample/richness_<screen without enum_>.csv, with the columns of data/sample/richness_2017_2022.csv
"""
import os
import sys

import duckdb
import pandas as pd

from whatif import web

SCREEN = sys.argv[1] if len(sys.argv) > 1 else "enum_scope2"
DIR = f"data/{SCREEN}/"
YEARS = [2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025]
SIGNAL = ["about", "career", "rnd", "news"]

db = DIR + ".richness.duckdb"
for f in (db, db + ".wal"):
    if os.path.exists(f):
        os.remove(f)
con = duckdb.connect(db)
con.execute("SET memory_limit = '4GB'")
con.execute("SET threads = 4")
con.execute("SET preserve_insertion_order = false")
con.execute(f"SET temp_directory = '{DIR}.duckdb_tmp'")
con.create_function("role", web.role, ["VARCHAR"], "VARCHAR")
csv = f"read_csv('{DIR}captures.csv', header=true, all_varchar=true, delim=',', quote='\"', escape='\"')"
con.execute(f"""CREATE TABLE caps AS
    SELECT slug, year, u, role(url) AS role FROM (
        SELECT slug, CAST(year AS INTEGER) AS year, any_value(url) AS url,
               rtrim(regexp_replace(regexp_replace(url, '^[a-z]+://', ''), '^www\\.', ''), '/') AS u
        FROM {csv}
        WHERE mime = 'html' AND CAST(year AS INTEGER) IN ({", ".join(map(str, YEARS))})
        GROUP BY slug, year, u)""")
print(con.execute("SELECT count(*), count(DISTINCT slug) FROM caps").fetchone(), "html page-years, firms", flush=True)

p = pd.read_csv(DIR + "enum_summary.csv").pivot_table(index="slug", columns="year", values="unique_urls", fill_value=0)


def types(yrs):
    """Number of signal page types present in the years."""
    q = con.execute(f"""SELECT slug, count(DISTINCT role) FROM caps
                        WHERE year IN ({", ".join(map(str, yrs))}) AND role IN ({", ".join(repr(r) for r in SIGNAL)})
                        GROUP BY slug""").fetchall()
    return pd.Series(dict(q)).reindex(p.index).fillna(0).astype(int)


def count(role, yrs):
    """Distinct pages of a role in the years."""
    q = con.execute(f"""SELECT slug, count(DISTINCT u) FROM caps
                        WHERE role = '{role}' AND year IN ({", ".join(map(str, yrs))}) GROUP BY slug""").fetchall()
    return pd.Series(dict(q)).reindex(p.index).fillna(0).astype(int)


out = pd.DataFrame({"types_2016_17": types([2016, 2017]), "types_2017": types([2017]), "types_2022": types([2022]),
                    "pages_2016_17": p.reindex(columns=[2016, 2017], fill_value=0).max(axis=1),
                    "pages_2017": p.get(2017, 0), "pages_2022": p.get(2022, 0),
                    "offer_pages_pre": count("offer", [2017, 2018, 2019]),
                    "offer_pages_post": count("offer", [2022, 2023, 2024, 2025]),
                    "news_pre": count("news", [2018, 2019]), "news_post": count("news", [2023, 2024, 2025])})
out.to_csv(f"data/sample/richness_{SCREEN.removeprefix('enum_')}.csv")
con.close()
for f in (db, db + ".wal"):
    if os.path.exists(f):
        os.remove(f)
print(f"{len(out)} candidates")
print(out.describe().loc[["25%", "50%", "75%"]].round(0).to_string())
