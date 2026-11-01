"""Eligible firms of the third archive screen and the pointers of their pages.

The rule of code/48_select_150.py is applied to the screen of data/enum_scope3 (richness from code/49_richness_scope2.py
enum_scope3), with 2026 added to the years that must hold at least 15 pages, since it carries the fates of the products
of 2024. Firms already in the pool are left out, and so are well-known brands and the sites of foreign groups, found by
the list below after reading the names of the eligible firms. The captures of the eligible firms in the years used are
written to a file of their own, which keeps the reading of pointers in whatif/web.py small.

Outputs: data/sample/pool_scope3.csv, data/enum_scope3/captures_pool.csv,
whatif-organizations/data/closed/sample_scope3.csv
"""
import pandas as pd

YEARS = [2017, 2018, 2019, 2022, 2023, 2024, 2025, 2026]
EXCLUDE = set(open("data/sample/exclude_scope3.txt").read().split()) if __import__("os").path.exists(
    "data/sample/exclude_scope3.txt") else set()
s = pd.read_csv("data/enum_scope3/enum_summary.csv")
p = s.pivot_table(index="slug", columns="year", values="unique_urls", fill_value=0).reindex(columns=YEARS, fill_value=0)
r = pd.read_csv("data/sample/richness_scope3.csv").set_index("slug").reindex(p.index)
ok = ((p[[2017, 2022]].min(axis=1) >= 15) & (p[[2018, 2019, 2023, 2024, 2025, 2026]].min(axis=1) >= 15)
      & (r[["offer_pages_pre", "offer_pages_post"]].min(axis=1) >= 3) & (r[["news_pre", "news_post"]].min(axis=1) >= 3))
f = pd.read_csv("data/sample/companies_scope3.csv").set_index("slug").loc[ok[ok].index]
f = f[~f.index.isin(pd.read_csv("data/sample/companies_pool252.csv").slug) & ~f.domain.isin(EXCLUDE)]
f.join(r).to_csv("data/sample/pool_scope3.csv")
f[["domain"]].to_csv("../whatif-organizations/data/closed/sample_scope3.csv", index=False)
keep = set(f.index)
for i, c in enumerate(pd.read_csv("data/enum_scope3/captures.csv", chunksize=2_000_000, dtype=str)):
    c = c[c.slug.isin(keep) & c.year.isin([str(y) for y in YEARS])]
    c.to_csv("data/enum_scope3/captures_pool.csv", mode="w" if i == 0 else "a", header=i == 0, index=False)
print(f"{int(ok.sum())} eligible, {len(f)} after exclusions")
print(f.regime.value_counts().to_string())
