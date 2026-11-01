"""Draw the pool of the third screen: about 200 eligible firms, stratified by regime, in a fixed random order so that
exclusions added later (data/sample/exclude_scope3.txt: global groups, consumer brands, platforms, media, public
bodies) remove firms without reshuffling the rest.

Outputs: data/sample/companies_pool3_200.csv, whatif-organizations/data/closed/sample_pool3.csv
"""
import numpy as np
import pandas as pd

N, MIN, SEED = 200, 8, 20261031
f = pd.read_csv("data/sample/pool_scope3.csv", index_col=0)
f = f.sample(frac=1, random_state=SEED)                          # one fixed order
ex = set(open("data/sample/exclude_scope3.txt").read().split())
# large firms with a foreign or global domain are mostly groups the model would recognise; large .de firms stay
foreign_large = (f["size"] == "large") & ~f.domain.str.endswith(".de")
ok = f[~f.domain.isin(ex) & ~foreign_large]
print(f"{int(foreign_large.sum())} large firms with a non-.de domain left out by rule")
cells = ok.groupby("regime").size()
quota = np.minimum(cells, np.maximum(np.minimum(MIN, cells), np.floor(cells / cells.sum() * N))).astype(int)
while quota.sum() < min(N, len(ok)):
    quota[(cells - quota).idxmax()] += 1
pick = pd.concat([ok[ok.regime == r].head(int(k)) for r, k in quota.items()])
pick.to_csv("data/sample/companies_pool3_200.csv")
pick[["domain"]].to_csv("../whatif-organizations/data/closed/sample_pool3.csv", index=False)
print(f"{len(ex)} excluded, {len(ok)} eligible, {len(pick)} drawn")
print(quota.to_string())
print(", ".join(sorted(pick.domain)))
