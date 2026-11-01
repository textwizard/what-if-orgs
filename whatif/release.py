"""Export of the public release: everything a reader needs to rebuild the pages and rerun the pipeline, without the
page text and without the variables of the sample frame.

Written to data/public/<name>/ for the firms of a list:
  firms.csv                 domain and slug of each firm (CC BY 4.0)
  pointers/<slug>.csv       Common Crawl pointer of every page read: year, url, warc_filename, warc_offset, warc_length
  codings/<slug>.json       products, statements with quotes, and profiles, as coded (coding_dir of config record)
  context/<slug>.json       situation of the firm per window, industry indicators, relatedness
  division.csv              NACE division from the website per firm and context year (without the frame's division)
  value_<coder>.csv         theory of value per firm and context year
  products.csv              product trajectories with prominence by year and the year of the drop
  decisions_<model>.csv     every decision of the simulation on the record with probability, decision, and reason
  record/<model>/           weights, validation, the gap set, coherence, the anchor contrast, the own-history run (memory.csv) and its hot stove tables
  answers/<hash>.json       model answers of every call of these firms, keyed by the prompt hash, with the prompt text
                            removed; a reader who rebuilds the pages from the pointers gets the same hashes and can
                            run the pipeline from these answers
Run from the repository: python -m whatif.release firms.csv [name]
"""
import csv
import json
import shutil
import sys

import pandas as pd

from whatif import llm
from whatif.config import CONFIG, ROOT

CLOSED = ROOT / "data" / "closed"


def main(path, name="final"):
    out = ROOT / "data" / "public" / name
    out.mkdir(parents=True, exist_ok=True)
    firms = pd.read_csv(path)
    firms["slug"] = firms.domain.str.replace(".", "-", regex=False)
    slugs = set(firms.slug)
    firms[["domain", "slug"]].to_csv(out / "firms.csv", index=False)
    for sub in ("pointers", "codings", "context", "answers"):
        (out / sub).mkdir(exist_ok=True)
    for s in slugs:
        shutil.copy(CLOSED / "pointers" / f"{s}.csv", out / "pointers" / f"{s}.csv")
        shutil.copy(CLOSED / CONFIG["record"]["coding_dir"] / f"{s}.json", out / "codings" / f"{s}.json")
        if (CLOSED / "context" / f"{s}.json").exists():
            c = json.loads((CLOSED / "context" / f"{s}.json").read_text())
            for w in c["windows"].values():
                w.pop("legal_name", None)
            (out / "context" / f"{s}.json").write_text(json.dumps(c, ensure_ascii=False, indent=1))
    div = pd.read_csv(CLOSED / "division.csv", dtype=str)
    div[div.slug.isin(slugs)].drop(columns=["division_frame", "agree"], errors="ignore").to_csv(out / "division.csv", index=False)
    for coder in ("opus", "gpt54"):
        f = CLOSED / "funnel" / f"value_{coder}.csv"
        if f.exists():
            v = pd.read_csv(f)
            v[v.slug.isin(slugs)].to_csv(out / f"value_{coder}.csv", index=False)
    p = pd.read_csv(CLOSED / "products.csv")
    p[p.slug.isin(slugs)].to_csv(out / "products.csv", index=False)
    keys = set()
    for m in CONFIG["record"]["models"]:
        d = CLOSED / "record" / m
        if not (d / "decisions.csv").exists():
            continue
        dec = pd.read_csv(d / "decisions.csv")
        dec[dec.slug.isin(slugs)].to_csv(out / f"decisions_{m}.csv", index=False)
        (out / "record" / m).mkdir(parents=True, exist_ok=True)
        for name in ["weights_common_F.csv", "weights_common_B.csv", "weights_regime_F.csv", "weights_regime_B.csv", "weights_firm_F.csv",
                     "weights_firm_B.csv", "validation_F.csv", "validation_B.csv", "coherence_F.csv", "coherence_B.csv", "anchor_contrast.csv",
                     "hotstove.csv", "hotstove_firms.csv", "memory.csv"]:
            if (d / name).exists():
                shutil.copy(d / name, out / "record" / m / name)
    # answers of every call that names one of the firms in its purpose, with the prompt removed
    n = 0
    for line in open(llm.LOG):
        r = json.loads(line)
        if any(s in r["purpose"] for s in slugs):
            keys.add(r["key"])
    for key in keys:
        f = llm.CACHE / f"{key}.json"
        if f.exists():
            a = json.loads(f.read_text())
            (out / "answers" / f"{key}.json").write_text(json.dumps(
                {"model": a["model"], "effort": a["effort"], "answer": a["answer"]}, ensure_ascii=False))
            n += 1
    print(f"{len(slugs)} firms, {n} answers, written to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "final")
