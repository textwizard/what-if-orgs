"""Re-identification test of the anonymized profiles, and recall test of the fate of products.

In the re-identification test, the model receives the same description of the firm and of its new products that the
what-if organization receives in the first window with innovations, and is asked to name the firm. A firm counts as re-identified if the answer contains the
distinctive part of its domain. In the recall test, the model receives the name it gave for a re-identified firm and
the name of each innovation of the firm, and is asked whether the firm still offered the product two years after its
launch. Recall that exceeds chance would mean that the model can draw on the fate of the products.

Outputs: data/closed/reidentification.csv, data/closed/recall.csv (with --recall)
"""
import sys

import pandas as pd

from whatif import llm
from whatif.config import ROOT
from whatif.products import OUT as PRODUCTS
from whatif.simulate import describe, jobs

SYSTEM = ("You receive the description of a real German firm and of products it launched. If you can identify the "
          "firm, give its name and website. Return {\"name\": ... or null, \"website\": ... or null, "
          "\"confidence\": 0-100} as a JSON object.")


RECALL = ("You receive the name of a German firm and of a product it launched in a given year. If you know whether the "
          "firm still offered this product two years after the launch, say so. Return {\"still_offered\": true, false, "
          "or null if you do not know, \"confidence\": 0-100} as a JSON object.")


def recall(slug, firm, prod):
    rows = []
    for _, r in prod[(prod.slug == slug) & prod.innovation & (prod.entry <= 2023)].iterrows():
        name = r.names.split("|")[0]
        a = llm.ask(RECALL, f"Firm: {firm}\nProduct: {name}\nLaunched: {r.entry}", purpose=f"recall {slug}")
        rows.append({"slug": slug, "product": name, "entry": r.entry, "answer": a.get("still_offered"),
                     "confidence": a.get("confidence"), "offered": pd.isna(r["drop"]) or r["drop"] > r.entry + 2})
    return rows


def probe(slug, profile, products):
    text = describe(profile) + "\n\nProducts\n" + "\n".join(f"- {p}" for p in products)
    a = llm.ask(SYSTEM, text, purpose=f"reidentification {slug}")
    core = slug.rsplit("-", 1)[0].replace("-", "")
    answer = f"{a.get('name') or ''} {a.get('website') or ''}".lower().replace("-", "").replace(" ", "")
    return {"slug": slug, "name": a.get("name"), "website": a.get("website"), "confidence": a.get("confidence"),
            "identified": len(core) > 3 and core in answer}


if __name__ == "__main__":
    slugs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    prod = pd.read_csv(PRODUCTS)
    first = {}
    for j in jobs(slugs):                                  # the description of the first window with innovations
        first.setdefault(j[0], (j[6], j[5]))
    rows = [probe(s, *first[s]) for s in slugs if s in first]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "closed" / "reidentification.csv", index=False)
    print(df.to_string(index=False))
    print(f"re-identified {int(df.identified.sum())} of {len(df)}, spent {llm.spent():.2f} USD")
    if "--recall" in sys.argv:
        r = pd.DataFrame([x for _, f in df[df.identified].iterrows() for x in recall(f.slug, f["name"], prod)])
        r.to_csv(ROOT / "data" / "closed" / "recall.csv", index=False)
        known = r[r.answer.notna()]
        print(f"{len(r)} products, answered {len(known)}, correct {(known.answer == known.offered).mean():.2f}, "
              f"share still offered {r.offered.mean():.2f}, spent {llm.spent():.2f} USD")
