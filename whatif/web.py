"""Select, fetch, and clean the archived pages of each firm.

The input is a list of Common Crawl captures per firm (url, year, crawl, and the WARC pointer), produced by the
enumeration of the Common Crawl index. For each firm and year, the latest capture of each URL is kept, pages are
assigned a role from their path, and at most a fixed number of pages per role is selected. Each selected page is
fetched by its WARC pointer, which allows anyone to retrieve the identical bytes, and its main text is extracted.
Lines that recur on many pages of the same firm and year, such as menus and footers, are removed.

Outputs: data/cache/pages/<domain>/<year>/<sha>.html.gz (raw pages, local only),
         data/closed/pages/<domain>.jsonl (one line per page with year, url, role, title, and text),
         data/closed/pointers/<slug>.csv (the WARC pointer of every page used).
"""
import csv
import gzip
import hashlib
import io
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests
import trafilatura

from whatif.config import CONFIG, PATHS, ROOT

W = CONFIG["web"]
CACHE = ROOT / "data" / "cache" / "pages"
PAGES = ROOT / "data" / "closed" / "pages"
POINTERS = ROOT / "data" / "closed" / "pointers"
ROLES = {  # the first matching pattern decides, and paths without segments are the home page
    "imprint": r"impressum|imprint|legal[-_]?notice|legal",
    "career": r"karriere|career|jobs?\b|stellen|bewerb|ausbildung|vacanc",
    "about": r"ueber-uns|uber-uns|about|unternehmen|company|team|geschichte|history|philosophie",
    "offer": r"produkt|product|leistung|service|loesung|losung|solution|sortiment|angebot|shop|portfolio|technolog",
    "rnd": r"forschung|entwicklung|research|innovation|labor|\blab\b|f-und-e|r-and-d",
    "news": r"news|aktuell|presse|press|blog|neuigkeit|referenz|reference|projekt|project",
}
DENY = re.compile(r"datenschutz|privacy|agb|terms|cookie|login|warenkorb|cart|checkout|sitemap|feed|\.xml$|\.pdf$", re.I)


HOME = {"de", "en", "de-de", "en-us", "en-gb", "home", "start", "startseite", "index", "default"}


def role(url):
    path = re.sub(r"\.(html?|php|aspx?)$", "", urlparse(url).path.lower())
    segments = [x for x in path.split("/") if x and x not in HOME]
    if not segments:
        return "home"
    path = "/".join(segments)
    for r, pattern in ROLES.items():
        if re.search(pattern, path):
            return r
    return "other"


def select(domains):
    """Latest capture per URL and year, at most W['per_role'] pages per role and year."""
    latest = {}
    keep = {s.replace(".", "-") for s in domains}
    for f in PATHS["captures"]:
        for c in csv.DictReader(open(f, encoding="utf-8")):
            if c["slug"] not in keep:                       # read only the firms asked for; the pointer files are large
                continue
            if c["mime"] != "html" or DENY.search(c["url"]) or (c["denied"] == "1" and role(c["url"]) != "imprint"):
                continue
            key = (c["slug"], c["year"], c["url"].split("://", 1)[-1].removeprefix("www.").rstrip("/"))
            if key not in latest or c["capture_ts"] > latest[key]["capture_ts"]:
                latest[key] = c
    chosen = defaultdict(list)
    for (slug, year, _), c in sorted(latest.items(), key=lambda kv: int(kv[1]["depth"])):
        r = role(c["url"])
        if sum(1 for x in chosen[slug, year] if x["role"] == r) < W["per_role"][r]:
            chosen[slug, year].append({**c, "role": r})
    return {k: v for k, v in chosen.items() if k[0] in keep}


def fetch(c):
    """Raw HTML of one capture, read from the cache or fetched by its WARC pointer."""
    sha = hashlib.sha1(f'{c["warc_filename"]}{c["warc_offset"]}'.encode()).hexdigest()[:16]
    path = CACHE / c["slug"] / c["year"] / f"{sha}.html.gz"
    if path.exists():
        return gzip.decompress(path.read_bytes()).decode("utf-8", "replace")
    start, length = int(c["warc_offset"]), int(c["warc_length"])
    for attempt in range(W["retries"]):        # Common Crawl throttles bursts, and a sleeping laptop loses its network
        try:
            r = requests.get(f'https://data.commoncrawl.org/{c["warc_filename"]}', timeout=60,
                             headers={"Range": f"bytes={start}-{start + length - 1}", "User-Agent": W["user_agent"]})
            if r.status_code not in (403, 429, 500, 502, 503, 504):
                break
        except requests.RequestException:
            if attempt == W["retries"] - 1:
                raise
        time.sleep(min(W["backoff"] * 2 ** attempt, 600))
    time.sleep(W["delay"])
    r.raise_for_status()
    record = gzip.GzipFile(fileobj=io.BytesIO(r.content)).read()
    body = record.split(b"\r\n\r\n", 2)[-1]                 # WARC headers, HTTP headers, payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(body))
    return body.decode("utf-8", "replace")


def text(html):
    t = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=True) or ""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    return (m.group(1).strip() if m else ""), t


def clean(pages):
    """Remove lines that recur on at least W['boiler_share'] of the pages of a firm-year."""
    counts = Counter(line for p in pages for line in set(p["text"].splitlines()) if line.strip())
    limit = max(2, W["boiler_share"] * len(pages))
    for p in pages:
        p["text"] = "\n".join(l for l in p["text"].splitlines() if l.strip() and counts[l] < limit)
    return pages


def main(domains, years=None):
    """Pages of the given domains, in the given years or all; slugs are the domains with dots replaced by hyphens."""
    chosen = {k: v for k, v in select(domains).items() if not years or int(k[1]) in years}
    jobs = [c for cs in chosen.values() for c in cs]
    print(f"{len(chosen)} firm-years, {len(jobs)} pages to read")

    def work(c):
        try:
            title, body = text(fetch(c))
            return {**c, "title": title, "text": body}
        except Exception as e:                              # failed pages are logged and skipped
            return {**c, "title": "", "text": "", "error": str(e)[:200]}
    with ThreadPoolExecutor(W["workers"]) as ex:
        done = list(ex.map(work, jobs))

    PAGES.mkdir(parents=True, exist_ok=True)
    by_firm = defaultdict(lambda: defaultdict(list))
    for p in done:
        by_firm[p["slug"]][p["year"]].append(p)
    for slug, years in by_firm.items():
        with open(PAGES / f"{slug}.jsonl", "w", encoding="utf-8") as f:
            for year in sorted(years):
                for p in clean([p for p in years[year] if p["text"]]):
                    f.write(json.dumps({k: p[k] for k in ("year", "url", "role", "title", "text", "capture_ts")},
                                       ensure_ascii=False) + "\n")
        POINTERS.mkdir(parents=True, exist_ok=True)
        with open(POINTERS / f"{slug}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["year", "url", "warc_filename", "warc_offset", "warc_length"])
            w.writerows([p["year"], p["url"], p["warc_filename"], p["warc_offset"], p["warc_length"]]
                        for ys in years.values() for p in ys)
    failed = [{k: p[k] for k in ("slug", "year", "url", "error")} for p in done if "error" in p]
    if failed:
        with open(ROOT / "data" / "closed" / "web_failures.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(failed[0]))
            w.writeheader()
            w.writerows(failed)
    print(f"read {len(done) - len(failed)} pages, {len(failed)} failed")


if __name__ == "__main__":
    import sys
    years = next(([int(y) for y in a.split("=")[1].split(",")] for a in sys.argv if a.startswith("--years=")), None)
    main([r["domain"] for r in csv.DictReader(open(sys.argv[1]))], years)
