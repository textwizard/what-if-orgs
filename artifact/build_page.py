"""Page of the research artifact.

Reads data.json (or data_public.json with --public) and architecture.svg, injects them into template.html, and writes
index_local.html (or index.html with --public, the page that is published). Run build_data.py first whenever the data
changed; analysis/figure_architecture.py redraws the figure.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
PUBLIC = "--public" in sys.argv


def svg_of(path):
    if not path.exists():
        return ""
    s = re.sub(r"<\?xml[^>]*\?>|<!DOCTYPE[^>]*>", "", path.read_text(), flags=re.S).strip()
    return re.sub(r'<svg([^>]*?)\s(width|height)="[^"]*"', r"<svg\1", s, count=2)


def main():
    data = (HERE / ("data_public.json" if PUBLIC else "data.json")).read_text()
    page = (HERE / "template.html").read_text().replace("<!--ARCH-->", svg_of(HERE / "architecture.svg")).replace("/*DATA*/null", data)
    out = HERE / ("index.html" if PUBLIC else "index_local.html")
    out.write_text(page)
    print(f"{out}, {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
