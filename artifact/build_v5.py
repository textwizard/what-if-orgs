"""Build explorer v5 (four pages: Strategy, Simulator, Firms, Validation) from the data that build_v3.py wrote.

Reads data_v3.json (or data_public.json with --public) and architecture.svg, injects them into template_v5.html, and
writes index_v5.html (or public/index_v5.html). Run build_v3.py first whenever the data changed; code/61 redraws the figure.
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
    data = (HERE / ("data_public.json" if PUBLIC else "data_v3.json")).read_text()
    page = (HERE / "template_v5.html").read_text().replace("<!--ARCH-->", svg_of(HERE / "architecture.svg")).replace("/*DATA*/null", data)
    out = HERE / "public" / "index_v5.html" if PUBLIC else HERE / "index_v5.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    print(f"{out}, {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
