# artifact

The research artifact, a single page with all data embedded, served at https://textwizard.github.io/what-if-orgs/.

| File | Content |
|---|---|
| `index.html` | The published page |
| `template.html` | The page without data: layout, charts, and text |
| `build_data.py` | Writes the data of the page from the repository (`--public` for the published version) |
| `build_page.py` | Injects the data and the architecture figure into the template |
| `architecture.svg` | Figure 1, drawn by `analysis/figure_architecture.py` |

To rebuild: `PYTHONPATH=. python artifact/build_data.py --public && python artifact/build_page.py --public`.
