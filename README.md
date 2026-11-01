# What-If Organizations

Code, public data, and research artifact for *What-If Organizations: Synthetic Agents for Organization Science Research* (Dahlke and Calvino, Stage 1 proposal to the *Organization Science* special issue AI-Assisted Frontiers in Organizational Science).

What-if organizations are language-model agents that represent specific real organizations. Each agent combines a language model, anchors measured from the archived websites of the organization, and an assigned behavioral theory given as a short text. The reference organization reads the firm's own record of new products year by year and decides about each of them; what-if organizations are the same firm simulated under alternative theories, without its anchors, or with its own earlier decisions as its past in place of the record. The proof of concept applies the agents to how 50 German firms respond to the early results of their new products, measured on their archived websites from 2017 to 2026.

## What is in the repository

| Path | Content |
|---|---|
| `whatif/` | The pipeline as a Python package; each module documents its step in its docstring |
| `config.yaml` | Design of the study: windows and decision years, theories and framings, anchors, models and settings, the spending limit |
| `protocols/` | Coding protocols given to the language model (products and statements per year, situation, strategy, theory of value, NACE division, relatedness, aliases) |
| `theories/` | Theory texts in both framings, considerations (F) and conditions (B) |
| `pilot/` | Numbered analysis scripts of the proof of concept (diagnostics, anchor contrast, hot stove analysis, figures) |
| `artifact/` | The research artifact: the built page (`index.html`), its template, and the build scripts |
| `data/public/final/` | The public data of the 50 firms (see Data) |
| `docs/design.md` | The design record: every decision of the design with its reasons, in the order taken |
| `paths.example.yaml` | Template for the local paths to sources that cannot be redistributed |

## The pipeline

| Module | Step |
|---|---|
| `whatif/eurostat.py` | Download of the Eurostat turnover tables |
| `whatif/sample.py` | Sample frame and draws |
| `whatif/web.py` | Selection, retrieval, and cleaning of archived pages from Common Crawl pointers |
| `whatif/llm.py` | Interface to the language models: batch calls, response cache keyed by the hash of model, settings, and prompt, call log, spending limit |
| `whatif/coding.py` | Coding of products, statements with quotes, strategy, and characteristics from the pages |
| `whatif/context.py` | Situation of the industry and the firm in the context year |
| `whatif/division.py` | NACE division of the firm inferred from its website |
| `whatif/products.py` | Product trajectories: prominence by year, innovations, lapses, and drops |
| `whatif/record.py` | The simulation on the record: decisions about the real new products of each firm, year by year, under no theory and each theory, with and without anchors |
| `whatif/memory.py` | The simulation on the firm's own history: the second decision year with the agent's own first-year decisions as its past, drawn from its probabilities, and the record-as-memory control |
| `whatif/weights.py` | Mixing weights of the conditions per firm, their validation after the pandemic, the pandemic gap set, and the coherence with the theory of value |
| `whatif/release.py` | Export of the public data |

Older modules of the instrument tests (`simulate.py`, `outcomes.py`, `twin.py`, `probes.py`, `analysis.py`) are kept for the record of the design.

## Reproduction

```
pip install -r requirements.txt
cp paths.example.yaml paths.yaml                 # set the local paths
python -m whatif.web firms.csv                    # pages from the Common Crawl pointers (a CSV with a column domain)
python -m whatif.coding slugs.txt                 # add --plan for a dry run without model calls
python -m whatif.products slugs.txt
python -m whatif.record firms.csv [--plan]        # the simulation on the record; --plan prints the cost
python -m whatif.memory firms.csv [--plan]        # the simulation on the firm's own history
python -m whatif.weights claude-opus-5-5
PYTHONPATH=. python pilot/55_record_diagnostics.py claude-opus-5-5   # the pilot scripts import the package from the repository root
python pilot/58_anchor_contrast.py claude-opus-5-5
python pilot/60_hot_stove.py claude-opus-5-5
```

The enumeration of the Common Crawl index, which lists the captures of each firm, uses a shared client of the authors and is released separately; the pointers it produced are published, so the pages can be rebuilt without it.

Two paths lead to the results. The quick path reads the published decisions and tables in `data/public/final/` and rebuilds every figure of the artifact without a model key. The full path rebuilds the pages from the published Common Crawl pointers, runs the coding and the simulations from the published model answers, which are keyed by the hash of their prompt, and reruns any call live with a key. Every call of the proof of concept went through the batch interface of its provider and is cached, so that the pipeline runs from the cache once the answers exist.

## Data

`data/public/final/` holds, for the 50 firms: the firm list; the Common Crawl pointer of every page read (crawl, URL, WARC file, offset, length); every coding with its quotes; the situation of the firm per window; the NACE division from the website; the theory of value; the product trajectories with the year of the drop; every decision of the simulation on the record and on the firm's own history, with probability, decision, and reason, per model; the weights, validation, gap, coherence, anchor contrast, and hot stove tables; and the model answers of every call with the prompt text removed. Page text is rebuilt from the pointers rather than redistributed.

The sample frame and the variables that define its cells come from the firm panel of Dahlke et al. (2026), which builds on a licensed company database and web indicators; they stay in `data/closed/`, which is excluded from the repository, and nothing derived from them is published beyond the firm list. The analysis uses no variable of the frame: the industry of each firm is inferred from its website.

## Licences

Code is released under the MIT licence (`LICENSE`); the data in `data/public/` under CC BY 4.0 (`DATA_LICENSE.md`). The theory texts and protocols are part of the code.

## Citation

Dahlke, J., and Calvino, F. (2026). *What-if organizations: Synthetic agents for organization science research*. Stage 1 proposal, Organization Science.

The views expressed are those of the authors and cannot be attributed to the OECD or its member countries.
