# What-If Organizations

Code, public data, and research artifact for *What-If Organizations: Synthetic Agents for Organization Science Research* (Dahlke and Calvino, Stage 1 proposal to the *Organization Science* special issue AI-Assisted Frontiers in Organizational Science).

What-if organizations are language-model agents that represent specific real organizations. Each agent combines a language model, anchors measured from the archived websites of the organization, and an assigned behavioral theory given as a short text. The reference organization reads the firm's own record of new products year by year and decides about each of them; what-if organizations are the same firm simulated under alternative theories, without its anchors, or with its own earlier decisions as its past in place of the record. The proof of concept applies the agents to how 50 German firms respond to the early results of their new products, measured on their archived websites from 2017 to 2026.

## What is in the repository

| Path | Content |
|---|---|
| `whatif/` | The pipeline as a Python package, one module per step |
| `analysis/` | The scripts that produce the reported results and figures |
| `artifact/` | The research artifact: the published page, its template, and the build scripts |
| `protocols/` | The coding protocols given to the language model |
| `theories/` | The theory texts in both framings |
| `data/public/final/` | The public data of the 50 firms |
| `config.yaml` | The design of the study: windows and decision years, theories and framings, anchors, models, and the spending limit |
| `paths.example.yaml` | Template for the local paths to sources that cannot be redistributed |

Each folder carries a README that describes its files.

## Reproduction

```
pip install -r requirements.txt
cp paths.example.yaml paths.yaml                      # set the local paths
export PYTHONPATH=.
python -m whatif.web data/public/final/firms.csv       # pages from the Common Crawl pointers
python -m whatif.coding firms.txt                      # add --plan for a dry run without model calls
python -m whatif.products firms.txt
python -m whatif.record data/public/final/firms.csv    # the simulation on the record; --plan prints the cost
python -m whatif.memory data/public/final/firms.csv    # the simulation on the firm's own history
python -m whatif.weights claude-opus-5-5
python analysis/diagnostics.py claude-opus-5-5
python analysis/anchor_contrast.py claude-opus-5-5
python analysis/hot_stove.py claude-opus-5-5
python artifact/build_data.py --public && python artifact/build_page.py --public
```

Two paths lead to the results. The quick path reads the published decisions and tables in `data/public/final/` and rebuilds every figure of the artifact without a model key. The full path rebuilds the pages from the published Common Crawl pointers, runs the coding and the simulations from the published model answers, which are keyed by the hash of their prompt, and reruns any call live with a key. Every call of the proof of concept went through the batch interface of its provider, or through parallel single calls where the provider offers none, and is cached, so that the pipeline runs from the cache once the answers exist. The enumeration of the Common Crawl index, which lists the captures of each firm, uses a shared client of the authors and is released separately; the pointers it produced are published.

## Data

`data/public/final/` holds, for the 50 firms: the firm list; the Common Crawl pointer of every page read (crawl, URL, WARC file, offset, length); every coding with its quotes; the situation of the firm per window; the NACE division from the website; the theory of value; the product trajectories with the year of the drop; every decision of the simulation on the record and on the firm's own history, with probability, decision, and reason, per model (the own-history run of DeepSeek V4 Pro covers 20 of the 79 firm-windows at the time of this release and is completed in the next); the weights, validation, gap, coherence, anchor contrast, and hot stove tables; and the model answers of every call with the prompt text removed. Page text is rebuilt from the pointers rather than redistributed.

The sample frame and the variables that define its cells come from the firm panel of Dahlke et al. (2026), which builds on a licensed company database and web indicators; they stay in `data/closed/`, which is excluded from the repository, and nothing derived from them is published beyond the firm list. The analysis uses no variable of the frame: the industry of each firm is inferred from its website.

## Licences

Code is released under the MIT licence (`LICENSE`); the data in `data/public/` under CC BY 4.0 (`DATA_LICENSE.md`). The theory texts and protocols are part of the code.
