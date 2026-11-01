# whatif

The pipeline as a Python package. Each module is one step and documents itself in its docstring.

| Module | Step |
|---|---|
| `config.py` | Reads `config.yaml` and `paths.yaml` |
| `eurostat.py` | Download of the Eurostat turnover tables that describe the situation of each industry |
| `sample.py` | The sample frame and the draw of firms, documented with inclusion probabilities |
| `web.py` | Selection, retrieval, and cleaning of archived pages from Common Crawl pointers |
| `llm.py` | Interface to the language models: batch calls, a cache keyed by the hash of model, settings, and prompt, the call log, and the spending limit |
| `coding.py` | Coding of products, statements with quotes, stated strategy, and characteristics from the pages |
| `context.py` | Situation of the industry and of the firm in the context year of each window |
| `division.py` | NACE division of the firm, read from its website |
| `products.py` | Product trajectories: prominence by year, innovations, lapses, and withdrawals |
| `record.py` | The simulation on the record: decisions about the real new products of each firm, year by year, under no theory and each theory, with and without anchors |
| `memory.py` | The simulation on the firm's own history: the second decision year with the agent's own first-year decisions as its past |
| `weights.py` | Mixing weights of the conditions, their validation on the held-out window, and the coherence with the theory of value |
| `release.py` | Export of the public data |
