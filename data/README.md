# data

`public/final/` holds the public data of the 50 firms. Everything that comes from the licensed sample frame stays outside the repository.

| Path | Content |
|---|---|
| `firms.csv` | The firm list |
| `pointers/` | The Common Crawl pointer of every page read: crawl, URL, file, offset, length |
| `codings/` | Every coding of products, statements, and profiles, with quotes |
| `context/` | The situation of each firm per window, with quotes |
| `division.csv` | The NACE division of each firm, read from its website |
| `value_opus.csv`, `value_gpt54.csv` | The theory of value by two coders |
| `products.csv` | The product trajectories with the year of withdrawal |
| `decisions_<model>.csv` | Every decision of the simulation on the record, with probability, decision, and reason |
| `record/<model>/` | The weights, validation, coherence, anchor contrast, and own-history tables, and `memory.csv`, the decisions on the firm's own history |
| `answers/` | The model answer of every call, keyed by the hash of its prompt, with the prompt text removed |

Page text is rebuilt from the pointers rather than redistributed.
