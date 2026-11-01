"""Rule-based twin of the what-if organizations, after the learning model of Denrell and March (2001).

The twin estimates the result of each product as the mean of a prior and the results it has observed, keeps a product
while the estimate is at or above the aspiration level, and drops it otherwise. Without forgone outcomes, a dropped
product is no longer observed and its estimate stays where it was. With forgone outcomes, the twin keeps observing
dropped products and reintroduces one when its estimate returns to the aspiration level. The twin faces the same
results as the language-model agents, so that differences between them arise from the decision alone.
"""
from whatif.config import CONFIG

T = CONFIG["twin"]
SCALE = CONFIG["outcomes"]["scale"]


def run(results, forgone):
    """Decisions per product and period for a matrix of results in percent of plan."""
    decisions = []
    for row in results:
        total, n, held, out = T["prior"] * T["weight"], T["weight"], True, []
        for pct in row:
            if held or forgone:
                total, n = total + (pct - 100) / SCALE, n + 1
            belief = total / n
            if held:
                held = belief >= T["aspiration"]
                out.append("keep" if held else "drop")
            elif forgone and belief >= T["aspiration"]:
                held = True
                out.append("reintroduce")
            else:
                out.append("stay dropped")
        decisions.append(out)
    return decisions
