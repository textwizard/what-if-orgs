"""Outcome model of the simulation.

The result of product k in period t is y_kt = mu_k + d_k [t = 1] + e_kt. The potential mu_k is high or low with equal
probability and is unknown to the agent, the shift d_k moves the first result toward an early failure or an early
success with equal probability, and e_kt is normal noise with a smaller spread in the first period. Results are
reported to the agent in percent of plan, 100 + scale y. All draws depend only on firm, window, and run, so that the
language-model agents under every condition and the rule-based twin face identical results.
"""
import hashlib

import numpy as np

from whatif.config import CONFIG

O = CONFIG["outcomes"]


def draws(slug, window, run, n):
    seed = int(hashlib.sha256(f"{slug}|{window}|{run}".encode()).hexdigest()[:12], 16)
    rng = np.random.default_rng(seed)
    high = rng.random(n) < O["share_high"]
    failure = rng.random(n) < O["share_early_failure"]
    mu = np.where(high, O["mu"], -O["mu"])
    y = mu[:, None] + rng.normal(0, O["noise"], (n, O["periods"]))
    y[:, 0] = mu + np.where(failure, -O["shift"], O["shift"]) + rng.normal(0, O["noise_first"], n)
    percent = np.clip(np.round(100 + O["scale"] * y), 5, 300).astype(int)
    return {"high": high.tolist(), "early": ["failure" if f else "success" for f in failure],
            "percent": percent.tolist()}
