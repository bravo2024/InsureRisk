# Loss distribution fitting: gamma, log-normal, exponential
# Picks the best distribution via KS statistic
from src.core import fit_distribution_mle, var, tvar
import numpy as np

def fit_and_evaluate(data, seed=42):
    losses = data["losses"]
    candidates = ["gamma", "lognorm", "expon"]
    results = {}
    for d in candidates:
        results[d] = fit_distribution_mle(losses, d)
    best = min(candidates, key=lambda d: results[d]["ks_stat"])
    met = results[best]
    return (
        {"results": results},
        {
            "best_dist": best,
            "ks_stat": met["ks_stat"],
            "n_losses": int((losses > 0).sum()),
            "var_95": var(0.95, best, met["params"]),
            "tvar_95": tvar(0.95, best, met["params"]),
        },
    )