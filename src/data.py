# Actuarial loss data: territory, age, claim severity
import numpy as np
import pandas as pd

def make_synthetic(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    territory = rng.choice(["urban", "rural", "suburban"], n)
    age = rng.integers(18, 80, n)
    # ~10% zero-loss policies, rest gamma-distributed
    has_loss = rng.random(n) > 0.1
    losses = np.zeros(n)
    losses[has_loss] = rng.gamma(2, 2000, has_loss.sum()).round(2)
    return {
        "losses": losses,
        "features": ["age"],
        "n_samples": n,
        "loss_rate": float((losses > 0).mean()),
    }