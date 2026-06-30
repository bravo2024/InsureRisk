"""core.py — Actuarial loss distribution metrics for InsureRisk (EXL).

Fits and evaluates loss distributions for insurance risk pricing.
Uses MLE for distribution fitting and statistical tests for selection.
NOT generic classification. Reference: Ohlsson & Johansson (2010).
"""
from __future__ import annotations
import numpy as np; from scipy.stats import kstest


def fit_distribution_mle(losses, dist_name="gamma"):
    """Fit a parametric distribution to loss data via MLE and compute KS stat."""
    import scipy.stats as st
    loss = np.asarray(losses, float)
    loss = loss[loss > 0]
    if len(loss) < 5:
        return {"name": dist_name, "params": [], "ks_stat": 0.0, "ks_pval": 0.0}
    dists = {"gamma": st.gamma, "expon": st.expon, "lognorm": st.lognorm, "pareto": st.pareto}
    dist = dists.get(dist_name, st.gamma)
    params = dist.fit(loss)
    ks = kstest(loss, dist_name, args=params)
    return {"name": dist_name, "params": [float(p) for p in params], "ks_stat": float(ks.statistic), "ks_pval": float(ks.pvalue)}


def var(alpha, dist_name, params):
    """Value-at-Risk at confidence level alpha."""
    import scipy.stats as st
    dists = {"gamma": st.gamma, "expon": st.expon, "lognorm": st.lognorm, "pareto": st.pareto}
    dist = dists.get(dist_name, st.gamma)
    return float(dist.ppf(alpha, *params))


def tvar(alpha, dist_name, params):
    """Tail Value-at-Risk (expected shortfall) at confidence level alpha."""
    import scipy.stats as st; import scipy.integrate as integrate
    dists = {"gamma": st.gamma, "expon": st.expon, "lognorm": st.lognorm, "pareto": st.pareto}
    dist = dists.get(dist_name, st.gamma)
    q = dist.ppf(alpha, *params)
    # TVaR = E[X | X > q] = (1/(1-alpha)) * integral_q^inf x*f(x) dx
    expected, _ = integrate.quad(lambda x: x * dist.pdf(x, *params), q, dist.ppf(0.9999, *params))
    return float(expected / (1 - alpha))