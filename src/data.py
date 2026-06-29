from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_NAMES = ["age", "vehicle_value", "driving_experience_yrs", "annual_mileage", "prior_claims_3yr", "credit_score", "region_risk_score", "policy_tenure_months", "vehicle_type", "gender", "marital_status"]
CATEGORICAL_FEATURES = ["vehicle_type", "gender", "marital_status"]
NUMERICAL_FEATURES = ["age", "vehicle_value", "driving_experience_yrs", "annual_mileage", "prior_claims_3yr", "credit_score", "region_risk_score", "policy_tenure_months"]

def make_synthetic(n=10000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.normal(42, 14, size=n).clip(18, 85).astype(int),
        "vehicle_value": rng.lognormal(mean=9.8, sigma=0.6, size=n).clip(5000, 80000).astype(int),
        "driving_experience_yrs": rng.uniform(0, 50, size=n).round(1),
        "annual_mileage": rng.lognormal(mean=9.3, sigma=0.5, size=n).clip(1000, 50000).astype(int),
        "prior_claims_3yr": rng.poisson(lam=0.4, size=n).clip(0, 5),
        "credit_score": rng.normal(680, 60, size=n).clip(350, 850).astype(int),
        "region_risk_score": rng.uniform(1, 10, size=n).round(2),
        "policy_tenure_months": rng.gamma(shape=2, scale=12, size=n).clip(1, 120).astype(int),
        "vehicle_type": rng.choice(["sedan", "suv", "hatchback", "luxury"], size=n, p=[0.35, 0.25, 0.30, 0.10]),
        "gender": rng.choice(["M", "F"], size=n),
        "marital_status": rng.choice(["single", "married", "divorced"], size=n, p=[0.30, 0.55, 0.15]),
    })
    claims = np.clip(df["prior_claims_3yr"], 0, 3); cs = (df["credit_score"] - 350) / 500
    age = df["age"] / 85; exp = df["driving_experience_yrs"] / 50; mileage = np.log(df["annual_mileage"] / 1000) / 3
    region = df["region_risk_score"] / 10; vehicle = (df["vehicle_value"] / 80000)
    log_odds = -2.5 + 0.8 * claims - 0.5 * cs + 0.3 * region + 0.4 * vehicle - 0.3 * exp + 0.15 * mileage + rng.normal(0, 0.5, size=n)
    prob = 1 / (1 + np.exp(-log_odds))
    y = (prob > np.percentile(prob, 78)).astype(np.float64)
    return {"X": df, "y": y, "features": FEATURE_NAMES, "df": df.assign(claim=y), "categorical_features": CATEGORICAL_FEATURES, "numerical_features": NUMERICAL_FEATURES, "n_samples": n, "n_features": len(FEATURE_NAMES), "positive_rate": y.mean()}
