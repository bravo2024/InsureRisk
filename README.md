# InsureRisk

> Insurance risk segmentation with Solvency II premium pricing framework.

Trains four classifiers on synthetic insurance policy data to predict claim likelihood. Dashboard provides risk segmentation, loss ratio analysis, risk-based premium pricing, and actuarial solvency modelling.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.898 |
| Gini | 0.796 |
| KS Statistic | 0.683 |
| F1 Score | 0.696 |
| Accuracy | 0.837 |

5-fold CV AUC: 0.889 ± 0.012. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Policy overview, claim distribution, key risk factor summary |
| **Model Lab** | Multi-model comparison, ROC curves, calibration plots, CV results |
| **Risk Segmentation** | Policy count by risk segment, loss ratio calculation, expected loss by segment |
| **Pricing** | Risk-based premium modelling, premium distribution by claim outcome, Solvency II ruin probability and SCR framework |

## Repo Structure

```
InsureRisk/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic insurance policy dataset: prior claims (3yr), credit score, region risk score, vehicle value, driving experience, annual mileage, and policy type.

## License

MIT
