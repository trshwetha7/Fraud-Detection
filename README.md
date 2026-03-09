# Fraud Detection Analysis

This project builds an end-to-end fraud detection workflow on transaction data from the PaySim dataset. The goal is not only to train a strong classifier, but also to approach the problem the way a fraud data scientist would: understand the transaction behavior, engineer interpretable features, avoid leakage, compare models, tune the decision threshold, and explain what the model is actually doing.

## Project goal

Fraud detection is an imbalanced classification problem. Only a small share of transactions are fraudulent, which means plain accuracy is not a useful metric on its own. In practice, the challenge is to catch as much fraud as possible without overwhelming analysts or customers with too many false alerts.

This project focuses on:

- exploratory analysis of transaction behavior
- interpretable feature engineering
- careful handling of leakage-prone fields
- model comparison on imbalanced data
- threshold tuning based on fraud-detection tradeoffs
- prediction export for downstream review

## Dataset

The analysis uses the public **PaySim** fraud dataset available on Kaggle:

- [PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)

The source data contains transaction-level fields such as:

- `step`
- `type`
- `amount`
- `nameOrig`
- `nameDest`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

For this project, the notebook builds a practical modeling sample by keeping all fraud rows and a reproducible sample of non-fraud rows. That keeps the workflow fast enough to run locally while preserving the full positive class.

## What is in the analysis

The notebook walks through the full fraud modeling process:

1. Load and sample the PaySim dataset.
2. Inspect class balance, transaction types, amount patterns, and time-based patterns.
3. Engineer readable features such as `hour_of_day`, `day_index`, `log_amount`, `amount_over_200k`, and `is_transfer_or_cash_out`.
4. Exclude raw identifiers and leakage-prone balance fields from modeling.
5. Train and compare Logistic Regression and Random Forest models.
6. Tune the probability threshold instead of treating `0.50` as fixed.
7. Evaluate performance on a held-out test set.
8. Interpret feature importance and export scored predictions.

## Modeling choices

Some project decisions are intentional and important:

- Raw account identifiers are excluded because they do not generalize well and can make the model memorize specific entities.
- Balance fields are excluded from modeling because they can leak target information and make performance look better than it would be in a real fraud setting.
- Precision-recall metrics are emphasized because fraud detection is highly imbalanced.
- Threshold tuning is included because the right operating point depends on how much missed fraud versus false alarms the business can tolerate.

## Results snapshot

The notebook currently produces the following test-set results for the selected Random Forest model:

- ROC-AUC: `0.9687`
- PR-AUC: `0.8068`
- Precision: `0.5539`
- Recall: `0.8138`
- F-beta: `0.7439`
- Tuned decision threshold: `0.61`

Confusion matrix counts at the tuned threshold:

- True negatives: `24,341`
- False positives: `1,077`
- False negatives: `306`
- True positives: `1,337`

These numbers show a model that catches a large share of fraud while keeping the false positive rate manageable for further review.

## Repository structure

```text
.
|-- data/
|   |-- processed/
|   |-- raw/
|-- notebooks/
|   `-- fraud_detection_analysis.ipynb
|-- reports/
|   `-- figures/
|-- scripts/
|   |-- generate_notebook.py
|   `-- run_jupyter_lab.sh
|-- src/
|   |-- data.py
|   |-- features.py
|   `-- modeling.py
`-- requirements.txt
```

## Running the project locally

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset CSV in one of these locations:

```text
data/raw/paysim.csv
```

or:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```

Start JupyterLab:

```bash
bash scripts/run_jupyter_lab.sh
```

Then open:

```text
notebooks/fraud_detection_analysis.ipynb
```

## Key takeaways

- Feature engineering should stay close to real transaction behavior so the model remains explainable.
- Leakage checks matter as much as model performance.
- Threshold selection is a core part of fraud modeling, not a cosmetic final step.
- A useful fraud project should show both predictive strength and operational understanding.

## Author

**Shwetha Tinnium Raju**

If you want to reuse or extend this work, feel free to adapt the notebook and modeling pipeline for different fraud scenarios or production-style scoring workflows.
