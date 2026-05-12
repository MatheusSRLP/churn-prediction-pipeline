![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

# Customer Churn Prediction Pipeline

End-to-end binary classification pipeline for predicting customer churn in subscription-based businesses — from raw data to calibrated probabilities and SHAP-driven business recommendations.

---

## Business Problem

Acquiring a new customer costs **5–7× more** than retaining an existing one. For any subscription-based business — SaaS, gyms, aesthetic clinics, telecom — identifying customers likely to churn *before they leave* enables targeted interventions: discount offers, re-engagement campaigns, priority support outreach.

This project answers a concrete business question:

> **Which customers are most likely to churn in the next billing cycle, and what are the primary drivers of their churn risk?**

The pipeline produces **calibrated churn probabilities** that enable risk-based customer segmentation, so retention budgets are allocated where they have the highest expected return — not wasted on customers who would have stayed anyway.

---

## Approach

Models are trained in a **Baseline → Complexity** progression, so each step is justified against the previous one:

| Model | Rationale |
|---|---|
| Logistic Regression | Interpretable baseline; coefficients map directly to log-odds |
| Random Forest | Non-linear ensemble; robust to outliers; native feature importance |
| XGBoost | State-of-the-art gradient boosting; handles imbalance via `scale_pos_weight` |

Class imbalance (~26.5% positive rate) is addressed with **SMOTE applied exclusively to the training set** and `class_weight="balanced"` / `scale_pos_weight` in each respective model. The primary evaluation metric is **F2-score** (recall weighted 2×), because missing a churner costs more than a false alarm.

---

## Key Results

Evaluated on a held-out test set of **1,409 customers** (stratified 80/20 split from 7,043 total).

| Model | ROC-AUC | PR-AUC | F1 | F2 | Recall | Precision |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.8357** | **0.6357** | 0.6132 | 0.6190 | 0.6230 | 0.6036 |
| Random Forest | 0.8240 | 0.6069 | 0.5844 | 0.5946 | 0.6016 | 0.5682 |
| XGBoost | 0.8151 | 0.6106 | 0.5605 | 0.5659 | 0.5695 | 0.5518 |

**Best model for ranking churners:** Logistic Regression (ROC-AUC 0.8357, PR-AUC 0.6357).

**Best model for deployment:** XGBoost with **optimised threshold = 0.10 → F2 = 0.7331**.

Threshold optimisation shifts the decision boundary so the model catches far more true churners at the cost of some additional false alarms — the correct trade-off when the cost of missing a churner (lost lifetime value) outweighs the cost of an unneeded retention offer (discount spend).

---

## Selected Visualisations

### ROC Curves — All Models

![ROC Curves](https://raw.githubusercontent.com/MatheusSRLP/churn-prediction-pipeline/main/docs/figures/roc_curves.png)

### SHAP Summary — Global Feature Importance (XGBoost)

![SHAP Summary](https://raw.githubusercontent.com/MatheusSRLP/churn-prediction-pipeline/main/docs/figures/shap_summary.png)

### Calibration Curves — Reliability Diagram

![Calibration Curves](https://raw.githubusercontent.com/MatheusSRLP/churn-prediction-pipeline/main/docs/figures/calibration_curves.png)

---

## Business Recommendations

Derived from SHAP analysis on the XGBoost model:

**1. Prioritise Month-to-Month Customers in Their First 12 Months**
Contract type and tenure are the top SHAP drivers. A month-to-month customer in their first year faces zero switching costs and no lock-in. Action: offer a discounted annual upgrade at month 3–6, before churn intent crystallises. Converting 10% of this segment to one-year contracts has an outsized retention impact.

**2. Bundle Add-On Services for Fiber Optic Subscribers**
Fiber optic customers paying high monthly charges without Online Security, Tech Support, or Device Protection show elevated SHAP values for churn — they perceive low value for money. Action: include one add-on service free for the first three months. Increased perceived value reduces price-driven switching and cross-sells into stickier service tiers.

**3. Use Calibrated Probabilities for Risk-Tiered Campaigns**
The calibration curves confirm XGBoost's probabilities are reliable enough to segment customers into three risk tiers:
- **Low** (p < 0.30): automated email touchpoint
- **Medium** (0.30–0.60): personalised outreach
- **High** (p > 0.60): direct contact + targeted offer

This prevents wasting the retention budget on customers who would have stayed regardless.

---

## How to Run

**1. Clone and install dependencies**

```bash
git clone https://github.com/MatheusSRLP/churn-prediction-pipeline.git
cd churn-prediction-pipeline
pip install -r requirements.txt
```

**2. Download the dataset**

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in `data/raw/`.

**3. Run the test suite**

```bash
pip install pytest-cov
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

**4. Build and execute notebooks**

```bash
python scripts/build_notebooks.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_model_comparison.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_interpretability.ipynb
```

**5. Open notebooks interactively**

```bash
jupyter lab
```

---

## Architecture

```
churn-prediction-pipeline/
├── data/
│   ├── raw/                          # original CSV — never modified, never committed
│   └── processed/                    # cleaned and engineered features
├── src/
│   ├── data/
│   │   ├── loader.py                 # load_raw_data() with schema validation
│   │   └── cleaner.py                # clean_data() + validate_clean_data()
│   ├── features/
│   │   └── engineering.py            # engineer_features() + get_feature_columns()
│   ├── models/
│   │   ├── trainer.py                # build_* + train_all_models() + save_model()
│   │   ├── evaluator.py              # evaluate_model() + compare_models() + find_optimal_threshold()
│   │   └── tuner.py                  # hyperparameter search
│   └── visualization/
│       └── plots.py                  # 9 plot functions (EDA + evaluation + SHAP)
├── notebooks/
│   ├── 01_eda.ipynb                  # exploratory analysis and churn profiling
│   ├── 02_feature_engineering.ipynb  # encoding, SMOTE, train/test split
│   ├── 03_model_comparison.ipynb     # model training and side-by-side evaluation
│   └── 04_interpretability.ipynb     # SHAP + calibration + threshold optimisation
├── scripts/
│   └── build_notebooks.py            # generates all .ipynb files from source
├── outputs/
│   ├── figures/                      # all exported charts (dpi=300)
│   ├── models/                       # serialised model artifacts (joblib)
│   └── reports/                      # train/test split and metrics CSVs
├── docs/figures/                     # charts embedded in this README
├── tests/
│   ├── test_cleaner.py               # 17 tests
│   ├── test_engineering.py           # 32 tests
│   └── test_evaluator.py             # 31 tests
├── .env.example
├── requirements.txt
└── CLAUDE.md
```

---

## Design Decisions

**Pipeline to prevent data leakage**
All preprocessing (scaling, encoding) is wrapped in a scikit-learn `Pipeline`. The pipeline is fit exclusively on training data; `transform` is called on validation and test sets. This guarantees that no information from held-out sets influences preprocessing parameters — a common source of optimistic bias in ML portfolios.

**SMOTE applied only on the training set**
SMOTE generates synthetic minority-class samples by interpolating between existing ones. Applying it before splitting would contaminate the test set with synthetic neighbours of training samples — inflating recall metrics artificially. Here SMOTE is applied strictly after the train/test split, on `X_train` only.

**F2 over F1 as primary metric**
Accuracy is misleading on the 73.5/26.5% split: a constant "No Churn" classifier scores 73.5% and detects zero churners. F1 treats false positives and false negatives equally. F2 (β=2) weights recall twice as heavily as precision — reflecting the business reality that the cost of missing a churner (lost lifetime value) outweighs the cost of an unnecessary retention offer (discount spend). Threshold is optimised to maximise F2 rather than F1.

**Calibration analysis before recommending deployment**
A high ROC-AUC does not mean predicted probabilities are reliable. Calibration curves (reliability diagrams) verify that a model predicting 0.7 churn probability is correct roughly 70% of the time. Uncalibrated scores produce misleading risk tiers. All three models are evaluated on calibration before any deployment recommendation is made.

**Logistic Regression as interpretable baseline**
Beyond being a benchmark, LR coefficients provide log-odds interpretability. For stakeholder presentations where "the model said so" is insufficient, LR lets analysts map a coefficient directly to "each additional month of tenure reduces churn odds by X%". This baseline also revealed that a linear decision boundary is surprisingly competitive — ROC-AUC 0.8357 — which itself is an insight worth documenting.

---

## Tech Stack

![pandas](https://img.shields.io/badge/pandas-2.1.4-150458?logo=pandas)
![numpy](https://img.shields.io/badge/numpy-1.26.2-013243?logo=numpy)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11.0-green)
![shap](https://img.shields.io/badge/SHAP-0.44.0-blueviolet)
![matplotlib](https://img.shields.io/badge/matplotlib-3.8.2-blue)
![seaborn](https://img.shields.io/badge/seaborn-0.13.0-blue)
![joblib](https://img.shields.io/badge/joblib-1.3.2-lightgrey)
![pytest](https://img.shields.io/badge/pytest-7.4.4-0A9EDC)

---

## Author

**Matheus Roratos** — Statistics student (5th semester, Brazil)

Hands-on experience in AI automation, Python, and production data pipelines. Building a portfolio of rigorous, business-framed ML projects targeting international data science roles.

**This project portfolio:**
- [`enem-inequality-analysis`](https://github.com/MatheusSRLP/enem-inequality-analysis) — statistical inference and regression on Brazilian public education data
- [`financial-reports-pipeline`](https://github.com/MatheusSRLP/financial-reports-pipeline) — LLM + SQL engineering pipeline for financial report extraction
- [`churn-prediction-pipeline`](https://github.com/MatheusSRLP/churn-prediction-pipeline) — supervised ML, class imbalance, interpretability *(this project)*
