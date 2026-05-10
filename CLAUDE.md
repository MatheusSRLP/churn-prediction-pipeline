# CLAUDE.md — Customer Churn Prediction Pipeline

## Project Context

Production-grade ML portfolio project demonstrating end-to-end binary classification:
data ingestion → exploratory analysis → feature engineering → model progression
(Logistic Regression → Random Forest → XGBoost) → interpretability (SHAP) →
calibration analysis → business recommendations.

**Developer:** Statistics student (5th semester, Brazil) with hands-on experience in
AI automation, Python, and real data pipelines. This is the third project in a
portfolio targeting international data science roles.

**Target audience:** Technical recruiters and senior ML engineers at international
companies. They expect: rigorous model evaluation, justified decisions, business
framing of technical results, and production-thinking code.

**Portfolio context:** Complements two existing projects —
- `enem-inequality-analysis`: statistical inference and regression on public data
- `financial-reports-pipeline`: LLM + SQL engineering pipeline

This project adds: supervised ML, class imbalance handling, hyperparameter tuning,
model interpretability, and calibration — the core ML engineering skillset.

---

## Problem Statement

> Acquiring a new customer costs 5–7x more than retaining an existing one.
> For subscription-based service businesses (aesthetic clinics, gyms, SaaS),
> identifying customers likely to churn before they leave enables targeted
> retention actions — discounts, re-engagement campaigns, personalized outreach.
> This project builds a binary classification pipeline to predict customer churn
> with calibrated probabilities, enabling risk-based prioritization of retention efforts.

**Business question:**
> Which customers are most likely to churn in the next billing cycle,
> and what are the primary drivers of their churn risk?

---

## Dataset

**Source:** IBM Telco Customer Churn — available on Kaggle
**URL:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
**Size:** 7,043 customers × 21 features
**Target:** `Churn` (Yes/No) — ~26.5% positive rate (imbalanced)

**Feature groups:**
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Services: PhoneService, MultipleLines, InternetService, OnlineSecurity,
  OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- Account: Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, tenure

**Business reframing for README:**
The dataset represents subscription service customers. The findings apply directly
to any recurring-revenue business — aesthetic clinics with treatment packages,
gyms, SaaS products. README will frame insights in this broader context.

---

## Architecture

```
churn-prediction-pipeline/
├── data/
│   ├── raw/                          # original CSV — never modified, never committed
│   └── processed/                    # cleaned and engineered features
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # data loading and validation
│   │   └── cleaner.py                # cleaning pipeline and type coercion
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py            # feature engineering and encoding
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py                # training pipeline for all models
│   │   ├── evaluator.py              # metrics, curves, calibration
│   │   └── tuner.py                  # hyperparameter search
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                  # all plot functions (EDA + model)
├── notebooks/
│   ├── 01_eda.ipynb                  # exploratory analysis and churn profiling
│   ├── 02_feature_engineering.ipynb  # encoding, scaling, imbalance handling
│   ├── 03_model_comparison.ipynb     # LR → RF → XGBoost with evaluation
│   └── 04_interpretability.ipynb     # SHAP values + calibration curves
├── outputs/
│   ├── figures/                      # exported charts
│   │   └── .gitkeep
│   ├── models/                       # serialized trained models
│   │   └── .gitkeep
│   └── reports/                      # metrics tables and CSVs
│       └── .gitkeep
├── docs/
│   └── figures/                      # charts for README display
├── tests/
│   ├── __init__.py
│   ├── test_cleaner.py
│   ├── test_engineering.py
│   └── test_evaluator.py
├── .gitignore
├── CLAUDE.md
├── README.md
└── requirements.txt
```

---

## Stack

```
# Core ML
scikit-learn==1.3.2
xgboost==2.0.3
imbalanced-learn==0.11.0      # SMOTE for class imbalance

# Interpretability
shap==0.44.0

# Data
pandas==2.1.4
numpy==1.26.2

# Visualization
matplotlib==3.8.2
seaborn==0.13.0

# Persistence
joblib==1.3.2                  # model serialization

# Dev & Testing
pytest==7.4.4
pytest-cov==4.1.0
```

**Dependency philosophy:** no AutoML, no heavy frameworks. Every modeling decision
is explicit and justified. scikit-learn Pipeline for reproducibility — no leakage.

---

## Modeling Strategy

### Baseline → Complexity progression
| Model | Reason |
|---|---|
| Logistic Regression | Interpretable baseline; coefficients as log-odds |
| Random Forest | Non-linear ensemble; robust to outliers; natural feature importance |
| XGBoost | State-of-the-art gradient boosting; handles imbalance via `scale_pos_weight` |

### Class imbalance handling
- **SMOTE** applied only on training set — never on validation or test
- `scale_pos_weight` in XGBoost = negative/positive ratio
- Evaluation prioritizes **ROC-AUC** and **PR-AUC** over accuracy (accuracy is misleading on imbalanced data)

### Evaluation metrics (all models)
- ROC-AUC
- PR-AUC (Precision-Recall — more informative for imbalanced classes)
- F1-score at threshold 0.5
- F2-score (weights recall 2× — missing a churner is costlier than a false alarm)
- Confusion matrix
- Calibration curve (reliability diagram)

### No data leakage rule
All preprocessing (scaling, encoding, SMOTE) lives inside a scikit-learn `Pipeline`.
The pipeline is fit **only on training data**. Validation and test sets are transformed
using the fitted pipeline — never fit on them.

---

## Feature Engineering

### Derived features to create
```python
# tenure groups
tenure_group: "0-12m", "12-24m", "24-48m", "48-60m", "60m+"

# service count
total_services: sum of all binary service columns (0–8)

# charges ratio
charges_per_month: TotalCharges / tenure (average monthly spend)

# high value flag
is_high_value: MonthlyCharges > 75th percentile
```

### Encoding strategy
- Binary categoricals → LabelEncoder (Yes/No, Male/Female)
- Multi-class categoricals → OneHotEncoder (drop='first' to avoid dummy trap)
- Numericals → StandardScaler inside Pipeline

### What NOT to do
- Never encode target variable `Churn` before train/test split
- Never apply SMOTE before splitting
- Never use TotalCharges without handling the empty string edge case
  (7 customers with tenure=0 have TotalCharges=" " — must coerce to 0.0)

---

## Code Standards (NON-NEGOTIABLE)

### scikit-learn Pipeline for everything
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=42, max_iter=1000))
])
# fit only on X_train — Pipeline handles transform of X_val/X_test
pipeline.fit(X_train, y_train)
```

### random_state=42 everywhere stochastic
```python
train_test_split(..., random_state=42)
RandomForestClassifier(..., random_state=42)
XGBClassifier(..., random_state=42)
SMOTE(..., random_state=42)
```

### Type hints on every public function
```python
def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, float]:
```

### Metrics returned as dict, never printed inline
```python
# CORRECT
def evaluate_model(...) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "f2": fbeta_score(y_test, y_pred, beta=2),
    }

# NEVER
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

### Model persistence with metadata
```python
import joblib
from datetime import datetime

model_artifact = {
    "model": pipeline,
    "feature_names": feature_names,
    "trained_at": datetime.utcnow().isoformat(),
    "metrics": metrics_dict,
    "dataset_shape": X_train.shape,
}
joblib.dump(model_artifact, "outputs/models/xgboost_v1.joblib")
```

### Logging, never print()
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Training %s on %d samples...", model_name, len(X_train))
```

---

## Visualization Standards

### Global style
```python
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
sns.set_palette("colorblind")
```

### Required plots (each as a separate function in plots.py)

**EDA:**
- `plot_churn_distribution()` — pie + count with percentage labels
- `plot_churn_by_contract()` — grouped bar: churn rate by contract type
- `plot_tenure_distribution()` — histogram by churn label (overlapping, alpha=0.6)
- `plot_charges_boxplot()` — MonthlyCharges distribution by churn label
- `plot_correlation_heatmap()` — numerical features only, annotated

**Model evaluation:**
- `plot_roc_curves()` — all 3 models on same axes, AUC in legend
- `plot_pr_curves()` — all 3 models, AP in legend
- `plot_confusion_matrix()` — one per model, normalized
- `plot_calibration_curves()` — reliability diagram for all 3 models
- `plot_threshold_analysis()` — F1, F2, Precision, Recall vs threshold for best model

**Interpretability:**
- `plot_shap_summary()` — beeswarm plot (global feature importance)
- `plot_shap_waterfall()` — single prediction explanation (highest-risk customer)
- `plot_shap_dependence()` — tenure × churn probability interaction

---

## Notebooks Narrative

### 01_eda.ipynb
1. Dataset overview (shape, dtypes, missing values)
2. Target distribution — churn rate with business context
3. Churn by contract type — the most predictive single feature
4. Tenure analysis — survival intuition
5. Monthly charges distribution by churn label
6. Correlation heatmap of numerical features
7. **Key findings:** 3–5 insights in markdown with numbers

### 02_feature_engineering.ipynb
1. TotalCharges edge case — coerce empty strings to 0.0
2. Derived features: tenure_group, total_services, charges_per_month
3. Encoding pipeline construction
4. Train/test split (80/20, stratified on target)
5. Class imbalance visualization before/after SMOTE
6. Final feature matrix shape and dtypes

### 03_model_comparison.ipynb
1. Logistic Regression — fit, evaluate, coefficients as business drivers
2. Random Forest — fit, evaluate, feature importance bar chart
3. XGBoost — fit with scale_pos_weight, evaluate
4. Side-by-side metrics table (all models)
5. ROC curves on same plot
6. PR curves on same plot
7. **Conclusion:** which model to deploy and why

### 04_interpretability.ipynb
1. SHAP explainer for best model (XGBoost)
2. Global: beeswarm summary plot
3. Global: mean |SHAP| bar chart (feature ranking)
4. Local: waterfall plot for highest-risk customer
5. Dependence plot: tenure × churn probability
6. Calibration curves — are predicted probabilities reliable?
7. Threshold analysis — optimal threshold for F2 maximization
8. **Business recommendations:** top 3 actionable insights from SHAP

---

## Git Convention

### Branch strategy
```
main              — always stable
feat/data
feat/features
feat/models
feat/interpretability
```

### Commit messages
```
feat: add data loader with TotalCharges edge case handling
feat: implement feature engineering pipeline with derived features
feat: train logistic regression baseline with full evaluation metrics
feat: add random forest with feature importance visualization
feat: add XGBoost with scale_pos_weight for class imbalance
feat: implement SHAP interpretability and calibration analysis
docs: write README with business framing and model results
test: add unit tests for cleaner and evaluator
```

### Never commit
```gitignore
data/raw/
*.csv
outputs/models/
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
outputs/figures/*
outputs/reports/*
!outputs/figures/.gitkeep
!outputs/reports/.gitkeep
!outputs/models/.gitkeep
```

---

## README Structure (English, public-facing)

1. Status badge + Python badge
2. Title + one-line description
3. **Business problem** — cost of churn, retention ROI
4. **Approach** — model progression rationale
5. **Key results** — ROC-AUC, PR-AUC, top SHAP features (real numbers after training)
6. **Selected visualizations** — ROC curves + SHAP summary inline
7. **Business recommendations** — 3 actionable insights from interpretability
8. **How to run** — step by step with real commands
9. **Architecture** — directory tree with explanations
10. **Design decisions** — Pipeline to prevent leakage, SMOTE only on train,
    F2 over F1, calibration rationale
11. **Tech stack** — badges
12. **Author** — LinkedIn + links to other two portfolio projects

---

## How to Work With This Assistant

### Each session, tell me:
- Which module you are building
- Any conceptual doubt before writing code
- Paste code for review before committing

### I will always:
- Reject any preprocessing applied before train/test split
- Reject accuracy as the primary metric on imbalanced data
- Reject models saved without metadata
- Reject SHAP analysis without business interpretation
- Question any hyperparameter choice that isn't justified

### I will never:
- Let data leakage pass in a Pipeline
- Accept a model comparison without PR-AUC alongside ROC-AUC
- Generate code "that works but nobody understands"
- Skip calibration analysis before recommending deployment
