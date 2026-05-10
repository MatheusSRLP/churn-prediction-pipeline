import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def build_logistic_regression() -> Pipeline:
    """Return an unfitted Pipeline: StandardScaler → LogisticRegression.

    class_weight="balanced" compensates for the ~26.5% positive rate without
    requiring SMOTE — a simpler baseline that still handles imbalance.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced",
        )),
    ])


def build_random_forest() -> Pipeline:
    """Return an unfitted Pipeline: StandardScaler → RandomForestClassifier.

    n_estimators=200 for stable feature importances.
    class_weight="balanced" mirrors the logistic regression approach.
    n_jobs=-1 uses all available CPU cores.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])


def build_xgboost(scale_pos_weight: float) -> Pipeline:
    """Return an unfitted Pipeline: StandardScaler → XGBClassifier.

    Parameters
    ----------
    scale_pos_weight : float
        Ratio count(negative) / count(positive). Tells XGBoost how much
        more to penalise false negatives — the XGBoost-native imbalance strategy.
        Caller must compute this from the training set only.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )),
    ])


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, Pipeline]:
    """Train Logistic Regression, Random Forest and XGBoost on the training set.

    scale_pos_weight for XGBoost is derived exclusively from y_train to prevent
    any leakage from the validation set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (post-encoding, pre-scaling — Pipeline handles scaling).
    y_train : pd.Series
        Training labels (0/1).
    X_val : pd.DataFrame
        Validation feature matrix — never used for fitting.
    y_val : pd.Series
        Validation labels — never used for fitting.

    Returns
    -------
    dict[str, Pipeline]
        Keys: "logistic_regression", "random_forest", "xgboost".
        Each value is a fitted Pipeline.
    """
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    logger.info(
        "Training set: %d samples (%d positive, %d negative) — scale_pos_weight=%.4f",
        len(y_train),
        (y_train == 1).sum(),
        (y_train == 0).sum(),
        scale_pos_weight,
    )

    specs = [
        ("logistic_regression", build_logistic_regression()),
        ("random_forest", build_random_forest()),
        ("xgboost", build_xgboost(scale_pos_weight)),
    ]

    trained: dict[str, Pipeline] = {}
    for name, pipeline in specs:
        logger.info("Training %s on %d samples...", name, len(X_train))
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        logger.info("  %s trained in %.2fs", name, elapsed)
        trained[name] = pipeline

    return trained


def save_model(
    model: Pipeline,
    model_name: str,
    metrics: dict[str, float],
    feature_names: list[str],
    output_dir: Path = Path("outputs/models"),
) -> Path:
    """Persist a fitted Pipeline with full metadata using joblib.

    The artifact is a plain dict so it can be inspected without re-importing
    the trainer module:
        artifact = joblib.load(path)
        pipeline = artifact["model"]

    Parameters
    ----------
    model : Pipeline
        A fitted scikit-learn Pipeline.
    model_name : str
        Used as the filename stem (spaces replaced with underscores).
    metrics : dict[str, float]
        Evaluation metrics to embed in the artifact.
    feature_names : list[str]
        Ordered list of column names the model was trained on.
    output_dir : Path
        Directory for the .joblib file. Created if it does not exist.

    Returns
    -------
    Path
        Absolute path to the saved artifact.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = model_name.replace(" ", "_") + ".joblib"
    output_path = output_dir / filename

    artifact = {
        "model": model,
        "model_name": model_name,
        "feature_names": feature_names,
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "metrics": metrics,
        "n_features": len(feature_names),
    }

    joblib.dump(artifact, output_path)
    logger.info("Model artifact saved to %s", output_path)
    return output_path.resolve()
