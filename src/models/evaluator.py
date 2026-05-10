import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, float]:
    """Evaluate a fitted Pipeline on a held-out set.

    Predicted probabilities are used for threshold-independent metrics
    (ROC-AUC, PR-AUC). All class-prediction metrics use threshold=0.5.

    Parameters
    ----------
    model : Pipeline
        A fitted scikit-learn Pipeline whose last step exposes predict_proba.
    X_test : pd.DataFrame
        Feature matrix — never fit on, only transformed by the pipeline.
    y_test : pd.Series
        Ground-truth binary labels (0/1).
    model_name : str
        Human-readable name logged with the results.

    Returns
    -------
    dict[str, float]
        Keys: roc_auc, pr_auc, f1, f2, precision, recall, accuracy.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "f2": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }

    logger.info("Evaluation results for %s:", model_name)
    for name, value in metrics.items():
        logger.info("  %-12s %.4f", name, value)

    return metrics


def compare_models(
    results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Build a comparison DataFrame from per-model metrics dicts.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        Mapping of model_name → metrics dict (as returned by evaluate_model).

    Returns
    -------
    pd.DataFrame
        One row per model, columns are metric names, sorted by roc_auc
        descending, all values rounded to 4 decimal places.
    """
    df = pd.DataFrame(results).T
    df.index.name = "model"
    df = df.sort_values("roc_auc", ascending=False)
    df = df.round(4)
    return df


def find_optimal_threshold(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = "f2",
) -> tuple[float, float]:
    """Search for the decision threshold that maximises a given metric.

    Thresholds are tested from 0.10 to 0.90 in steps of 0.01.

    Parameters
    ----------
    model : Pipeline
        A fitted scikit-learn Pipeline.
    X_val : pd.DataFrame
        Validation feature matrix.
    y_val : pd.Series
        Ground-truth binary labels (0/1).
    metric : str
        One of: "f1", "f2", "precision", "recall", "accuracy".
        Defaults to "f2" (weights recall 2× — missing a churner costs more).

    Returns
    -------
    tuple[float, float]
        (optimal_threshold, best_metric_value)

    Raises
    ------
    ValueError
        If metric is not one of the supported options.
    """
    supported = {"f1", "f2", "precision", "recall", "accuracy"}
    if metric not in supported:
        raise ValueError(f"metric must be one of {supported}, got '{metric}'")

    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.01)

    best_threshold = thresholds[0]
    best_value = -np.inf

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if metric == "f1":
            value = f1_score(y_val, y_pred, zero_division=0)
        elif metric == "f2":
            value = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
        elif metric == "precision":
            value = precision_score(y_val, y_pred, zero_division=0)
        elif metric == "recall":
            value = recall_score(y_val, y_pred, zero_division=0)
        else:  # accuracy
            value = accuracy_score(y_val, y_pred)

        if value > best_value:
            best_value = value
            best_threshold = threshold

    logger.info(
        "Optimal threshold for %s: %.2f (%.4f)", metric, best_threshold, best_value
    )
    return float(round(best_threshold, 2)), float(best_value)
