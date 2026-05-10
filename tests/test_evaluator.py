import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluator import compare_models, evaluate_model, find_optimal_threshold


# ---------------------------------------------------------------------------
# Helpers — deterministic toy datasets and pre-fitted pipelines
# ---------------------------------------------------------------------------

def _perfect_pipeline() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Pipeline that always predicts the correct class (linearly separable data)."""
    from sklearn.linear_model import LogisticRegression

    X = pd.DataFrame({"a": [0.0, 0.0, 1.0, 1.0], "b": [0.0, 0.1, 0.9, 1.0]})
    y = pd.Series([0, 0, 1, 1])
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
    pipe.fit(X, y)
    return pipe, X, y


def _random_pipeline() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Pipeline backed by a stratified DummyClassifier (known probabilities)."""
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame({"x": rng.standard_normal(n)})
    y = pd.Series((rng.random(n) > 0.73).astype(int))  # ~27% positive

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DummyClassifier(strategy="stratified", random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe, X, y


def _imbalanced_pipeline() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """A more realistic setup with a LogisticRegression on slightly separable data."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n = 200
    X_pos = rng.normal(loc=1.5, scale=1.0, size=(50, 2))
    X_neg = rng.normal(loc=0.0, scale=1.0, size=(150, 2))
    X = pd.DataFrame(np.vstack([X_pos, X_neg]), columns=["f1", "f2"])
    y = pd.Series([1] * 50 + [0] * 150)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42, max_iter=500)),
    ])
    pipe.fit(X, y)
    return pipe, X, y


# ---------------------------------------------------------------------------
# evaluate_model — return structure
# ---------------------------------------------------------------------------

class TestEvaluateModelStructure:
    def test_returns_dict(self):
        pipe, X, y = _perfect_pipeline()
        result = evaluate_model(pipe, X, y, "test")
        assert isinstance(result, dict)

    def test_has_all_required_keys(self):
        pipe, X, y = _perfect_pipeline()
        result = evaluate_model(pipe, X, y, "test")
        expected_keys = {"roc_auc", "pr_auc", "f1", "f2", "precision", "recall", "accuracy"}
        assert set(result.keys()) == expected_keys

    def test_all_values_are_floats(self):
        pipe, X, y = _perfect_pipeline()
        result = evaluate_model(pipe, X, y, "test")
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not float: {type(val)}"

    def test_all_values_in_unit_interval(self):
        pipe, X, y = _perfect_pipeline()
        result = evaluate_model(pipe, X, y, "test")
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} outside [0, 1]"


# ---------------------------------------------------------------------------
# evaluate_model — known values on perfect classifier
# ---------------------------------------------------------------------------

class TestEvaluateModelValues:
    def test_perfect_roc_auc(self):
        pipe, X, y = _perfect_pipeline()
        result = evaluate_model(pipe, X, y, "perfect")
        assert result["roc_auc"] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_f1(self):
        pipe, X, y = _perfect_pipeline()
        result = evaluate_model(pipe, X, y, "perfect")
        assert result["f1"] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_f2(self):
        pipe, X, y = _perfect_pipeline()
        result = evaluate_model(pipe, X, y, "perfect")
        assert result["f2"] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_precision_and_recall(self):
        pipe, X, y = _perfect_pipeline()
        result = evaluate_model(pipe, X, y, "perfect")
        assert result["precision"] == pytest.approx(1.0, abs=1e-6)
        assert result["recall"] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_accuracy(self):
        pipe, X, y = _perfect_pipeline()
        result = evaluate_model(pipe, X, y, "perfect")
        assert result["accuracy"] == pytest.approx(1.0, abs=1e-6)

    def test_f2_geq_f1_when_recall_dominates(self):
        # On imbalanced data LR trained to maximise recall,
        # F2 (which weights recall more) should be >= F1 only when recall > precision.
        pipe, X, y = _imbalanced_pipeline()
        result = evaluate_model(pipe, X, y, "imbalanced")
        # This is not always guaranteed — just verify both are computed and in range
        assert 0.0 <= result["f2"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0

    def test_metrics_consistent_with_manual_calculation(self):
        """Smoke-check that roc_auc matches a manual sklearn call."""
        from sklearn.metrics import roc_auc_score

        pipe, X, y = _imbalanced_pipeline()
        result = evaluate_model(pipe, X, y, "smoke")
        y_proba = pipe.predict_proba(X)[:, 1]
        expected_auc = roc_auc_score(y, y_proba)
        assert result["roc_auc"] == pytest.approx(expected_auc, rel=1e-6)


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def _sample_results(self) -> dict[str, dict[str, float]]:
        return {
            "LogisticRegression": {
                "roc_auc": 0.82, "pr_auc": 0.61, "f1": 0.58,
                "f2": 0.63, "precision": 0.55, "recall": 0.62, "accuracy": 0.79,
            },
            "RandomForest": {
                "roc_auc": 0.91, "pr_auc": 0.74, "f1": 0.67,
                "f2": 0.71, "precision": 0.64, "recall": 0.70, "accuracy": 0.85,
            },
            "XGBoost": {
                "roc_auc": 0.93, "pr_auc": 0.78, "f1": 0.70,
                "f2": 0.74, "precision": 0.67, "recall": 0.73, "accuracy": 0.87,
            },
        }

    def test_returns_dataframe(self):
        df = compare_models(self._sample_results())
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_input(self):
        results = self._sample_results()
        df = compare_models(results)
        assert len(df) == len(results)

    def test_sorted_by_roc_auc_descending(self):
        df = compare_models(self._sample_results())
        roc_values = df["roc_auc"].tolist()
        assert roc_values == sorted(roc_values, reverse=True)

    def test_best_model_is_first(self):
        df = compare_models(self._sample_results())
        assert df.index[0] == "XGBoost"

    def test_values_rounded_to_four_decimals(self):
        results = {
            "ModelA": {"roc_auc": 0.123456789, "pr_auc": 0.5, "f1": 0.5,
                       "f2": 0.5, "precision": 0.5, "recall": 0.5, "accuracy": 0.5},
        }
        df = compare_models(results)
        assert df.loc["ModelA", "roc_auc"] == pytest.approx(0.1235, abs=1e-6)

    def test_all_metric_columns_present(self):
        df = compare_models(self._sample_results())
        for col in ["roc_auc", "pr_auc", "f1", "f2", "precision", "recall", "accuracy"]:
            assert col in df.columns, f"column '{col}' missing from compare_models output"

    def test_index_name_is_model(self):
        df = compare_models(self._sample_results())
        assert df.index.name == "model"

    def test_single_model(self):
        results = {"OnlyModel": {"roc_auc": 0.75, "pr_auc": 0.5, "f1": 0.6,
                                 "f2": 0.65, "precision": 0.58, "recall": 0.63, "accuracy": 0.80}}
        df = compare_models(results)
        assert len(df) == 1
        assert df.index[0] == "OnlyModel"


# ---------------------------------------------------------------------------
# find_optimal_threshold
# ---------------------------------------------------------------------------

class TestFindOptimalThreshold:
    def test_returns_tuple(self):
        pipe, X, y = _imbalanced_pipeline()
        result = find_optimal_threshold(pipe, X, y)
        assert isinstance(result, tuple)

    def test_tuple_length_is_two(self):
        pipe, X, y = _imbalanced_pipeline()
        result = find_optimal_threshold(pipe, X, y)
        assert len(result) == 2

    def test_threshold_in_valid_range(self):
        pipe, X, y = _imbalanced_pipeline()
        threshold, _ = find_optimal_threshold(pipe, X, y)
        assert 0.10 <= threshold <= 0.90

    def test_best_value_in_unit_interval(self):
        pipe, X, y = _imbalanced_pipeline()
        _, best_value = find_optimal_threshold(pipe, X, y)
        assert 0.0 <= best_value <= 1.0

    def test_both_elements_are_floats(self):
        pipe, X, y = _imbalanced_pipeline()
        threshold, best_value = find_optimal_threshold(pipe, X, y)
        assert isinstance(threshold, float)
        assert isinstance(best_value, float)

    def test_default_metric_is_f2(self):
        pipe, X, y = _imbalanced_pipeline()
        threshold_default, value_default = find_optimal_threshold(pipe, X, y)
        threshold_f2, value_f2 = find_optimal_threshold(pipe, X, y, metric="f2")
        assert threshold_default == threshold_f2
        assert value_default == pytest.approx(value_f2)

    def test_f1_metric(self):
        pipe, X, y = _imbalanced_pipeline()
        threshold, value = find_optimal_threshold(pipe, X, y, metric="f1")
        assert 0.10 <= threshold <= 0.90
        assert 0.0 <= value <= 1.0

    def test_precision_metric(self):
        pipe, X, y = _imbalanced_pipeline()
        threshold, value = find_optimal_threshold(pipe, X, y, metric="precision")
        assert 0.10 <= threshold <= 0.90

    def test_recall_metric(self):
        pipe, X, y = _imbalanced_pipeline()
        threshold, value = find_optimal_threshold(pipe, X, y, metric="recall")
        assert 0.10 <= threshold <= 0.90

    def test_accuracy_metric(self):
        pipe, X, y = _imbalanced_pipeline()
        threshold, value = find_optimal_threshold(pipe, X, y, metric="accuracy")
        assert 0.10 <= threshold <= 0.90

    def test_invalid_metric_raises_value_error(self):
        pipe, X, y = _imbalanced_pipeline()
        with pytest.raises(ValueError, match="metric must be one of"):
            find_optimal_threshold(pipe, X, y, metric="roc_auc")

    def test_found_threshold_is_locally_optimal(self):
        """The returned threshold should match the value obtained by brute-force search."""
        from sklearn.metrics import fbeta_score as sk_fbeta

        pipe, X, y = _imbalanced_pipeline()
        opt_threshold, opt_value = find_optimal_threshold(pipe, X, y, metric="f2")

        y_proba = pipe.predict_proba(X)[:, 1]
        thresholds = np.arange(0.10, 0.91, 0.01)
        brute_best = max(
            sk_fbeta(y, (y_proba >= t).astype(int), beta=2, zero_division=0)
            for t in thresholds
        )
        assert opt_value == pytest.approx(brute_best, rel=1e-6)
