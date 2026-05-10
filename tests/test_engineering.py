import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    SERVICE_COLUMNS,
    _safe_divide,
    engineer_features,
    get_feature_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_clean_df(**overrides) -> pd.DataFrame:
    """Return a minimal cleaned DataFrame (6 rows) with known values."""
    base = {
        "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
        "SeniorCitizen": ["No", "Yes", "No", "No", "Yes", "No"],
        "Partner": ["Yes", "No", "Yes", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "No", "No", "Yes"],
        # tenure spans all 5 bins: 0-12, 13-24, 25-48, 49-60, 61+
        "tenure": [0, 6, 18, 36, 55, 72],
        "PhoneService": ["No", "Yes", "Yes", "Yes", "No", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes", "No", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No", "Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "No", "Yes", "No internet service", "No", "Yes"],
        "OnlineBackup": ["Yes", "No", "No", "No internet service", "Yes", "No"],
        "DeviceProtection": ["No", "Yes", "No", "No internet service", "No", "Yes"],
        "TechSupport": ["No", "No", "No", "No internet service", "No", "No"],
        "StreamingTV": ["No", "Yes", "No", "No internet service", "Yes", "No"],
        "StreamingMovies": ["No", "Yes", "No", "No internet service", "Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "No"],
        "PaymentMethod": [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
            "Electronic check", "Mailed check",
        ],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 99.65],
        "TotalCharges": [0.0, 341.70, 969.30, 1519.80, 3889.35, 7174.80],
        "Churn": [0, 0, 0, 0, 1, 1],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# tenure_group
# ---------------------------------------------------------------------------

class TestTenureGroup:
    def test_column_exists(self):
        df = engineer_features(_make_clean_df())
        assert "tenure_group" in df.columns

    def test_tenure_zero_maps_to_first_bin(self):
        df = engineer_features(_make_clean_df())
        # row 0: tenure=0 → "0-12m"
        assert df.loc[0, "tenure_group"] == "0-12m"

    def test_all_five_bins_represented(self):
        # tenure values [0, 6, 18, 36, 55, 72] cover all 5 labels
        df = engineer_features(_make_clean_df())
        expected = {"0-12m", "13-24m", "25-48m", "49-60m", "61m+"}
        assert set(df["tenure_group"]) == expected

    def test_known_bin_assignments(self):
        # tenure: 0→0-12m, 6→0-12m, 18→13-24m, 36→25-48m, 55→49-60m, 72→61m+
        df = engineer_features(_make_clean_df())
        expected = ["0-12m", "0-12m", "13-24m", "25-48m", "49-60m", "61m+"]
        assert df["tenure_group"].tolist() == expected

    def test_dtype_is_string(self):
        df = engineer_features(_make_clean_df())
        # pandas 2.1+ may return StringDtype instead of object — both are valid string dtypes
        assert pd.api.types.is_string_dtype(df["tenure_group"])


# ---------------------------------------------------------------------------
# total_services
# ---------------------------------------------------------------------------

class TestTotalServices:
    def test_column_exists(self):
        df = engineer_features(_make_clean_df())
        assert "total_services" in df.columns

    def test_range_zero_to_eight(self):
        df = engineer_features(_make_clean_df())
        assert df["total_services"].between(0, 8).all()

    def test_no_service_customer_scores_zero(self):
        # A customer with PhoneService=No and all internet services = No internet service
        row = {
            "PhoneService": ["No"],
            "MultipleLines": ["No phone service"],
            "OnlineSecurity": ["No internet service"],
            "OnlineBackup": ["No internet service"],
            "DeviceProtection": ["No internet service"],
            "TechSupport": ["No internet service"],
            "StreamingTV": ["No internet service"],
            "StreamingMovies": ["No internet service"],
        }
        df_row = _make_clean_df(**{k: v * 6 for k, v in row.items()})
        df = engineer_features(df_row)
        assert (df["total_services"] == 0).all()

    def test_all_service_customer_scores_eight(self):
        row = {
            "PhoneService": ["Yes"],
            "MultipleLines": ["Yes"],
            "OnlineSecurity": ["Yes"],
            "OnlineBackup": ["Yes"],
            "DeviceProtection": ["Yes"],
            "TechSupport": ["Yes"],
            "StreamingTV": ["Yes"],
            "StreamingMovies": ["Yes"],
        }
        df_row = _make_clean_df(**{k: v * 6 for k, v in row.items()})
        df = engineer_features(df_row)
        assert (df["total_services"] == 8).all()

    def test_known_values(self):
        df = engineer_features(_make_clean_df())
        # row 0: PhoneService=No, MultipleLines=No phone service, OnlineBackup=Yes → 1 service
        assert df.loc[0, "total_services"] == 1
        # row 1: PhoneService=Yes, StreamingTV=Yes, StreamingMovies=Yes,
        #         DeviceProtection=Yes → 4 services
        assert df.loc[1, "total_services"] == 4


# ---------------------------------------------------------------------------
# charges_per_month
# ---------------------------------------------------------------------------

class TestChargesPerMonth:
    def test_column_exists(self):
        df = engineer_features(_make_clean_df())
        assert "charges_per_month" in df.columns

    def test_normal_calculation(self):
        # row 1: TotalCharges=341.70, tenure=6 → 56.95
        df = engineer_features(_make_clean_df())
        assert df.loc[1, "charges_per_month"] == pytest.approx(341.70 / 6, rel=1e-4)

    def test_no_nans(self):
        df = engineer_features(_make_clean_df())
        assert df["charges_per_month"].isna().sum() == 0

    def test_no_infinities(self):
        df = engineer_features(_make_clean_df())
        assert np.isinf(df["charges_per_month"]).sum() == 0


# ---------------------------------------------------------------------------
# _safe_divide
# ---------------------------------------------------------------------------

class TestSafeDivide:
    def test_normal_division(self):
        num = pd.Series([100.0, 200.0])
        den = pd.Series([10.0, 20.0])
        result = _safe_divide(num, den)
        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(10.0)

    def test_zero_denominator_returns_zero(self):
        num = pd.Series([99.0, 50.0])
        den = pd.Series([0.0, 5.0])
        result = _safe_divide(num, den)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(10.0)

    def test_all_zero_denominators(self):
        num = pd.Series([1.0, 2.0, 3.0])
        den = pd.Series([0.0, 0.0, 0.0])
        result = _safe_divide(num, den)
        assert (result == 0.0).all()

    def test_no_nan_or_inf_produced(self):
        num = pd.Series([0.0, 50.0, 100.0])
        den = pd.Series([0.0, 0.0, 5.0])
        result = _safe_divide(num, den)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()


# ---------------------------------------------------------------------------
# is_high_value
# ---------------------------------------------------------------------------

class TestIsHighValue:
    def test_column_exists(self):
        df = engineer_features(_make_clean_df())
        assert "is_high_value" in df.columns

    def test_dtype_is_integer(self):
        df = engineer_features(_make_clean_df())
        assert pd.api.types.is_integer_dtype(df["is_high_value"])

    def test_values_only_zero_or_one(self):
        df = engineer_features(_make_clean_df())
        assert set(df["is_high_value"].unique()).issubset({0, 1})

    def test_threshold_is_75th_percentile(self):
        df_raw = _make_clean_df()
        df = engineer_features(df_raw)
        threshold = df_raw["MonthlyCharges"].quantile(0.75)
        expected = (df_raw["MonthlyCharges"] > threshold).astype(int)
        pd.testing.assert_series_equal(
            df["is_high_value"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_highest_charge_is_high_value(self):
        df = engineer_features(_make_clean_df())
        # row 5: MonthlyCharges=99.65 — highest in fixture, must be 1
        assert df.loc[5, "is_high_value"] == 1

    def test_lowest_charge_is_not_high_value(self):
        df = engineer_features(_make_clean_df())
        # row 0: MonthlyCharges=29.85 — lowest in fixture, must be 0
        assert df.loc[0, "is_high_value"] == 0


# ---------------------------------------------------------------------------
# get_feature_columns
# ---------------------------------------------------------------------------

class TestGetFeatureColumns:
    def test_returns_three_keys(self):
        cols = get_feature_columns()
        assert set(cols.keys()) == {"numerical", "binary", "categorical"}

    def test_numerical_contains_core_features(self):
        cols = get_feature_columns()
        for feature in ["tenure", "MonthlyCharges", "TotalCharges", "total_services", "charges_per_month"]:
            assert feature in cols["numerical"], f"{feature} missing from numerical"

    def test_categorical_contains_all_service_columns(self):
        cols = get_feature_columns()
        for svc in SERVICE_COLUMNS:
            assert svc in cols["categorical"], f"{svc} missing from categorical"

    def test_categorical_contains_account_features(self):
        cols = get_feature_columns()
        for feature in ["Contract", "PaymentMethod", "PaperlessBilling"]:
            assert feature in cols["categorical"], f"{feature} missing from categorical"

    def test_categorical_contains_tenure_group(self):
        cols = get_feature_columns()
        assert "tenure_group" in cols["categorical"]

    def test_binary_contains_is_high_value(self):
        cols = get_feature_columns()
        assert "is_high_value" in cols["binary"]

    def test_no_overlap_between_groups(self):
        cols = get_feature_columns()
        num = set(cols["numerical"])
        bin_ = set(cols["binary"])
        cat = set(cols["categorical"])
        assert num.isdisjoint(bin_), f"numerical ∩ binary: {num & bin_}"
        assert num.isdisjoint(cat), f"numerical ∩ categorical: {num & cat}"
        assert bin_.isdisjoint(cat), f"binary ∩ categorical: {bin_ & cat}"

    def test_churn_not_in_any_group(self):
        cols = get_feature_columns()
        all_features = (
            set(cols["numerical"]) | set(cols["binary"]) | set(cols["categorical"])
        )
        assert "Churn" not in all_features, "Churn (target) must not appear in feature columns"
