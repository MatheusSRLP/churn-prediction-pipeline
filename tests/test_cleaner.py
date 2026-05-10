import pandas as pd
import pytest

from src.data.cleaner import clean_data, validate_clean_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_df(**overrides) -> pd.DataFrame:
    """Return a minimal valid raw DataFrame (2 rows) for testing."""
    base = {
        "customerID": ["1111-AAAAA", "2222-BBBBB"],
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "tenure": [1, 34],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No phone service", "No"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "No"],
        "StreamingTV": ["No", "No"],
        "StreamingMovies": ["No", "No"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95],
        "TotalCharges": ["29.85", "1889.5"],
        "Churn": ["No", "Yes"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# TotalCharges edge case
# ---------------------------------------------------------------------------

class TestTotalChargesCoercion:
    def test_empty_string_becomes_zero(self):
        df_raw = _make_raw_df(TotalCharges=[" ", "1889.5"])
        df = clean_data(df_raw)
        assert df["TotalCharges"].iloc[0] == 0.0

    def test_numeric_string_parsed_correctly(self):
        df_raw = _make_raw_df(TotalCharges=["29.85", "1889.5"])
        df = clean_data(df_raw)
        assert df["TotalCharges"].iloc[0] == pytest.approx(29.85)
        assert df["TotalCharges"].iloc[1] == pytest.approx(1889.5)

    def test_no_nans_after_coercion(self):
        df_raw = _make_raw_df(TotalCharges=[" ", " "])
        df = clean_data(df_raw)
        assert df["TotalCharges"].isna().sum() == 0

    def test_dtype_is_float(self):
        df_raw = _make_raw_df(TotalCharges=["29.85", "1889.5"])
        df = clean_data(df_raw)
        assert pd.api.types.is_float_dtype(df["TotalCharges"])


# ---------------------------------------------------------------------------
# SeniorCitizen conversion
# ---------------------------------------------------------------------------

class TestSeniorCitizenConversion:
    def test_zero_becomes_no(self):
        df_raw = _make_raw_df(SeniorCitizen=[0, 0])
        df = clean_data(df_raw)
        assert (df["SeniorCitizen"] == "No").all()

    def test_one_becomes_yes(self):
        df_raw = _make_raw_df(SeniorCitizen=[1, 1])
        df = clean_data(df_raw)
        assert (df["SeniorCitizen"] == "Yes").all()

    def test_mixed_mapping(self):
        df_raw = _make_raw_df(SeniorCitizen=[0, 1])
        df = clean_data(df_raw)
        assert df["SeniorCitizen"].tolist() == ["No", "Yes"]


# ---------------------------------------------------------------------------
# Churn encoding
# ---------------------------------------------------------------------------

class TestChurnEncoding:
    def test_yes_becomes_one(self):
        df_raw = _make_raw_df(Churn=["Yes", "Yes"])
        df = clean_data(df_raw)
        assert (df["Churn"] == 1).all()

    def test_no_becomes_zero(self):
        df_raw = _make_raw_df(Churn=["No", "No"])
        df = clean_data(df_raw)
        assert (df["Churn"] == 0).all()

    def test_mixed_encoding(self):
        df_raw = _make_raw_df(Churn=["No", "Yes"])
        df = clean_data(df_raw)
        assert df["Churn"].tolist() == [0, 1]

    def test_churn_dtype_is_integer(self):
        df_raw = _make_raw_df(Churn=["No", "Yes"])
        df = clean_data(df_raw)
        assert pd.api.types.is_integer_dtype(df["Churn"])


# ---------------------------------------------------------------------------
# customerID removal
# ---------------------------------------------------------------------------

class TestCustomerIDRemoval:
    def test_customer_id_not_in_columns(self):
        df_raw = _make_raw_df()
        df = clean_data(df_raw)
        assert "customerID" not in df.columns

    def test_column_count_reduced_by_one(self):
        df_raw = _make_raw_df()
        df = clean_data(df_raw)
        assert len(df.columns) == len(df_raw.columns) - 1


# ---------------------------------------------------------------------------
# validate_clean_data
# ---------------------------------------------------------------------------

class TestValidateCleanData:
    def test_passes_on_valid_data(self):
        df = clean_data(_make_raw_df())
        validate_clean_data(df)  # should not raise

    def test_raises_on_nan(self):
        df = clean_data(_make_raw_df())
        df.loc[0, "tenure"] = float("nan")
        with pytest.raises(AssertionError, match="NaN"):
            validate_clean_data(df)

    def test_raises_on_invalid_churn_value(self):
        df = clean_data(_make_raw_df())
        df.loc[0, "Churn"] = 99
        with pytest.raises(AssertionError, match="Churn"):
            validate_clean_data(df)

    def test_raises_on_negative_total_charges(self):
        df = clean_data(_make_raw_df())
        df.loc[0, "TotalCharges"] = -1.0
        with pytest.raises(AssertionError, match="TotalCharges"):
            validate_clean_data(df)
