import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline to the raw Telco churn DataFrame.

    Transformations applied in order:
    1. TotalCharges — coerce to float, fill empty strings / NaN with 0.0
    2. SeniorCitizen — map 0/1 → "No"/"Yes" for consistency with other binary columns
    3. Churn — map "Yes"/"No" → 1/0 (numeric target)
    4. customerID — drop (not a feature)
    5. Strip leading/trailing whitespace from all string columns

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by load_raw_data().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    df = df.copy()

    # 1. TotalCharges: coerce empty strings to NaN then fill with 0.0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_coerced = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    logger.info(
        "TotalCharges: coerced %d non-numeric value(s) to 0.0", n_coerced
    )

    # 2. SeniorCitizen: 0/1 → "No"/"Yes"
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    logger.info("SeniorCitizen: mapped 0/1 → No/Yes")

    # 3. Churn: "Yes"/"No" → 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    logger.info("Churn: mapped Yes/No → 1/0")

    # 4. customerID: drop — not a feature
    df = df.drop(columns=["customerID"])
    logger.info("customerID: dropped")

    # 5. Strip whitespace from all object columns
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
    logger.info("Stripped whitespace from %d string column(s)", len(str_cols))

    logger.info("Cleaning complete — final shape: %s", df.shape)
    return df


def validate_clean_data(df: pd.DataFrame) -> None:
    """Assert post-cleaning invariants on the cleaned DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by clean_data().

    Raises
    ------
    AssertionError
        If any NaN values remain, Churn contains values other than 0/1,
        or TotalCharges contains negative values.
    """
    assert df.isnull().sum().sum() == 0, (
        f"NaN values remain after cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}"
    )

    invalid_churn = set(df["Churn"].unique()) - {0, 1}
    assert not invalid_churn, (
        f"Churn column contains unexpected values: {invalid_churn}"
    )

    assert (df["TotalCharges"] >= 0).all(), (
        "TotalCharges contains negative values"
    )
