import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Bins and labels for tenure_group — right-inclusive, left-exclusive
# [0, 12] [13, 24] [25, 48] [49, 60] [61, inf)
_TENURE_BINS = [0, 12, 24, 48, 60, np.inf]
_TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49-60m", "61m+"]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two Series element-wise; returns 0.0 where denominator is 0."""
    return np.where(denominator == 0, 0.0, numerator / denominator)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features on top of the cleaned DataFrame.

    Transformations applied in order:
    1. tenure_group — categorical bin of tenure in months
    2. total_services — count of active services per customer (0–8)
    3. charges_per_month — TotalCharges / tenure (0.0 when tenure == 0)
    4. is_high_value — 1 if MonthlyCharges > 75th percentile, else 0

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame as returned by clean_data().

    Returns
    -------
    pd.DataFrame
        DataFrame with four additional columns; original columns preserved.
    """
    df = df.copy()

    # 1. tenure_group
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=_TENURE_BINS,
        labels=_TENURE_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)
    logger.info("tenure_group: created with bins %s", _TENURE_BINS)

    # 2. total_services — map any "Yes" to 1, everything else to 0
    service_flags = df[SERVICE_COLUMNS].apply(
        lambda col: col.map(lambda v: 1 if v == "Yes" else 0)
    )
    df["total_services"] = service_flags.sum(axis=1)
    logger.info("total_services: summed %d service columns", len(SERVICE_COLUMNS))

    # 3. charges_per_month — safe divide avoids NaN when tenure == 0
    df["charges_per_month"] = _safe_divide(df["TotalCharges"], df["tenure"])
    logger.info("charges_per_month: computed TotalCharges / tenure (0.0 for tenure=0)")

    # 4. is_high_value — threshold derived from the received DataFrame only
    threshold = df["MonthlyCharges"].quantile(0.75)
    df["is_high_value"] = (df["MonthlyCharges"] > threshold).astype(int)
    logger.info(
        "is_high_value: MonthlyCharges > %.2f (75th pct) → %d high-value customers",
        threshold,
        df["is_high_value"].sum(),
    )

    logger.info("Feature engineering complete — shape: %s", df.shape)
    return df


def get_feature_columns() -> dict[str, list[str]]:
    """Return the column classification used to build the preprocessing Pipeline.

    Returns
    -------
    dict with keys:
        "numerical"   — columns to pass through StandardScaler
        "binary"      — already-encoded 0/1 columns (pass-through)
        "categorical" — columns to pass through OneHotEncoder (drop='first')
    """
    numerical = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "total_services",
        "charges_per_month",
    ]

    # Binary string columns (Yes/No) — handled by OneHotEncoder with drop='first',
    # which effectively produces a single 0/1 column per binary feature.
    # SeniorCitizen was mapped to "No"/"Yes" strings in cleaner.py, so it
    # belongs here alongside the other categorical columns.
    categorical = [
        # Demographics
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        # Services
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        # Account
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        # Derived
        "tenure_group",
    ]

    binary = [
        "is_high_value",
    ]

    return {
        "numerical": numerical,
        "binary": binary,
        "categorical": categorical,
    }
