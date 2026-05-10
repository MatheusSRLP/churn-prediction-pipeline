import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_PATH = os.getenv("DATA_RAW_PATH", "data/raw")

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def load_raw_data() -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV and validate its schema.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with all 21 original columns.

    Raises
    ------
    FileNotFoundError
        If the CSV file is not found at the expected path.
    ValueError
        If any expected column is missing from the loaded file.
    """
    csv_path = Path(RAW_DATA_PATH) / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    logger.info("Loading raw data from %s", csv_path)
    df = pd.read_csv(csv_path)

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    logger.info("Loaded dataset — shape: %s", df.shape)
    logger.info("Column dtypes:\n%s", df.dtypes.to_string())

    return df
