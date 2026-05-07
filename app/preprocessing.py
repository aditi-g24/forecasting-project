"""
preprocessing.py
----------------
Handles all data loading, cleaning, and feature engineering for the
time-series forecasting pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load sales data from an Excel or CSV file."""
    path = Path(filepath)
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    # Rename the sales column
    if "total" in df.columns:
        df.rename(columns={"total": "sales"}, inplace=True)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLEANING & RESAMPLING
# ──────────────────────────────────────────────────────────────────────────────

def clean_and_resample(df: pd.DataFrame, state: str, freq: str = "W") -> pd.DataFrame:
    """
    Filter to a single state, resample to a regular weekly frequency,
    and fill any gaps via linear interpolation.
    """
    state_df = df[df["state"] == state].copy()
    state_df = state_df.sort_values("date").drop_duplicates("date")
    state_df = state_df.set_index("date")[["sales"]]

    # Resample to weekly (Sunday week-end) and sum within the window
    weekly = state_df.resample(freq).sum()

    # Replace zeros that resulted from empty bins with NaN, then interpolate
    weekly["sales"] = weekly["sales"].replace(0, np.nan)
    weekly["sales"] = weekly["sales"].interpolate(method="linear").ffill().bfill()

    weekly.reset_index(inplace=True)
    weekly.rename(columns={"date": "ds", "sales": "y"}, inplace=True)
    return weekly


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and lag features required by ML models (XGBoost / LSTM).

    Features added
    --------------
    lag_1, lag_4, lag_8, lag_13    : recent + quarterly lags (weekly data)
    rolling_mean_4, rolling_mean_13: 4-week and 13-week rolling averages
    rolling_std_4                  : short-term volatility
    week_of_year                   : seasonality signal (1-52)
    month                          : month number (1-12)
    quarter                        : quarter (1-4)
    year                           : calendar year
    trend                          : simple integer index
    """
    df = df.copy().sort_values("ds").reset_index(drop=True)
    y = df["y"]

    df["lag_1"]  = y.shift(1)
    df["lag_4"]  = y.shift(4)
    df["lag_8"]  = y.shift(8)
    df["lag_13"] = y.shift(13)

    df["rolling_mean_4"]  = y.shift(1).rolling(4).mean()
    df["rolling_mean_13"] = y.shift(1).rolling(13).mean()
    df["rolling_std_4"]   = y.shift(1).rolling(4).std()

    df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)
    df["month"]        = df["ds"].dt.month
    df["quarter"]      = df["ds"].dt.quarter
    df["year"]         = df["ds"].dt.year
    df["trend"]        = np.arange(len(df))

    # Drop rows that have NaN lags (first ~13 rows)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


FEATURE_COLS = [
    "lag_1", "lag_4", "lag_8", "lag_13",
    "rolling_mean_4", "rolling_mean_13", "rolling_std_4",
    "week_of_year", "month", "quarter", "year", "trend",
]


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def train_test_split_ts(df: pd.DataFrame, test_weeks: int = 8):
    """
    Chronological split — last `test_weeks` rows become the test set.
    No shuffling; time order is preserved to avoid data leakage.
    """
    split = len(df) - test_weeks
    train = df.iloc[:split].copy()
    test  = df.iloc[split:].copy()
    return train, test


# ──────────────────────────────────────────────────────────────────────────────
# FUTURE DATE GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def make_future_dates(last_date: pd.Timestamp, weeks: int = 8) -> pd.DataFrame:
    """Generate a DataFrame of `weeks` future weekly dates starting after last_date."""
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                 periods=weeks, freq="W")
    return pd.DataFrame({"ds": future_dates})
