from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_NORMALIZED_COLUMNS = (
    "timestamp",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
OPTIONAL_NORMALIZED_COLUMNS = ("tick_volume", "spread")
NUMERIC_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    *OPTIONAL_NORMALIZED_COLUMNS,
)


def validate_normalized_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return GoldMind's canonical, provider-neutral OHLCV schema.

    The dataset must represent one symbol and timeframe, have unique
    chronological UTC timestamps, finite numeric values, valid OHLC geometry,
    and non-negative volume/spread fields. Unexpected provider-specific
    columns are rejected so they cannot silently leak into stored research data.
    """
    if frame.empty:
        raise ValueError("market-data frame is empty")

    result = frame.copy()
    normalized_columns = [str(column).strip().lower() for column in result.columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError("duplicate column names after normalization")
    result.columns = normalized_columns

    allowed = set(REQUIRED_NORMALIZED_COLUMNS + OPTIONAL_NORMALIZED_COLUMNS)
    unexpected = sorted(set(result.columns) - allowed)
    if unexpected:
        raise ValueError(f"unexpected normalized market-data columns: {unexpected}")

    missing = sorted(set(REQUIRED_NORMALIZED_COLUMNS) - set(result.columns))
    if missing:
        raise ValueError(f"normalized market-data missing required columns: {missing}")

    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if result["timestamp"].isna().any():
        raise ValueError("invalid timestamp detected")
    if result["timestamp"].duplicated().any():
        duplicates = result.loc[result["timestamp"].duplicated(keep=False), "timestamp"]
        raise ValueError(f"duplicate timestamps detected: {duplicates.iloc[0].isoformat()}")
    if not result["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps are out of order")

    for column in ("symbol", "timeframe"):
        values = result[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"{column} must contain non-empty values")
        if values.nunique(dropna=False) != 1:
            raise ValueError(f"a normalized dataset must contain exactly one {column}")
        result[column] = values.astype(str)

    numeric_columns = [column for column in NUMERIC_COLUMNS if column in result.columns]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        values = result[column].to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite numeric values")

    if (
        result["high"].to_numpy(dtype="float64")
        < result[["open", "close"]].max(axis=1).to_numpy(dtype="float64")
    ).any():
        raise ValueError("high is below open/close")
    if (
        result["low"].to_numpy(dtype="float64")
        > result[["open", "close"]].min(axis=1).to_numpy(dtype="float64")
    ).any():
        raise ValueError("low is above open/close")
    if (result["low"] > result["high"]).any():
        raise ValueError("low is above high")

    for column in ("volume", *OPTIONAL_NORMALIZED_COLUMNS):
        if column in result.columns and (result[column] < 0).any():
            raise ValueError(f"{column} cannot contain negative values")

    ordered = list(REQUIRED_NORMALIZED_COLUMNS)
    ordered.extend(column for column in OPTIONAL_NORMALIZED_COLUMNS if column in result.columns)
    return result[ordered].reset_index(drop=True)
