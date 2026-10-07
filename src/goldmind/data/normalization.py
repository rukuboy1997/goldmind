from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close")
OPTIONAL_COLUMNS = ("volume", "tick_volume", "spread")


def normalize_ohlcv(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """Normalize provider output into GoldMind's canonical OHLCV schema."""
    if frame.empty:
        raise ValueError("market-data frame is empty")

    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if result["timestamp"].isna().any():
        raise ValueError("invalid timestamp detected")

    numeric_columns = ["open", "high", "low", "close", *[c for c in OPTIONAL_COLUMNS if c in result]]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if result[numeric_columns].isna().any().any():
        raise ValueError("invalid numeric market-data value detected")

    if result["timestamp"].duplicated().any():
        duplicates = result.loc[result["timestamp"].duplicated(keep=False), "timestamp"]
        raise ValueError(f"duplicate timestamps detected: {duplicates.iloc[0].isoformat()}")

    if not result["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps are out of order")

    result = result.reset_index(drop=True)

    if (result["high"] < result[["open", "close"]].max(axis=1)).any():
        raise ValueError("high is below open/close")
    if (result["low"] > result[["open", "close"]].min(axis=1)).any():
        raise ValueError("low is above open/close")
    if (result["low"] > result["high"]).any():
        raise ValueError("low is above high")

    if "volume" not in result:
        result["volume"] = 0.0

    result.insert(1, "symbol", symbol)
    result.insert(2, "timeframe", timeframe)

    ordered = ["timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"]
    for column in ("tick_volume", "spread"):
        if column in result:
            ordered.append(column)

    return result[ordered]


def to_utc_datetime(value: datetime) -> datetime:
    """Convert a datetime to timezone-aware UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
