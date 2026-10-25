from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from goldmind.data.validation import validate_normalized_ohlcv

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
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    if not isinstance(timeframe, str) or not timeframe.strip():
        raise ValueError("timeframe must be a non-empty string")

    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    if len(set(result.columns)) != len(result.columns):
        raise ValueError("duplicate column names after normalization")

    missing = [column for column in REQUIRED_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if result["timestamp"].isna().any():
        raise ValueError("invalid timestamp detected")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        *(column for column in OPTIONAL_COLUMNS if column in result.columns),
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    # MT5 provider output commonly has no real_volume on some instruments.
    # GoldMind uses a stable numeric schema and records that absence as zero.
    if "volume" not in result.columns:
        result["volume"] = 0.0

    # Never persist source-specific columns or trust provider-supplied metadata.
    result = result.drop(columns=["symbol", "timeframe"], errors="ignore")
    result.insert(1, "symbol", symbol.strip())
    result.insert(2, "timeframe", timeframe.strip())

    ordered = [
        "timestamp",
        "symbol",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
        "volume",
        *(column for column in ("tick_volume", "spread") if column in result.columns),
    ]
    return validate_normalized_ohlcv(result[ordered])


def to_utc_datetime(value: datetime) -> datetime:
    """Convert a datetime to timezone-aware UTC; naive values are interpreted as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
