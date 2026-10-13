from __future__ import annotations

import numpy as np
import pandas as pd

PIVOT_COLUMNS = (
    "pivot_index",
    "timestamp",
    "confirmation_index",
    "confirmed_at",
    "type",
    "price",
)


def find_swings(
    frame: pd.DataFrame,
    *,
    left_bars: int = 2,
    right_bars: int = 2,
) -> pd.DataFrame:
    """Find local swing highs and lows in chronological OHLC data.

    The pivot bar is timestamped where the extreme occurred. The
    confirmed_at field records the final right-hand bar required to
    confirm it, allowing consumers to avoid look-ahead bias.

    For a tied plateau, the first qualifying extreme is selected: a
    pivot must be strictly more extreme than the left window and may
    tie the right window.
    """
    if isinstance(left_bars, bool) or not isinstance(left_bars, int) or left_bars < 1:
        raise ValueError("left_bars must be a positive integer")
    if isinstance(right_bars, bool) or not isinstance(right_bars, int) or right_bars < 1:
        raise ValueError("right_bars must be a positive integer")

    required = {"timestamp", "high", "low"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if frame.empty or len(frame) <= left_bars + right_bars:
        return pd.DataFrame(columns=PIVOT_COLUMNS)

    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("invalid timestamp detected")
    if timestamps.duplicated().any():
        raise ValueError("timestamps must be unique")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps must be in chronological order")

    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(high).all() or not np.isfinite(low).all():
        raise ValueError("high and low must contain finite numeric values")
    if (low > high).any():
        raise ValueError("low cannot be above high")

    rows: list[dict[str, object]] = []
    for index in range(left_bars, len(frame) - right_bars):
        left_slice = slice(index - left_bars, index)
        right_slice = slice(index + 1, index + right_bars + 1)
        pivot_time = timestamps.iloc[index]
        confirmation_index = index + right_bars
        confirmation_time = timestamps.iloc[confirmation_index]

        if (
            high[index] > high[left_slice].max()
            and high[index] >= high[right_slice].max()
        ):
            rows.append(
                {
                    "pivot_index": index,
                    "timestamp": pivot_time,
                    "confirmation_index": confirmation_index,
                    "confirmed_at": confirmation_time,
                    "type": "HIGH",
                    "price": float(high[index]),
                }
            )

        if (
            low[index] < low[left_slice].min()
            and low[index] <= low[right_slice].min()
        ):
            rows.append(
                {
                    "pivot_index": index,
                    "timestamp": pivot_time,
                    "confirmation_index": confirmation_index,
                    "confirmed_at": confirmation_time,
                    "type": "LOW",
                    "price": float(low[index]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=PIVOT_COLUMNS)

    return (
        pd.DataFrame(rows, columns=PIVOT_COLUMNS)
        .sort_values(["pivot_index", "type"], kind="stable")
        .reset_index(drop=True)
    )
