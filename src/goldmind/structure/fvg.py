from __future__ import annotations

import math

import numpy as np
import pandas as pd

FVG_COLUMNS = (
    "left_index",
    "middle_index",
    "right_index",
    "left_timestamp",
    "middle_timestamp",
    "formed_at",
    "direction",
    "zone_bottom",
    "zone_top",
    "gap_size",
)


def detect_fair_value_gaps(
    bars: pd.DataFrame,
    *,
    min_gap_size: float = 0.0,
) -> pd.DataFrame:
    """Detect three-candle Fair Value Gaps (FVGs) in chronological OHLC data.

    Bullish: current low is above the high two candles earlier.
    Bearish: current high is below the low two candles earlier.

    The FVG is recorded only when its third candle is present. ``formed_at``
    uses that candle's timestamp; no later candles are required to identify
    the initial zone. This function does not decide whether the zone remains
    unmitigated, offers a trade, or has predictive value.
    """
    if isinstance(min_gap_size, bool) or not isinstance(min_gap_size, (int, float)):
        raise ValueError("min_gap_size must be a finite non-negative number")
    if not math.isfinite(float(min_gap_size)) or min_gap_size < 0:
        raise ValueError("min_gap_size must be a finite non-negative number")

    required = {"timestamp", "high", "low"}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")

    if bars.empty or len(bars) < 3:
        return pd.DataFrame(columns=FVG_COLUMNS)

    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("bar timestamps contain invalid values")
    if timestamps.duplicated().any():
        raise ValueError("bar timestamps must be unique")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("bar timestamps must be chronological")

    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(high).all() or not np.isfinite(low).all():
        raise ValueError("high and low must be finite numeric values")
    if (low > high).any():
        raise ValueError("low cannot be above high")

    rows: list[dict[str, object]] = []
    threshold = float(min_gap_size)
    for right_index in range(2, len(bars)):
        left_index = right_index - 2
        if low[right_index] > high[left_index]:
            gap_size = float(low[right_index] - high[left_index])
            if gap_size >= threshold and gap_size > 0:
                rows.append(
                    {
                        "left_index": left_index,
                        "middle_index": right_index - 1,
                        "right_index": right_index,
                        "left_timestamp": timestamps.iloc[left_index],
                        "middle_timestamp": timestamps.iloc[right_index - 1],
                        "formed_at": timestamps.iloc[right_index],
                        "direction": "BULLISH",
                        "zone_bottom": float(high[left_index]),
                        "zone_top": float(low[right_index]),
                        "gap_size": gap_size,
                    }
                )
        elif high[right_index] < low[left_index]:
            gap_size = float(low[left_index] - high[right_index])
            if gap_size >= threshold and gap_size > 0:
                rows.append(
                    {
                        "left_index": left_index,
                        "middle_index": right_index - 1,
                        "right_index": right_index,
                        "left_timestamp": timestamps.iloc[left_index],
                        "middle_timestamp": timestamps.iloc[right_index - 1],
                        "formed_at": timestamps.iloc[right_index],
                        "direction": "BEARISH",
                        "zone_bottom": float(high[right_index]),
                        "zone_top": float(low[left_index]),
                        "gap_size": gap_size,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=FVG_COLUMNS)
    return pd.DataFrame(rows, columns=FVG_COLUMNS).reset_index(drop=True)
