from __future__ import annotations

import numpy as np
import pandas as pd

STRUCTURE_EVENT_COLUMNS = (
    "bar_index",
    "timestamp",
    "event",
    "direction",
    "broken_level_type",
    "level_price",
    "close",
    "pivot_index",
    "pivot_timestamp",
    "pivot_confirmed_at",
    "trend_before",
    "trend_after",
)

_SWING_COLUMNS = (
    "pivot_index",
    "timestamp",
    "confirmation_index",
    "confirmed_at",
    "type",
    "price",
)


def detect_structure_breaks(
    bars: pd.DataFrame,
    swings: pd.DataFrame,
) -> pd.DataFrame:
    """Detect close-confirmed BOS and CHOCH events without look-ahead.

    Only the most recently formed confirmed swing high and low are tracked.
    A level becomes available after its confirmation candle has closed, so
    it can only be broken by a later candle. A level is consumed on its first
    close-through; wick-only sweeps do not count as BOS/CHOCH.

    The first detected break is labeled BOS while trend is UNKNOWN. A break
    against an established direction is labeled CHOCH and flips the trend.
    """
    required_bars = {"timestamp", "close"}
    missing_bars = sorted(required_bars - set(bars.columns))
    if missing_bars:
        raise ValueError(f"bars missing required columns: {missing_bars}")

    missing_swings = sorted(set(_SWING_COLUMNS) - set(swings.columns))
    if missing_swings:
        raise ValueError(f"swings missing required columns: {missing_swings}")

    if bars.empty:
        return pd.DataFrame(columns=STRUCTURE_EVENT_COLUMNS)

    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("bar timestamps contain invalid values")
    if timestamps.duplicated().any():
        raise ValueError("bar timestamps must be unique")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("bar timestamps must be chronological")

    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(close).all():
        raise ValueError("close prices must be finite numeric values")

    if swings.empty:
        return pd.DataFrame(columns=STRUCTURE_EVENT_COLUMNS)

    pivots = swings.copy()
    pivots["timestamp"] = pd.to_datetime(pivots["timestamp"], utc=True, errors="coerce")
    pivots["confirmed_at"] = pd.to_datetime(pivots["confirmed_at"], utc=True, errors="coerce")
    if pivots[["timestamp", "confirmed_at"]].isna().any().any():
        raise ValueError("swing timestamps contain invalid values")

    for column in ("pivot_index", "confirmation_index"):
        pivots[column] = pd.to_numeric(pivots[column], errors="coerce")
        values = pivots[column].to_numpy(dtype="float64")
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain integer indices")
        pivots[column] = pivots[column].astype("int64")

    pivots["price"] = pd.to_numeric(pivots["price"], errors="coerce")
    if not np.isfinite(pivots["price"].to_numpy(dtype="float64")).all():
        raise ValueError("swing prices must be finite numeric values")
    if not pivots["type"].isin(["HIGH", "LOW"]).all():
        raise ValueError("swing type must be HIGH or LOW")

    if ((pivots["pivot_index"] < 0) | (pivots["pivot_index"] >= len(bars))).any():
        raise ValueError("pivot_index is outside the bars range")
    if ((pivots["confirmation_index"] <= pivots["pivot_index"]) | (pivots["confirmation_index"] >= len(bars))).any():
        raise ValueError("confirmation_index must follow the pivot and be inside the bars range")

    for row in pivots.itertuples(index=False):
        if row.timestamp != timestamps.iloc[row.pivot_index]:
            raise ValueError("swing timestamp does not match pivot_index in bars")
        if row.confirmed_at != timestamps.iloc[row.confirmation_index]:
            raise ValueError("confirmed_at does not match confirmation_index in bars")

    pivots = pivots.sort_values(["confirmation_index", "pivot_index"], kind="stable")
    records = pivots.to_dict(orient="records")
    next_pivot = 0
    active_high: dict[str, object] | None = None
    active_low: dict[str, object] | None = None
    trend = "UNKNOWN"
    events: list[dict[str, object]] = []

    for bar_index, current_close in enumerate(close):
        # A pivot confirmed on this candle is only available from the next one.
        while next_pivot < len(records) and int(records[next_pivot]["confirmation_index"]) < bar_index:
            pivot = records[next_pivot]
            side = str(pivot["type"])
            if side == "HIGH":
                if active_high is None or int(pivot["pivot_index"]) > int(active_high["pivot_index"]):
                    active_high = pivot
            else:
                if active_low is None or int(pivot["pivot_index"]) > int(active_low["pivot_index"]):
                    active_low = pivot
            next_pivot += 1

        if bar_index == 0:
            continue

        previous_close = close[bar_index - 1]
        broken: dict[str, object] | None = None
        direction: str | None = None

        if active_high is not None:
            level = float(active_high["price"])
            if previous_close <= level and current_close > level:
                broken = active_high
                direction = "BULLISH"
                active_high = None

        if active_low is not None:
            level = float(active_low["price"])
            if previous_close >= level and current_close < level:
                # OHLC close cannot normally cross both valid active sides;
                # retain an explicit guard for malformed/inverted structure.
                if broken is not None:
                    broken = None
                    direction = None
                    active_high = None
                else:
                    broken = active_low
                    direction = "BEARISH"
                    active_low = None

        if broken is None or direction is None:
            continue

        event = "CHOCH" if trend not in ("UNKNOWN", direction) else "BOS"
        previous_trend = trend
        trend = direction
        events.append(
            {
                "bar_index": bar_index,
                "timestamp": timestamps.iloc[bar_index],
                "event": event,
                "direction": direction,
                "broken_level_type": str(broken["type"]),
                "level_price": float(broken["price"]),
                "close": float(current_close),
                "pivot_index": int(broken["pivot_index"]),
                "pivot_timestamp": broken["timestamp"],
                "pivot_confirmed_at": broken["confirmed_at"],
                "trend_before": previous_trend,
                "trend_after": trend,
            }
        )

    if not events:
        return pd.DataFrame(columns=STRUCTURE_EVENT_COLUMNS)
    return pd.DataFrame(events, columns=STRUCTURE_EVENT_COLUMNS)
