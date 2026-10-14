from __future__ import annotations

import numpy as np
import pandas as pd

LIQUIDITY_SWEEP_COLUMNS = (
    "bar_index",
    "timestamp",
    "sweep_type",
    "liquidity_type",
    "direction",
    "level_price",
    "sweep_extreme",
    "close",
    "pivot_index",
    "pivot_timestamp",
    "pivot_confirmed_at",
)

_SWING_COLUMNS = (
    "pivot_index",
    "timestamp",
    "confirmation_index",
    "confirmed_at",
    "type",
    "price",
)


def detect_liquidity_sweeps(
    bars: pd.DataFrame,
    swings: pd.DataFrame,
) -> pd.DataFrame:
    """Find wick-through-and-reject events at confirmed swing liquidity.

    A swing high is treated as buy-side liquidity. A sweep requires the
    candle to open at/below the level, trade strictly above it, and close
    back below it. Swing lows use the mirrored rule for sell-side liquidity.

    A level is only available after its confirmation candle closes. Each
    level can produce at most one event: it is retired after a qualifying
    sweep or after a candle closes through it. This is a first-pass model;
    equal-high/low pools and session liquidity are not clustered here.
    """
    required_bars = {"timestamp", "open", "high", "low", "close"}
    missing_bars = sorted(required_bars - set(bars.columns))
    if missing_bars:
        raise ValueError(f"bars missing required columns: {missing_bars}")

    missing_swings = sorted(set(_SWING_COLUMNS) - set(swings.columns))
    if missing_swings:
        raise ValueError(f"swings missing required columns: {missing_swings}")

    if bars.empty:
        return pd.DataFrame(columns=LIQUIDITY_SWEEP_COLUMNS)

    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("bar timestamps contain invalid values")
    if timestamps.duplicated().any():
        raise ValueError("bar timestamps must be unique")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("bar timestamps must be chronological")

    price_columns = [
        pd.to_numeric(bars[column], errors="coerce").to_numpy(dtype="float64")
        for column in ("open", "high", "low", "close")
    ]
    open_prices, high_prices, low_prices, close_prices = price_columns
    if any(not np.isfinite(values).all() for values in price_columns):
        raise ValueError("OHLC prices must be finite numeric values")
    if (high_prices < np.maximum(open_prices, close_prices)).any():
        raise ValueError("high cannot be below open or close")
    if (low_prices > np.minimum(open_prices, close_prices)).any():
        raise ValueError("low cannot be above open or close")
    if (low_prices > high_prices).any():
        raise ValueError("low cannot be above high")

    if swings.empty:
        return pd.DataFrame(columns=LIQUIDITY_SWEEP_COLUMNS)

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
    if (
        (pivots["confirmation_index"] <= pivots["pivot_index"])
        | (pivots["confirmation_index"] >= len(bars))
    ).any():
        raise ValueError("confirmation_index must follow the pivot and be inside the bars range")

    for row in pivots.itertuples(index=False):
        if row.timestamp != timestamps.iloc[row.pivot_index]:
            raise ValueError("swing timestamp does not match pivot_index in bars")
        if row.confirmed_at != timestamps.iloc[row.confirmation_index]:
            raise ValueError("confirmed_at does not match confirmation_index in bars")
        if row.type == "HIGH" and not np.isclose(row.price, high_prices[row.pivot_index]):
            raise ValueError("HIGH swing price does not match pivot bar high")
        if row.type == "LOW" and not np.isclose(row.price, low_prices[row.pivot_index]):
            raise ValueError("LOW swing price does not match pivot bar low")

    pivots = pivots.sort_values(["confirmation_index", "pivot_index"], kind="stable")
    pivot_records = pivots.to_dict(orient="records")
    next_pivot = 0
    active_highs: list[dict[str, object]] = []
    active_lows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []

    for bar_index in range(len(bars)):
        # A pivot confirmed on the current candle only becomes usable later.
        while (
            next_pivot < len(pivot_records)
            and int(pivot_records[next_pivot]["confirmation_index"]) < bar_index
        ):
            pivot = pivot_records[next_pivot]
            if pivot["type"] == "HIGH":
                active_highs.append(pivot)
            else:
                active_lows.append(pivot)
            next_pivot += 1

        current_open = open_prices[bar_index]
        current_high = high_prices[bar_index]
        current_low = low_prices[bar_index]
        current_close = close_prices[bar_index]
        remaining_highs: list[dict[str, object]] = []

        for pivot in active_highs:
            level = float(pivot["price"])
            if current_high > level:
                if current_open <= level and current_close < level:
                    events.append(
                        {
                            "bar_index": bar_index,
                            "timestamp": timestamps.iloc[bar_index],
                            "sweep_type": "BUY_SIDE_SWEEP",
                            "liquidity_type": "BUY_SIDE",
                            "direction": "BEARISH",
                            "level_price": level,
                            "sweep_extreme": float(current_high),
                            "close": float(current_close),
                            "pivot_index": int(pivot["pivot_index"]),
                            "pivot_timestamp": pivot["timestamp"],
                            "pivot_confirmed_at": pivot["confirmed_at"],
                        }
                    )
                # A penetrated level is retired even if its close is not a
                # strict rejection; that first attempt is no longer reusable.
                continue
            remaining_highs.append(pivot)
        active_highs = remaining_highs

        remaining_lows: list[dict[str, object]] = []
        for pivot in active_lows:
            level = float(pivot["price"])
            if current_low < level:
                if current_open >= level and current_close > level:
                    events.append(
                        {
                            "bar_index": bar_index,
                            "timestamp": timestamps.iloc[bar_index],
                            "sweep_type": "SELL_SIDE_SWEEP",
                            "liquidity_type": "SELL_SIDE",
                            "direction": "BULLISH",
                            "level_price": level,
                            "sweep_extreme": float(current_low),
                            "close": float(current_close),
                            "pivot_index": int(pivot["pivot_index"]),
                            "pivot_timestamp": pivot["timestamp"],
                            "pivot_confirmed_at": pivot["confirmed_at"],
                        }
                    )
                # A penetrated level is retired even if its close is not a
                # strict rejection; that first attempt is no longer reusable.
                continue
            remaining_lows.append(pivot)
        active_lows = remaining_lows

    if not events:
        return pd.DataFrame(columns=LIQUIDITY_SWEEP_COLUMNS)
    return pd.DataFrame(events, columns=LIQUIDITY_SWEEP_COLUMNS).sort_values(
        ["bar_index", "liquidity_type", "pivot_index"], kind="stable"
    ).reset_index(drop=True)
