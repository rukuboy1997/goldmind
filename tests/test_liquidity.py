import pandas as pd
import pytest

from goldmind.structure.liquidity import detect_liquidity_sweeps


def make_high_sweep_bars() -> pd.DataFrame:
    rows = [
        (9.0, 10.0, 8.0, 9.0),
        (10.0, 12.0, 9.0, 11.0),
        (10.0, 11.5, 9.5, 10.5),
        (10.4, 11.0, 9.8, 10.2),
        (11.0, 12.6, 10.5, 11.4),
        (11.4, 11.7, 10.8, 11.0),
    ]
    return make_bars(rows)


def make_bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-10-12T00:00:00Z",
                periods=len(rows),
                freq="5min",
            ),
            "open": [row[0] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[3] for row in rows],
        }
    )


def make_swing(
    bars: pd.DataFrame,
    *,
    pivot_index: int,
    confirmation_index: int,
    swing_type: str,
    price: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pivot_index": pivot_index,
                "timestamp": bars.loc[pivot_index, "timestamp"],
                "confirmation_index": confirmation_index,
                "confirmed_at": bars.loc[confirmation_index, "timestamp"],
                "type": swing_type,
                "price": price,
            }
        ]
    )


def test_buy_side_liquidity_sweep_needs_rejection_close() -> None:
    bars = make_high_sweep_bars()
    swings = make_swing(
        bars,
        pivot_index=1,
        confirmation_index=2,
        swing_type="HIGH",
        price=12.0,
    )

    events = detect_liquidity_sweeps(bars, swings)

    assert len(events) == 1
    event = events.iloc[0]
    assert event["bar_index"] == 4
    assert event["sweep_type"] == "BUY_SIDE_SWEEP"
    assert event["liquidity_type"] == "BUY_SIDE"
    assert event["direction"] == "BEARISH"
    assert event["level_price"] == 12.0
    assert event["sweep_extreme"] == 12.6
    assert event["close"] < event["level_price"]


def test_sell_side_liquidity_sweep_is_bullish_rejection() -> None:
    bars = make_bars(
        [
            (9.0, 10.0, 8.0, 9.0),
            (9.0, 10.0, 8.0, 8.5),
            (8.5, 9.2, 8.2, 8.6),
            (8.6, 9.0, 8.3, 8.7),
            (9.0, 9.4, 7.5, 8.5),
            (8.5, 9.0, 8.2, 8.7),
        ]
    )
    swings = make_swing(
        bars,
        pivot_index=1,
        confirmation_index=2,
        swing_type="LOW",
        price=8.0,
    )

    events = detect_liquidity_sweeps(bars, swings)

    assert len(events) == 1
    event = events.iloc[0]
    assert event["bar_index"] == 4
    assert event["sweep_type"] == "SELL_SIDE_SWEEP"
    assert event["liquidity_type"] == "SELL_SIDE"
    assert event["direction"] == "BULLISH"
    assert event["sweep_extreme"] == 7.5
    assert event["close"] > event["level_price"]


def test_close_through_consumes_level_without_sweep() -> None:
    bars = make_high_sweep_bars()
    bars.loc[3, ["open", "high", "low", "close"]] = [12.1, 12.6, 11.8, 12.3]
    swings = make_swing(
        bars,
        pivot_index=1,
        confirmation_index=2,
        swing_type="HIGH",
        price=12.0,
    )

    events = detect_liquidity_sweeps(bars, swings)

    assert events.empty


def test_misaligned_swing_confirmation_timestamp_is_rejected() -> None:
    bars = make_high_sweep_bars()
    swings = make_swing(
        bars,
        pivot_index=1,
        confirmation_index=2,
        swing_type="HIGH",
        price=12.0,
    )
    swings.loc[0, "confirmed_at"] = bars.loc[1, "timestamp"]

    with pytest.raises(ValueError, match="confirmed_at does not match"):
        detect_liquidity_sweeps(bars, swings)
