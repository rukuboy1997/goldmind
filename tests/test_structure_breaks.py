import pandas as pd
import pytest

from goldmind.structure.breaks import detect_structure_breaks


def make_bars() -> pd.DataFrame:
    closes = [10.0, 11.0, 10.5, 10.8, 10.0, 13.0, 12.0, 10.0, 8.0]
    highs = [10.8, 12.0, 11.3, 11.6, 10.8, 13.8, 12.8, 10.8, 8.8]
    lows = [9.2, 10.3, 9.7, 10.0, 9.0, 12.0, 11.0, 9.0, 7.0]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-10-12T00:00:00Z",
                periods=len(closes),
                freq="5min",
            ),
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def make_swings(bars: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pivot_index": 1,
                "timestamp": bars.loc[1, "timestamp"],
                "confirmation_index": 3,
                "confirmed_at": bars.loc[3, "timestamp"],
                "type": "HIGH",
                "price": 12.0,
            },
            {
                "pivot_index": 4,
                "timestamp": bars.loc[4, "timestamp"],
                "confirmation_index": 6,
                "confirmed_at": bars.loc[6, "timestamp"],
                "type": "LOW",
                "price": 9.0,
            },
        ]
    )


def test_close_breaks_generate_bos_then_choch() -> None:
    bars = make_bars()
    swings = make_swings(bars)

    events = detect_structure_breaks(bars, swings)

    assert events["event"].tolist() == ["BOS", "CHOCH"]
    assert events["direction"].tolist() == ["BULLISH", "BEARISH"]
    assert events["bar_index"].tolist() == [5, 8]
    assert events["broken_level_type"].tolist() == ["HIGH", "LOW"]
    assert events["level_price"].tolist() == [12.0, 9.0]
    assert events["trend_before"].tolist() == ["UNKNOWN", "BULLISH"]
    assert events["trend_after"].tolist() == ["BULLISH", "BEARISH"]
    assert events.iloc[0]["bar_index"] > 3
    assert events.iloc[1]["bar_index"] > 6


def test_wick_through_without_close_does_not_count_as_break() -> None:
    bars = make_bars()
    bars.loc[4, "high"] = 13.0
    bars.loc[4, "close"] = 11.0
    bars.loc[5, "close"] = 11.5
    swings = make_swings(bars).iloc[[0]].reset_index(drop=True)

    events = detect_structure_breaks(bars, swings)

    assert events.empty


def test_empty_swings_returns_empty_events() -> None:
    bars = make_bars()
    swings = pd.DataFrame(
        columns=[
            "pivot_index",
            "timestamp",
            "confirmation_index",
            "confirmed_at",
            "type",
            "price",
        ]
    )

    events = detect_structure_breaks(bars, swings)

    assert events.empty
    assert "event" in events.columns


def test_misaligned_confirmation_timestamp_is_rejected() -> None:
    bars = make_bars()
    swings = make_swings(bars)
    swings.loc[0, "confirmed_at"] = bars.loc[2, "timestamp"]

    with pytest.raises(ValueError, match="confirmed_at does not match"):
        detect_structure_breaks(bars, swings)
