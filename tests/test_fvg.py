import pandas as pd
import pytest

from goldmind.structure.fvg import detect_fair_value_gaps


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


def test_detects_bullish_fvg_and_records_zone() -> None:
    bars = make_bars(
        [
            (100.0, 101.0, 99.0, 100.0),
            (101.0, 106.0, 100.0, 105.0),
            (106.0, 108.0, 102.0, 107.0),
        ]
    )

    result = detect_fair_value_gaps(bars)

    assert len(result) == 1
    gap = result.iloc[0]
    assert gap["direction"] == "BULLISH"
    assert gap["left_index"] == 0
    assert gap["middle_index"] == 1
    assert gap["right_index"] == 2
    assert gap["zone_bottom"] == 101.0
    assert gap["zone_top"] == 102.0
    assert gap["gap_size"] == 1.0
    assert gap["formed_at"] == bars.loc[2, "timestamp"]


def test_detects_bearish_fvg_and_records_zone() -> None:
    bars = make_bars(
        [
            (109.0, 110.0, 108.0, 109.0),
            (108.0, 109.0, 103.0, 104.0),
            (104.0, 107.0, 101.0, 102.0),
        ]
    )

    result = detect_fair_value_gaps(bars)

    assert len(result) == 1
    gap = result.iloc[0]
    assert gap["direction"] == "BEARISH"
    assert gap["zone_bottom"] == 107.0
    assert gap["zone_top"] == 108.0
    assert gap["gap_size"] == 1.0


def test_minimum_gap_size_filters_small_gaps() -> None:
    bars = make_bars(
        [
            (100.0, 101.0, 99.0, 100.0),
            (101.0, 106.0, 100.0, 105.0),
            (106.0, 108.0, 102.0, 107.0),
        ]
    )

    assert detect_fair_value_gaps(bars, min_gap_size=1.01).empty
    assert len(detect_fair_value_gaps(bars, min_gap_size=1.0)) == 1


def test_fewer_than_three_bars_returns_empty_result() -> None:
    bars = make_bars([(100.0, 101.0, 99.0, 100.0), (101.0, 102.0, 100.0, 101.0)])

    result = detect_fair_value_gaps(bars)

    assert result.empty
    assert "direction" in result.columns


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf")])
def test_invalid_minimum_gap_size_is_rejected(threshold: float) -> None:
    bars = make_bars([(100.0, 101.0, 99.0, 100.0)] * 3)

    with pytest.raises(ValueError, match="finite non-negative"):
        detect_fair_value_gaps(bars, min_gap_size=threshold)


def test_duplicate_timestamps_are_rejected() -> None:
    bars = make_bars(
        [
            (100.0, 101.0, 99.0, 100.0),
            (101.0, 106.0, 100.0, 105.0),
            (106.0, 108.0, 102.0, 107.0),
        ]
    )
    bars.loc[2, "timestamp"] = bars.loc[1, "timestamp"]

    with pytest.raises(ValueError, match="unique"):
        detect_fair_value_gaps(bars)
