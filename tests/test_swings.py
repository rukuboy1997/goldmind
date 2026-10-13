import pandas as pd
import pytest

from goldmind.structure.swings import find_swings


def make_frame(highs: list[float], lows: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-10-12T00:00:00Z",
                periods=len(highs),
                freq="5min",
            ),
            "high": highs,
            "low": lows,
        }
    )


def test_pivots_include_confirmation_time() -> None:
    frame = make_frame(
        highs=[5, 6, 10, 7, 6],
        lows=[4, 3, 1, 3, 4],
    )

    result = find_swings(frame, left_bars=1, right_bars=1)

    high = result.loc[result["type"] == "HIGH"].iloc[0]
    low = result.loc[result["type"] == "LOW"].iloc[0]

    assert high["pivot_index"] == 2
    assert high["price"] == 10.0
    assert high["confirmation_index"] == 3
    assert high["timestamp"] == frame.loc[2, "timestamp"]
    assert high["confirmed_at"] == frame.loc[3, "timestamp"]

    assert low["pivot_index"] == 2
    assert low["price"] == 1.0


def test_flat_high_plateau_selects_first_pivot() -> None:
    frame = make_frame(
        highs=[1, 3, 3, 2, 1],
        lows=[0, 1, 2, 1, 0],
    )

    result = find_swings(frame, left_bars=1, right_bars=1)
    highs = result.loc[result["type"] == "HIGH", "pivot_index"].tolist()

    assert highs == [1]


def test_too_short_frame_returns_no_pivots() -> None:
    frame = make_frame(highs=[2, 3, 2], lows=[1, 2, 1])

    result = find_swings(frame, left_bars=2, right_bars=2)

    assert result.empty
    assert list(result.columns) == [
        "pivot_index",
        "timestamp",
        "confirmation_index",
        "confirmed_at",
        "type",
        "price",
    ]


@pytest.mark.parametrize(
    ("left_bars", "right_bars"),
    [(0, 1), (1, 0), (-1, 2)],
)
def test_invalid_window_sizes_are_rejected(left_bars: int, right_bars: int) -> None:
    frame = make_frame(highs=[2, 3, 2], lows=[1, 2, 1])

    with pytest.raises(ValueError, match="positive integer"):
        find_swings(frame, left_bars=left_bars, right_bars=right_bars)


def test_duplicate_timestamps_are_rejected() -> None:
    frame = make_frame(highs=[1, 3, 2, 1], lows=[0, 1, 1, 0])
    frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"]

    with pytest.raises(ValueError, match="unique"):
        find_swings(frame, left_bars=1, right_bars=1)
