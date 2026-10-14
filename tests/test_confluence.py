import pandas as pd
import pytest

from goldmind.structure.confluence import detect_confluence_candidates


def make_inputs(
    *,
    sweep_direction: str = "BULLISH",
    structure_direction: str = "BULLISH",
    structure_event: str = "CHOCH",
    fvg_direction: str = "BULLISH",
    sweep_index: int = 1,
    structure_index: int = 3,
    fvg_index: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-10-12T00:00:00Z",
                periods=8,
                freq="5min",
            ),
            "open": [100.0] * 8,
            "high": [101.0] * 8,
            "low": [99.0] * 8,
            "close": [100.0] * 8,
        }
    )
    sweeps = pd.DataFrame(
        [
            {
                "bar_index": sweep_index,
                "timestamp": bars.loc[sweep_index, "timestamp"],
                "sweep_type": "SELL_SIDE_SWEEP" if sweep_direction == "BULLISH" else "BUY_SIDE_SWEEP",
                "direction": sweep_direction,
                "level_price": 99.5,
                "sweep_extreme": 99.0 if sweep_direction == "BULLISH" else 101.5,
            }
        ]
    )
    structures = pd.DataFrame(
        [
            {
                "bar_index": structure_index,
                "timestamp": bars.loc[structure_index, "timestamp"],
                "event": structure_event,
                "direction": structure_direction,
            }
        ]
    )
    fvgs = pd.DataFrame(
        [
            {
                "right_index": fvg_index,
                "formed_at": bars.loc[fvg_index, "timestamp"],
                "direction": fvg_direction,
                "zone_bottom": 100.5,
                "zone_top": 101.0,
                "gap_size": 0.5,
            }
        ]
    )
    return bars, structures, sweeps, fvgs


def test_detects_ordered_sweep_choch_fvg_candidate() -> None:
    bars, structures, sweeps, fvgs = make_inputs()

    result = detect_confluence_candidates(bars, structures, sweeps, fvgs)

    assert len(result) == 1
    candidate = result.iloc[0]
    assert candidate["direction"] == "BULLISH"
    assert candidate["status"] == "WATCH_ONLY"
    assert candidate["sweep_bar_index"] == 1
    assert candidate["structure_bar_index"] == 3
    assert candidate["structure_event"] == "CHOCH"
    assert candidate["fvg_right_index"] == 4
    assert candidate["signal_bar_index"] == 4
    assert candidate["bars_from_sweep"] == 3
    assert candidate["bars_from_structure"] == 1


def test_mismatched_directions_are_not_a_candidate() -> None:
    bars, structures, sweeps, fvgs = make_inputs(structure_direction="BEARISH")

    result = detect_confluence_candidates(bars, structures, sweeps, fvgs)

    assert result.empty


def test_fvg_before_choch_is_not_used() -> None:
    bars, structures, sweeps, fvgs = make_inputs(fvg_index=2)

    result = detect_confluence_candidates(bars, structures, sweeps, fvgs)

    assert result.empty


def test_bos_without_choch_is_not_a_reversal_candidate() -> None:
    bars, structures, sweeps, fvgs = make_inputs(structure_event="BOS")

    result = detect_confluence_candidates(bars, structures, sweeps, fvgs)

    assert result.empty


def test_events_outside_confluence_window_are_ignored() -> None:
    bars, structures, sweeps, fvgs = make_inputs(fvg_index=4)

    result = detect_confluence_candidates(
        bars, structures, sweeps, fvgs, max_bars_after_sweep=2
    )

    assert result.empty


def test_event_timestamp_must_match_bar_index() -> None:
    bars, structures, sweeps, fvgs = make_inputs()
    structures.loc[0, "timestamp"] = bars.loc[2, "timestamp"]

    with pytest.raises(ValueError, match="timestamp does not match"):
        detect_confluence_candidates(bars, structures, sweeps, fvgs)


@pytest.mark.parametrize("window", [0, -1, True])
def test_invalid_confluence_window_is_rejected(window: int) -> None:
    bars, structures, sweeps, fvgs = make_inputs()

    with pytest.raises(ValueError, match="positive integer"):
        detect_confluence_candidates(
            bars, structures, sweeps, fvgs, max_bars_after_sweep=window
        )


def test_duplicate_sweep_sequence_keeps_most_recent_sweep_once() -> None:
    bars, structures, sweeps, fvgs = make_inputs()
    later_sweep = {
        "bar_index": 2,
        "timestamp": bars.loc[2, "timestamp"],
        "sweep_type": "SELL_SIDE_SWEEP",
        "direction": "BULLISH",
        "level_price": 99.0,
        "sweep_extreme": 98.8,
    }
    sweeps = pd.concat([sweeps, pd.DataFrame([later_sweep])], ignore_index=True)

    result = detect_confluence_candidates(bars, structures, sweeps, fvgs)

    assert len(result) == 1
    assert result.iloc[0]["sweep_bar_index"] == 2
    assert result.iloc[0]["candidate_id"] == 1

