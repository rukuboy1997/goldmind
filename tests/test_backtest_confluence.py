import pandas as pd
import pytest

from goldmind.backtesting.confluence import simulate_confluence_backtest


def make_bars(*, highs=None, lows=None, periods=5):
    opens = [100.0] * periods
    closes = [100.0] * periods
    if highs is None:
        highs = [101.0] * periods
    if lows is None:
        lows = [99.0] * periods
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-10-12T00:00:00Z", periods=periods, freq="5min"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "spread": [10.0] * periods,
        }
    )


def make_candidate(*, direction="BULLISH", signal_index=1, sweep_index=0, extreme=99.0):
    return {
        "candidate_id": 1,
        "direction": direction,
        "signal_bar_index": signal_index,
        "sweep_bar_index": sweep_index,
        "sweep_extreme": extreme,
        "fvg_gap_size": 0.5,
    }


def test_signal_enters_next_bar_and_target_is_two_r():
    bars = make_bars(
        highs=[101.0, 101.0, 101.0, 103.0, 101.0],
        lows=[99.0, 99.5, 99.5, 99.0, 99.5],
    )
    trades, counters = simulate_confluence_backtest(
        bars,
        pd.DataFrame([make_candidate()]),
        point_size=0.01,
        stop_buffer=0.2,
        reward_risk=2.0,
        slippage=0.0,
        max_hold_bars=4,
    )
    assert counters["trades"] == 1
    assert trades.loc[0, "entry_bar_index"] == 2
    assert trades.loc[0, "exit_bar_index"] == 3
    assert trades.loc[0, "exit_reason"] == "TARGET"
    assert trades.loc[0, "r_multiple"] == pytest.approx(2.0)


def test_stop_is_assumed_first_when_both_stop_and_target_hit():
    bars = make_bars(
        highs=[101.0, 101.0, 103.0, 101.0, 101.0],
        lows=[99.0, 99.5, 98.5, 99.5, 99.5],
    )
    trades, _ = simulate_confluence_backtest(
        bars,
        pd.DataFrame([make_candidate()]),
        slippage=0.0,
        max_hold_bars=4,
    )
    assert trades.loc[0, "exit_reason"] == "STOP_AMBIGUOUS"
    assert trades.loc[0, "r_multiple"] < 0


def test_short_trade_uses_ask_side_for_exit_and_can_hit_target():
    bars = make_bars(
        highs=[102.0, 101.0, 101.0, 100.0, 100.0],
        lows=[99.0, 99.5, 99.5, 97.0, 98.5],
    )
    trades, _ = simulate_confluence_backtest(
        bars,
        pd.DataFrame([make_candidate(direction="BEARISH", extreme=101.0)]),
        slippage=0.0,
        max_hold_bars=4,
    )
    assert trades.loc[0, "exit_reason"] == "TARGET"
    assert trades.loc[0, "r_multiple"] == pytest.approx(2.0)


def test_signal_during_open_trade_is_skipped_if_entry_precedes_or_matches_exit_bar():
    bars = make_bars(
        highs=[101.0, 101.0, 101.0, 103.0, 101.0],
        lows=[99.0, 99.5, 99.5, 99.0, 99.5],
    )
    second = make_candidate(signal_index=2, sweep_index=1, extreme=99.0)
    second["candidate_id"] = 2
    trades, counters = simulate_confluence_backtest(
        bars,
        pd.DataFrame([make_candidate(), second]),
        slippage=0.0,
        max_hold_bars=4,
    )
    assert len(trades) == 1
    assert counters["skipped_overlap"] == 1


def test_candidate_without_a_next_bar_is_skipped():
    bars = make_bars(periods=3)
    trades, counters = simulate_confluence_backtest(
        bars,
        pd.DataFrame([make_candidate(signal_index=2, sweep_index=1)]),
    )
    assert trades.empty
    assert counters["skipped_no_entry_bar"] == 1


def test_invalid_spread_is_rejected():
    bars = make_bars()
    bars.loc[2, "spread"] = -1
    with pytest.raises(ValueError, match="spread must contain"):
        simulate_confluence_backtest(bars, pd.DataFrame([make_candidate()]))
