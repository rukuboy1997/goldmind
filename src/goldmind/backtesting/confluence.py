from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

TRADE_COLUMNS = (
    "candidate_id",
    "direction",
    "signal_bar_index",
    "signal_bar_open_utc",
    "entry_bar_index",
    "entry_bar_open_utc",
    "entry_price",
    "stop_price",
    "target_price",
    "initial_risk_price",
    "sweep_bar_index",
    "sweep_extreme",
    "fvg_gap_size",
    "exit_bar_index",
    "exit_bar_open_utc",
    "exit_price",
    "exit_reason",
    "bars_held",
    "pnl_price",
    "r_multiple",
)


def _finite_number(value: object, *, name: str, minimum: float | None = None) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return numeric


def simulate_confluence_backtest(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    point_size: float = 0.01,
    stop_buffer: float = 0.20,
    reward_risk: float = 2.0,
    slippage: float = 0.05,
    max_hold_bars: int = 48,
) -> tuple[pd.DataFrame, Mapping[str, int]]:
    """Simulate next-bar entries for WATCH_ONLY confluence candidates.

    Modeling assumptions for the initial bar-based experiment:
    - OHLC values are treated as Bid-side prices; verify this for the feed.
    - The MT5 spread column is in broker points and is converted using
      point_size. Bar-level spread is only an approximation of executable
      spread at a specific tick.
    - Long entries use Ask (Bid open + spread); short entries use Bid open.
    - Long exits are triggered against Bid; short exits against Ask
      (Bid OHLC + the current bar's spread estimate).
    - Signal candle must close before entry; entry is at the next bar open.
    - Stops/targets are checked with OHLC. If both are touched within a bar,
      assume the stop occurs first. Gap-through stops use the worse opening
      price. Target orders are filled at target, not at a potentially better
      gap price. Market fills get adverse slippage; target fills do not.
    - One position at a time. Signals whose entry bar is not after the prior
      exit bar are skipped. Trades still open at the data boundary are marked
      DATA_END and liquidated at the last close for reporting.

    Returned R multiples are net of modeled spread/slippage price effects;
    they are not account-currency P&L and do not model financing, commissions,
    latency, tick-level price paths, or position sizing.
    """
    point_size = _finite_number(point_size, name="point_size", minimum=1e-12)
    stop_buffer = _finite_number(stop_buffer, name="stop_buffer", minimum=0.0)
    reward_risk = _finite_number(reward_risk, name="reward_risk", minimum=1e-12)
    slippage = _finite_number(slippage, name="slippage", minimum=0.0)
    if isinstance(max_hold_bars, bool) or not isinstance(max_hold_bars, int) or max_hold_bars < 1:
        raise ValueError("max_hold_bars must be a positive integer")

    required_bars = {"timestamp", "open", "high", "low", "close", "spread"}
    missing_bars = sorted(required_bars - set(bars.columns))
    if missing_bars:
        raise ValueError(f"bars missing required columns: {missing_bars}")
    required_candidates = {
        "candidate_id", "direction", "signal_bar_index", "sweep_bar_index", "sweep_extreme"
    }
    missing_candidates = sorted(required_candidates - set(candidates.columns))
    if missing_candidates and not candidates.empty:
        raise ValueError(f"candidates missing required columns: {missing_candidates}")

    counters = {
        "candidates": int(len(candidates)),
        "trades": 0,
        "skipped_overlap": 0,
        "skipped_invalid_stop": 0,
        "skipped_no_entry_bar": 0,
    }
    if bars.empty:
        raise ValueError("bars must not be empty")
    if candidates.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS), counters

    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("bar timestamps must be valid, unique, and chronological")

    price_columns: dict[str, np.ndarray] = {}
    for column in ("open", "high", "low", "close"):
        values = pd.to_numeric(bars[column], errors="coerce").to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError(f"{column} prices must be finite")
        price_columns[column] = values
    open_prices = price_columns["open"]
    high_prices = price_columns["high"]
    low_prices = price_columns["low"]
    close_prices = price_columns["close"]
    if (high_prices < np.maximum(open_prices, close_prices)).any():
        raise ValueError("high cannot be below open or close")
    if (low_prices > np.minimum(open_prices, close_prices)).any():
        raise ValueError("low cannot be above open or close")
    if (low_prices > high_prices).any():
        raise ValueError("low cannot be above high")

    spread_points = pd.to_numeric(bars["spread"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(spread_points).all() or (spread_points < 0).any():
        raise ValueError("spread must contain finite, non-negative broker point values")
    spread_prices = spread_points * point_size

    signal_indices = pd.to_numeric(candidates["signal_bar_index"], errors="coerce")
    sweep_indices = pd.to_numeric(candidates["sweep_bar_index"], errors="coerce")
    sweep_extremes = pd.to_numeric(candidates["sweep_extreme"], errors="coerce")
    for name, values in (
        ("signal_bar_index", signal_indices),
        ("sweep_bar_index", sweep_indices),
    ):
        numeric = values.to_numpy(dtype="float64")
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"candidate {name} must contain integer indices")
        if ((values < 0) | (values >= len(bars))).any():
            raise ValueError(f"candidate {name} is outside the bars range")
    if not np.isfinite(sweep_extremes.to_numpy(dtype="float64")).all():
        raise ValueError("candidate sweep_extreme must be finite")
    if (sweep_indices >= signal_indices).any():
        raise ValueError("candidate sweep_bar_index must precede signal_bar_index")

    working = candidates.copy()
    working["signal_bar_index"] = signal_indices.astype("int64")
    working["sweep_bar_index"] = sweep_indices.astype("int64")
    working["sweep_extreme"] = sweep_extremes.astype("float64")
    if not working["direction"].isin(["BULLISH", "BEARISH"]).all():
        raise ValueError("candidate direction must be BULLISH or BEARISH")
    working = working.sort_values(["signal_bar_index", "candidate_id"], kind="stable")

    trades: list[dict[str, object]] = []
    last_exit_index = -1

    for candidate in working.to_dict(orient="records"):
        signal_index = int(candidate["signal_bar_index"])
        entry_index = signal_index + 1
        if entry_index >= len(bars):
            counters["skipped_no_entry_bar"] += 1
            continue
        if entry_index <= last_exit_index:
            counters["skipped_overlap"] += 1
            continue

        direction = str(candidate["direction"])
        sweep_index = int(candidate["sweep_bar_index"])
        sweep_extreme = float(candidate["sweep_extreme"])
        spread_at_entry = float(spread_prices[entry_index])

        if direction == "BULLISH":
            entry_price = open_prices[entry_index] + spread_at_entry + slippage
            stop_price = sweep_extreme - stop_buffer
        else:
            # Sweep extreme is derived from bid OHLC; approximate its Ask-side
            # equivalent with the spread observed on the sweep bar.
            stop_price = sweep_extreme + float(spread_prices[sweep_index]) + stop_buffer
            entry_price = open_prices[entry_index] - slippage

        initial_risk = (
            entry_price - stop_price if direction == "BULLISH" else stop_price - entry_price
        )
        if not math.isfinite(initial_risk) or initial_risk <= 0:
            counters["skipped_invalid_stop"] += 1
            continue

        target_price = (
            entry_price + reward_risk * initial_risk
            if direction == "BULLISH"
            else entry_price - reward_risk * initial_risk
        )
        final_index = min(len(bars) - 1, entry_index + max_hold_bars - 1)
        exit_index: int | None = None
        exit_price: float | None = None
        exit_reason: str | None = None

        for bar_index in range(entry_index, final_index + 1):
            bar_spread = float(spread_prices[bar_index])
            if direction == "BULLISH":
                side_open = open_prices[bar_index]
                side_high = high_prices[bar_index]
                side_low = low_prices[bar_index]
                side_close = close_prices[bar_index]
            else:
                side_open = open_prices[bar_index] + bar_spread
                side_high = high_prices[bar_index] + bar_spread
                side_low = low_prices[bar_index] + bar_spread
                side_close = close_prices[bar_index] + bar_spread

            stop_at_open = side_open <= stop_price if direction == "BULLISH" else side_open >= stop_price
            target_at_open = side_open >= target_price if direction == "BULLISH" else side_open <= target_price
            stop_touched = side_low <= stop_price if direction == "BULLISH" else side_high >= stop_price
            target_touched = side_high >= target_price if direction == "BULLISH" else side_low <= target_price

            if stop_at_open:
                exit_index = bar_index
                exit_price = side_open - slippage if direction == "BULLISH" else side_open + slippage
                exit_reason = "STOP_GAP"
                break
            if target_at_open:
                exit_index = bar_index
                exit_price = target_price
                exit_reason = "TARGET"
                break
            if stop_touched and target_touched:
                # Without tick data, do not assume the favorable target came first.
                exit_index = bar_index
                exit_price = stop_price - slippage if direction == "BULLISH" else stop_price + slippage
                exit_reason = "STOP_AMBIGUOUS"
                break
            if stop_touched:
                exit_index = bar_index
                exit_price = stop_price - slippage if direction == "BULLISH" else stop_price + slippage
                exit_reason = "STOP"
                break
            if target_touched:
                exit_index = bar_index
                exit_price = target_price
                exit_reason = "TARGET"
                break

        if exit_index is None:
            exit_index = final_index
            if final_index == len(bars) - 1 and entry_index + max_hold_bars - 1 >= len(bars):
                exit_reason = "DATA_END"
            else:
                exit_reason = "TIME_EXIT"
            if direction == "BULLISH":
                exit_price = close_prices[exit_index] - slippage
            else:
                exit_price = close_prices[exit_index] + float(spread_prices[exit_index]) + slippage

        pnl_price = (
            float(exit_price) - float(entry_price)
            if direction == "BULLISH"
            else float(entry_price) - float(exit_price)
        )
        trades.append(
            {
                "candidate_id": candidate["candidate_id"],
                "direction": direction,
                "signal_bar_index": signal_index,
                "signal_bar_open_utc": timestamps.iloc[signal_index],
                "entry_bar_index": entry_index,
                "entry_bar_open_utc": timestamps.iloc[entry_index],
                "entry_price": float(entry_price),
                "stop_price": float(stop_price),
                "target_price": float(target_price),
                "initial_risk_price": float(initial_risk),
                "sweep_bar_index": sweep_index,
                "sweep_extreme": sweep_extreme,
                "fvg_gap_size": float(candidate.get("fvg_gap_size", float("nan"))),
                "exit_bar_index": exit_index,
                "exit_bar_open_utc": timestamps.iloc[exit_index],
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "bars_held": exit_index - entry_index + 1,
                "pnl_price": float(pnl_price),
                "r_multiple": float(pnl_price / initial_risk),
            }
        )
        last_exit_index = exit_index

    counters["trades"] = len(trades)
    return pd.DataFrame(trades, columns=TRADE_COLUMNS), counters
