from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from goldmind.backtesting import simulate_confluence_backtest
from goldmind.data.storage import load_ohlcv
from goldmind.structure import (
    detect_confluence_candidates,
    detect_fair_value_gaps,
    detect_liquidity_sweeps,
    detect_structure_breaks,
    find_swings,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a research-only, bar-based backtest of GoldMind confluence candidates."
    )
    parser.add_argument("--input", default="data/processed/XAUUSD_M5.parquet")
    parser.add_argument("--output", default="data/processed/backtest_trades.csv")
    parser.add_argument("--point-size", type=float, default=0.01,
                        help="price units per broker point; verify against your broker symbol (default assumption: 0.01)")
    parser.add_argument("--stop-buffer", type=float, default=0.20,
                        help="price units beyond the sweep extreme (baseline: 0.20)")
    parser.add_argument("--reward-risk", type=float, default=2.0,
                        help="target distance divided by initial price risk")
    parser.add_argument("--slippage", type=float, default=0.05,
                        help="adverse slippage per market fill in price units; baseline assumption, not measured")
    parser.add_argument("--max-hold-bars", type=int, default=48,
                        help="maximum number of M5 bars to hold a trade")
    parser.add_argument("--left-bars", type=int, default=2)
    parser.add_argument("--right-bars", type=int, default=2)
    parser.add_argument("--confluence-window", type=int, default=12)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    bars = load_ohlcv(input_path)

    swings = find_swings(bars, left_bars=args.left_bars, right_bars=args.right_bars)
    structures = detect_structure_breaks(bars, swings)
    sweeps = detect_liquidity_sweeps(bars, swings)
    fvgs = detect_fair_value_gaps(bars)
    candidates = detect_confluence_candidates(
        bars, structures, sweeps, fvgs,
        max_bars_after_sweep=args.confluence_window,
    )
    trades, counters = simulate_confluence_backtest(
        bars,
        candidates,
        point_size=args.point_size,
        stop_buffer=args.stop_buffer,
        reward_risk=args.reward_risk,
        slippage=args.slippage,
        max_hold_bars=args.max_hold_bars,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_path, index=False)

    print("=" * 64)
    print("GOLDMIND - RESEARCH-ONLY CONFLUENCE BACKTEST")
    print("NO LIVE ORDERS PLACED")
    print("=" * 64)
    print(f"Input: {input_path}")
    print(f"Trades CSV: {output_path}")
    print(f"Candles: {len(bars):,}")
    print(f"Candidates detected: {counters['candidates']:,}")
    print(f"Trades simulated: {counters['trades']:,}")
    print(f"Skipped due to overlapping position: {counters['skipped_overlap']:,}")
    print(f"Skipped due to invalid entry/stop geometry: {counters['skipped_invalid_stop']:,}")
    print(f"Skipped because no next candle exists: {counters['skipped_no_entry_bar']:,}")
    print("\nBASELINE ASSUMPTIONS")
    print("Entry: next candle open, only after signal candle has closed")
    print("Direction: bullish=long, bearish=short")
    print(f"Stop buffer beyond sweep extreme: {args.stop_buffer:.3f} price units")
    print(f"Target: {args.reward_risk:.2f}R")
    print(f"Adverse slippage: {args.slippage:.3f} price units per market fill")
    print(f"Maximum holding period: {args.max_hold_bars} bars")
    print(f"Historical bar spread median: {bars['spread'].median():.2f} broker points")
    print(f"Spread conversion point size: {args.point_size:.4f}")
    print("OHLC side and historical bar spread timing are approximations; verify feed semantics.")
    print("When stop and target both touch in a bar, stop is assumed first.")

    if trades.empty:
        print("\nNo trades were simulated with these rules.")
        return

    r = pd.to_numeric(trades["r_multiple"], errors="coerce")
    wins = r[r > 0]
    losses = r[r < 0]
    equity = r.cumsum()
    peaks = pd.concat([pd.Series([0.0]), equity.reset_index(drop=True)]).cummax().iloc[1:].reset_index(drop=True)
    drawdown = equity.reset_index(drop=True) - peaks
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    print("\nRESULTS (price-risk R multiples after modeled spread/slippage)")
    print(f"Closed/reconciled trades: {len(trades):,}")
    print(f"Win rate: {(r > 0).mean() * 100:.2f}%")
    print(f"Net total R: {r.sum():.2f}")
    print(f"Expectancy: {r.mean():.3f} R/trade")
    print(f"Average winning trade: {wins.mean():.3f} R" if not wins.empty else "Average winning trade: n/a")
    print(f"Average losing trade: {losses.mean():.3f} R" if not losses.empty else "Average losing trade: n/a")
    print(f"Profit factor (R): {profit_factor:.3f}")
    print(f"Maximum drawdown: {abs(float(drawdown.min())):.2f} R")
    print(f"Median holding bars: {trades['bars_held'].median():.1f}")
    print("\nEXIT REASONS")
    print(trades["exit_reason"].value_counts().to_string())
    print("\nThese results are a baseline simulation, not evidence of a profitable edge.")
    print("Repeat with different costs and unseen dates before drawing conclusions.")


if __name__ == "__main__":
    main()
