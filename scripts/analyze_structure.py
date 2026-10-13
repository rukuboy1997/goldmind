from __future__ import annotations

import argparse
from pathlib import Path

from goldmind.data.storage import load_ohlcv
from goldmind.structure import detect_structure_breaks, find_swings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze confirmed swings and BOS/CHOCH events in a GoldMind Parquet dataset."
    )
    parser.add_argument("--input", default="data/processed/XAUUSDmicro_M5.parquet")
    parser.add_argument("--left-bars", type=int, default=2)
    parser.add_argument("--right-bars", type=int, default=2)
    args = parser.parse_args()

    input_path = Path(args.input)
    bars = load_ohlcv(input_path)
    swings = find_swings(
        bars,
        left_bars=args.left_bars,
        right_bars=args.right_bars,
    )
    events = detect_structure_breaks(bars, swings)

    print(f"Input: {input_path}")
    print(f"Candles: {len(bars):,}")
    print(f"Confirmed swing points: {len(swings):,}")
    print(f"Structure breaks: {len(events):,}")

    if swings.empty:
        print("\nNo confirmed swing points found.")
    else:
        print("\nRecent swing points:")
        print(swings.tail(15).to_string(index=False))

    if events.empty:
        print("\nNo close-confirmed BOS/CHOCH events found in this dataset.")
    else:
        print("\nBOS/CHOCH events:")
        print(events.to_string(index=False))


if __name__ == "__main__":
    main()
