from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from goldmind.config import DEFAULT_MARKET
from goldmind.data.mt5 import MT5DataProvider
from goldmind.data.normalization import normalize_ohlcv, to_utc_datetime
from goldmind.data.storage import save_ohlcv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch historical XAUUSD data from MT5."
    )
    parser.add_argument("--symbol", default=DEFAULT_MARKET.symbol)
    parser.add_argument(
        "--timeframe",
        choices=DEFAULT_MARKET.timeframes,
        default="M5",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="UTC datetime, e.g. 2026-01-01T00:00:00+00:00",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="UTC datetime, e.g. 2026-01-31T23:59:59+00:00",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    start = to_utc_datetime(datetime.fromisoformat(args.start))
    end = to_utc_datetime(datetime.fromisoformat(args.end))

    if end <= start:
        raise ValueError("--end must be later than --start")

    provider = MT5DataProvider()
    raw = provider.fetch_ohlcv(args.symbol, args.timeframe, start, end)
    normalized = normalize_ohlcv(
        raw,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )

    output = Path(
        args.output
        or f"data/processed/{args.symbol}_{args.timeframe}.parquet"
    )
    save_ohlcv(normalized, output)
    print(f"Saved {len(normalized):,} bars to {output}")


if __name__ == "__main__":
    main()
