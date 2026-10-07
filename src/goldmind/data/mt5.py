from __future__ import annotations

from datetime import datetime

import pandas as pd

from goldmind.data.base import MarketDataProvider

_TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
}


class MT5DataProvider(MarketDataProvider):
    """Historical-data adapter for a locally running MetaTrader 5 terminal."""

    def __init__(self) -> None:
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise RuntimeError(
                "MetaTrader5 is not installed. Install GoldMind with: "
                "pip install -e '.[mt5]'"
            ) from exc
        self.mt5 = mt5

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        if timeframe not in _TIMEFRAME_MAP:
            raise ValueError(f"unsupported timeframe: {timeframe}")

        if not self.mt5.initialize():
            error = self.mt5.last_error()
            raise RuntimeError(f"MT5 initialize failed: {error}")

        try:
            if not self.mt5.symbol_select(symbol, True):
                error = self.mt5.last_error()
                raise RuntimeError(f"MT5 symbol_select failed for {symbol}: {error}")

            mt5_timeframe = getattr(self.mt5, _TIMEFRAME_MAP[timeframe])
            rates = self.mt5.copy_rates_range(symbol, mt5_timeframe, start, end)
            if rates is None:
                error = self.mt5.last_error()
                raise RuntimeError(f"MT5 copy_rates_range failed: {error}")

            frame = pd.DataFrame(rates)
            if frame.empty:
                return frame

            frame["timestamp"] = pd.to_datetime(frame["time"], unit="s", utc=True)
            rename = {"real_volume": "volume", "tick_volume": "tick_volume"}
            frame = frame.rename(columns=rename)
            columns = ["timestamp", "open", "high", "low", "close", "volume", "tick_volume"]
            if "spread" in frame:
                columns.append("spread")
            return frame[columns]
        finally:
            self.mt5.shutdown()
