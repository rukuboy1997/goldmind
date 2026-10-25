from __future__ import annotations

from datetime import datetime

import pandas as pd

from goldmind.data.base import MarketDataProvider
from goldmind.data.normalization import to_utc_datetime

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
        symbol = symbol.strip() if isinstance(symbol, str) else ""
        timeframe = timeframe.strip().upper() if isinstance(timeframe, str) else ""
        if not symbol:
            raise ValueError("symbol must be a non-empty string")
        if timeframe not in _TIMEFRAME_MAP:
            raise ValueError(f"unsupported timeframe: {timeframe}")

        start_utc = to_utc_datetime(start)
        end_utc = to_utc_datetime(end)
        if end_utc <= start_utc:
            raise ValueError("end must be later than start")

        if not self.mt5.initialize():
            error = self.mt5.last_error()
            raise RuntimeError(f"MT5 initialize failed: {error}")

        try:
            if not self.mt5.symbol_select(symbol, True):
                error = self.mt5.last_error()
                raise RuntimeError(f"MT5 symbol_select failed for {symbol}: {error}")

            mt5_timeframe = getattr(self.mt5, _TIMEFRAME_MAP[timeframe])
            rates = self.mt5.copy_rates_range(symbol, mt5_timeframe, start_utc, end_utc)
            if rates is None:
                error = self.mt5.last_error()
                raise RuntimeError(f"MT5 copy_rates_range failed: {error}")

            frame = pd.DataFrame(rates)
            if frame.empty:
                return frame

            required_source_columns = {"time", "open", "high", "low", "close"}
            missing = sorted(required_source_columns - set(frame.columns))
            if missing:
                raise RuntimeError(f"MT5 rates missing required fields: {missing}")

            frame["timestamp"] = pd.to_datetime(frame["time"], unit="s", utc=True, errors="coerce")
            if frame["timestamp"].isna().any():
                raise ValueError("MT5 rates contain invalid timestamps")

            # Only retain bars in the requested half-open interval [start, end).
            # Some terminal feeds can return the candle opening at the end time.
            frame = frame.loc[
                (frame["timestamp"] >= pd.Timestamp(start_utc))
                & (frame["timestamp"] < pd.Timestamp(end_utc))
            ].copy()
            if frame.empty:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

            if "real_volume" in frame.columns:
                frame["volume"] = frame["real_volume"]
            elif "volume" not in frame.columns:
                # MT5 may report no real volume for some OTC instruments.
                frame["volume"] = 0.0

            rename = {"tick_volume": "tick_volume"}
            frame = frame.rename(columns=rename)
            columns = ["timestamp", "open", "high", "low", "close", "volume"]
            for optional in ("tick_volume", "spread"):
                if optional in frame.columns:
                    columns.append(optional)
            return frame[columns].reset_index(drop=True)
        finally:
            self.mt5.shutdown()
