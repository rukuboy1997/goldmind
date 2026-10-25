from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from goldmind.data.mt5 import MT5DataProvider
from goldmind.data.storage import load_ohlcv


def make_rates() -> list[dict[str, float | int]]:
    start = int(pd.Timestamp("2026-01-01T00:00:00Z").timestamp())
    return [
        {
            "time": start + offset,
            "open": 2000.0 + index,
            "high": 2002.0 + index,
            "low": 1999.0 + index,
            "close": 2001.0 + index,
            "tick_volume": 100 + index,
            "spread": 12,
            "real_volume": 0,
        }
        for index, offset in enumerate((0, 300, 600))
    ]


def fake_mt5_module(rates: list[dict[str, object]]) -> tuple[types.ModuleType, dict[str, object]]:
    module = types.ModuleType("MetaTrader5")
    calls: dict[str, object] = {"initialize": 0, "shutdown": 0, "ranges": []}
    module.TIMEFRAME_M1 = 1
    module.TIMEFRAME_M5 = 5
    module.TIMEFRAME_M15 = 15
    module.TIMEFRAME_H1 = 60
    module.TIMEFRAME_H4 = 240

    def initialize() -> bool:
        calls["initialize"] = int(calls["initialize"]) + 1
        return True

    def shutdown() -> None:
        calls["shutdown"] = int(calls["shutdown"]) + 1

    def symbol_select(symbol: str, enable: bool) -> bool:
        calls["selected_symbol"] = symbol
        calls["selected"] = enable
        return True

    def copy_rates_range(
        symbol: str,
        timeframe: int,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, object]]:
        ranges = calls["ranges"]
        assert isinstance(ranges, list)
        ranges.append((symbol, timeframe, start, end))
        return rates

    module.initialize = initialize
    module.shutdown = shutdown
    module.symbol_select = symbol_select
    module.copy_rates_range = copy_rates_range
    module.last_error = lambda: (-1, "mock MT5 error")
    return module, calls


def test_fetch_cli_mt5_to_normalization_to_parquet_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake, calls = fake_mt5_module(make_rates())
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)

    from scripts import fetch_data

    output = tmp_path / "processed" / "XAUUSD_M5.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_data.py",
            "--symbol",
            "XAUUSD",
            "--timeframe",
            "M5",
            "--start",
            "2026-01-01T00:00:00+00:00",
            "--end",
            "2026-01-01T00:10:00+00:00",
            "--output",
            str(output),
        ],
    )

    fetch_data.main()

    stored = load_ohlcv(output)
    assert len(stored) == 2  # The 00:10 bar is excluded by the half-open range.
    assert stored["timestamp"].tolist() == list(
        pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"], utc=True)
    )
    assert stored["symbol"].eq("XAUUSD").all()
    assert stored["timeframe"].eq("M5").all()
    assert stored["spread"].tolist() == [12, 12]
    assert stored["tick_volume"].tolist() == [100, 101]
    assert "time" not in stored.columns
    assert "real_volume" not in stored.columns
    assert output.exists()
    assert "Saved 2 bars" in capsys.readouterr().out
    assert calls["initialize"] == 1
    assert calls["shutdown"] == 1
    assert calls["selected_symbol"] == "XAUUSD"
    assert calls["selected"] is True

    requested = calls["ranges"]
    assert isinstance(requested, list)
    assert requested[0][0:2] == ("XAUUSD", fake.TIMEFRAME_M5)
    assert requested[0][2].tzinfo == timezone.utc
    assert requested[0][3].tzinfo == timezone.utc


def test_mt5_adapter_always_shuts_down_when_response_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, calls = fake_mt5_module([{"open": 1.0, "close": 1.0}])
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    provider = MT5DataProvider()

    with pytest.raises(RuntimeError, match="missing required fields"):
        provider.fetch_ohlcv(
            "XAUUSD",
            "M5",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
        )

    assert calls["shutdown"] == 1


def test_mt5_adapter_rejects_invalid_range_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, calls = fake_mt5_module(make_rates())
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    provider = MT5DataProvider()

    with pytest.raises(ValueError, match="end must be later"):
        provider.fetch_ohlcv(
            "XAUUSD",
            "M5",
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

    assert calls["initialize"] == 0
    assert calls["shutdown"] == 0
