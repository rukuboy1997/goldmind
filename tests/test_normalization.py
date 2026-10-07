import pandas as pd
import pytest

from goldmind.data.normalization import normalize_ohlcv


def test_normalize_sorts_and_adds_metadata() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:01:00Z", "open": 2, "high": 3, "low": 1, "close": 2.5},
            {"timestamp": "2026-01-01T00:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5},
        ]
    )

    result = normalize_ohlcv(frame, symbol="XAUUSD", timeframe="M5")

    assert result["timestamp"].is_monotonic_increasing
    assert list(result["symbol"].unique()) == ["XAUUSD"]
    assert list(result["timeframe"].unique()) == ["M5"]
    assert "volume" in result.columns


def test_duplicate_timestamps_are_rejected() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5},
            {"timestamp": "2026-01-01T00:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5},
        ]
    )

    with pytest.raises(ValueError, match="duplicate timestamps"):
        normalize_ohlcv(frame, symbol="XAUUSD", timeframe="M5")


def test_invalid_ohlc_is_rejected() -> None:
    frame = pd.DataFrame(
        [{"timestamp": "2026-01-01T00:00:00Z", "open": 2, "high": 1, "low": 0.5, "close": 1.5}]
    )

    with pytest.raises(ValueError, match="high"):
        normalize_ohlcv(frame, symbol="XAUUSD", timeframe="M5")
