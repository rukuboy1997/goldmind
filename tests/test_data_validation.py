from __future__ import annotations

import pandas as pd
import pytest

from goldmind.data.normalization import normalize_ohlcv
from goldmind.data.storage import load_ohlcv, save_ohlcv
from goldmind.data.validation import validate_normalized_ohlcv


def make_raw_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="5min"),
            "open": [2000.0, 2001.0, 2002.0],
            "high": [2002.0, 2003.0, 2004.0],
            "low": [1999.0, 2000.0, 2001.0],
            "close": [2001.0, 2002.0, 2003.0],
            "tick_volume": [100, 110, 120],
            "spread": [12, 11, 13],
            "provider_internal_id": ["a", "b", "c"],
        }
    )


def test_normalize_rejects_non_finite_price() -> None:
    raw = make_raw_bars()
    raw.loc[1, "close"] = float("inf")

    with pytest.raises(ValueError, match="close must contain finite"):
        normalize_ohlcv(raw, symbol="XAUUSD", timeframe="M5")


@pytest.mark.parametrize("column", ["volume", "tick_volume", "spread"])
def test_normalize_rejects_negative_activity_or_cost_fields(column: str) -> None:
    raw = make_raw_bars()
    raw[column] = [1, -1, 2]

    with pytest.raises(ValueError, match=f"{column} cannot contain negative"):
        normalize_ohlcv(raw, symbol="XAUUSD", timeframe="M5")


def test_normalize_drops_provider_specific_columns_and_adds_metadata() -> None:
    normalized = normalize_ohlcv(make_raw_bars(), symbol="XAUUSD", timeframe="M5")

    assert "provider_internal_id" not in normalized.columns
    assert normalized["symbol"].eq("XAUUSD").all()
    assert normalized["timeframe"].eq("M5").all()
    assert str(normalized["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_normalize_rejects_duplicate_columns_after_case_normalization() -> None:
    raw = pd.DataFrame(
        [[pd.Timestamp("2026-01-01T00:00:00Z"), 2000, 2001, 1999, 2000, 2001]],
        columns=["timestamp", "open", "high", "low", "close", "OPEN"],
    )

    with pytest.raises(ValueError, match="duplicate column names"):
        normalize_ohlcv(raw, symbol="XAUUSD", timeframe="M5")


def test_storage_round_trip_validates_and_preserves_canonical_data(tmp_path) -> None:
    normalized = normalize_ohlcv(make_raw_bars(), symbol="XAUUSD", timeframe="M5")
    output = tmp_path / "nested" / "bars.parquet"

    saved_path = save_ohlcv(normalized, output)
    loaded = load_ohlcv(saved_path)

    pd.testing.assert_frame_equal(loaded, normalized)
    assert saved_path.exists()
    assert loaded["timestamp"].is_monotonic_increasing
    assert loaded["timestamp"].is_unique


def test_storage_refuses_invalid_dataset(tmp_path) -> None:
    normalized = normalize_ohlcv(make_raw_bars(), symbol="XAUUSD", timeframe="M5")
    normalized.loc[1, "high"] = 1000.0
    destination = tmp_path / "invalid.parquet"

    with pytest.raises(ValueError, match="high is below open/close"):
        save_ohlcv(normalized, destination)
    assert not destination.exists()


def test_storage_loader_revalidates_existing_parquet(tmp_path) -> None:
    normalized = normalize_ohlcv(make_raw_bars(), symbol="XAUUSD", timeframe="M5")
    normalized.loc[1, "spread"] = -1
    source = tmp_path / "bad.parquet"
    normalized.to_parquet(source, index=False)

    with pytest.raises(ValueError, match="spread cannot contain negative"):
        load_ohlcv(source)


def test_validation_rejects_mixed_symbols_and_timeframes() -> None:
    normalized = normalize_ohlcv(make_raw_bars(), symbol="XAUUSD", timeframe="M5")
    normalized.loc[1, "symbol"] = "XAUUSDmicro"

    with pytest.raises(ValueError, match="exactly one symbol"):
        validate_normalized_ohlcv(normalized)
