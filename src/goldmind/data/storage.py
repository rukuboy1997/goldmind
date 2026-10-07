from pathlib import Path

import pandas as pd


def save_ohlcv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist normalized OHLCV data as Parquet."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return destination


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load a previously saved normalized OHLCV dataset."""
    return pd.read_parquet(path)
