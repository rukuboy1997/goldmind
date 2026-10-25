from pathlib import Path
from uuid import uuid4

import pandas as pd

from goldmind.data.validation import validate_normalized_ohlcv


def save_ohlcv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Validate and atomically persist canonical OHLCV data as Parquet."""
    normalized = validate_normalized_ohlcv(frame)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix or ".parquet"
    temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp{suffix}")
    try:
        normalized.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load and validate a previously saved normalized OHLCV dataset."""
    frame = pd.read_parquet(path)
    return validate_normalized_ohlcv(frame)
