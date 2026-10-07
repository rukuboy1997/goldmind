from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class OHLCVBar:
    """Provider-neutral OHLCV bar."""

    timestamp: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    tick_volume: int | None = None
    spread: float | None = None

    def validate(self) -> None:
        if self.high < max(self.open, self.close):
            raise ValueError("high must be >= open and close")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be <= open and close")
        if self.low > self.high:
            raise ValueError("low must be <= high")
        if self.volume < 0:
            raise ValueError("volume cannot be negative")
        if self.tick_volume is not None and self.tick_volume < 0:
            raise ValueError("tick_volume cannot be negative")
