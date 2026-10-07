from dataclasses import dataclass


@dataclass(frozen=True)
class MarketConfig:
    """Canonical market configuration for the first GoldMind release."""

    symbol: str = "XAUUSD"
    timeframes: tuple[str, ...] = ("M1", "M5", "M15", "H1", "H4")
    primary_execution_timeframe: str = "M5"
    timezone: str = "UTC"


DEFAULT_MARKET = MarketConfig()
