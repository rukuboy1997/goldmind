# GoldMind

GoldMind is a research-first AI trading intelligence system specialized in **XAUUSD (Gold)**.

The goal is not to predict every candle or promise perfect trades. GoldMind is designed to combine market data, multi-timeframe price action, liquidity/market-structure analysis, macroeconomic context, setup validation, historical statistics, and risk management into disciplined, explainable trade decisions.

## Core principle

> **No setup is better than a bad setup.**

GoldMind must be able to return **NO TRADE** when confluence, data quality, risk/reward, or market conditions are insufficient.

## Development phases

1. **Data Foundation** — collect and normalize XAUUSD OHLCV/tick data and macro events.
2. **Market Structure Engine** — swings, BOS/CHOCH, liquidity, FVGs, POIs, sessions.
3. **Backtesting Engine** — deterministic historical testing with realistic spread/slippage assumptions.
4. **Signal Engine** — score setups and produce structured signals.
5. **AI Analyst** — explain and challenge the quantitative setup rather than replacing deterministic calculations.
6. **Paper Trading** — live signal monitoring without real-money execution.
7. **Performance Learning** — record every signal and evaluate which conditions create genuine statistical edge.
8. **Optional Execution** — only after extensive validation and with independent risk controls.

## Architecture

```
Market Data ──┐
              ├──> Data Layer ──> Feature/Structure Engine
Macro Data ───┘                         │
                                       ▼
                              Setup & Confluence Engine
                                       │
                              ┌────────┴────────┐
                              ▼                 ▼
                         Backtester        Risk Engine
                              │                 │
                              └────────┬────────┘
                                       ▼
                                  AI Analyst
                                       │
                                       ▼
                              Signal / NO TRADE
```

## Technology direction

- Python for research, data engineering, quantitative analysis and ML.
- PostgreSQL/TimescaleDB-compatible design for historical market data.
- MetaTrader 5 as one possible live market-data/execution bridge. The official Python integration exposes ticks, bars, symbol information, account information and market depth. See the MQL5 documentation: https://www.mql5.com/en/docs/python_metatrader5
- LLMs are used for analysis/explanation and orchestration, not as the sole source of trading decisions.

## Safety

GoldMind is a research and decision-support system. Backtests do not guarantee future performance. Live trading should remain disabled until the system has passed out-of-sample, walk-forward, paper-trading, and risk-control validation.
