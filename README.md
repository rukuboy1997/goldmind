# GoldMind

GoldMind is a research-first AI trading intelligence system specialized in **XAUUSD (Gold)**.

The goal is not to predict every candle or promise perfect trades. GoldMind combines market data, price action, liquidity/market-structure analysis, setup validation, historical statistics, and risk management into disciplined, explainable trade research.

## Core principle

> **No setup is better than a bad setup.**

GoldMind may return **NO SETUP** when confluence or data quality is insufficient. Current confluence candidates are marked `WATCH_ONLY`; they are not live trade instructions.

## Current implementation

- Historical OHLCV retrieval through a local MetaTrader 5 terminal.
- UTC timestamp and OHLC validation, plus Parquet data storage.
- Swing, close-confirmed BOS/CHOCH, liquidity-sweep, and Fair Value Gap detectors.
- A confluence detector for the ordered sweep -> CHOCH -> same-direction FVG sequence.
- A bar-based research backtester that models spread and adverse slippage.
- Unit tests for core data, structure, confluence, and backtesting behavior.

The AI analyst, macroeconomic event layer, paper-trading service, strategy-learning loop, and live execution are not yet implemented. Backtests do not establish that the strategy has an edge.

## Setup

Use Python 3.11 or newer. The MetaTrader 5 Python package is supported here on Windows, and a local MT5 terminal must be installed and available to the logged-in account.

```bash
python -m venv .venv
```

Activate the virtual environment, then install the project and developer dependencies:

```bash
python -m pip install -e ".[dev,mt5]"
```

Check that MT5 is reachable and identify the gold symbol provided by your broker:

```bash
python scripts/check_mt5.py
```

Brokers may expose gold as `XAUUSD`, `XAUUSDmicro`, or another symbol. Use the exact name shown by your terminal. GoldMind's default symbol is `XAUUSD`.

## Fetch, analyze, and backtest

All three commands use the shared default dataset path `data/processed/XAUUSD_M5.parquet`.

First fetch historical M5 candles. Replace the example dates with the UTC interval you want to research; the end time is exclusive.

```bash
python scripts/fetch_data.py --symbol XAUUSD --timeframe M5 --start 2026-10-01T00:00:00+00:00 --end 2026-10-25T00:00:00+00:00
```

Then analyze market structure and review watch-only candidates:

```bash
python scripts/analyze_structure.py
```

Run the historical confluence backtest:

```bash
python scripts/backtest_confluence.py
```

The default trade report is written to `data/processed/backtest_trades.csv`. Override `--input` and `--output` to use another dataset or location. If you fetch a broker-specific symbol such as `XAUUSDmicro`, pass its generated Parquet path explicitly to the analysis and backtest scripts.

The initial backtest defaults are research assumptions, not measured execution conditions. Confirm the broker's symbol specification and the meaning of the MT5 spread/point values before interpreting results. The simulation does not model tick-level intrabar paths, commissions, financing, latency, or position sizing. If stop and target both fall within one candle, it assumes the stop happens first.

## Run tests

```bash
python -m pytest
```

## Development phases

1. **Data foundation** — reproducible historical data and robust provider/data-quality checks.
2. **Market structure engine** — swings, BOS/CHOCH, liquidity, FVGs, and later session/POI logic.
3. **Backtesting engine** — deterministic historical testing with realistic costs and explicit assumptions.
4. **Signal engine** — score setups and produce structured research signals.
5. **AI analyst** — explain and challenge quantitative setups without replacing deterministic calculations.
6. **Paper trading** — monitor candidate outcomes without real-money execution.
7. **Performance learning** — evaluate which conditions produce a repeatable statistical edge.
8. **Optional execution** — only after out-of-sample, walk-forward, paper-trading, and independent risk-control validation.

## Architecture

```
Market Data -> Data Layer -> Feature/Structure Engine
                                  |
                                  v
                         Setup & Confluence Engine
                                  |
                         +--------+--------+
                         v                 v
                     Backtester        Risk Engine
                         +--------+--------+
                                  v
                              AI Analyst
                                  v
                           Signal / NO SETUP
```

## Technology direction

- Python for research, data engineering, quantitative analysis, and ML.
- Parquet files for the current local research workflow; PostgreSQL/TimescaleDB may be introduced later.
- MetaTrader 5 as a historical market-data adapter and a possible future bridge. See the [MQL5 Python integration documentation](https://www.mql5.com/en/docs/python_metatrader5).
- LLMs for analysis and explanation, not as the sole source of trading decisions.

## Safety

GoldMind is a research and decision-support system. Backtests do not guarantee future performance. Keep live trading disabled until the strategy has passed out-of-sample, walk-forward, paper-trading, and risk-control validation.
