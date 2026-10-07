from __future__ import annotations

import MetaTrader5 as mt5


def main() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        print(f"MetaTrader5 package: {mt5.__version__}")
        print(f"Terminal version: {mt5.version()}")

        symbols = mt5.symbols_get()
        if symbols is None:
            raise RuntimeError(f"symbols_get failed: {mt5.last_error()}")

        candidates = sorted(
            symbol.name
            for symbol in symbols
            if "XAU" in symbol.name.upper() or "GOLD" in symbol.name.upper()
        )

        print("Gold symbol candidates:")
        for name in candidates:
            print(f"  {name}")

        if not candidates:
            print("No XAU/GOLD symbol was found in the connected terminal.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
