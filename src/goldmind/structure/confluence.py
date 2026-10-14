from __future__ import annotations

import numpy as np
import pandas as pd

CONFLUENCE_COLUMNS = (
    "candidate_id",
    "direction",
    "status",
    "signal_bar_index",
    "signal_bar_timestamp",
    "sweep_bar_index",
    "sweep_timestamp",
    "sweep_type",
    "swept_level",
    "sweep_extreme",
    "structure_bar_index",
    "structure_timestamp",
    "structure_event",
    "fvg_right_index",
    "fvg_formed_at",
    "fvg_zone_bottom",
    "fvg_zone_top",
    "fvg_gap_size",
    "bars_from_sweep",
    "bars_from_structure",
)


def _validated_event_frame(
    frame: pd.DataFrame,
    *,
    label: str,
    index_column: str,
    timestamp_column: str,
    bar_timestamps: pd.Series,
) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result

    indices = pd.to_numeric(result[index_column], errors="coerce")
    values = indices.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} {index_column} must contain integer indices")
    result[index_column] = indices.astype("int64")
    if ((result[index_column] < 0) | (result[index_column] >= len(bar_timestamps))).any():
        raise ValueError(f"{label} {index_column} is outside the bars range")

    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column], utc=True, errors="coerce"
    )
    if result[timestamp_column].isna().any():
        raise ValueError(f"{label} timestamps contain invalid values")
    for row in result.itertuples(index=False):
        index_value = int(getattr(row, index_column))
        if getattr(row, timestamp_column) != bar_timestamps.iloc[index_value]:
            raise ValueError(f"{label} timestamp does not match its bar index")
    return result


def detect_confluence_candidates(
    bars: pd.DataFrame,
    structure_events: pd.DataFrame,
    liquidity_events: pd.DataFrame,
    fvgs: pd.DataFrame,
    *,
    max_bars_after_sweep: int = 12,
) -> pd.DataFrame:
    """Find reversal candidates with a causal sweep -> CHOCH -> FVG sequence.

    A candidate requires a liquidity sweep, then a later CHOCH in the sweep's
    directional bias, then a same-direction FVG formed on the CHOCH bar or
    within the configured window after the sweep. Each sweep contributes at
    most one candidate: the first complete sequence found.

    The output is WATCH_ONLY, not an entry signal. Timestamps label candle
    opening times; a sequence is only observable after its final candle closes.
    There are no entry/stop/target prices and no claim of profitability.
    """
    if (
        isinstance(max_bars_after_sweep, bool)
        or not isinstance(max_bars_after_sweep, int)
        or max_bars_after_sweep < 1
    ):
        raise ValueError("max_bars_after_sweep must be a positive integer")

    if "timestamp" not in bars.columns:
        raise ValueError("bars missing required column: timestamp")
    required_structures = {"bar_index", "timestamp", "event", "direction"}
    required_sweeps = {
        "bar_index", "timestamp", "sweep_type", "direction", "level_price", "sweep_extreme"
    }
    required_fvgs = {
        "right_index", "formed_at", "direction", "zone_bottom", "zone_top", "gap_size"
    }
    missing = sorted(required_structures - set(structure_events.columns))
    if missing:
        raise ValueError(f"structure_events missing required columns: {missing}")
    missing = sorted(required_sweeps - set(liquidity_events.columns))
    if missing:
        raise ValueError(f"liquidity_events missing required columns: {missing}")
    missing = sorted(required_fvgs - set(fvgs.columns))
    if missing:
        raise ValueError(f"fvgs missing required columns: {missing}")

    if bars.empty:
        return pd.DataFrame(columns=CONFLUENCE_COLUMNS)
    bar_timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    if bar_timestamps.isna().any():
        raise ValueError("bar timestamps contain invalid values")
    if bar_timestamps.duplicated().any():
        raise ValueError("bar timestamps must be unique")
    if not bar_timestamps.is_monotonic_increasing:
        raise ValueError("bar timestamps must be chronological")

    structures = _validated_event_frame(
        structure_events,
        label="structure_events",
        index_column="bar_index",
        timestamp_column="timestamp",
        bar_timestamps=bar_timestamps,
    )
    sweeps = _validated_event_frame(
        liquidity_events,
        label="liquidity_events",
        index_column="bar_index",
        timestamp_column="timestamp",
        bar_timestamps=bar_timestamps,
    )
    gap_rows = _validated_event_frame(
        fvgs,
        label="fvgs",
        index_column="right_index",
        timestamp_column="formed_at",
        bar_timestamps=bar_timestamps,
    )

    if not structures.empty and not structures["direction"].isin(["BULLISH", "BEARISH"]).all():
        raise ValueError("structure event direction must be BULLISH or BEARISH")
    if not sweeps.empty and not sweeps["direction"].isin(["BULLISH", "BEARISH"]).all():
        raise ValueError("liquidity sweep direction must be BULLISH or BEARISH")
    if not gap_rows.empty and not gap_rows["direction"].isin(["BULLISH", "BEARISH"]).all():
        raise ValueError("FVG direction must be BULLISH or BEARISH")

    for column in ("level_price", "sweep_extreme"):
        if not sweeps.empty:
            sweeps[column] = pd.to_numeric(sweeps[column], errors="coerce")
            if not np.isfinite(sweeps[column].to_numpy(dtype="float64")).all():
                raise ValueError(f"liquidity sweep {column} must be finite")

    for column in ("zone_bottom", "zone_top", "gap_size"):
        if not gap_rows.empty:
            gap_rows[column] = pd.to_numeric(gap_rows[column], errors="coerce")
            if not np.isfinite(gap_rows[column].to_numpy(dtype="float64")).all():
                raise ValueError(f"FVG {column} must be finite")

    if not gap_rows.empty and (
        (gap_rows["zone_bottom"] >= gap_rows["zone_top"])
        | (gap_rows["gap_size"] <= 0)
    ).any():
        raise ValueError("FVG zones must have positive width and gap size")

    chochs = structures.loc[structures["event"] == "CHOCH"].sort_values(
        "bar_index", kind="stable"
    )
    sweeps = sweeps.sort_values("bar_index", kind="stable")
    gap_rows = gap_rows.sort_values("right_index", kind="stable")
    candidates: list[dict[str, object]] = []

    for sweep in sweeps.to_dict(orient="records"):
        sweep_index = int(sweep["bar_index"])
        direction = str(sweep["direction"])
        last_allowed_index = min(
            len(bars) - 1, sweep_index + max_bars_after_sweep
        )
        possible_chochs = chochs.loc[
            (chochs["direction"] == direction)
            & (chochs["bar_index"] > sweep_index)
            & (chochs["bar_index"] <= last_allowed_index)
        ]
        matched: dict[str, object] | None = None

        for choch in possible_chochs.to_dict(orient="records"):
            structure_index = int(choch["bar_index"])
            possible_gaps = gap_rows.loc[
                (gap_rows["direction"] == direction)
                & (gap_rows["right_index"] >= structure_index)
                & (gap_rows["right_index"] <= last_allowed_index)
            ]
            if possible_gaps.empty:
                continue
            gap = possible_gaps.iloc[0].to_dict()
            signal_index = int(gap["right_index"])
            matched = {
                "candidate_id": len(candidates) + 1,
                "direction": direction,
                "status": "WATCH_ONLY",
                "signal_bar_index": signal_index,
                "signal_bar_timestamp": bar_timestamps.iloc[signal_index],
                "sweep_bar_index": sweep_index,
                "sweep_timestamp": sweep["timestamp"],
                "sweep_type": str(sweep["sweep_type"]),
                "swept_level": float(sweep["level_price"]),
                "sweep_extreme": float(sweep["sweep_extreme"]),
                "structure_bar_index": structure_index,
                "structure_timestamp": choch["timestamp"],
                "structure_event": "CHOCH",
                "fvg_right_index": signal_index,
                "fvg_formed_at": gap["formed_at"],
                "fvg_zone_bottom": float(gap["zone_bottom"]),
                "fvg_zone_top": float(gap["zone_top"]),
                "fvg_gap_size": float(gap["gap_size"]),
                "bars_from_sweep": signal_index - sweep_index,
                "bars_from_structure": signal_index - structure_index,
            }
            break

        if matched is not None:
            candidates.append(matched)

    if not candidates:
        return pd.DataFrame(columns=CONFLUENCE_COLUMNS)
    return pd.DataFrame(candidates, columns=CONFLUENCE_COLUMNS).sort_values(
        ["signal_bar_index", "sweep_bar_index"], kind="stable"
    ).reset_index(drop=True)
