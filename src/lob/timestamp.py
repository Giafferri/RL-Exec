# =============================================================
# lob/timestamp.py
#
# Timestamp and order book snapshot handling for the LOB environment.
# =============================================================

import pandas as pd
from typing import Dict, List, Any

N_LEVELS = 20  # number of depth levels per side

def _has_wide_schema(df: pd.DataFrame) -> bool:
    """Return True if DataFrame follows the WIDE schema (one row per timestamp).

    Checks for sentinel columns (e.g., `bids_distance_0`, `asks_distance_0`).
    """
    return "bids_distance_0" in df.columns and "asks_distance_0" in df.columns


def _safe_get(row: pd.Series, prefix: str, lvl: int, default: float = 0.0) -> float:
    """Safely read a numeric value from a column named `{prefix}_{lvl}`.

    Args:
        row: Source pandas Series (one LOB snapshot row).
        prefix: Column family prefix (e.g., 'bids_distance').
        lvl: Depth level index.
        default: Fallback value when the column is missing or NaN.

    Returns:
        float: Parsed value or `default` if unavailable.
    """
    col = f"{prefix}_{lvl}"
    try:
        v = row[col]
    except Exception:
        return float(default)
    try:
        v = float(v)
        # NaN guard
        if v != v:
            return float(default)
        return v
    except Exception:
        return float(default)


def _build_from_wide_row(row: pd.Series) -> Dict[str, List[Dict[str, Any]]]:
    """Construct depth snapshots from a single WIDE-format row.

    Produces two lists of dictionaries with per-level metadata. Ordering is:
      • `bid_values`: level 0 (best) → 19 (farther).
      • `ask_values`: level 19 → 0 (far→near) for backward compatibility with
        downstream code that expects `reversed(ask_values)` to yield the best first.

    Args:
        row: Wide-format row containing midpoint and per-level fields.

    Returns:
        dict: `{ 'ask_values': List[Dict[str, Any]], 'bid_values': List[Dict[str, Any]] }`.
    """
    mid = float(row.get("midpoint", 0.0))

    ask_values: List[Dict[str, Any]] = []
    bid_values: List[Dict[str, Any]] = []

    # BIDS: level 0 (best) -> 19
    for lvl in range(N_LEVELS):
        dist = _safe_get(row, "bids_distance", lvl, 0.0)
        notional = _safe_get(row, "bids_notional", lvl, 0.0)
        cancel_n = _safe_get(row, "bids_cancel_notional", lvl, 0.0)
        limit_n  = _safe_get(row, "bids_limit_notional", lvl, 0.0)
        market_n = _safe_get(row, "bids_market_notional", lvl, 0.0)
        price = mid * (1.0 + dist) if mid else 0.0
        size = (notional / price) if price else 0.0
        bid_values.append({
            "side": "BID",
            "level": lvl,
            "midpoint_USD": mid,
            "distance_to_mid": dist,
            "notional_USD": notional,
            "size_BTC": size,
            "cancel_notional_USD": cancel_n,
            "limit_notional_USD": limit_n,
            "market_notional_USD": market_n,
        })

    # ASKS: iterate 19 -> 0 so stored list is far -> near (compat with reversed() usage downstream)
    for lvl in range(N_LEVELS - 1, -1, -1):
        dist = _safe_get(row, "asks_distance", lvl, 0.0)
        notional = _safe_get(row, "asks_notional", lvl, 0.0)
        cancel_n = _safe_get(row, "asks_cancel_notional", lvl, 0.0)
        limit_n  = _safe_get(row, "asks_limit_notional", lvl, 0.0)
        market_n = _safe_get(row, "asks_market_notional", lvl, 0.0)
        price = mid * (1.0 + dist) if mid else 0.0
        size = (notional / price) if price else 0.0
        ask_values.append({
            "side": "ASK",
            "level": lvl,
            "midpoint_USD": mid,
            "distance_to_mid": dist,
            "notional_USD": notional,
            "size_BTC": size,
            "cancel_notional_USD": cancel_n,
            "limit_notional_USD": limit_n,
            "market_notional_USD": market_n,
        })

    return {"ask_values": ask_values, "bid_values": bid_values}


def get_values(ts: int, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Return ask/bid snapshots for the given timestamp.

    Supports two dataframe schemas:
      • WIDE: one row per timestamp with per-level columns (preferred).
      • LONG: multiple rows per timestamp with columns ['side','level', ...].

    Behavior:
      • If no exact `timestamp_ns` match is found, the function snaps to the nearest
        available timestamp (useful for tiny drifts like ±80 ns).
      • Output always includes exactly `N_LEVELS` entries per side when available.
      • Ordering: `bid_values` best→farther (0→19); `ask_values` far→near (19→0).

    Args:
        ts: Target timestamp in nanoseconds.
        df: Source LOB dataframe containing at least 'timestamp_ns'.

    Returns:
        dict: `{'ask_values': List[Dict[str, Any]], 'bid_values': List[Dict[str, Any]]}`.

    Raises:
        KeyError: If 'timestamp_ns' is missing from `df`.
    """
    if "timestamp_ns" not in df.columns:
        raise KeyError("DataFrame must contain a 'timestamp_ns' column")

    rows = df[df["timestamp_ns"] == ts]
    if rows.empty:
        # Fallback: use the nearest timestamp available (handles tiny drifts)
        s = df["timestamp_ns"]
        idx_near = (s - ts).abs().idxmin()
        ts = int(s.loc[idx_near])
        rows = df[s == ts]

    # WIDE schema: 1 matching row
    if _has_wide_schema(df):
        row = rows.iloc[0]
        return _build_from_wide_row(row)

    # LONG schema fallback (backward-compat)
    # We assume rows for this ts are contiguous and contain 20 bids then 20 asks
    rows_sorted = rows.sort_values(["side", "level"]) if {"side", "level"}.issubset(rows.columns) else rows

    # BIDS: level 0..19
    bid_rows = rows_sorted[rows_sorted["side"].str.upper() == "BID"].sort_values("level")
    # ASKS: far -> near (19..0)
    ask_rows = rows_sorted[rows_sorted["side"].str.upper() == "ASK"].sort_values("level", ascending=False)

    def _map_row(r: pd.Series) -> Dict[str, Any]:
        return {
            "side": str(r["side"]).upper(),
            "level": int(r["level"]),
            "midpoint_USD": float(r["midpoint_USD"]),
            "distance_to_mid": float(r["distance_to_mid"]),
            "notional_USD": float(r["notional_USD"]),
            "size_BTC": float(r.get("size_BTC", 0.0)),
            "cancel_notional_USD": float(r.get("cancel_notional_USD", 0.0)),
            "limit_notional_USD": float(r.get("limit_notional_USD", 0.0)),
            "market_notional_USD": float(r.get("market_notional_USD", 0.0)),
        }

    ask_values = [_map_row(r) for _, r in ask_rows.head(N_LEVELS).iterrows()]
    bid_values = [_map_row(r) for _, r in bid_rows.head(N_LEVELS).iterrows()]

    return {"ask_values": ask_values, "bid_values": bid_values}