# =============================================================
# rl/obs_builder
# 
# Build observations for the RL agent trading on a LOB.
# =============================================================

from typing import Dict, Any, List
import numpy as np

def _safe_float(x: Any, default: float = 0.0) -> float:
    """Best-effort float cast with NaN/None handling; returns `default` on failure."""
    try:
        if x is None:
            return float(default)
        fx = float(x)
        if fx != fx:  # NaN check
            return float(default)
        return fx
    except Exception:
        return float(default)


def _pad_truncate(arr: List[float], n: int, pad_value: float = 0.0) -> np.ndarray:
    """Return a 1D float32 array of length `n`, truncating or right-padding with `pad_value`."""
    arr = np.asarray(list(arr), dtype=np.float32)
    if arr.shape[0] >= n:
        return arr[:n]
    out = np.full((n,), pad_value, dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out


def _near_to_far(levels: List[dict]) -> List[dict]:
    """Sort depth levels by |distance_to_mid| so index 0 is closest to the mid."""
    return sorted(levels or [], key=lambda lv: abs(_safe_float(lv.get("distance_to_mid", 0.0))))


def build_obs_depth_20(values_now: Dict[str, Any], cash: float, btc: float, n_levels: int = 20, time_left_ratio: float | None = None, backlog_ratio: float | None = None, include_aux: bool = False) -> np.ndarray:
    """
    Build a depth-20 observation with six feature families per side.

    Parameters:

    values_now : dict
        LOB snapshot with keys like 'ask_values', 'bid_values', and optionally
        'mid'/'midpoint', 'ask', 'bid'. Each level dict may contain
        'distance_to_mid', 'size_BTC', 'notional_USD', 'cancel_notional_USD',
        'limit_notional_USD', 'market_notional_USD', and 'price'.
    cash, btc : float
        Current cash and BTC inventory.
    n_levels : int, default 20
        Number of depth levels per side to include (truncated/padded as needed).
    time_left_ratio : float | None
        Optional fraction of episode time remaining in [0, 1]. Used only when
        `include_aux=True`.
    backlog_ratio : float | None
        Optional backlog as a fraction of initial inventory (>=0). Used only when
        `include_aux=True`.
    include_aux : bool, default False
        If True, append [time_left_ratio, backlog_ratio].

    Returns:

    np.ndarray
        Float32 vector of length 243:
            [cash, btc, midpoint,
             ASK(near→far): 20×[distance_to_mid, size_BTC, notional_USD,
                                 cancel_notional_USD, limit_notional_USD, market_notional_USD],
             BID(near→far): same 20×6 block]
        If `include_aux=True`, two extras are appended, for a total of 245.

    Notes:

    The midpoint is taken from 'mid'/'midpoint' if present; otherwise it falls back
    to the average of best ask/bid when available. All numeric inputs are sanitized via
    `_safe_float` and arrays are float32.
    """
    ask_levels = _near_to_far(values_now.get("ask_values", []) or [])[:n_levels]
    bid_levels = _near_to_far(values_now.get("bid_values", []) or [])[:n_levels]

    # Midpoint: prefer top-level mid/midpoint, else fallback to best ask/bid average, else 0.
    midpoint = _safe_float(values_now.get("mid", values_now.get("midpoint", 0.0)))

    if midpoint == 0.0:
        a0 = 0.0
        b0 = 0.0
        if ask_levels:
            a0 = _safe_float(ask_levels[0].get("price", values_now.get("ask", 0.0)))
        else:
            a0 = _safe_float(values_now.get("ask", 0.0))
        if bid_levels:
            b0 = _safe_float(bid_levels[0].get("price", values_now.get("bid", 0.0)))
        else:
            b0 = _safe_float(values_now.get("bid", 0.0))
        if a0 > 0.0 and b0 > 0.0:
            midpoint = 0.5 * (a0 + b0)

    def _stack_side(levels: List[dict]) -> np.ndarray:
        dist = _pad_truncate([_safe_float(lv.get("distance_to_mid", 0.0)) for lv in levels], n_levels)
        size = _pad_truncate([_safe_float(lv.get("size_BTC", 0.0)) for lv in levels], n_levels)
        noti = _pad_truncate([_safe_float(lv.get("notional_USD", 0.0)) for lv in levels], n_levels)
        canc = _pad_truncate([_safe_float(lv.get("cancel_notional_USD", 0.0)) for lv in levels], n_levels)
        limt = _pad_truncate([_safe_float(lv.get("limit_notional_USD", 0.0)) for lv in levels], n_levels)
        mart = _pad_truncate([_safe_float(lv.get("market_notional_USD", 0.0)) for lv in levels], n_levels)
        return np.concatenate([dist, size, noti, canc, limt, mart]).astype(np.float32)

    ask_feat = _stack_side(ask_levels)
    bid_feat = _stack_side(bid_levels)

    head = np.array([float(cash), float(btc), max(0.0, midpoint)], dtype=np.float32)
    obs = np.concatenate([head, ask_feat, bid_feat]).astype(np.float32)

    if include_aux:
        try:
            tlr = float(time_left_ratio)
        except Exception:
            tlr = 0.0
        try:
            blr = float(backlog_ratio)
        except Exception:
            blr = 0.0
        tlr = float(np.clip(tlr, 0.0, 1.0))
        blr = max(0.0, blr)
        obs = np.concatenate([obs, np.array([tlr, blr], dtype=np.float32)], axis=0)

    return obs