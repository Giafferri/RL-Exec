# =============================================================
# lob/performance.py
#
# Performance and valuation functions for the LOB environment.
# =============================================================

def _safe_float(x, default: float = 0.0) -> float:
    """
    Best-effort float conversion with NaN guard.

    Parameters:

    x : Any
        Value to convert.
    default : float
        Fallback value returned on conversion failure or NaN.

    Returns:

    float
        Parsed float or `default` if parsing fails or yields NaN.
    """
    try:
        fx = float(x)
        if fx != fx:  # NaN
            return float(default)
        return fx
    except Exception:
        return float(default)


def _level_price(lv: dict, mid_fallback: float = 0.0) -> float:
    """
    Best-effort extraction of a level price.

    Preference order:
    1) explicit `price`
    2) `midpoint_USD * (1 + distance_to_mid)`
    3) `mid_fallback`

    Parameters:

    lv : dict
        A depth level record (e.g., from bid/ask ladders).
    mid_fallback : float
        Midpoint to fall back on when fields are missing.

    Returns:

    float
        A non-negative price estimate (0.0 if unavailable).
    """
    p = lv.get("price")
    if p is not None:
        return _safe_float(p)
    mid = _safe_float(lv.get("midpoint_USD", mid_fallback))
    dist = _safe_float(lv.get("distance_to_mid", 0.0))
    if mid > 0.0:
        return mid * (1.0 + dist)
    return _safe_float(mid_fallback)


def _level_size_btc(lv: dict, price_hint: float = 0.0) -> float:
    """
    Return level size in BTC; if absent, derive from `notional_USD / price`.

    Parameters:

    lv : dict
        Depth level record containing `size_BTC` or `notional_USD`.
    price_hint : float
        Price used when deriving BTC from notional; if non-positive, size
        cannot be inferred from `notional_USD`.

    Returns:

    float
        Non-negative size in BTC (0.0 if unavailable).
    """
    size = lv.get("size_BTC")
    if size is not None:
        return max(0.0, _safe_float(size))
    notional = _safe_float(lv.get("notional_USD", 0.0))
    price = _safe_float(price_hint)
    if price <= 0.0 and notional > 0.0:
        # Can't infer BTC size without a valid price
        return 0.0
    return max(0.0, notional / price) if price > 0.0 else 0.0


def _best_midpoint(values_at_ts: dict) -> float:
    """
    Robust midpoint extraction from a snapshot dict.

    Preference order:
    1. direct fields: `mid`, `midpoint`, `midpoint_USD`
    2. best ask/bid `price` average (if both available)
    3. level-0 `midpoint_USD` from ask or bid ladders

    Returns:

    float
        Midpoint in USD if available, else 0.0.
    """
    # Direct fields commonly present
    for k in ("mid", "midpoint", "midpoint_USD"):
        v = values_at_ts.get(k)
        if v is not None:
            mv = _safe_float(v)
            if mv > 0.0:
                return mv
    ask_values = values_at_ts.get("ask_values") or []
    bid_values = values_at_ts.get("bid_values") or []
    # Try explicit best prices
    a_price = _safe_float(ask_values[0].get("price")) if ask_values else 0.0
    b_price = _safe_float(bid_values[0].get("price")) if bid_values else 0.0
    if a_price > 0.0 and b_price > 0.0:
        return 0.5 * (a_price + b_price)
    # As a last resort use midpoint_USD at level 0 if present
    a_mid = _safe_float(ask_values[0].get("midpoint_USD")) if ask_values else 0.0
    b_mid = _safe_float(bid_values[0].get("midpoint_USD")) if bid_values else 0.0
    if a_mid > 0.0:
        return a_mid
    if b_mid > 0.0:
        return b_mid
    return 0.0


# --- Main API -----------------------------------------------------------------

def get_performance(initial_cash, initial_btc, current_cash, current_btc, values_at_ts, goal, target, duration):
    """
    Compute portfolio value, PnL, and goal achievement at a timestamp.

    Parameters
    ----------
    initial_cash : float
        Starting cash (USD).
    initial_btc : float
        Starting BTC inventory.
    current_cash : float
        Cash after executing the schedule (USD).
    current_btc : float
        BTC after executing the schedule.
    values_at_ts : dict
        Snapshot dict at the terminal timestamp (includes ladders/mid).
    goal : {'btc', 'cash'}
        Liquidation objective; checked against `target`.
    target : float
        Residual fraction target (e.g., 0.10 means keep 10%).
    duration : int
        Episode duration in steps/seconds (metadata only).

    Returns
    -------
    dict
        A mapping with mark price, portfolio value, PnL (abs/%), goal flag,
        and the inputs echoed for traceability.
    """
    mid = _best_midpoint(values_at_ts)

    # If we still have no price, treat as zero-change to avoid NaNs
    if mid <= 0.0:
        total_value = float(current_cash)
        initial_value = float(initial_cash)
    else:
        total_value = float(current_cash) + float(current_btc) * mid
        initial_value = float(initial_cash) + float(initial_btc) * mid

    pnl = total_value - initial_value
    pnl_fraction = (pnl / initial_value) if initial_value != 0.0 else 0.0
    pnl_percentage = pnl_fraction * 100.0

    # Determine if the schedule goal was met (by inventory/cash, not PnL)
    if goal == 'cash':
        achieved = float(current_cash) <= float(initial_cash) * float(target)
    else:  # goal == 'btc'
        achieved = float(current_btc) <= float(initial_btc) * float(target)

    return {
        'cash_at_t': float(current_cash),
        'btc_at_t': float(current_btc),
        'initial_cash': float(initial_cash),
        'initial_btc': float(initial_btc),
        'mark_price': float(mid),
        'total_portfolio_value': float(total_value),
        'pnl': float(pnl),
        'pnl_fraction': float(pnl_fraction),
        'pnl_percentage': float(pnl_percentage),
        'achieved_goal': bool(achieved),
        'duration': int(duration),
        'target': float(target),
        'goal': str(goal),
    }


def slippage(values, quantity_BTC=None, quantity_USD=None, side="buy"):
    """
    Estimate slippage of a market order from depth ladders.

    Slippage is expressed in percent relative to mid:
    - For buys: positive means paying above mid.
    - For sells: positive means receiving below mid.

    Parameters
    ----------
    values : dict
        Snapshot dict with `ask_values`, `bid_values`, and mid fields.
    quantity_BTC : float, optional
        Order size in BTC. Mutually exclusive with `quantity_USD`.
    quantity_USD : float, optional
        Order notional in USD; converted to BTC using mid.
    side : {'buy', 'sell'}
        Market order side.

    Returns
    -------
    float | None
        Slippage in percent if enough depth and a valid mid are available;
        otherwise `None`.
    """
    ask_values = values.get('ask_values', []) or []
    bid_values = values.get('bid_values', []) or []

    mid = _best_midpoint(values)
    if mid <= 0.0:
        return None

    # Determine desired size in BTC
    if quantity_BTC is None:
        if quantity_USD is None:
            raise ValueError("Provide quantity_BTC or quantity_USD")
        quantity_BTC = float(quantity_USD) / mid if mid > 0.0 else 0.0
    else:
        quantity_BTC = float(quantity_BTC)

    if quantity_BTC <= 0.0:
        return 0.0

    # Build (price, size) ladder for the chosen side
    fills = []
    if side == "buy":
        # Best ask first: ascending price
        for lv in sorted(ask_values, key=lambda x: _level_price(x, mid)):
            p = _level_price(lv, mid)
            s = _level_size_btc(lv, p)
            if p > 0.0 and s > 0.0:
                fills.append((p, s))
    else:
        # Best bid first: descending price
        for lv in sorted(bid_values, key=lambda x: _level_price(x, mid), reverse=True):
            p = _level_price(lv, mid)
            s = _level_size_btc(lv, p)
            if p > 0.0 and s > 0.0:
                fills.append((p, s))

    if not fills:
        return None

    remain = quantity_BTC
    notional = 0.0
    filled = 0.0

    for p, s in fills:
        take = s if s <= remain else remain
        notional += take * p
        filled += take
        remain -= take
        if remain <= 1e-12:
            break

    if filled < max(1e-12, quantity_BTC - 1e-12):
        return None  # not enough depth

    exec_price = notional / filled if filled > 0.0 else 0.0
    if exec_price <= 0.0:
        return None

    if side == "buy":
        return (exec_price - mid) / mid * 100.0
    else:
        return (mid - exec_price) / mid * 100.0


def print_performance_summary(performance):
    """
    Print a compact, human-readable performance summary.

    Parameters
    ----------
    performance : dict
        Mapping produced by `get_performance`.
    """
    mp = float(performance.get('mark_price', 0.0))
    cash = float(performance.get('cash_at_t', 0.0))
    btc = float(performance.get('btc_at_t', 0.0))
    tpv = float(performance.get('total_portfolio_value', 0.0))
    pnl = float(performance.get('pnl', 0.0))
    pnl_pct = float(performance.get('pnl_percentage', 0.0))
    achieved = bool(performance.get('achieved_goal', False))

    print("[lob.perf] Performance summary")
    print(f"  mark_price_usd={mp:.2f}")
    print(f"  final_cash_usd={cash:.2f}")
    print(f"  final_btc={btc}")
    print(f"  total_portfolio_value_usd={tpv:.2f}")
    print(f"  pnl_usd={pnl:.2f}  pnl_pct={pnl_pct:.2f}%")
    print(f"  achieved_goal={'Yes' if achieved else 'No'}")