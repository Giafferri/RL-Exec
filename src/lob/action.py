
# =============================================================
# lob/action.py
#
# Action execution functions for the LOB environment.
# =============================================================

import pandas as pd
import numpy as np

def _safe_float(x, default=0.0):
    """
    Convert to float robustly, returning `default` on NaN/Inf or conversion failure.
    """
    try:
        fx = float(x)
        if fx != fx or fx == float("inf") or fx == float("-inf"):
            return default
        return fx
    except Exception:
        return default


def _level_price(level: dict, side: str):
    """
    Return a robust per-level price.

    Prefers an explicit `price` if present and valid; otherwise falls back to
    `midpoint_USD * (1 + distance_to_mid)`. The `side` argument is only used
    for semantic clarity and sorting elsewhere.
    """
    p = level.get("price", None)
    try:
        p = float(p) if p is not None else None
    except Exception:
        p = None
    if p is None or p <= 0.0:
        mid = _safe_float(level.get("midpoint_USD", 0.0), 0.0)
        dist = _safe_float(level.get("distance_to_mid", 0.0), 0.0)
        p = mid * (1.0 + dist)
    return max(0.0, float(p))


def _level_size_btc(level: dict, fallback_price: float | None = None) -> float:
    """
    Return the level size in BTC.

    Prefers `size_BTC` if available; otherwise derives it as `notional_USD / price`.
    A `fallback_price` can be provided to avoid recomputing the level price.
    """
    if "size_BTC" in level:
        return max(0.0, _safe_float(level.get("size_BTC", 0.0), 0.0))
    notional = _safe_float(level.get("notional_USD", 0.0), 0.0)
    price = fallback_price if fallback_price is not None else _level_price(level, side="ask")
    if price <= 0:
        return 0.0
    return max(0.0, notional / price)


def _sorted_by_best_price(levels: list[dict], side: str) -> list[dict]:
    """
    Sort depth so the best prices are consumed first.

    - For asks (buying): ascending price (cheapest first)
    - For bids (selling): descending price (highest first)
    """
    if not levels:
        return []
    reverse = True if side == "bid" else False
    return sorted(levels, key=lambda lv: _level_price(lv, side=side), reverse=reverse)

def transaction_cost(notional_usd, is_maker=False):
    """
    Compute the transaction fee (USD) for a given notional (USD).

    Args:
        notional_usd: Trade notional in USD (spent for buy, received for sell).
        is_maker: Whether the order is maker (rebate tier) or taker (default).

    Returns:
        Fee in USD (enforces a small minimum ticket fee).
    """
    maker_fee_bps = 2.5   # 0.025%
    taker_fee_bps = 7.0   # 0.07%
    min_ticket_usd = 50.0  # minimum fee per order in USD to discourage micro trades

    fee_rate_bps = maker_fee_bps if is_maker else taker_fee_bps
    fee_usd = notional_usd * (fee_rate_bps / 1e4)
    return max(fee_usd, min_ticket_usd)

def buy(amount, ask_values, current_cash, current_btc):
    """
    Execute a BUY sweep against the ask side, walking the book with partial fills.

    Args:
        amount: USD budget to spend **excluding fees**.
        ask_values: List of ask levels (dicts) with at least price/size or notional.
        current_cash: Current wallet cash in USD.
        current_btc: Current wallet BTC.

    Returns:
        [new_cash_usd, new_btc, filled_btc, "buy" or "hold"]
    """
    if amount <= 0 or current_cash <= 0:
        return [current_cash, current_btc, 0.0, "hold"]

    # Ensure we can at least pay fees on the intended spend; if not, trim the spend
    est_fee = transaction_cost(amount, is_maker=False)
    if amount + est_fee > current_cash:
        amount = max(0.0, current_cash - est_fee)
        if amount <= 0:
            return [current_cash, current_btc, 0.0, "hold"]

    total_cost = 0.0  # USD spent on BTC (excluding fees)
    total_btc = 0.0

    # Consume from best (cheapest) ask outward
    for ask in _sorted_by_best_price(ask_values, side="ask"):
        price = _level_price(ask, side="ask")
        size_btc = _level_size_btc(ask, fallback_price=price)
        if price <= 0.0 or size_btc <= 0.0:
            continue

        remaining_budget = max(0.0, amount - total_cost)
        if remaining_budget <= 0.0:
            break

        lvl_notional = size_btc * price
        if lvl_notional <= remaining_budget:
            fill_btc = size_btc
            fill_cost = lvl_notional
        else:
            fill_btc = remaining_budget / max(1e-12, price)
            fill_cost = remaining_budget

        if fill_btc <= 0.0:
            continue

        total_btc += fill_btc
        total_cost += fill_cost

    if total_btc <= 0.0:
        return [current_cash, current_btc, 0.0, "hold"]

    fee_usd = transaction_cost(total_cost, is_maker=False)
    if total_cost + fee_usd > current_cash:
        # Scale down proportionally to fit cash (rare but safe)
        max_spend = max(0.0, current_cash - fee_usd)
        if max_spend <= 0.0:
            return [current_cash, current_btc, 0.0, "hold"]
        scale = max_spend / max(1e-12, total_cost)
        total_cost = max_spend
        total_btc *= scale

    current_cash -= (total_cost + fee_usd)
    current_btc += total_btc

    return [current_cash, current_btc, total_btc, "buy"]

def sell(amount, bid_values, current_cash, current_btc):
    """
    Execute a SELL sweep against the bid side, walking the book with partial fills.

    Args:
        amount: BTC amount to sell **excluding fees** (capped at holdings).
        bid_values: List of bid levels (dicts) with at least price/size or notional.
        current_cash: Current wallet cash in USD.
        current_btc: Current wallet BTC.

    Returns:
        [new_cash_usd, new_btc, filled_btc, "sell" or "hold"]
    """
    if amount <= 0 or current_btc <= 0:
        return [current_cash, current_btc, 0.0, "hold"]

    # Do not exceed holdings
    amount = min(amount, current_btc)

    total_revenue = 0.0 # USD received before fees
    total_btc_sold = 0.0

    # Consume from best (highest) bid outward
    for bid in _sorted_by_best_price(bid_values, side="bid"):
        price = _level_price(bid, side="bid")
        size_btc = _level_size_btc(bid, fallback_price=price)
        if price <= 0.0 or size_btc <= 0.0:
            continue

        remaining = max(0.0, amount - total_btc_sold)
        if remaining <= 0.0:
            break

        fill_btc = min(size_btc, remaining)
        total_btc_sold += fill_btc
        total_revenue += fill_btc * price

    if total_btc_sold <= 0.0:
        return [current_cash, current_btc, 0.0, "hold"]

    fee_usd = transaction_cost(total_revenue, is_maker=False)
    if total_revenue <= fee_usd:
        # If fees would exceed revenue, skip trade
        return [current_cash, current_btc, 0.0, "hold"]

    current_cash += (total_revenue - fee_usd)
    current_btc -= total_btc_sold

    return [current_cash, current_btc, total_btc_sold, "sell"]


def choose_action(values, current_cash, current_btc):
    """
    Minimal CLI helper for manual experiments (not used in training).

    Prompts the user to buy in USD, sell in BTC, hold, or quit; then executes
    the selected action against the provided LOB snapshot.
    """
    input_action = input("Action [b=buy USD→BTC, s=sell BTC→USD, h=hold, q=quit]: ").strip().lower()

    # Get order book values at the given timestamp
    ask_values = values['ask_values']
    bid_values = values['bid_values']

    if input_action == 'b':
        amount = float(input("USD amount to spend (excluding fees): "))
        return buy(amount, ask_values, current_cash, current_btc)
    elif input_action == 's':
        amount = float(input("BTC amount to sell (excluding fees): "))
        return sell(amount, bid_values, current_cash, current_btc)
    elif input_action == 'h':
        return [current_cash, current_btc, 0, "hold"]
    elif input_action == 'q':
        exit()
    else:
        pass