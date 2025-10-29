
# =============================================================
# lob/indicators.py
#
# Indicator functions for limit order book (LOB) data analysis.
# =============================================================

import pandas as pd

# =============================================================
# Best bids and asks
# =============================================================

def best_bid(bid_values):
    """
    Return the best bid distance-to-mid (most aggressive bid) from a bid ladder.

    Args:
        bid_values (list[dict]): Bid levels at a timestamp (closest-to-mid first), each with
            'size_BTC' and 'distance_to_mid'.

    Returns:
        float | None: Distance-to-mid of the best bid, or None if no positive-size level exists.
    """
    if not bid_values:
        return None
    
    # Get the best bid (highest bid price with non-zero size)
    best_bid = None
    for lvl in bid_values:
        if lvl['size_BTC'] == 0:
            continue
        else:
            best_bid = lvl['distance_to_mid']
            break
    
    return best_bid

def best_ask(ask_values):
    """
    Return the best ask distance-to-mid (most aggressive ask) from an ask ladder.

    Args:
        ask_values (list[dict]): Ask levels at a timestamp (closest-to-mid first), each with
            'size_BTC' and 'distance_to_mid'.

    Returns:
        float | None: Distance-to-mid of the best ask, or None if no positive-size level exists.
    """
    if not ask_values:
        return None
    
    # Get the best ask (lowest ask price with non-zero size)
    best_ask = None
    for lvl in reversed(ask_values):
        if lvl['size_BTC'] == 0:
            continue
        else:
            best_ask = lvl['distance_to_mid']
            break
    
    return best_ask

def best_bid_size(bid_values):
    """
    Return the size (BTC) available at the best bid.

    Args:
        bid_values (list[dict]): Bid levels at a timestamp.

    Returns:
        float | None: Size at the best bid, or None if no positive-size bid exists.
    """
    if not bid_values:
        return None
    
    # Get the size of the best bid (highest bid price with non-zero size)
    best_bid_size = None
    for lvl in bid_values:
        if lvl['size_BTC'] == 0:
            continue
        else:
            best_bid_size = lvl['size_BTC']
            break
    
    return best_bid_size

def best_ask_size(ask_values):
    """
    Return the size (BTC) available at the best ask.

    Args:
        ask_values (list[dict]): Ask levels at a timestamp.

    Returns:
        float | None: Size at the best ask, or None if no positive-size ask exists.
    """
    if not ask_values:
        return None
    
    # Get the size of the best ask (lowest ask price with non-zero size)
    best_ask_size = None
    for lvl in reversed(ask_values):
        if lvl['size_BTC'] == 0:
            continue
        else:
            best_ask_size = lvl['size_BTC']
            break
    
    return best_ask_size

# =============================================================
# Midpoints & spreads
# =============================================================

def midpoint(values):
    """
    Return the mid price (USD) using the precomputed `midpoint_USD` carried on the top level.

    Args:
        values (dict): Dict with keys 'bid_values' and 'ask_values'.

    Returns:
        float | None: Midpoint in USD, or None if unavailable.
    """
    ask_values = values['ask_values']
    bid_values = values['bid_values']
    if ask_values:
        return ask_values[0]['midpoint_USD']
    if bid_values:
        return bid_values[0]['midpoint_USD']
    return None

def VAMP(values):
    """
    Compute a volume-adjusted mid price (VAMP) across visible levels.

    Defined as the average of side-wise volume-weighted prices:
        VAMP = 0.5 * (sum(notional_ask)/sum(size_ask) + sum(notional_bid)/sum(size_bid))

    Args:
        values (dict): Dict with 'bid_values' and 'ask_values'.

    Returns:
        float | None: VAMP in USD, or None if either side has zero total size.
    """
    ask_values = values['ask_values']
    bid_values = values['bid_values']

    # Check if there are valid ask and bid values
    if not ask_values or not bid_values:
        return None

    # Ensure that there are non-zero sizes for both asks and bids
    if sum(ask['size_BTC'] for ask in ask_values) == 0 or sum(bid['size_BTC'] for bid in bid_values) == 0:
        return None
    
    # Calculate the volume weighted midpoint
    weighted_total_ask = sum(ask['notional_USD'] for ask in ask_values) / sum(ask['size_BTC'] for ask in ask_values)
    weighted_total_bid = sum(bid['notional_USD'] for bid in bid_values) / sum(bid['size_BTC'] for bid in bid_values)
    
    return (weighted_total_ask + weighted_total_bid) / 2

def VAMP_var_midpoint(values):
    """
    VAMP deviation from midpoint, expressed in percent.

    Args:
        values (dict): Dict with 'bid_values' and 'ask_values'.

    Returns:
        float | None: 100 * (VAMP - midpoint) / midpoint, or None if inputs are invalid.
    """
    vamp = VAMP(values)
    midpoint_price = midpoint(values)

    if vamp is None or midpoint_price is None or midpoint_price == 0:
        return None
    
    return (vamp - midpoint_price) / midpoint_price * 100

def VAMP_ask(ask_values):
    """
    Volume-weighted average ask price across visible levels.

    Args:
        ask_values (list[dict]): Ask levels at a timestamp.

    Returns:
        float: USD price; returns 0 when total ask size is zero.
    """
    total_size = sum(ask['size_BTC'] for ask in ask_values)
    if total_size == 0:
        return 0
    total_notional = sum(ask['notional_USD'] for ask in ask_values)
    return total_notional / total_size

def VAMP_ask_var_midpoint(ask_values):
    """
    Deviation of the volume-weighted ask price from the midpoint, in percent.

    Args:
        ask_values (list[dict]): Ask levels at a timestamp.

    Returns:
        float | None: 100 * (VAMP_ask - midpoint) / midpoint, or None if inputs are invalid.
    """
    vamp_ask = VAMP_ask(ask_values)
    if not ask_values or ask_values[0]['midpoint_USD'] == 0:
        return None
    midpoint_price = ask_values[0]['midpoint_USD']

    if vamp_ask is None or midpoint_price is None or midpoint_price == 0:
        return None
    
    return (vamp_ask - midpoint_price) / midpoint_price * 100

def VAMP_bid(bid_values):
    """
    Volume-weighted average bid price across visible levels.

    Args:
        bid_values (list[dict]): Bid levels at a timestamp.

    Returns:
        float: USD price; returns 0 when total bid size is zero.
    """
    total_size = sum(bid['size_BTC'] for bid in bid_values)
    if total_size == 0:  # protect against division by zero
        return 0
    total_notional = sum(bid['notional_USD'] for bid in bid_values)
    return total_notional / total_size

def VAMP_bid_var_midpoint(bid_values):
    """
    Deviation of the volume-weighted bid price from the midpoint, in percent.

    Args:
        bid_values (list[dict]): Bid levels at a timestamp.

    Returns:
        float | None: 100 * (VAMP_bid - midpoint) / midpoint, or None if inputs are invalid.
    """
    vamp_bid = VAMP_bid(bid_values)
    if not bid_values or bid_values[0]['midpoint_USD'] == 0:
        return None
    midpoint_price = bid_values[0]['midpoint_USD']

    if vamp_bid is None or midpoint_price is None or midpoint_price == 0:
        return None
    
    return (vamp_bid - midpoint_price) / midpoint_price * 100

def spread(values):
    """
    Distance-to-mid spread between best ask and best bid.

    Args:
        values (dict): Dict with 'bid_values' and 'ask_values'.

    Returns:
        float | None: (best_ask_distance - best_bid_distance), or None if either is missing.
    """
    ask_values = values['ask_values']
    bid_values = values['bid_values']
    return (best_ask(ask_values) - best_bid(bid_values)) if (best_ask(ask_values) is not None and best_bid(bid_values) is not None) else None

def normalized_spread(values):
    """
    Normalized spread in percent using distance-to-mid and the midpoint reference.

    Computes 100 * (best_ask_distance - best_bid_distance) / midpoint_USD.

    Args:
        values (dict): Dict with 'bid_values' and 'ask_values'.

    Returns:
        float | None: Normalized spread (%), or None if inputs are invalid.
    """
    bid_values = values.get('bid_values', [])
    ask_values = values.get('ask_values', [])

    best_bid_dist = best_bid(bid_values)
    best_ask_dist = best_ask(ask_values)

    if best_bid_dist is None or best_ask_dist is None:
        return None

    # Use midpoint as reference: take first available midpoint from ask or bid
    midpoint_val = None
    if ask_values:
        midpoint_val = ask_values[0].get('midpoint_USD')
    elif bid_values:
        midpoint_val = bid_values[0].get('midpoint_USD')

    if midpoint_val is None or midpoint_val == 0:
        return None

    # Normalized spread in percentage
    return (best_ask_dist - best_bid_dist) / midpoint_val * 100

# =============================================================
# Depths, Imbalances, and Ratios
# =============================================================

def bid_depth(bid_values):
    """
    Total bid-side depth (USD notional) across visible levels.

    Args:
        bid_values (list[dict]): Bid levels at a timestamp.

    Returns:
        float | None: Sum of 'notional_USD' if any level has positive notional; otherwise None.
    """
    if not bid_values:
        return None
    return sum(bid['notional_USD'] for bid in bid_values) if any(bid['notional_USD'] > 0 for bid in bid_values) else None

def ask_depth(ask_values):
    """
    Total ask-side depth (USD notional) across visible levels.

    Args:
        ask_values (list[dict]): Ask levels at a timestamp.

    Returns:
        float | None: Sum of 'notional_USD' if any level has positive notional; otherwise None.
    """
    if not ask_values:
        return None
    return sum(ask['notional_USD'] for ask in ask_values) if any(ask['notional_USD'] > 0 for ask in ask_values) else None

def liquidity_ratio(values):
    """
    Liquidity ratio defined as bid_depth / ask_depth (using USD notional).

    Args:
        values (dict): Dict with 'bid_values' and 'ask_values'.

    Returns:
        float | None: Ratio, or None if either side depth is unavailable.
    """
    ask_values = values['ask_values']
    bid_values = values['bid_values']
    return bid_depth(bid_values) / ask_depth(ask_values) if (bid_depth(bid_values) is not None and ask_depth(ask_values) is not None) else None

# =============================================================
# Standard Deviations
# =============================================================

def std_side(values):
    """
    Standard deviation of USD notional across levels on one side.

    Args:
        values (list[dict]): Levels on a single side (bid or ask).

    Returns:
        float | None: Standard deviation, or None if input is empty.
    """
    if not values:
        return None
    
    notional = [lvl['notional_USD'] for lvl in values]
    if not notional:
        return None
    
    return pd.Series(notional).std()

# =============================================================
# Temporal Dynamics (t & t-1)
# =============================================================

def delta_spread(values, previous_values):
    """
    Change in distance-to-mid spread between two consecutive snapshots.

    Args:
        values (dict): Current snapshot with 'bid_values' and 'ask_values'.
        previous_values (dict): Previous snapshot.

    Returns:
        float | None: Current spread minus previous spread, or None on invalid inputs.
    """
    current_spread = spread(values)
    previous_spread = spread(previous_values)

    if current_spread is None or previous_spread is None:
        return None
    
    return current_spread - previous_spread

def delta_midpoint(values, previous_values):
    """
    Change in midpoint (USD) between two snapshots.

    Args:
        values (dict): Current snapshot.
        previous_values (dict): Previous snapshot.

    Returns:
        float | None: Current midpoint minus previous midpoint, or None if unavailable.
    """
    current_midpoint = midpoint(values)
    previous_midpoint = midpoint(previous_values)

    if current_midpoint is None or previous_midpoint is None:
        return None
    
    return current_midpoint - previous_midpoint

def delta_VAMP(values, previous_values):
    """
    Change in VAMP (USD) between two snapshots.

    Args:
        values (dict): Current snapshot.
        previous_values (dict): Previous snapshot.

    Returns:
        float | None: Current VAMP minus previous VAMP, or None if unavailable.
    """
    current_vamp = VAMP(values)
    previous_vamp = VAMP(previous_values)

    if current_vamp is None or previous_vamp is None:
        return None
    
    return current_vamp - previous_vamp

def delta_std_side(side_values, previous_side_values):
    """
    Change in the standard deviation of USD notional on one side.

    Args:
        side_values (list[dict]): Current side levels.
        previous_side_values (list[dict]): Previous side levels.

    Returns:
        float | None: Current std minus previous std, or None if unavailable.
    """
    current_std = std_side(side_values)
    previous_std = std_side(previous_side_values)

    if current_std is None or previous_std is None:
        return None
    
    return current_std - previous_std

# =============================================================
# Others
# =============================================================

def micro_price(values):
    """
    Micro-price-like metric at top-of-book using distance-to-mid and sizes.

    Uses:
        micro_metric = (d_ask * Q_bid + d_bid * Q_ask) / (Q_bid + Q_ask),
    where d_* are distance-to-mid (ask positive, bid negative) and Q_* are sizes at the best levels.

    Args:
        values (dict): Dict with 'bid_values' and 'ask_values'.

    Returns:
        float | None: Dimensionless micro metric, or None on invalid inputs.
    """
    ask_values = values['ask_values']
    bid_values = values['bid_values']

    # Basic presence checks
    if not ask_values or not bid_values:
        return None

    # Extract best distances and sizes
    p_bid = best_bid(bid_values)
    p_ask = best_ask(ask_values)
    q_bid = best_bid_size(bid_values)
    q_ask = best_ask_size(ask_values)

    # Validate extracted values
    if (p_bid is None or p_ask is None or q_bid is None or q_ask is None):
        return None

    denom = q_bid + q_ask
    if denom == 0:
        return None

    return (p_ask * q_bid + p_bid * q_ask) / denom

# =============================================================
# Imbalance of the book
# =============================================================

def imbalance_top_of_book(values, threshold=0.8):
    """
    Top-of-book imbalance in [-1, 1] with a qualitative label.

    imbalance = (Q_bid - Q_ask) / (Q_bid + Q_ask)
    A threshold labels regimes as "buy pressure", "sell pressure", or "balanced".

    Args:
        values (dict): Dict with 'bid_values' and 'ask_values'.
        threshold (float): Absolute threshold for qualitative labeling.

    Returns:
        tuple[float | None, str]: (imbalance value or None, interpretation string).
    """
    q_bid = values and values['bid_values'] and values['bid_values'][0]['size_BTC']
    q_ask = values and values['ask_values'] and values['ask_values'][-1]['size_BTC']

    if not q_bid or not q_ask or (q_bid + q_ask) == 0:
        return None, "No data"

    imbalance = (q_bid - q_ask) / (q_bid + q_ask)

    # Interpretation
    if imbalance >= threshold:
        interpretation = "Book tilted to BID (buy pressure)"
    elif imbalance <= -threshold:
        interpretation = "Book tilted to ASK (sell pressure)"
    else:
        interpretation = "Book balanced"

    return float(imbalance), interpretation

def imbalance_multi_levels(values, n_levels=5, threshold=0.6):
    """
    Multi-level imbalance over the first/last n levels on each side.

    imbalance = (Σ Q_bid - Σ Q_ask) / (Σ Q_bid + Σ Q_ask)

    Args:
        values (dict): Dict with 'bid_values' and 'ask_values'.
        n_levels (int): Number of levels per side to aggregate.
        threshold (float): Absolute threshold for qualitative labeling.

    Returns:
        tuple[float | None, str]: (imbalance value or None, interpretation string).
    """
    bid_values = values.get('bid_values', [])[:n_levels]
    ask_values = values.get('ask_values', [])[-n_levels:]


    total_bid = sum(lvl['size_BTC'] for lvl in bid_values)
    total_ask = sum(lvl['size_BTC'] for lvl in ask_values)


    if (total_bid + total_ask) == 0:
      return None, "No data"


    imbalance = (total_bid - total_ask) / (total_bid + total_ask)


    if imbalance >= threshold:
     interpretation = "Book tilted to BID (buy pressure)"
    elif imbalance <= -threshold:
     interpretation = "Book tilted to ASK (sell pressure)"
    else:
     interpretation = "Book balanced"

    return float(imbalance), interpretation

def order_flow_imbalance(current_values, previous_values, n_levels=1, threshold=0.05):
    """
    Order Flow Imbalance (OFI) between two snapshots using size_BTC over n_levels.

    Args:
        current_values (dict): Current snapshot.
        previous_values (dict): Previous snapshot.
        n_levels (int): Levels per side to include (top for bids, top for asks).
        threshold (float): Absolute threshold for labeling.

    Returns:
        tuple[float | None, str]: (OFI value or None, label: "Buy pressure" / "Sell pressure" / "Balanced").
    """
    bid_now = sum(l['size_BTC'] for l in current_values.get('bid_values', [])[:n_levels])
    ask_now = sum(l['size_BTC'] for l in current_values.get('ask_values', [])[-n_levels:])
    bid_prev = sum(l['size_BTC'] for l in previous_values.get('bid_values', [])[:n_levels])
    ask_prev = sum(l['size_BTC'] for l in previous_values.get('ask_values', [])[-n_levels:])

    delta_bid = bid_now - bid_prev
    delta_ask = ask_now - ask_prev

    denom = delta_bid + delta_ask
    if denom == 0:
        return None, "No change"

    ofi = (delta_bid - delta_ask) / denom

    # Interpretation based on threshold
    if ofi > threshold:
        interpretation = "Buy pressure"
    elif ofi < -threshold:
        interpretation = "Sell pressure"
    else:
        interpretation = "Balanced"

    return float(ofi), interpretation

# =============================================================
# Slippage
# =============================================================

def slippage(values, quantity_BTC, side="buy"):
    """
    Estimate percent slippage for a market order executed against displayed depth.

    The execution price is computed by walking the book until the requested size is filled.
    Slippage is reported as 100 * (exec_price - mid) / mid for buys (sign flipped for sells).

    Args:
        values (dict): Dict with 'bid_values' and 'ask_values'.
        quantity_BTC (float): Order size in BTC.
        side (str): "buy" or "sell".

    Returns:
        float | None: Slippage in percent, or None if depth is insufficient or mid is missing.
    """
    bid_values = values.get('bid_values', [])
    ask_values = values.get('ask_values', [])

    midpoint_val = None
    if ask_values:
        midpoint_val = ask_values[0].get('midpoint_USD')
    elif bid_values:
        midpoint_val = bid_values[0].get('midpoint_USD')

    if midpoint_val is None or midpoint_val == 0:
        return None

    depth = ask_values if side == "buy" else bid_values
    if not depth:
        return None

    filled = 0.0
    cost = 0.0
    levels = depth if side == "buy" else reversed(depth)
    for lvl in levels:
        size = lvl['size_BTC']
        price = lvl['midpoint_USD'] + lvl['distance_to_mid']  # approx actual price
        if filled + size >= quantity_BTC:
            cost += (quantity_BTC - filled) * price
            filled = quantity_BTC
            break
        else:
            cost += size * price
            filled += size

    if filled < quantity_BTC:
        return None  # not enough depth

    exec_price = cost / quantity_BTC
    return (exec_price - midpoint_val) / midpoint_val * 100 if side == "buy" else (midpoint_val - exec_price) / midpoint_val * 100

# =============================================================
# Orderbook Slope using distance to mid
# =============================================================

def orderbook_slope(values, n_levels=5):
    """
    Estimate side-wise order book slope using distance_to_mid and size_BTC over n_levels.

    For each side, slope ≈ |distance_to_mid(first) - distance_to_mid(last)| / depth_size,
    providing a crude steepness measure near the top levels.

    Args:
        values (dict): Dict with 'bid_values' and 'ask_values'.
        n_levels (int): Number of levels per side to consider.

    Returns:
        dict | tuple: {'BID': float, 'ASK': float} on success; (None, None) if insufficient data.
    """
    bid_values = values.get('bid_values', [])[:n_levels]
    ask_values = values.get('ask_values', [])[-n_levels:]

    if not bid_values or not ask_values:
        return None, None

    bid_depth = sum(l['size_BTC'] for l in bid_values)
    ask_depth = sum(l['size_BTC'] for l in ask_values)

    bid_slope = abs((bid_values[0]['distance_to_mid'] - bid_values[-1]['distance_to_mid']) / bid_depth if bid_depth > 0 else None)
    ask_slope = abs((ask_values[-1]['distance_to_mid'] - ask_values[0]['distance_to_mid']) / ask_depth if ask_depth > 0 else None)

    return {
        "BID": round(float(bid_slope), 6),
        "ASK": round(float(ask_slope), 6)
    }

# =============================================================
# Book Pressure Index (BPI)
# =============================================================

def book_pressure_index(values, n_levels=5):
    """
    Book Pressure Index (BPI) using size_BTC weighted by inverse distance_to_mid.

    BPI &gt; 1 suggests buying pressure; BPI &lt; 1 suggests selling pressure.

    Args:
        values (dict): Dict with 'bid_values' and 'ask_values'.
        n_levels (int): Levels per side to include.

    Returns:
        float | None: Pressure index, or None if denominator is zero.
    """
    bid_values = values.get('bid_values', [])[:n_levels]
    ask_values = values.get('ask_values', [])[-n_levels:]

    num = sum(l['size_BTC'] / abs(l['distance_to_mid']) for l in bid_values if l['distance_to_mid'] < 0)
    den = sum(l['size_BTC'] / abs(l['distance_to_mid']) for l in ask_values if l['distance_to_mid'] > 0)

    if den == 0:
        return None
    return num / den

# =============================================================
# Values for RL environment
# =============================================================

def get_rl_values(values, previous_values=None):
    """
    Compute a feature dictionary for the RL environment from one snapshot (and optionally deltas).

    Includes:
      - Top-of-book & mid: best_{bid,ask}, sizes, midpoint
      - Spreads & microstructure: spread, normalized_spread, micro_price
      - Depths & ratios: bid_depth, ask_depth, liquidity_ratio, book_pressure_index
      - VAMP family: vamp, vamp_{ask,bid}, and % deviations
      - Stats: std_{bid,ask}
      - Imbalances: imbalance_top_of_book, imbalance_multi_levels
      - Optional deltas vs previous snapshot and multi-horizon deltas if `values["history"]` is present.

    Args:
        values (dict): Current snapshot with 'bid_values' and 'ask_values'.
        previous_values (dict | None): Previous snapshot for deltas.

    Returns:
        dict: Feature name to value.
    """
    d = {
        # top & mid
        "best_bid": best_bid(values.get('bid_values', [])),
        "best_ask": best_ask(values.get('ask_values', [])),
        "best_bid_size": best_bid_size(values.get('bid_values', [])),
        "best_ask_size": best_ask_size(values.get('ask_values', [])),
        "midpoint": midpoint(values),

        # spreads & microstructure
        "spread": spread(values),
        "normalized_spread": normalized_spread(values),
        "micro_price": micro_price(values),

        # depths & ratios
        "bid_depth": bid_depth(values.get('bid_values', [])),
        "ask_depth": ask_depth(values.get('ask_values', [])),
        "liquidity_ratio": liquidity_ratio(values),
        "book_pressure_index": book_pressure_index(values),

        # VAMP family
        "vamp": VAMP(values),
        "vamp_var_midpoint": VAMP_var_midpoint(values),
        "vamp_ask": VAMP_ask(values.get('ask_values', [])),
        "vamp_ask_var_midpoint": VAMP_ask_var_midpoint(values.get('ask_values', [])),
        "vamp_bid": VAMP_bid(values.get('bid_values', [])),
        "vamp_bid_var_midpoint": VAMP_bid_var_midpoint(values.get('bid_values', [])),

        # stats
        "std_bid": std_side(values.get('bid_values', [])),
        "std_ask": std_side(values.get('ask_values', [])),

        # imbalance
        "imbalance_top_of_book": imbalance_top_of_book(values)[0],
        "imbalance_multi_levels": imbalance_multi_levels(values)[0],
    }

    # If previous_values is given at this step
    if previous_values is not None:
        d.update({
            "delta_spread": delta_spread(values, previous_values),
            "delta_midpoint": delta_midpoint(values, previous_values),
            "delta_vamp": delta_VAMP(values, previous_values),
            "delta_std_bid": delta_std_side(values.get('bid_values', []), previous_values.get('bid_values', [])),
            "delta_std_ask": delta_std_side(values.get('ask_values', []), previous_values.get('ask_values', [])),
            "orderflow_imbalance": order_flow_imbalance(values, previous_values)[0],
        })

    # === Deltas multi-horizons ===
    if previous_values is not None and "history" in values:
        history = values["history"]

        if len(history) >= 10:
            d.update({
                "delta_spread_10": delta_spread(values, history[-10]),
                "delta_midpoint_10": delta_midpoint(values, history[-10]),
                "delta_vamp_10": delta_VAMP(values, history[-10]),
            })

        if len(history) >= 100:
            d.update({
                "delta_spread_100": delta_spread(values, history[-100]),
                "delta_midpoint_100": delta_midpoint(values, history[-100]),
                "delta_vamp_100": delta_VAMP(values, history[-100]),
            })

        if len(history) >= 1000:
            d.update({
                "delta_spread_1000": delta_spread(values, history[-1000]),
                "delta_midpoint_1000": delta_midpoint(values, history[-1000]),
                "delta_vamp_1000": delta_VAMP(values, history[-1000]),
            })

    return d
