# =============================================================
# rl/reward.py
#
# Reward shaping for the RL agent trading on a LOB.
# =============================================================

from __future__ import annotations

# -----------------------------
# Tunable weights
# -----------------------------
PNL_WEIGHT_STEP = 100.0
PNL_WEIGHT_FINAL = 5.0

# Schedule adherence penalty coefficient (per-step)
SCHEDULE_PENALTY = 0.03

# Final shortfall penalty (if target not met at episode end)
FINAL_SHORTFALL_PENALTY = 1000.0
AHEAD_BONUS = 0.005 

# Delta clipping to reduce variance of the PnL term
DELTA_CLIP = 0.02
REWARD_CLIP_ABS = 10.0

# Trade-aware shaping multipliers
TRADE_DELTA_GAIN_MULT = 1.5 
TRADE_DELTA_LOSS_MULT = 0.5
SLIPPAGE_COEF = 0.02 
SLIPPAGE_NEG_BONUS = 0.002

_EPS = 1e-9

# -----------------------------
# Helpers
# -----------------------------

def _progress_ratio(step: int, duration: int) -> float:
    """Return progress ∈ [0, 1] given the current `step` and `duration`.

    Args:
        step: 1-based step index within the episode.
        duration: Total number of steps in the episode.

    Returns:
        A float in [0, 1] (safe when `duration <= 0`).
    """
    if duration <= 0:
        return 1.0
    r = step / float(max(1, duration))
    return 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)


def _expected_remaining(initial: float, target: float, progress: float) -> float:
    """Linear schedule of remaining quantity.

    The schedule linearly moves from `initial` at t=0 to `initial*target` at t=1.

    Args:
        initial: Starting quantity (cash or BTC, depending on the goal).
        target: Target fraction to keep at the end (e.g., 0.10 means keep 10%).
        progress: Fraction of episode elapsed in [0, 1].

    Returns:
        Expected remaining quantity at the given progress.
    """
    return float(initial) * (1.0 - (1.0 - float(target)) * progress)


def _schedule_slack_ratio(perf: dict, step: int) -> float:
    """How far we are *behind* a linear schedule, as a ratio in [0, +∞).

    For `goal='btc'`, remaining is measured in BTC; for `goal='cash'`, in USD notional.
    The ratio is defined as max(0, current - expected) divided by the total to liquidate.
    Returns 0 when on or ahead of schedule.

    Args:
        perf: Dictionary with fields from the environment (see `compute_reward_at_t`).
        step: Current step index (1-based).

    Returns:
        Non-negative float; 0 means on/ahead of schedule.
    """
    progress = _progress_ratio(step, int(perf.get('duration', 0)))
    goal = perf.get('goal', 'btc')
    target = float(perf.get('target', 0.0))

    if goal == 'cash':
        initial = float(perf.get('initial_cash', 0.0))
        current = float(perf.get('cash_at_t', 0.0))
    else:
        initial = float(perf.get('initial_btc', 0.0))
        current = float(perf.get('btc_at_t', 0.0))

    expected_rem = _expected_remaining(initial, target, progress)
    slack = current - expected_rem
    total_to_liq = initial * max(0.0, (1.0 - target))

    if total_to_liq <= _EPS:
        return 0.0
    return max(0.0, slack) / max(total_to_liq, _EPS)


def _schedule_ahead_ratio(perf: dict, step: int) -> float:
    """How far we are *ahead of* a linear schedule, as a ratio in [0, +∞).

    Defined symmetrically to `_schedule_slack_ratio` as max(0, expected - current)
    divided by the total to liquidate.

    Args:
        perf: Dictionary with fields from the environment (see `compute_reward_at_t`).
        step: Current step index (1-based).

    Returns:
        Non-negative float; 0 means on/behind schedule.
    """
    progress = _progress_ratio(step, int(perf.get('duration', 0)))
    goal = perf.get('goal', 'btc')
    target = float(perf.get('target', 0.0))

    if goal == 'cash':
        initial = float(perf.get('initial_cash', 0.0))
        current = float(perf.get('cash_at_t', 0.0))
    else:
        initial = float(perf.get('initial_btc', 0.0))
        current = float(perf.get('btc_at_t', 0.0))

    expected_rem = _expected_remaining(initial, target, progress)
    ahead = expected_rem - current
    total_to_liq = initial * max(0.0, (1.0 - target))

    if total_to_liq <= _EPS:
        return 0.0
    return max(0.0, ahead) / max(total_to_liq, _EPS)


# -----------------------------
# Main reward functions
# -----------------------------

def compute_reward_at_t(performance: dict, sum_rewards: float = 0.0, step: int = 1, previous_pnl: float = 0.0):
    """Per-step reward combining PnL, schedule adherence, and execution quality.

    The signal is dominated by a delta-PnL term (variance-clipped), shaped by:
      • behind/ahead of a linear schedule (penalty/bonus scaled by urgency),
      • trade-aware boosts/damps (bigger weight when a trade occurs),
      • slippage penalties/bonuses on trade steps,
      • damping of over-intense sell bursts late in the episode.

    Preferred inputs from `performance`:
      - 'pnl_fraction' (float): mark-to-market PnL relative to reference.
      - 'time_left_ratio' (float): remaining time fraction, if provided by the env.
      - 'backlog_signed' (float): signed deviation vs schedule (pos=behind, neg=ahead).
      - 'trade_qty_btc' (float), 'trade_side' ('sell'/'buy'), 'trade_slippage_pct' (float).
      - 'initial_btc'/'initial_cash', 'btc_at_t'/'cash_at_t', 'goal', 'target', 'duration'.

    Args:
        performance: Dictionary emitted by the environment at the current step.
        sum_rewards: Running sum of rewards so far (used to return an updated total).
        step: 1-based step index.
        previous_pnl: PnL fraction at the previous step (for delta computation).

    Returns:
        A list [reward_at_t, updated_sum_rewards, current_pnl_fraction].
    """
    pnl_frac = float(performance.get('pnl_fraction', 0.0))

    # Progress & urgency
    progress = _progress_ratio(step, int(performance.get('duration', 0)))
    # Prefer env-provided time_left_ratio; fallback to 1 - progress
    time_left_ratio = float(performance.get('time_left_ratio', max(0.0, 1.0 - progress)))
    urgency = 1.0 - time_left_ratio  # 0 early, 1 at the end
    urg2 = urgency * urgency

    # PnL component (delta-based after t=1), with delta clipping to reduce variance
    if step == 1:
        delta_used = pnl_frac
    else:
        delta_used = pnl_frac - float(previous_pnl)

    # clip delta_used to tame variance
    if delta_used > DELTA_CLIP:
        delta_used = DELTA_CLIP
    elif delta_used < -DELTA_CLIP:
        delta_used = -DELTA_CLIP

    base_reward = PNL_WEIGHT_STEP * ((1.0 + delta_used) ** 2 - 1.0)

    # Trade-aware boosting of the delta term
    trade_qty = float(performance.get('trade_qty_btc', 0.0) or 0.0)
    trade_qty = abs(trade_qty)

    # Normalize trade size by planned liquidation in BTC to make the shaping scale-free
    init_btc = float(performance.get('initial_btc', 0.0) or 0.0)
    tgt = float(performance.get('target', 0.0) or 0.0)
    denom_btc = init_btc * max(0.0, 1.0 - tgt)
    if denom_btc <= _EPS:
        denom_btc = max(init_btc, 1.0)
    trade_intensity = trade_qty / denom_btc
    if trade_intensity < 0.0:
        trade_intensity = 0.0
    if trade_intensity > 1.0:
        trade_intensity = 1.0

    if trade_qty > 0.0:
        if delta_used >= 0.0:
            # Strongly reward trades that produce positive mark-to-market delta
            base_reward *= (1.0 + TRADE_DELTA_GAIN_MULT * trade_intensity)
        else:
            # Damp negative deltas on trade steps to avoid over-penalizing exploration
            damp = 1.0 - (1.0 - TRADE_DELTA_LOSS_MULT) * trade_intensity
            base_reward *= max(0.0, damp)

    # Schedule shaping: prefer signed backlog from env if present
    # - backlog_signed: signed deviation vs schedule (pos = behind, neg = ahead)
    # - backlog_ratio: legacy non-negative backlog 
    if 'backlog_signed' in performance:
        b = float(performance.get('backlog_signed', 0.0))
        behind = max(0.0, b)
        ahead  = max(0.0, -b)
    elif 'backlog_ratio' in performance:
        # Legacy path: backlog_ratio is non-negative 
        b = float(performance.get('backlog_ratio', 0.0))
        behind = max(0.0, b)
        ahead  = 0.0
    else:
        # Fallback on internal computation
        behind = _schedule_slack_ratio(performance, step)
        ahead  = _schedule_ahead_ratio(performance, step)

    # Penalty grows with urgency; keep gentle early, stronger late
    penalty = SCHEDULE_PENALTY * (0.15 + 0.85 * urg2) * (behind ** 2)
    bonus   = AHEAD_BONUS       * (0.10 + 0.90 * urg2) * (ahead  ** 2)

    # Slippage shaping (only when a trade happened)
    slippage_pct = float(performance.get('trade_slippage_pct', 0.0) or 0.0)
    if trade_qty > 0.0:
        # Penalize bad (positive) slippage more as time runs out; tiny bonus for negative slippage
        slip_pen   = SLIPPAGE_COEF      * (0.10 + 0.90 * urg2) * max(0.0,  slippage_pct) * trade_intensity
        slip_bonus = SLIPPAGE_NEG_BONUS * (0.10 + 0.90 * urg2) * max(0.0, -slippage_pct) * trade_intensity
    else:
        slip_pen, slip_bonus = 0.0, 0.0

    # Dump penalty: penalize overshoot intensity (sell bursts), stronger late
    DUMP_INTENSITY_COEF = 0.02
    dump_penalty = 0.0
    if trade_qty > 0.0 and str(performance.get('trade_side')) == 'sell':
        # recommended intensity grows with behind and urgency
        rec_intensity = min(1.0, max(0.0, (behind + 0.05) * (0.40 + 0.60 * urgency)))
        overshoot = max(0.0, trade_intensity - rec_intensity)
        dump_penalty = DUMP_INTENSITY_COEF * (0.25 + 0.75 * urg2) * (overshoot ** 2)

    reward_at_t = base_reward - penalty + bonus - slip_pen + slip_bonus - dump_penalty

    # Clip the final per-step reward to stabilize gradients
    if reward_at_t > REWARD_CLIP_ABS:
        reward_at_t = REWARD_CLIP_ABS
    elif reward_at_t < -REWARD_CLIP_ABS:
        reward_at_t = -REWARD_CLIP_ABS

    updated_sum = (sum_rewards if step > 1 else 0.0) + reward_at_t
    return [reward_at_t, updated_sum, pnl_frac]


def compute_final_reward(performance: dict, sum_rewards: float = 0.0) -> float:
    """Terminal reward with a PnL term and a bounded shortfall penalty.

    The penalty scales with the fraction of the *total to liquidate* that remains
    above the target level at episode end, and is capped by `FINAL_SHORTFALL_PENALTY`.

    Args:
        performance: Dictionary with terminal fields (goal, target, initial/current state).
        sum_rewards: (Unused in the computation; kept for interface symmetry).

    Returns:
        Final scalar reward (float).
    """
    pnl_frac = float(performance.get('pnl_fraction', 0.0))
    final_reward = PNL_WEIGHT_FINAL * (((1.0 + pnl_frac) ** 2 - 1.0) / 2.0)

    goal = performance.get('goal', 'btc')
    target = float(performance.get('target', 0.0))

    if goal == 'cash':
        initial = float(performance.get('initial_cash', 0.0))
        current = float(performance.get('cash_at_t', 0.0))
    else:
        initial = float(performance.get('initial_btc', 0.0))
        current = float(performance.get('btc_at_t', 0.0))

    target_level = initial * target
    total_to_liq = max(0.0, initial - target_level)

    if total_to_liq <= _EPS:
        return final_reward

    # Remaining above the target level
    remaining = max(0.0, current - target_level)
    # Ratio in [0,1] when using total-to-liquidate as denominator
    shortfall_ratio = remaining / max(total_to_liq, _EPS)

    final_reward -= FINAL_SHORTFALL_PENALTY * shortfall_ratio
    return final_reward