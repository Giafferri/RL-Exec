
# =============================================================
# rl/env.py
#
# Custom Gym environment for RL agent trading on a Limit Order Book (LOB).
# =============================================================

from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lob.timestamp import get_values
from lob.performance import get_performance, slippage
from rl.reward import compute_reward_at_t, compute_final_reward
import lob.action as sim_action
import lob.indicators as ind
from rl.obs_builder import build_obs_depth_20

class RLExecEnv(gym.Env):
    """
    Gymnasium environment for execution on historical LOB with endogenous impact.

    Actions:
        0=HOLD, 1=BUY, 2=SELL (opposite-side trades optionally blocked).

    Observations:
        Depth-20 snapshot (both sides) plus optional auxiliary scheduling features
        and precomputed indicators (columns prefixed with 'ind_').

    Rewards:
        Dense implementation-shortfall style; terminal shaping on leftover inventory.

    Notes:
        - One episode = fixed intraday window [start_idx, start_idx+duration).
        - Uses `lob.timestamp.get_values` for snapshots and `lob.action` for fills.
    """
    metadata = {"render_modes": []}
    ACTION_MEANINGS = {0: "HOLD", 1: "BUY", 2: "SELL"}

    def _best_mid(self, values: dict) -> float:
        """Return a robust midpoint from a LOB snapshot.

        Args:
            values (dict): Snapshot with 'bid_values'/'ask_values' and optional
                'mid'/'midpoint'/'midpoint_USD'.
        Returns:
            float: Midpoint price (>0) if derivable, else 0.0.
        """
        try:
            mid = values.get("mid") or values.get("midpoint") or values.get("midpoint_USD")
            if mid is not None:
                mid = float(mid)
                if mid > 0.0 and mid == mid:
                    return mid
        except Exception:
            pass
        ask_values = values.get("ask_values") or []
        bid_values = values.get("bid_values") or []
        try:
            a = float(ask_values[0].get("price", 0.0)) if ask_values else 0.0
        except Exception:
            a = 0.0
        try:
            b = float(bid_values[0].get("price", 0.0)) if bid_values else 0.0
        except Exception:
            b = 0.0
        if a > 0.0 and b > 0.0:
            return 0.5 * (a + b)
        # fallback to midpoint_USD at level 0 if present
        try:
            a_mid = float(ask_values[0].get("midpoint_USD", 0.0)) if ask_values else 0.0
        except Exception:
            a_mid = 0.0
        try:
            b_mid = float(bid_values[0].get("midpoint_USD", 0.0)) if bid_values else 0.0
        except Exception:
            b_mid = 0.0
        if a_mid > 0.0:
            return a_mid
        if b_mid > 0.0:
            return b_mid
        return 0.0

    def _normalize_day_df(self, df):
        """Return a day DataFrame normalized to one row per timestamp (int64 ns), sorted and de-duplicated.

        Coerces 'timestamp_ns' to int64 nanoseconds and drops duplicate timestamps.
        """
        import pandas as pd
        import numpy as np
        if df is None or len(df) == 0:
            return df
        # Coerce timestamp_ns to int64 ns
        if 'timestamp_ns' in df.columns:
            if not np.issubdtype(df['timestamp_ns'].dtype, np.integer):
                # Try strict ISO first, then fallback
                try:
                    df['timestamp_ns'] = pd.to_datetime(df['timestamp_ns'], utc=True, format='ISO8601').astype('int64')
                except Exception:
                    df['timestamp_ns'] = pd.to_datetime(df['timestamp_ns'], utc=True, errors='coerce').astype('int64')
            else:
                # If it is integer-like but not int64, cast
                df['timestamp_ns'] = df['timestamp_ns'].astype('int64')
            df = df.sort_values('timestamp_ns').drop_duplicates('timestamp_ns')
        return df

    # ------------------------------------------------------------------
    # Indicator integration
    # ------------------------------------------------------------------
    def _infer_indicator_schema(self, df_hint=None):
        """Infer indicator columns (prefixed with 'ind_') and set `self._ind_cols`/`self._ind_dim`.

        The search order is: provided `df_hint`, else the active `self.df`, else the
        first of `self.df_list`, else the first file in `self.df_paths`.
        """
        import os
        import pandas as pd

        if df_hint is None:
            df_hint = getattr(self, 'df', None)
            if df_hint is None and getattr(self, 'df_list', None):
                try:
                    df_hint = self.df_list[0]
                except Exception:
                    df_hint = None
            if df_hint is None and getattr(self, 'df_paths', None):
                try:
                    path = self.df_paths[0]
                    ext = os.path.splitext(path)[1].lower()
                    if ext in (".feather", ".ft", ".fth"):
                        df_hint = pd.read_feather(path)
                    elif ext in (".parquet", ".pq"):
                        df_hint = pd.read_parquet(path)
                    else:
                        df_hint = pd.read_csv(path)
                    df_hint = self._normalize_day_df(df_hint)
                except Exception:
                    df_hint = None

        cols = list(df_hint.columns) if df_hint is not None else []
        ind_cols = [c for c in cols if c.startswith('ind_') and c != 'ind_timestamp_ns']
        ind_cols.sort()
        self._ind_cols = ind_cols
        self._ind_dim = len(ind_cols)

    def _build_indicator_matrix_for_active_df(self):
        """Build and cache the per-row indicator matrix for the active day.

        Creates `self._ind_mat` of shape (num_rows, `self._ind_dim`) with float32 values;
        missing/invalid values map to 0. No effect when indicators are disabled or
        unavailable.
        """
        import numpy as np
        if not getattr(self, '_with_indicators', False):
            self._ind_mat = None
            return
        if self.df is None or len(self.df) == 0:
            self._ind_mat = None
            return
        if not getattr(self, '_ind_cols', None):
            # Infer from the active df if we don't have a schema yet
            self._infer_indicator_schema(self.df)
        d = int(getattr(self, '_ind_dim', 0) or 0)
        n = int(len(self.df))
        if d == 0 or n == 0:
            self._ind_mat = None
            return
        mat = np.zeros((n, d), dtype=np.float32)
        present = set(self.df.columns)
        for j, col in enumerate(self._ind_cols):
            if col in present:
                try:
                    arr = self.df[col].to_numpy(dtype=np.float32, copy=False)
                except Exception:
                    arr = np.asarray(self.df[col].values, dtype=np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                if arr.shape[0] == n:
                    mat[:, j] = arr
        self._ind_mat = mat

    def _compute_aux_scheduling_metrics(self):
        """Compute progress/time_left_ratio and liquidation backlog metrics based on current state.

        Returns:
            progress (float): in [0,1]
            time_left_ratio (float): in [0,1], 1 at start → 0 at end
            backlog_pos (float): max(0, signed_backlog)  (how far behind schedule, normalized)
            backlog_signed (float): signed deviation vs schedule (positive = behind, negative = ahead)
            scheduled_remaining (float): how much should remain now (units of asset or cash-to-liquidate)
            actual_remaining (float): how much actually remains now relative to target level
            total_to_liq (float): total to liquidate from t=0 to horizon
        """
        progress = float(self.step_in_episode) / max(1.0, float(self.duration))
        time_left_ratio = float(np.clip(1.0 - progress, 0.0, 1.0))

        if self.goal == 'btc':
            initial = float(self.initial_btc)
            current = float(self.current_btc)
        else:  # goal == 'cash'
            initial = float(self.initial_cash)
            current = float(self.current_cash)

        target_level = initial * float(self.target)
        total_to_liq = max(1e-9, initial - target_level)

        # Remaining to liquidate now
        actual_remaining = current - target_level

        # Linear schedule: how much should remain now
        scheduled_remaining = (1.0 - progress) * total_to_liq

        # Signed deviation normalized by total_to_liq
        backlog_signed = (actual_remaining - scheduled_remaining) / total_to_liq
        backlog_pos = max(0.0, backlog_signed)

        return (progress, time_left_ratio, backlog_pos, backlog_signed,
                scheduled_remaining, actual_remaining, total_to_liq)

    def __init__(
        self,
        df=None,
        df_list=None, # list of preloaded day DataFrames
        df_paths=None, # list of paths to day files (feather/parquet/csv)
        preload=False, # if True, preload df_paths into RAM once
        initial_cash=50_000_000,
        initial_btc=10_000,
        goal='btc',
        target=0.1,
        duration=3600,
        start_idx=40,
        trade_fraction=0.10,
        use_builder=False,
        obs_kind='depth20+ind',  # kept for backward compatibility but ignored internally
        include_aux=True,
        allow_opposite_side_trades=True,
        size_policy='adaptive',
        size_boost_backlog=0.8,
        size_damp_ahead=0.5,
        size_boost_urgency=1.0,
        min_trade_usd=500.0,
        min_size_factor=0.25,
        max_size_factor=4.0,
    ):
        """Initialize the execution environment.

        Args:
            df (pd.DataFrame | None): Active day (wide schema).
            df_list (list[pd.DataFrame] | None): Preloaded days.
            df_paths (list[str] | None): Paths to day files; used if not preloaded.
            preload (bool): If True, preload `df_paths` into RAM once.
            initial_cash (float): Starting USD cash.
            initial_btc (float): Starting BTC inventory.
            goal (str): 'btc' to liquidate BTC, or 'cash' to spend cash.
            target (float): Residual fraction to reach by horizon (e.g., 0.10).
            duration (int): Episode length in steps (seconds).
            start_idx (int): Start row index within the day.
            trade_fraction (float): Max fraction of remaining inventory/cash per step.
            use_builder (bool): Kept for backward compatibility; unused.
            obs_kind (str): Kept for backward compatibility; depth20+ind is always used.
            include_aux (bool): Include scheduling auxiliaries in observations.
            allow_opposite_side_trades (bool): If False, opposite-side actions are converted to HOLD.
            size_policy (str): 'adaptive' or 'fixed' sizing policy.
            size_boost_backlog (float): Size boost when behind schedule.
            size_damp_ahead (float): Size damping when ahead of schedule.
            size_boost_urgency (float): Size boost as time runs out.
            min_trade_usd (float): Gate tiny trades below this notional.
            min_size_factor (float): Lower bound on adaptive size multiplier.
            max_size_factor (float): Upper bound on adaptive size multiplier.
        """
        super(RLExecEnv, self).__init__()

        # Initialize environment parameters
        self.df = df  # current active day
        self.df_list = list(df_list) if df_list is not None else None
        self.df_paths = list(df_paths) if df_paths is not None else None
        self._df_cache = {}
        self.preload = bool(preload)

        if self.df_list is None and self.df_paths is None and self.df is None:
            raise ValueError("Provide df OR df_list OR df_paths to RLExecEnv.")

        if self.preload and self.df_paths:
            import os, pandas as pd
            for p in self.df_paths:
                if p not in self._df_cache:
                    ext = os.path.splitext(p)[1].lower()
                    if ext in (".feather", ".ft", ".fth"):
                        tmp = pd.read_feather(p)
                        self._df_cache[p] = self._normalize_day_df(tmp)
                    elif ext in (".parquet", ".pq"):
                        tmp = pd.read_parquet(p)
                        self._df_cache[p] = self._normalize_day_df(tmp)
                    else:
                        tmp = pd.read_csv(p)
                        self._df_cache[p] = self._normalize_day_df(tmp)

        self.initial_cash = float(initial_cash)
        self.initial_btc = float(initial_btc)
        self.goal = goal
        self.target = float(target)
        self.duration = int(duration)
        self.base_idx = int(start_idx)
        self.use_builder = use_builder
        self.N_LEVELS = 20 # Number of levels to consider for depth features
        # Observation space: forced to depth20+ind (single supported mode)
        self.obs_kind = 'depth20+ind'
        self.include_aux = bool(include_aux)

        # Always enable indicators in this mode
        self._with_indicators = True

        # Prepare indicator schema
        self._ind_cols = []
        self._ind_dim = 0
        self._ind_mat = None
        try:
            self._infer_indicator_schema()
        except Exception:
            self._ind_cols = []
            self._ind_dim = 0
            self._ind_mat = None

        # Build observation space for depth20 (20 levels per side * 6 metrics) + optional AUX + indicators
        base_dim = 3 + 2 * self.N_LEVELS * 6  # [cash, btc, midpoint] + ASK/BID blocks
        if self.include_aux:
            base_dim += 2  # time_left_ratio, backlog_ratio
        self.observation_dim = base_dim + int(self._ind_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

        # Allow BUYs during a BTC liquidation program and SELLs during a cash (buy) program
        # When False, opposite-side actions are converted to HOLD for clean comparisons.
        self.allow_opposite_side_trades = bool(allow_opposite_side_trades)

        # Adaptive sizing & trade gating
        self.size_policy = str(size_policy)
        self.size_boost_backlog = float(size_boost_backlog)
        self.size_damp_ahead = float(size_damp_ahead)
        self.size_boost_urgency = float(size_boost_urgency)
        self.min_trade_usd = float(min_trade_usd)
        self.min_size_factor = float(min_size_factor)
        self.max_size_factor = float(max_size_factor)

        # Label of the currently selected day (filled in reset/_pick_day_df)
        self._current_day_label = None

        self.randomize_start = False # can be toggled by the trainer (random start per episode)
        self.randomize_day = True # randomize trading day per episode when multiple days are provided

        # Discrete actions
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # At most this fraction of available cash/BTC can be traded per step
        self.trade_fraction = float(trade_fraction)

        # Initialize state variables
        self.current_cash = None
        self.current_btc = None
        self.current_ts = None
        self._values_t = None
        self.step_in_episode = 0
        self.previous_pnl = 0.0
        self.sum_rewards = 0.0

        if self.df is not None:
            self.df = self._normalize_day_df(self.df)

    def _pick_day_df(self, rng):
        """Select the active day DataFrame.

        When multiple days are available, randomly choose one (using the env RNG),
        load/normalize it, and set `_current_day_label` for logging; otherwise return
        the current `self.df`.

        Returns:
            pd.DataFrame: The selected/loaded day in wide schema.
        """
        import os, pandas as pd
        if self.df_list:
            idx = int(rng.integers(0, len(self.df_list))) if hasattr(rng, "integers") else int(np.random.randint(0, len(self.df_list)))
            try:
                self._current_day_label = f"list[{idx}]"
            except Exception:
                self._current_day_label = None
            return self._normalize_day_df(self.df_list[idx])
        if self.df_paths:
            idx = int(rng.integers(0, len(self.df_paths))) if hasattr(rng, "integers") else int(np.random.randint(0, len(self.df_paths)))
            path = self.df_paths[idx]
            try:
                import os as _os
                self._current_day_label = _os.path.basename(path)
            except Exception:
                self._current_day_label = None
            if path in self._df_cache:
                return self._df_cache[path]
            ext = os.path.splitext(path)[1].lower()
            df = pd.read_feather(path) if ext in (".feather", ".ft", ".fth") else (pd.read_parquet(path) if ext in (".parquet", ".pq") else pd.read_csv(path))
            df = self._normalize_day_df(df)
            self._df_cache[path] = df
            return df
        return self.df

    def _get_obs(self):
        """Assemble the observation vector for the current state.

        Composition:
            [cash, btc, midpoint] + depth20 features (both sides)
            [+ auxiliaries if enabled] [+ indicator vector].

        Returns:
            numpy.ndarray: Shape (`self.observation_dim`,), dtype float32.
        """
        import numpy as np
        values = self._values_t or {}

        # Auxiliary features for scheduling
        (progress, time_left_ratio, backlog_ratio, backlog_signed,
         scheduled_remaining, actual_remaining, total_to_liq) = self._compute_aux_scheduling_metrics()

        base = build_obs_depth_20(
            values_now=values,
            cash=self.current_cash,
            btc=self.current_btc,
            n_levels=self.N_LEVELS,
            time_left_ratio=(time_left_ratio if self.include_aux else None),
            backlog_ratio=(backlog_ratio if self.include_aux else None),
            include_aux=bool(self.include_aux),
        )

        # Append indicator vector (always enabled in this build)
        d = int(getattr(self, '_ind_dim', 0) or 0)
        if d > 0:
            if self._ind_mat is not None and 0 <= self.current_idx < self._ind_mat.shape[0]:
                ind_vec = self._ind_mat[self.current_idx]
            else:
                ind_vec = np.zeros((d,), dtype=np.float32)
            return np.concatenate([base.astype(np.float32), ind_vec.astype(np.float32)], axis=0)
        return base

    def reset(self, seed=None, options=None):
        """Reset the environment and start a new episode.

        Optionally accepts `options={'day_idx': int, 'start_idx': int, 'randomize_start': bool}`
        to control day selection and start index.

        Returns:
            tuple: (obs, info) where `obs` is the initial observation and `info` is an
            empty dict.

        Raises:
            RuntimeError: If the active day DataFrame is missing or empty.
        """
        super().reset(seed=seed)
        rng = getattr(self, "np_random", None)
        if rng is None:
            rng = np.random

        # Select/force a training day when multiple are available
        if options and (options.get("day_idx") is not None):
            di = int(options["day_idx"])
            if self.df_list:
                self.df = self.df_list[di % len(self.df_list)]
            elif self.df_paths:
                import os, pandas as pd
                path = self.df_paths[di % len(self.df_paths)]
                if path in self._df_cache:
                    self.df = self._df_cache[path]
                else:
                    ext = os.path.splitext(path)[1].lower()
                    if ext in (".feather", ".ft", ".fth"):
                        self.df = pd.read_feather(path)
                    elif ext in (".parquet", ".pq"):
                        self.df = pd.read_parquet(path)
                    else:
                        self.df = pd.read_csv(path)
                    self._df_cache[path] = self.df
            try:
                if self.df_list:
                    self._current_day_label = f"list[{di % len(self.df_list)}]"
                elif self.df_paths:
                    import os as _os
                    path = self.df_paths[di % len(self.df_paths)]
                    self._current_day_label = _os.path.basename(path)
                else:
                    self._current_day_label = "single"
            except Exception:
                self._current_day_label = None
        elif self.randomize_day:
            self.df = self._pick_day_df(rng)

        # Normalize active day to guarantee int64, sorted, unique timestamps
        self.df = self._normalize_day_df(self.df)

        if getattr(self, "_current_day_label", None) is None:
            self._current_day_label = "single"

        # Build per-day indicator matrix if needed
        if getattr(self, '_with_indicators', False):
            self._build_indicator_matrix_for_active_df()

        # Ensure a valid day DataFrame is present
        if self.df is None or len(self.df) == 0:
            raise RuntimeError("Active day DataFrame is empty or None.")

        start_idx_override = None
        if options and ('start_idx' in options) and (options['start_idx'] is not None):
            start_idx_override = int(options['start_idx'])
        else:
            rnd_flag = False
            if options and options.get('randomize_start'):
                rnd_flag = True
            elif getattr(self, 'randomize_start', False):
                rnd_flag = True

            if rnd_flag:
                max_start = max(0, len(self.df) - self.duration - 1)
                if max_start > 0:
                    # use gymnasium RNG if available for determinism across seeds
                    if hasattr(rng, 'integers'):
                        start_idx_override = int(rng.integers(0, max_start))
                    else:
                        start_idx_override = int(rng.randint(0, max_start))

        if start_idx_override is not None:
            self.base_idx = start_idx_override
        else:
            # Clamp base_idx to fit within the current day
            max_start = max(0, len(self.df) - self.duration - 1)
            if self.base_idx > max_start:
                self.base_idx = int(max_start)

        # Portfolio and time pointers
        self.current_cash = float(self.initial_cash)
        self.current_btc = float(self.initial_btc)
        self.current_idx = int(self.base_idx)
        self.current_ts = int(self.df['timestamp_ns'].iloc[self.current_idx])

        # Episode state
        self.step_in_episode = 0
        self.sum_rewards = 0.0

        # Snapshot the book at t
        self._values_t = get_values(self.current_ts, self.df)

        # Initialize previous_pnl with current cumulative pnl fraction
        perf0 = get_performance(
            self.initial_cash, self.initial_btc,
            self.current_cash, self.current_btc,
            self._values_t, self.goal, self.target, self.duration
        )
        self.previous_pnl = float(perf0.get('pnl_fraction', 0.0))

        # First observation
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        """Advance the environment by one step.

        Flow:
            1. Validate and possibly map the action (respecting constraints/gates).
            2. Execute trade on the current book snapshot.
            3. Advance time and refresh snapshot.
            4. Compute performance and dense reward.
            5. Check termination/truncation; add terminal reward if ending.

        Args:
            action (int): 0=HOLD, 1=BUY, 2=SELL.

        Returns:
            tuple: (next_obs, reward, terminated, truncated, info)
                next_obs (numpy.ndarray): Observation after the transition.
                reward (float): Per-step reward (incl. terminal bonus if ended).
                terminated (bool): True if horizon reached.
                truncated (bool): True if data ended early.
                info (dict): Diagnostics (PnL, inventory, reasons, scheduling metrics).
        """
        # 1. Validate action and read current snapshot
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        # Current snapshot at time t, we expect 'ask_values' and 'bid_values' in it
        values_t = self._values_t or {}
        ask_values = values_t.get('ask_values', [])
        bid_values = values_t.get('bid_values', [])

        # Adaptive trade sizing and micro-trade gating
        mid_price = self._best_mid(values_t)
        (prog0, tlr0, br0, b_signed0, _sch_rem0, _act_rem0, _tot_liq0) = self._compute_aux_scheduling_metrics()
        behind = max(0.0, b_signed0)
        ahead = max(0.0, -b_signed0)
        urgency = 1.0 - float(tlr0)
        size_factor = 1.0
        if str(self.size_policy).lower() == "adaptive":
            # Increase size when behind schedule and as urgency grows; gently damp when ahead
            size_factor = 1.0 + self.size_boost_backlog * behind - self.size_damp_ahead * ahead + self.size_boost_urgency * (urgency ** 2)
            if size_factor < self.min_size_factor:
                size_factor = self.min_size_factor
            if size_factor > self.max_size_factor:
                size_factor = self.max_size_factor

        # 2. Map action to simple, safe order sizes, with adaptive sizing
        base_buy_usd = self.trade_fraction * float(self.current_cash)
        base_sell_btc = self.trade_fraction * float(self.current_btc)
        buy_notional_usd = base_buy_usd * size_factor
        sell_quantity_btc = base_sell_btc * size_factor

        # Save backlog state before executing the action and init trade info
        (_, _, _, prev_backlog_signed, _, _, _) = self._compute_aux_scheduling_metrics()
        trade_qty_btc = 0.0
        trade_side = "hold"

        # 3. Execute the action
        # 0 = HOLD (no op), 1 = BUY using the ask side, 2 = SELL using the bid side
        #    Respect the allow_opposite_side_trades flag by converting forbidden actions to HOLD.
        effective_action = int(action)
        opposite_blocked = False
        if not self.allow_opposite_side_trades:
            if self.goal == 'btc' and effective_action == 1:
                # BTC liquidation program: block BUYs
                effective_action = 0
                opposite_blocked = True
            elif self.goal == 'cash' and effective_action == 2:
                # Cash spend / buy program: block SELLs
                effective_action = 0
                opposite_blocked = True

        # Micro-trade gating: block tiny trades relative to min_trade_usd
        small_trade_blocked = False
        if effective_action == 1:
            if buy_notional_usd < self.min_trade_usd:
                effective_action = 0
                small_trade_blocked = True
        elif effective_action == 2:
            est_notional = (sell_quantity_btc * mid_price) if mid_price > 0.0 else 0.0
            if est_notional < self.min_trade_usd:
                effective_action = 0
                small_trade_blocked = True
        if effective_action == 1 and buy_notional_usd > 0.0:
            res = sim_action.buy(
                amount=buy_notional_usd,
                ask_values=ask_values,
                current_cash=self.current_cash,
                current_btc=self.current_btc,
            )
            # Robust unpacking: support [cash, btc] or [cash, btc, qty_btc, "buy"]
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                cash, btc = res[0], res[1]
                self.current_cash, self.current_btc = float(cash), float(btc)
                if len(res) >= 4:
                    try:
                        trade_qty_btc = float(abs(res[2]))
                        trade_side = "buy"
                    except Exception:
                        trade_qty_btc = 0.0
                        trade_side = "hold"
        elif effective_action == 2 and sell_quantity_btc > 0.0:
            # Execute sell. As with buy(), the simulator may return extra info
            res = sim_action.sell(
                amount=sell_quantity_btc,
                bid_values=bid_values,
                current_cash=self.current_cash,
                current_btc=self.current_btc,
            )
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                cash, btc = res[0], res[1]
                self.current_cash, self.current_btc = float(cash), float(btc)
                if len(res) >= 4:
                    try:
                        trade_qty_btc = float(abs(res[2]))
                        trade_side = "sell"
                    except Exception:
                        trade_qty_btc = 0.0
                        trade_side = "hold"
        # else: HOLD: do nothing

        # Estimate execution slippage versus midpoint on the pre-trade book snapshot
        trade_slippage_pct = None
        if trade_qty_btc > 0.0 and trade_side in ("buy", "sell"):
            try:
                trade_slippage_pct = slippage(values_t, quantity_BTC=trade_qty_btc, side=trade_side)
            except Exception:
                trade_slippage_pct = None

        # 4. Advance time by one tick (t: t+1)
        self.step_in_episode += 1
        self.current_idx += 1

        # If we run out of data, we truncate the episode cleanly
        truncated = False
        done_reason = 'in_progress'
        if self.current_idx >= len(self.df):
            self.current_idx = len(self.df) - 1
            truncated = True
            done_reason = 'out_of_data'

        # Refresh timestamp and snapshot at the new index
        self.current_ts = int(self.df['timestamp_ns'].iloc[self.current_idx])
        self._values_t = get_values(self.current_ts, self.df)

        # 5. Compute performance and per step reward
        perf_t1 = get_performance(
            self.initial_cash, self.initial_btc,
            self.current_cash, self.current_btc,
            self._values_t, self.goal, self.target, self.duration,
        )
        (progress, time_left_ratio, backlog_ratio, backlog_signed,
         scheduled_remaining, actual_remaining, total_to_liq) = self._compute_aux_scheduling_metrics()
        perf_t1.update({
            "time_left_ratio": float(time_left_ratio),
            "backlog_ratio": float(backlog_ratio),      
            "backlog_signed": float(backlog_signed),  
            "progress": float(progress),
            "scheduled_remaining": float(scheduled_remaining),
            "actual_remaining": float(actual_remaining),
            "total_to_liq": float(total_to_liq),
            "prev_backlog_signed": float(prev_backlog_signed),
            "trade_qty_btc": float(trade_qty_btc),
            "trade_side": str(trade_side),
            "trade_slippage_pct": (None if trade_slippage_pct is None else float(trade_slippage_pct)),
        })

        # Dense reward at step t: t+1. `compute_reward_at_t` returns
        # [reward_at_t, updated_sum_rewards, current_pnl_fraction]
        reward_t, self.sum_rewards, pnl_cum = compute_reward_at_t(
            performance=perf_t1,
            sum_rewards=self.sum_rewards,
            step=self.step_in_episode,
            previous_pnl=self.previous_pnl,
        )
        # Keep track of the cumulative pnl for the next delta computation
        self.previous_pnl = pnl_cum

        # 6. Termination conditions
        # a. Horizon reached (episode length)
        terminated = False
        if self.step_in_episode >= self.duration:
            terminated = True
            done_reason = 'horizon'

        # If the episode ends (terminated or truncated), add terminal reward once
        if terminated or truncated:
            reward_t += compute_final_reward(perf_t1, sum_rewards=self.sum_rewards)

        # 7. Build next observation and info dict
        obs_next = self._get_obs()
        info = {
            "ts": self.current_ts,
            "pnl_pct": float(perf_t1.get('pnl_percentage', 0.0)),
            "pnl_frac": float(perf_t1.get('pnl_fraction', 0.0)),
            "achieved_goal": bool(perf_t1.get('achieved_goal', False)),
            "cash": float(self.current_cash),
            "btc": float(self.current_btc),
            "done_reason": done_reason,
            "action": int(action),
            "effective_action": int(effective_action),
            "opposite_blocked": bool(opposite_blocked),
            "step": int(self.step_in_episode),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "time_left_ratio": float(time_left_ratio),
            "backlog_ratio": float(backlog_ratio),
            "backlog_signed": float(backlog_signed),
            "progress": float(progress),
            "scheduled_remaining": float(scheduled_remaining),
            "actual_remaining": float(actual_remaining),
            "total_to_liq": float(total_to_liq),
            "goal": str(self.goal),
            "target": float(self.target),
            "trade_qty_btc": float(trade_qty_btc),
            "trade_side": str(trade_side),
            "trade_slippage_pct": (None if trade_slippage_pct is None else float(trade_slippage_pct)),
            "prev_backlog_signed": float(prev_backlog_signed),
            "size_factor": float(size_factor),
            "small_trade_blocked": bool(small_trade_blocked),
            "day_label": getattr(self, "_current_day_label", None),
        }

        return obs_next, float(reward_t), terminated, truncated, info