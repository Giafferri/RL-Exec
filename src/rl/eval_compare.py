# =============================================================
# rl/eval_compare.py
#
# Compare a trained RL agent against simple baselines (TWAP, VWAP-like)
# =============================================================

import os, glob, argparse
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from rl.env import RLExecEnv
from lob.timestamp import get_values
from lob.performance import get_performance
import lob.action as sim_action


# -------------------------------
# Data loading helpers
# -------------------------------

def _load_df(path: str) -> pd.DataFrame:
    """Load a single day file into a DataFrame.

    Supports Feather, Parquet, and CSV based on file extension.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".feather", ".ft", ".fth"]:
        return pd.read_feather(path)
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _resolve_datasets(single_path: str | None, list_paths: list[str] | None, glob_pattern: str | None):
    """Resolve dataset source(s) to either a single DataFrame or a list of paths.

    Returns:

    (df, df_paths)
        If a multi-day specification is provided via list/glob, returns (None, list_of_paths).
        Otherwise, loads the single file and returns (df, None).
    """
    df = None
    df_paths = None
    if list_paths or glob_pattern:
        paths: list[str] = []
        if list_paths:
            paths.extend(list_paths)
        if glob_pattern:
            paths.extend(sorted(glob.glob(glob_pattern)))
        seen = set()
        df_paths = [p for p in paths if not (p in seen or seen.add(p))]
        if len(df_paths) == 0:
            raise FileNotFoundError("No files matched (check --data-list / --data-glob).")
        return df, df_paths
    if single_path is None:
        raise ValueError("Provide --data or --data-list/--data-glob.")
    df = _load_df(single_path)
    return df, df_paths


def _make_env(
    df,
    df_paths,
    preload_days,
    initial_cash,
    initial_btc,
    goal,
    target,
    duration,
    start_idx,
    trade_fraction,
    randomize_start,
    randomize_day,
    obs_kind,
    allow_opposite_side_trades,
):
    """Factory for a single monitored RLExecEnv (for DummyVecEnv).

    Parameters mirror RLExecEnv; this returns a thunk `_init()` used by
    `DummyVecEnv([_init])`.
    """
    def _init():
        env = RLExecEnv(
            df=df,
            df_paths=df_paths,
            preload=preload_days,
            initial_cash=initial_cash,
            initial_btc=initial_btc,
            goal=goal,
            target=target,
            duration=duration,
            start_idx=start_idx,
            obs_kind=obs_kind,
        )
        env.trade_fraction = trade_fraction
        env.randomize_start = bool(randomize_start)
        env.randomize_day = bool(randomize_day)
        env.allow_opposite_side_trades = bool(allow_opposite_side_trades)
        return Monitor(env)

    return _init


# -------------------------------
# Baselines
# -------------------------------

def _episode_timestamps(df_day: pd.DataFrame, start_idx: int, duration: int):
    """Return the episode's nanosecond timestamps (exclusive end).

    Exactly `duration` timestamps if available, starting at `start_idx`.
    """
    if duration <= 0:
        return []
    end_idx_excl = min(len(df_day), start_idx + duration)  # exclusive end
    return df_day['timestamp_ns'].iloc[start_idx:end_idx_excl].tolist()


def _infer_day_label(df_day: pd.DataFrame) -> str:
    """Infer a YYYY-MM-DD (UTC) label from the day's first timestamp.

    Returns "unknown_day" if parsing fails.
    """
    try:
        t0 = int(df_day['timestamp_ns'].iloc[0])
        return str(pd.to_datetime(t0, unit='ns', utc=True).date())
    except Exception:
        return "unknown_day"


def _run_schedule_on_window(
    df_day: pd.DataFrame,
    ts_list: list[int],
    initial_cash: float,
    initial_btc: float,
    goal: str,
    target: float,
    per_step_sizes,
):
    """Execute a fixed schedule over `ts_list` with the simulator's costs.

    For `goal='btc'`, sells BTC on the bid; for `goal='cash'`, buys BTC on
    the ask spending USD. Returns the performance dict from `get_performance`.
    """
    cash = float(initial_cash)
    btc = float(initial_btc)

    for t, size in zip(ts_list, per_step_sizes):
        values = get_values(t, df_day)
        if goal == "btc":
            # Sell: `size` is BTC to sell this step
            if size > 0.0 and btc > 0.0:
                cash, btc, *_ = sim_action.sell(size, values.get('bid_values', []), cash, btc)
        else:
            # Reduce CASH: buy BTC spending `size` USD
            if size > 0.0 and cash > 0.0:
                cash, btc, *_ = sim_action.buy(size, values.get('ask_values', []), cash, btc)

    # Final mark-to-market
    last_ts = ts_list[-1]
    perf = get_performance(initial_cash, initial_btc, cash, btc, get_values(last_ts, df_day), goal, target, len(ts_list))
    return perf


def _twap_sizes(total_to_exec: float, n_steps: int):
    """TWAP sizes: equal quantity per step (BTC for sell, USD for buy)."""
    if n_steps <= 0 or total_to_exec <= 0.0:
        return [0.0] * max(0, n_steps)
    q = total_to_exec / n_steps
    return [q] * n_steps


def _vwap_like_sizes(
    total_to_exec: float,
    df_day: pd.DataFrame,
    ts_list: list[int],
    side: str,
    levels: int = 20,
):
    """Liquidity-weighted schedule (VWAP-like) using opposite-side depth.

    Weights are proportional to the sum of available BTC across the top `levels`
    on the opposite side (bid for sells, ask for buys). Falls back to notional/mid
    when size is unavailable.
    """
    if total_to_exec <= 0.0 or len(ts_list) == 0:
        return [0.0] * len(ts_list)

    weights = []
    for t in ts_list:
        v = get_values(t, df_day)
        lvls = (v.get('bid_values', []) if side == "sell" else v.get('ask_values', [])) or []
        lvls = lvls[:levels]
        # Sum of liquidity in BTC
        mid = v.get("mid") or v.get("midpoint") or v.get("midpoint_USD") or 0.0
        if not mid:
            # fallback best prices
            try:
                a = v.get('ask_values', [])[0].get('price', 0.0)
                b = v.get('bid_values', [])[0].get('price', 0.0)
                mid = 0.5 * (float(a) + float(b)) if a and b else 0.0
            except Exception:
                mid = 0.0
        liq_btc = 0.0
        for L in lvls:
            s = L.get("size_BTC", 0.0)
            if s and s > 0:
                liq_btc += float(s)
            else:
                noti = float(L.get("notional_USD", 0.0) or 0.0)
                if mid and mid > 0:
                    liq_btc += noti / mid
        weights.append(max(1e-9, liq_btc))

    W = float(np.sum(weights))
    if W <= 0.0:
        return _twap_sizes(total_to_exec, len(ts_list))

    # Share the total to execute according to weights
    return [total_to_exec * (w / W) for w in weights]


# -------------------------------
# Main
# -------------------------------

def main():
    """CLI entry-point: evaluate RL vs baselines and optionally export CSV."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--data-list", nargs="+")
    p.add_argument("--data-glob", type=str, default=None)
    p.add_argument("--preload-days", type=int, default=1)

    p.add_argument("--model", type=str, required=True)
    p.add_argument("--stats", type=str, required=True)

    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--deterministic", type=int, default=1)
    p.add_argument("--randomize-start", type=int, default=1)
    p.add_argument("--randomize-day", type=int, default=1)

    # Env params
    p.add_argument("--initial-cash", type=float, default=50_000_000)
    p.add_argument("--initial-btc", type=float, default=10_000)
    p.add_argument("--goal", type=str, choices=["btc", "cash"], default="btc")
    p.add_argument("--target", type=float, default=0.10)
    p.add_argument("--duration", type=int, default=3600)
    p.add_argument("--start-idx", type=int, default=40)
    p.add_argument("--trade-fraction", type=float, default=0.03)
    p.add_argument("--obs-kind", type=str, choices=["8d", "depth20", "8d+ind", "depth20+ind"], default="depth20+ind")
    p.add_argument(
        "--allow-opposite-side-trades",
        type=int,
        choices=[0, 1],
        default=1,
        help="1: allow BUY during BTC liquidation (goal='btc') and SELL during cash reduction (goal='cash'); 0: forbid.",
    )

    p.add_argument("--baselines", type=str, default="twap,vwap", help="comma list among: twap,vwap")
    p.add_argument("--vwap-levels", type=int, default=20)
    p.add_argument("--write-csv", type=str, default=None, help="Path to write per-episode results as CSV (includes day label).")
    p.add_argument("--per-day", type=int, default=0, help="1: iterate deterministically over each file/day exactly once (ignores global --episodes)")
    p.add_argument("--episodes-per-day", type=int, default=1, help="Number of episodes per day (used if --per-day=1)")
    p.add_argument("--max-days", type=int, default=0, help="Limit the number of days to iterate over (0 = all)")

    args = p.parse_args()

    # datasets & env set-up
    df, df_paths = _resolve_datasets(args.data, args.data_list, args.data_glob)
    if df is not None:
        assert "timestamp_ns" in df.columns
        print(f"[eval_compare] data: single-day with {len(df):,} rows.")
    else:
        print(f"[eval_compare] data: multi-day with {len(df_paths)} files.")
        sample = _load_df(df_paths[0])
        assert "timestamp_ns" in sample.columns

    print(f"[eval_compare] obs: kind={args.obs_kind}")
    print(f"[eval_compare] env: opposite_side_trades={bool(args.allow_opposite_side_trades)}")

    # Per-day deterministic evaluation
    if args.per_day and df_paths:
        print(f"[eval_compare] per-day: deterministic evaluation over {len(df_paths)} days (max_days={args.max_days or 'all'}).")
        want_twap = "twap" in args.baselines.lower()
        want_vwap = "vwap" in args.baselines.lower()

        all_rows = []
        model = None 

        # Ensure parent folder for CSV exists up-front 
        if args.write_csv:
            parent = os.path.dirname(args.write_csv)
            if parent:
                os.makedirs(parent, exist_ok=True)

        # Iterate files in deterministic order
        for i, path in enumerate(sorted(df_paths)):
            if args.max_days and i >= args.max_days:
                break

            df_day = _load_df(path)

            # Build an env bound to THIS day only (no randomize_day)
            venv_day = DummyVecEnv([
                _make_env(
                    df=df_day,
                    df_paths=None,
                    preload_days=False,
                    initial_cash=args.initial_cash,
                    initial_btc=args.initial_btc,
                    goal=args.goal,
                    target=args.target,
                    duration=args.duration,
                    start_idx=args.start_idx,
                    trade_fraction=args.trade_fraction,
                    randomize_start=bool(args.randomize_start),
                    randomize_day=False,
                    obs_kind=args.obs_kind,
                    allow_opposite_side_trades=args.allow_opposite_side_trades,
                )
            ])
            venv_day = VecNormalize.load(args.stats, venv_day)
            venv_day.training = False
            venv_day.norm_reward = False

            if model is None:
                model = PPO.load(args.model, env=venv_day)
            else:
                model.set_env(venv_day)

            inner = venv_day.venv if hasattr(venv_day, "venv") else venv_day
            monitor_env = inner.envs[0]
            core_env_day = getattr(monitor_env, "env", monitor_env)

            for ep in range(int(args.episodes_per_day)):
                obs = venv_day.reset()
                df_cur = core_env_day.df
                start_idx = core_env_day.base_idx
                duration = args.duration
                day_label = _infer_day_label(df_cur)

                done = False
                truncated = False
                ep_ret = 0.0
                last_info = {}

                # Rollout
                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=bool(args.deterministic))
                    step_out = venv_day.step(action)
                    if isinstance(step_out, (list, tuple)) and len(step_out) == 5:
                        obs, rewards, term_vec, trunc_vec, infos = step_out
                        done = bool(term_vec[0] or trunc_vec[0])
                        truncated = bool(trunc_vec[0])
                    else:
                        obs, rewards, dones_vec, infos = step_out
                        done = bool(dones_vec[0])
                        if isinstance(infos, (list, tuple)) and len(infos) > 0:
                            info0 = infos[0]
                            truncated = bool(info0.get("truncated", False) or info0.get("TimeLimit.truncated", False))
                        else:
                            truncated = False

                    r = float(rewards[0]) if isinstance(rewards, (np.ndarray, list)) else float(rewards)
                    ep_ret += r
                    if isinstance(infos, (list, tuple)) and len(infos) > 0:
                        last_info = infos[0]

                rl_pnl_pct = float(last_info.get('pnl_pct', 0.0))

                # Baselines computed on the exact same window
                ts_list = _episode_timestamps(df_cur, start_idx, duration)
                init_cash = float(args.initial_cash)
                init_btc = float(args.initial_btc)
                goal = args.goal
                target = float(args.target)

                results = {"rl_pnl%": rl_pnl_pct, "rl_ret": ep_ret}
                if goal == "btc":
                    total_to_exec = max(0.0, init_btc - init_btc * target)
                    if want_twap:
                        sizes = _twap_sizes(total_to_exec, len(ts_list))
                        perf = _run_schedule_on_window(df_cur, ts_list, init_cash, init_btc, goal, target, sizes)
                        results["twap_pnl%"] = perf["pnl_percentage"]
                    if want_vwap:
                        sizes = _vwap_like_sizes(total_to_exec, df_cur, ts_list, side="sell", levels=args.vwap_levels)
                        perf = _run_schedule_on_window(df_cur, ts_list, init_cash, init_btc, goal, target, sizes)
                        results["vwap_pnl%"] = perf["pnl_percentage"]
                else:
                    total_to_exec_usd = max(0.0, init_cash - init_cash * target)
                    if want_twap:
                        sizes = _twap_sizes(total_to_exec_usd, len(ts_list))
                        perf = _run_schedule_on_window(df_cur, ts_list, init_cash, init_btc, goal, target, sizes)
                        results["twap_pnl%"] = perf["pnl_percentage"]
                    if want_vwap:
                        sizes = _vwap_like_sizes(total_to_exec_usd, df_cur, ts_list, side="buy", levels=args.vwap_levels)
                        perf = _run_schedule_on_window(df_cur, ts_list, init_cash, init_btc, goal, target, sizes)
                        results["vwap_pnl%"] = perf["pnl_percentage"]

                results["day"] = day_label
                results["episode"] = ep + 1
                results["start_idx"] = start_idx
                results["duration"] = duration
                all_rows.append(results.copy())

                cols = ["rl_pnl%"] + [c for c in ["twap_pnl%", "vwap_pnl%"] if c in results]
                cols_str = "  ".join([f"{c}={results[c]:+.4f}" for c in cols])
                print(f"[eval_compare] {day_label} (ep {ep+1}/{int(args.episodes_per_day)}): {cols_str}")

        # Summary + CSV for per-day mode
        if all_rows:
            print("[eval_compare] summary:")
            dfres = pd.DataFrame(all_rows)
            for c in ["rl_pnl%", "rl_ret", "twap_pnl%", "vwap_pnl%"]:
                if c in dfres.columns:
                    arr = dfres[c].astype(float).values
                    print(f"[eval_compare]   {c}: mean={np.mean(arr):+.4f}  std={np.std(arr):.4f}  min={np.min(arr):+.4f}  max={np.max(arr):+.4f}")
            if args.write_csv:
                try:
                    dfres.to_csv(args.write_csv, index=False)
                    print(f"[eval_compare] wrote CSV: {args.write_csv}")
                except Exception as e:
                    print(f"[eval_compare] failed to write CSV '{args.write_csv}': {e}")
        return  # end per-day branch

    venv = DummyVecEnv([
        _make_env(
            df=df,
            df_paths=df_paths,
            preload_days=bool(args.preload_days),
            initial_cash=args.initial_cash,
            initial_btc=args.initial_btc,
            goal=args.goal,
            target=args.target,
            duration=args.duration,
            start_idx=args.start_idx,
            trade_fraction=args.trade_fraction,
            randomize_start=bool(args.randomize_start),
            randomize_day=bool(args.randomize_day),
            obs_kind=args.obs_kind,
            allow_opposite_side_trades=args.allow_opposite_side_trades,
        )
    ])

    venv = VecNormalize.load(args.stats, venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(args.model, env=venv)

    inner = venv.venv if hasattr(venv, "venv") else venv
    monitor_env = inner.envs[0]  # Monitor
    core_env = getattr(monitor_env, "env", monitor_env)  # RLExecEnv

    # parse baselines
    want_twap = "twap" in args.baselines.lower()
    want_vwap = "vwap" in args.baselines.lower()

    # run episodes
    rows = []
    all_rows = []
    for ep in range(args.episodes):
        obs = venv.reset()
        # capture the chosen day & starting index for this episode
        df_day = core_env.df
        start_idx = core_env.base_idx
        duration = args.duration
        day_label = _infer_day_label(df_day)

        done = False
        truncated = False
        ep_ret = 0.0
        last_info = {}

        # RL rollout
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            step_out = venv.step(action)
            if isinstance(step_out, (list, tuple)) and len(step_out) == 5:
                obs, rewards, term_vec, trunc_vec, infos = step_out
                done = bool(term_vec[0] or trunc_vec[0])
                truncated = bool(trunc_vec[0])
            else:
                obs, rewards, dones_vec, infos = step_out
                done = bool(dones_vec[0])
                if isinstance(infos, (list, tuple)) and len(infos) > 0:
                    info0 = infos[0]
                    truncated = bool(info0.get("truncated", False) or info0.get("TimeLimit.truncated", False))
                else:
                    truncated = False

            r = float(rewards[0]) if isinstance(rewards, (np.ndarray, list)) else float(rewards)
            ep_ret += r
            if isinstance(infos, (list, tuple)) and len(infos) > 0:
                last_info = infos[0]

        # RL result
        rl_pnl_pct = float(last_info.get('pnl_pct', 0.0))
        rl_pnl_frac = float(last_info.get('pnl_frac', 0.0))
        rl_cash = float(last_info.get('cash', 0.0))
        rl_btc = float(last_info.get('btc', 0.0))

        # Baselines on the same window
        ts_list = _episode_timestamps(df_day, start_idx, duration)
        init_cash = float(args.initial_cash)
        init_btc = float(args.initial_btc)
        goal = args.goal
        target = float(args.target)

        results = {"rl_pnl%": rl_pnl_pct, "rl_ret": ep_ret}

        if goal == "btc":
            total_to_exec = max(0.0, init_btc - init_btc * target)  # BTC to sell
            if want_twap:
                sizes = _twap_sizes(total_to_exec, len(ts_list))
                perf = _run_schedule_on_window(df_day, ts_list, init_cash, init_btc, goal, target, sizes)
                results["twap_pnl%"] = perf["pnl_percentage"]
            if want_vwap:
                sizes = _vwap_like_sizes(total_to_exec, df_day, ts_list, side="sell", levels=args.vwap_levels)
                perf = _run_schedule_on_window(df_day, ts_list, init_cash, init_btc, goal, target, sizes)
                results["vwap_pnl%"] = perf["pnl_percentage"]
        else:
            total_to_exec_usd = max(0.0, init_cash - init_cash * target)  # USD to spend (buy)
            if want_twap:
                sizes = _twap_sizes(total_to_exec_usd, len(ts_list))
                perf = _run_schedule_on_window(df_day, ts_list, init_cash, init_btc, goal, target, sizes)
                results["twap_pnl%"] = perf["pnl_percentage"]
            if want_vwap:
                sizes = _vwap_like_sizes(total_to_exec_usd, df_day, ts_list, side="buy", levels=args.vwap_levels)
                perf = _run_schedule_on_window(df_day, ts_list, init_cash, init_btc, goal, target, sizes)
                results["vwap_pnl%"] = perf["pnl_percentage"]

        # enrich results for CSV export
        results["day"] = day_label
        results["episode"] = ep + 1
        results["start_idx"] = start_idx
        results["duration"] = duration
        all_rows.append(results.copy())

        # print per-episode
        cols = ["rl_pnl%"] + [c for c in ["twap_pnl%", "vwap_pnl%"] if c in results]
        cols_str = "  ".join([f"{c}={results[c]:+.4f}" for c in cols])
        print(f"[eval_compare] Episode {ep+1}/{args.episodes}: {cols_str}")

        rows.append(results)

    # Summary
    if rows:
        print("[eval_compare] summary:")
        wanted = ["rl_pnl%", "rl_ret"]
        if want_twap:
            wanted.append("twap_pnl%")
        if want_vwap:
            wanted.append("vwap_pnl%")
        for c in wanted:
            vals = []
            for r in rows:
                if c in r:
                    try:
                        vals.append(float(r[c]))
                    except Exception:
                        pass
            if vals:
                arr = np.asarray(vals, dtype=float)
                print(f"[eval_compare]   {c}: mean={np.mean(arr):+.4f}  std={np.std(arr):.4f}  min={np.min(arr):+.4f}  max={np.max(arr):+.4f}")

    # CSV export
    if args.write_csv and all_rows:
        out_path = args.write_csv
        parent = os.path.dirname(out_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        try:
            pd.DataFrame(all_rows).to_csv(out_path, index=False)
            print(f"[eval_compare] wrote CSV: {out_path}")
        except Exception as e:
            print(f"[eval_compare] failed to write CSV '{out_path}': {e}")

if __name__ == "__main__":
    main()