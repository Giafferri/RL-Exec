# ======================================================================
# rl/train_sb3
# 
# Train a PPO agent using Stable Baselines3 on the LOB environment.
# ======================================================================

import os
import argparse
import numpy as np
import pandas as pd
import torch
import glob

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from rl.env import RLExecEnv

# -------------------------------
# Utils
# -------------------------------

def _load_df(path: str | None) -> pd.DataFrame:
    """Load a tabular dataset from feather/parquet/csv.

    Args:
        path: Path to the file. If None, try default fallbacks under `data/samples/`.

    Returns:
        A pandas DataFrame with the loaded data.

    Raises:
        ValueError: If the file extension is unsupported.
        FileNotFoundError: If no file is found when `path` is None and fallbacks are absent.
    """
    if path is not None:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".feather", ".ft", ".fth"]:
            return pd.read_feather(path)
        if ext in [".parquet", ".pq"]:
            return pd.read_parquet(path)
        if ext == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported file extension: {ext}. Use .feather, .parquet or .csv")

    # Fallbacks
    if os.path.exists("data/samples/lob.feather"):
        return pd.read_feather("data/samples/lob.feather")
    if os.path.exists("data/samples/lob.parquet"):
        return pd.read_parquet("data/samples/lob.parquet")
    if os.path.exists("data/samples/lob.csv"):
        return pd.read_csv("data/samples/lob.csv")
    raise FileNotFoundError("Could not find data file. Pass --data or place a file at data/samples/lob.feather|parquet|csv")


def _resolve_datasets(single_path: str | None, list_paths: list[str] | None, glob_pattern: str | None):
    """Resolve dataset sources into either a single DataFrame or a list of file paths.

    Prefers multi-day training when `list_paths` or `glob_pattern` are provided; otherwise
    falls back to a single-file mode.

    Args:
        single_path: Path to a single feather/parquet/csv file (single-day mode).
        list_paths: Optional list of file paths for multi-day mode.
        glob_pattern: Optional glob (e.g., 'data/202001*.feather') for multi-day mode.

    Returns:
        Tuple (df, df_list, df_paths):
            - df: The loaded DataFrame in single-file mode; otherwise None.
            - df_list: Reserved for preloaded DataFrames (unused here, always None).
            - df_paths: List of file paths in multi-day mode; otherwise None.

    Raises:
        FileNotFoundError: If multi-day is requested but no files match.
        ValueError: If neither single-file nor multi-day sources are provided.
    """
    df = None
    df_list = None
    df_paths = None

    if list_paths or glob_pattern:
        paths = []
        if list_paths:
            for p in list_paths:
                paths.extend(p if isinstance(p, (list, tuple)) else [p])
        if glob_pattern:
            paths.extend(sorted(glob.glob(glob_pattern)))
        # deduplicate preserving order
        seen = set()
        df_paths = [p for p in paths if not (p in seen or seen.add(p))]
        if len(df_paths) == 0:
            raise FileNotFoundError("No files matched for multi day training (check --data-list / --data-glob).")
        return df, df_list, df_paths

    # Single file mode
    if single_path is None:
        raise ValueError("Provide --data (single file) or --data-list/--data-glob for multiday.")
    df = _load_df(single_path)
    return df, df_list, df_paths


def _make_env_fn(
    df: pd.DataFrame | None,
    df_paths: list[str] | None,
    df_list: list[pd.DataFrame] | None,
    preload_days: bool,
    seed: int,
    idx: int,
    initial_cash: float,
    initial_btc: float,
    goal: str,
    target: float,
    duration: int,
    start_idx: int,
    trade_fraction: float,
    randomize_start: bool,
    randomize_day: bool,
    allow_opposite_side_trades: bool,
    obs_kind: str,
    size_policy: str,
    size_boost_backlog: float,
    size_damp_ahead: float,
    size_boost_urgency: float,
    min_trade_usd: float,
    min_size_factor: float,
    max_size_factor: float,
):
    """Return a thunk that builds one monitored RLExecEnv (for Dummy/Subproc VecEnv).

    The inner function constructs `RLExecEnv`, applies sizing and randomization knobs,
    seeds the environment, and wraps it with `Monitor`.

    Args:
        df, df_paths, df_list: Data sources (single day, list of paths, or preloaded list).
        preload_days: If True, eagerly loads multi-day files in the env constructor.
        seed: Base random seed.
        idx: Env index (added to seed for distinct streams).
        initial_cash, initial_btc: Starting inventory.
        goal: 'btc' to liquidate BTC, 'cash' to spend USD.
        target: Fraction to remain at the end (e.g., 0.10 keeps 10%).
        duration: Episode length in timesteps (1s per step).
        start_idx: Initial index within the day.
        trade_fraction: Max fraction tradable per step.
        randomize_start, randomize_day: Reset-time randomization flags.
        allow_opposite_side_trades: If True, permit opposite-side trades; else they map to HOLD.
        obs_kind: Observation variant (e.g., 'depth20+ind').
        size_policy: 'fixed' or 'adaptive' internal sizing policy.
        size_boost_backlog, size_damp_ahead, size_boost_urgency: Adaptive sizing modifiers.
        min_trade_usd, min_size_factor, max_size_factor: Execution and sizing thresholds.

    Returns:
        Callable[[], gym.Env]: A zero-arg function that returns a `Monitor`-wrapped env.
    """
    def _init():
        env = RLExecEnv(
            df=df,
            df_paths=df_paths,
            df_list=df_list,
            preload=preload_days,
            initial_cash=initial_cash,
            initial_btc=initial_btc,
            goal=goal,
            target=target,
            duration=duration,
            start_idx=start_idx,
            use_builder=False,
            obs_kind=obs_kind,
            allow_opposite_side_trades=bool(allow_opposite_side_trades),
        )
        env.trade_fraction = trade_fraction
        try:
            env.randomize_start = bool(randomize_start)
            env.randomize_day = bool(randomize_day)
        except Exception:
            pass
        # Sizing/shaping knobs
        try:
            env.size_policy = size_policy
            env.size_boost_backlog = float(size_boost_backlog)
            env.size_damp_ahead = float(size_damp_ahead)
            env.size_boost_urgency = float(size_boost_urgency)
            env.min_trade_usd = float(min_trade_usd)
            env.min_size_factor = float(min_size_factor)
            env.max_size_factor = float(max_size_factor)
        except Exception:
            pass
        env.reset(seed=seed + idx)
        return Monitor(env)
    return _init


def _select_device(device_arg: str) -> str:
    """Resolve device from CLI.

    'auto' prefers Apple MPS when available, then CUDA, else CPU.

    Args:
        device_arg: One of {'auto', 'cpu', 'cuda', 'mps'}.

    Returns:
        A lowercase device string usable by PyTorch/SB3.
    """
    if device_arg.lower() == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    # sanity fallback
    return device_arg.lower()


def _best_divisor(rollout_size: int, preferred: int) -> int:
    """Pick a large batch size that divides the rollout.

    Finds the largest divisor of `rollout_size` that is ≤ `preferred` and ≥ 64.
    Falls back to `rollout_size // 8` (at least 64) if none is found.

    Args:
        rollout_size: Total number of samples per rollout (n_steps × n_envs).
        preferred: The desired batch size.

    Returns:
        An integer batch size that divides `rollout_size`.
    """
    preferred = max(64, preferred)
    for b in range(min(preferred, rollout_size), 63, -1):
        if rollout_size % b == 0:
            return b
    return max(64, rollout_size // 8)

# -------------------------------
# Main
# -------------------------------

def main():
    """CLI entry-point: parse args, build VecEnv, train PPO, and save artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to feather/parquet/csv LOB dataset")
    parser.add_argument("--data-list", nargs="+", help="List of feather/parquet/csv files for multiday training")
    parser.add_argument("--data-glob", type=str, default=None, help="Glob pattern for multiday training, e.g. 'data/replay/202001*.feather'")
    parser.add_argument("--preload-days", type=int, default=0, help="1: preload all --data-list/--data-glob files in RAM")

    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs (1 = DummyVecEnv)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    # Env hyper-params
    parser.add_argument("--initial-cash", type=float, default=50_000_000)
    parser.add_argument("--initial-btc", type=float, default=10_000)
    parser.add_argument("--goal", type=str, choices=["btc", "cash"], default="btc")
    parser.add_argument("--target", type=float, default=0.10, help="Fraction to remain at end (e.g. 0.10)")
    parser.add_argument("--duration", type=int, default=3600, help="Episode length in steps (1s per step)")
    parser.add_argument("--start-idx", type=int, default=40)
    parser.add_argument("--trade-fraction", type=float, default=0.03, help="Max fraction of cash/BTC tradable per step")
    parser.add_argument("--allow-opposite-side-trades", type=int, default=1,
                        help="1: allow BUY while goal='btc' (sell program) and SELL while goal='cash' (buy program). 0: block opposite-side trades by converting them to HOLD.")
    parser.add_argument("--size-policy", type=str, choices=["fixed", "adaptive"], default="adaptive",
                        help="Trade sizing policy inside the env: fixed=use trade_fraction only; adaptive=modulate with schedule/urgency.")
    parser.add_argument("--size-boost-backlog", type=float, default=0.8,
                        help="Strength of size increase when behind schedule (adaptive only).")
    parser.add_argument("--size-damp-ahead", type=float, default=0.5,
                        help="Damp factor when ahead of schedule (adaptive only).")
    parser.add_argument("--size-boost-urgency", type=float, default=1.0,
                        help="Urgency multiplier as time runs out (adaptive only).")
    parser.add_argument("--min-trade-usd", type=float, default=500.0,
                        help="Minimum notional (USD) per trade; smaller intents are converted to HOLD to avoid fee waste.")
    parser.add_argument("--min-size-factor", type=float, default=0.25,
                        help="Lower bound for adaptive size factor relative to trade_fraction.")
    parser.add_argument("--max-size-factor", type=float, default=4.0,
                        help="Upper bound for adaptive size factor relative to trade_fraction.")

    # PPO hyper-params (sane defaults for first run)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.005, help="Entropy bonus coefficient (encourage exploration)")
    parser.add_argument("--target-kl", type=float, default=0.03, help="Early stop PPO epoch when KL > target")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument("--randomize-start-train", type=int, default=1,
                        help="1: randomize start_idx at every reset during training (better generalization)")
    parser.add_argument("--randomize-day-train", type=int, default=1,
                        help="1: randomize the trading day at every reset when multi-day is provided")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="Compute device for SB3/PyTorch. 'auto' prefers MPS on Apple, then CUDA, else CPU.")
    parser.add_argument("--hidden-sizes", type=str, default="512,512,256",
                        help="Comma-separated hidden layer sizes for the MLP policy, e.g. '256,256,256'.")
    parser.add_argument("--skip-check", action="store_true",
                        help="Skip SB3 check_env() to save startup time once the env is stable.")

    # Logging / saving
    parser.add_argument("--tb-log", type=str, default="runs/ppo_lob", help="TensorBoard log dir")
    parser.add_argument("--save-dir", type=str, default="models", help="Where to save model & stats")
    parser.add_argument("--checkpoint-freq", type=int, default=0, help="Save checkpoints every N steps (0 = disabled)")
    parser.add_argument(
        "--log-formats",
        type=str,
        default="stdout,tensorboard,csv",
        help="Comma-separated list of log sinks: stdout,tensorboard,csv",
    )

    args = parser.parse_args()

    # Optimized threading for multi env on macOS Silicon
    if args.n_envs > 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

    # Choose device
    device = _select_device(args.device)
    print(f"[device] using {device}")
    print("[obs] kind=depth20+ind")
    print(f"[env] opposite_side_trades={bool(args.allow_opposite_side_trades)}  size_policy={args.size_policy}  min_trade_usd={args.min_trade_usd}")

    os.makedirs(args.save_dir, exist_ok=True)
    set_random_seed(args.seed)

    # 1. Resolve data sources (single day vs multi day)
    df, df_list, df_paths = _resolve_datasets(args.data, args.data_list, args.data_glob)

    if df is not None:
        # Single file sanity checks
        assert "timestamp_ns" in df.columns, "DataFrame must contain a 'timestamp_ns' column"
        if args.start_idx + args.duration >= len(df):
            raise ValueError(
                f"Episode does not fit: start_idx({args.start_idx}) + duration({args.duration}) >= len(df)({len(df)})"
            )
        print(f"[data] Single day mode with {len(df):,} rows.")
    else:
        # Multi day summary (paths mode)
        print(f"[data] Multi day mode with {len(df_paths)} files.")
        if args.preload_days:
            # Quick header check on a sample file
            import pandas as pd
            sample = pd.read_feather(df_paths[0]) if df_paths[0].endswith((".feather",".ft",".fth")) \
                     else (pd.read_parquet(df_paths[0]) if df_paths[0].endswith((".parquet",".pq")) else pd.read_csv(df_paths[0]))
            assert "timestamp_ns" in sample.columns, "Each dataset must contain a 'timestamp_ns' column"

    # 2. Build a single env and check API
    if not args.skip_check:
        single_env = _make_env_fn(
            df=df,
            df_paths=df_paths,
            df_list=df_list,
            preload_days=bool(args.preload_days),
            seed=args.seed,
            idx=0,
            initial_cash=args.initial_cash,
            initial_btc=args.initial_btc,
            goal=args.goal,
            target=args.target,
            duration=args.duration,
            start_idx=args.start_idx,
            trade_fraction=args.trade_fraction,
            randomize_start=bool(args.randomize_start_train),
            randomize_day=bool(args.randomize_day_train),
            allow_opposite_side_trades=bool(args.allow_opposite_side_trades),
            obs_kind="depth20+ind",
            size_policy=args.size_policy,
            size_boost_backlog=args.size_boost_backlog,
            size_damp_ahead=args.size_damp_ahead,
            size_boost_urgency=args.size_boost_urgency,
            min_trade_usd=args.min_trade_usd,
            min_size_factor=args.min_size_factor,
            max_size_factor=args.max_size_factor,
        )()
        check_env(single_env, warn=True)
        try:
            # Display resolved observation dimension for quick sanity check
            print(f"[obs] dim={single_env.observation_space.shape[0]}")
        except Exception:
            pass

    # 3. Build VecEnv (Dummy for 1 env, Subproc for >1)
    make = lambda i: _make_env_fn(
        df=df,
        df_paths=df_paths,
        df_list=df_list,
        preload_days=bool(args.preload_days),
        seed=args.seed,
        idx=i,
        initial_cash=args.initial_cash,
        initial_btc=args.initial_btc,
        goal=args.goal,
        target=args.target,
        duration=args.duration,
        start_idx=args.start_idx,
        trade_fraction=args.trade_fraction,
        randomize_start=bool(args.randomize_start_train),
        randomize_day=bool(args.randomize_day_train),
        allow_opposite_side_trades=bool(args.allow_opposite_side_trades),
        obs_kind="depth20+ind",
        size_policy=args.size_policy,
        size_boost_backlog=args.size_boost_backlog,
        size_damp_ahead=args.size_damp_ahead,
        size_boost_urgency=args.size_boost_urgency,
        min_trade_usd=args.min_trade_usd,
        min_size_factor=args.min_size_factor,
        max_size_factor=args.max_size_factor,
    )

    if args.n_envs == 1:
        vec_env = DummyVecEnv([make(0)])
    else:
        vec_env = SubprocVecEnv([make(i) for i in range(args.n_envs)])

    # 4. Normalize observations & rewards
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Parse policy architecture from CLI and adjust batch size to rollout
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",") if x.strip()]
    rollout_size = args.n_steps * args.n_envs
    if rollout_size % args.batch_size != 0:
        new_bs = _best_divisor(rollout_size, args.batch_size)
        print(f"[train] Adjusting batch_size from {args.batch_size} to {new_bs} to divide rollout_size={rollout_size}.")
        args.batch_size = new_bs

    policy_kwargs = dict(net_arch=hidden_sizes)

    # 5. PPO model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gae_lambda=args.gae_lambda,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        learning_rate=args.lr,
        n_epochs=args.n_epochs,
        clip_range=args.clip_range,
        tensorboard_log=args.tb_log,
        seed=args.seed,
        device=device,
        policy_kwargs=policy_kwargs,
        target_kl=args.target_kl,
    )

    # Checkpointing
    callback = None
    if args.checkpoint_freq and args.checkpoint_freq > 0:
        callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=args.save_dir,
            name_prefix="ppo_lob_ckpt",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

    # Configure SB3 logger to also write CSV (progress.csv) in tb-log dir
    log_formats = [s.strip() for s in str(getattr(args, "log_formats", "stdout,tensorboard,csv")).split(",") if s.strip()]
    new_logger = configure(args.tb_log, log_formats)
    model.set_logger(new_logger)
    print(f"[log] writing logs to: {args.tb_log} (formats: {', '.join(log_formats)})")

    # 6. Training
    model.learn(total_timesteps=args.timesteps, callback=callback)

    # 7. Save model + VecNormalize stats
    model_path = os.path.join(args.save_dir, "ppo_lob_model")
    stats_path = os.path.join(args.save_dir, "vecnormalize.pkl")
    model.save(model_path)
    vec_env.save(stats_path)

    print(f"[save] model: {model_path}")
    print(f"[save] vecnormalize: {stats_path}")
    print(f"[tb] log dir: {args.tb_log}")


if __name__ == "__main__":
    main()