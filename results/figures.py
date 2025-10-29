# =============================================================
# results/figures.py
#
# Plotting utilities for training curves, indicators, and evaluation summaries.
# =============================================================

import argparse
import os
import sys
import math
from typing import Optional, List
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Helper logger function
def _log(level: str, msg: str):
    """Print a message with a level tag; INFO/WARN/ERROR go to stderr, others to stdout."""
    stream = sys.stderr if level in ("INFO", "WARN", "ERROR") else sys.stdout
    print(f"[{level}] {msg}", file=stream)

def _read_day(path: str | Path) -> pd.DataFrame:
    """Read a day file (feather/parquet/csv) and return a DataFrame."""
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".feather", ".ft", ".fth"):
        return pd.read_feather(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# Plot midpoints by month from LOB files
def plot_midpoints_by_month(data_glob: str, outdir: str):
    """Scan data_glob for day files, group by YYYYMM (from filename prefix if present, else from timestamp),
    and save a midpoint vs timestamp_s scatter/line per month.
    """
    files = sorted([p for p in Path(".").glob(data_glob) if p.is_file()]) if ("*" in data_glob or "?" in data_glob) else sorted([Path(data_glob)])
    if not files:
        _log("INFO", f"No files matched for price plotting: {data_glob}")
        return
    by_month: dict[str, list[pd.DataFrame]] = {}
    for fp in files:
        try:
            df = _read_day(fp)
        except Exception as e:
            _log("WARN", f"Failed to read {fp}: {e}")
            continue
        # unify midpoint column
        if "midpoint" not in df.columns:
            if "mid" in df.columns:
                df = df.rename(columns={"mid": "midpoint"})
            elif "midpoint_USD" in df.columns:
                df = df.rename(columns={"midpoint_USD": "midpoint"})
        if "timestamp_ns" not in df.columns or "midpoint" not in df.columns:
            _log("INFO", f"Skipping {fp}: missing timestamp_ns or midpoint")
            continue
        # timestamp in seconds
        ts = df["timestamp_ns"].astype("float64") / 1e9
        mid = pd.to_numeric(df["midpoint"], errors="coerce")
        tmp = pd.DataFrame({"timestamp_s": ts, "midpoint": mid})
        # infer YYYYMM from filename stem prefix or from timestamp
        stem = fp.stem
        month = None
        if len(stem) >= 6 and stem[:6].isdigit():
            month = stem[:6]
        else:
            try:
                # derive from first timestamp
                month = pd.to_datetime(int(df["timestamp_ns"].iloc[0]), unit="ns", utc=True).strftime("%Y%m")
            except Exception:
                month = "unknown"
        by_month.setdefault(month, []).append(tmp)
    for month, chunks in by_month.items():
        dfm = pd.concat(chunks, ignore_index=True)
        dfm = dfm.dropna()
        if dfm.empty:
            continue
        plt.figure(figsize=(10, 4))
        # Use a light scatter to avoid overplotting
        plt.plot(dfm["timestamp_s"].values, dfm["midpoint"].values, linestyle="None", marker=".", alpha=0.35, markersize=1.5)
        plt.xlabel("Timestamp (s)")
        plt.ylabel("Midpoint")
        plt.title(f"Midpoint vs Timestamp — {month}")
        plt.grid(True, linewidth=0.3, alpha=0.5)
        plt.tight_layout()
        outp = os.path.join(outdir, f"fig_midpoint_{month}.png")
        plt.savefig(outp, dpi=200, bbox_inches="tight")
        plt.close()
        _log("OK", f"Saved midpoint plot: {outp}")


# Plot indicator correlation heatmap using matplotlib only
def plot_indicator_heatmap(data_glob: str, indicator_cols: list[str], outdir: str):
    """
    Compute pairwise correlations for the requested indicators across all matched files
    and save a heatmap PNG plus a CSV of the correlation matrix.
    """
    files = sorted([p for p in Path(".").glob(data_glob) if p.is_file()]) if ("*" in data_glob or "?" in data_glob) else sorted([Path(data_glob)])
    if not files:
        _log("INFO", f"No files matched for indicator heatmap: {data_glob}")
        return
    # Collect only present columns to reduce memory
    collected: list[pd.DataFrame] = []
    present_any = set()
    for fp in files:
        try:
            df = _read_day(fp)
        except Exception as e:
            _log("WARN", f"Failed to read {fp}: {e}")
            continue
        cols = [c for c in indicator_cols if c in df.columns]
        if not cols:
            continue
        present_any.update(cols)
        collected.append(df[cols].apply(pd.to_numeric, errors="coerce"))
    if not collected:
        _log("INFO", "No requested indicator columns found across files; skipping heatmap.")
        return
    all_df = pd.concat(collected, ignore_index=True)
    all_df = all_df.dropna(how="all")
    if all_df.empty:
        _log("INFO", "Indicator DataFrame is empty after concat; skipping heatmap.")
        return
    corr = all_df.corr()
    # Save CSV of the correlation
    corr_csv = os.path.join(outdir, "corr_indicators.csv")
    corr.to_csv(corr_csv)
    # Plot heatmap using imshow
    plt.figure(figsize=(8, 8))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    labels = list(corr.columns)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels, rotation=0, fontsize=8)
    plt.title("Correlation heatmap", fontsize=12)
    # Annotate cells with the correlation values
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.values[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")
    plt.tight_layout()
    outp = os.path.join(outdir, "fig_heatmap_indicators.png")
    plt.savefig(outp, dpi=200, bbox_inches="tight")
    plt.close()
    _log("OK", f"Saved indicator heatmap: {outp}")


def _ecdf(arr: np.ndarray):
    """
    Return the empirical CDF (x, y) for a 1D numeric array; NaNs are dropped.
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    x = np.sort(arr)
    y = np.arange(1, len(x)+1) / float(len(x))
    return x, y

def _to_date_order(day_series: pd.Series) -> pd.Series:
    """
    Parse day labels (YYYY-MM-DD strings) to UTC datetimes for stable chronological sorting;
    returns a Series usable as a sort key.
    """
    # Try to parse YYYY-MM-DD strings to datetime for stable sorting; fallback to original
    try:
        return pd.to_datetime(day_series, errors="coerce", utc=True).fillna(pd.NaT)
    except Exception:
        return day_series

def _aggregate_eval_daily(df_eval: pd.DataFrame, baselines: list[str]) -> pd.DataFrame:
    """
    Accepts eval_compare CSV (episode- or day-level). Aggregates by 'day' to compute
    daily mean rl_pnl% and baseline pnl%, and differences in bps.
    Returns a DataFrame with columns:
      day, rl_pnl%, {baseline}, diff_{baseline}_bps
    """
    req = ["rl_pnl%", "day"]
    for r in req:
        if r not in df_eval.columns:
            raise ValueError(f"Missing required column '{r}' in eval CSV")
    # If multiple episodes per day exist, aggregate by mean
    g = df_eval.groupby("day", as_index=False).mean(numeric_only=True)
    # Compute diffs in bps
    for b in baselines:
        if b in g.columns:
            g[f"diff_{b}_bps"] = (g["rl_pnl%"] - g[b]) * 10000.0
    # Keep a deterministic order by date if parseable
    try:
        order = pd.to_datetime(g["day"], errors="coerce", utc=True)
        g = g.assign(_ord=order).sort_values("_ord").drop(columns=["_ord"])
    except Exception:
        g = g.sort_values("day")
    return g

def _plot_gap_bars(daily: pd.DataFrame, base_col: str, outdir: str):
    """
    Plot a sorted horizontal bar chart of daily RL-baseline gaps (in bps) for the given baseline
    and save the figure to disk.
    """
    col = f"diff_{base_col}_bps"
    if col not in daily.columns:
        return
    df = daily[["day", col]].dropna().copy()
    if df.empty:
        return
    df = df.sort_values(col)
    y_pos = np.arange(len(df))
    plt.figure(figsize=(8, max(3, 0.25*len(df)+1)))
    plt.barh(y_pos, df[col].values)
    plt.axvline(0.0, linestyle="--")
    plt.yticks(y_pos, df["day"].astype(str).tolist(), fontsize=7)
    mean_gap = np.nanmean(df[col].values)
    winrate = float(np.mean(df[col].values > 0.0))*100.0
    plt.xlabel("RL − baseline (bps)")
    plt.title(f"Daily advantage vs {base_col} — mean={mean_gap:.2f} bps, win-rate={winrate:.1f}%")
    plt.tight_layout()
    outp = os.path.join(outdir, f"fig_gap_bars_{base_col.replace('%','pct')}.png")
    plt.savefig(outp, dpi=200, bbox_inches="tight")
    plt.close()

def _plot_gap_hist_ecdf(daily: pd.DataFrame, base_cols: list[str], outdir: str):
    """
    Plot overlaid histograms and ECDFs of daily RL-baseline gaps (bps) for the provided baselines,
    and save both figures.
    """
    diffs = {}
    for b in base_cols:
        col = f"diff_{b}_bps"
        if col in daily.columns:
            arr = daily[col].astype(float).values
            arr = arr[~np.isnan(arr)]
            if arr.size:
                diffs[b] = arr
    if not diffs:
        return
    # Histogram overlay
    plt.figure(figsize=(8,4.5))
    for b, arr in diffs.items():
        plt.hist(arr, bins=20, alpha=0.5, label=b, density=True)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("RL − baseline (bps)")
    plt.ylabel("Density")
    plt.title("Distribution of daily advantages")
    plt.legend()
    plt.tight_layout()
    outp = os.path.join(outdir, "fig_gap_hist.png")
    plt.savefig(outp, dpi=200, bbox_inches="tight")
    plt.close()
    # ECDF overlay
    plt.figure(figsize=(8,4.5))
    for b, arr in diffs.items():
        x, y = _ecdf(arr)
        plt.plot(x, y, label=b)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("RL − baseline (bps)")
    plt.ylabel("ECDF")
    plt.title("ECDF of daily advantages")
    plt.legend()
    plt.tight_layout()
    outp = os.path.join(outdir, "fig_gap_ecdf.png")
    plt.savefig(outp, dpi=200, bbox_inches="tight")
    plt.close()

def _plot_scatter_rl_vs_base(daily: pd.DataFrame, base_col: str, outdir: str):
    """
    Scatter plot of per-day mean rl_pnl% versus a baseline pnl%; includes a 45-degree parity line.
    The figure is saved to disk.
    """
    if base_col not in daily.columns or "rl_pnl%" not in daily.columns:
        return
    x = daily[base_col].astype(float).values
    y = daily["rl_pnl%"].astype(float).values
    plt.figure(figsize=(5.5,5.5))
    plt.scatter(x, y, s=18, alpha=0.8)
    lim_lo = np.nanmin([x.min(), y.min()])
    lim_hi = np.nanmax([x.max(), y.max()])
    plt.plot([lim_lo, lim_hi], [lim_lo, lim_hi], linestyle="--")
    plt.xlabel(base_col)
    plt.ylabel("rl_pnl%")
    plt.title(f"RL vs {base_col} (per-day means)")
    plt.tight_layout()
    outp = os.path.join(outdir, f"fig_scatter_rl_vs_{base_col.replace('%','pct')}.png")
    plt.savefig(outp, dpi=200, bbox_inches="tight")
    plt.close()

def _plot_cumsum_advantage(daily: pd.DataFrame, base_col: str, outdir: str):
    """
    Plot the chronological cumulative sum of the daily RL-baseline gap (bps) for the given baseline
    and save the figure.
    """
    col = f"diff_{base_col}_bps"
    if col not in daily.columns:
        return
    # Ensure chronological order if possible
    try:
        dd = daily.assign(_ord=pd.to_datetime(daily["day"], errors="coerce", utc=True)).sort_values("_ord")
    except Exception:
        dd = daily.copy()
    arr = dd[col].astype(float).values
    cs = np.cumsum(arr)
    plt.figure(figsize=(8,4.5))
    plt.plot(cs)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Days (chronological)")
    plt.ylabel("Cumulative advantage (bps)")
    plt.title(f"Cumulative RL − {base_col} advantage")
    plt.tight_layout()
    outp = os.path.join(outdir, f"fig_gap_cumsum_{base_col.replace('%','pct')}.png")
    plt.savefig(outp, dpi=200, bbox_inches="tight")
    plt.close()

def export_eval_summary(daily: pd.DataFrame, base_cols: list[str], outdir: str):
    """
    Write a compact CSV with mean, median, standard deviation (bps) and win rate of daily RL-baseline
    gaps for each baseline column present.
    """
    rows = []
    for b in base_cols:
        col = f"diff_{b}_bps"
        if col in daily.columns:
            vals = daily[col].astype(float).values
            vals = vals[~np.isnan(vals)]
            if vals.size:
                rows.append({
                    "baseline": b,
                    "days": int(vals.size),
                    "diff_mean_bps": float(np.mean(vals)),
                    "diff_median_bps": float(np.median(vals)),
                    "diff_std_bps": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                    "pct_days_rl_better": float(np.mean(vals > 0.0))*100.0
                })
    if rows:
        outp = os.path.join(outdir, "eval_daily_summary.csv")
        pd.DataFrame(rows).to_csv(outp, index=False)
        _log("OK", f"Saved eval summary: {outp}")


def smooth_series(y: pd.Series, method: str = "ewma", window: int = 9, span: int = 10) -> pd.Series:
    """
    Smooth a pandas Series with EWMA (default) or a moving average and return the smoothed series.
    """
    if len(y) == 0:
        return y
    if method == "ma":
        w = max(1, int(window))
        return y.rolling(window=w, min_periods=max(1, w // 2), center=False).mean()
    s = max(1, int(span))
    return y.ewm(span=s, adjust=False).mean()


def ensure_monotonic_time(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """
    Sort by the x column, drop duplicate x values (keep the last), and reset the index.
    """
    if x_col in df.columns:
        df = df.sort_values(x_col).drop_duplicates(subset=[x_col], keep="last")
    return df.reset_index(drop=True)


def humanize_millions(x: float) -> str:
    """
    Format a numeric value in millions with one decimal (e.g., 2.3 for 2.3e6).
    """
    return f"{x/1e6:.1f}"


def plot_line(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str,
              out_path: str, hlines: Optional[List[float]] = None, logy: bool = False):
    """
    Generic helper to plot x/y with optional horizontal reference lines and optional log-y.
    Saves the plot to the provided file path.
    """
    plt.figure(figsize=(8, 4.5))
    plt.plot(x.values, y.values)
    if hlines:
        for h in hlines:
            if h is not None and not (isinstance(h, float) and (math.isnan(h) or math.isinf(h))):
                plt.axhline(h, linestyle="--")
    if logy:
        if np.all(np.asarray(y.values) > 0):
            plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    try:
        locs = plt.gca().get_xticks()
        labels = [humanize_millions(v) for v in locs]
        plt.gca().set_xticklabels(labels)
        plt.gca().set_xlim(left=np.nanmin(x.values), right=np.nanmax(x.values))
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    """
    CLI entry point: generate training curves, optional price/indicator figures, and evaluation
    summary plots from CSV inputs.
    """
    parser = argparse.ArgumentParser(description="Plot SB3 training curves from progress.csv")
    parser.add_argument("--progress", type=str, required=True, help="Path to progress.csv from SB3 CSVLogger")
    parser.add_argument("--outdir", type=str, default="plots_out", help="Directory to save figures")
    parser.add_argument("--smooth", type=str, default="ewma", choices=["ewma", "ma"], help="Smoothing method")
    parser.add_argument("--span", type=int, default=10, help="EWMA span (if --smooth=ewma)")
    parser.add_argument("--window", type=int, default=9, help="MA window (if --smooth=ma)")
    parser.add_argument("--target-kl", type=float, default=None, help="Optional target_kl (overrides CSV if provided)")
    parser.add_argument("--clip-range", type=float, default=None, help="Optional clip range (overrides CSV if provided)")
    parser.add_argument("--lob-glob", type=str, default="data/*.feather", help="Glob (or single path) for LOB day files with midpoint and indicators")
    parser.add_argument("--plot-prices", type=int, default=1, help="1 to export midpoint vs time plots per month")
    parser.add_argument("--plot-heatmap", type=int, default=1, help="1 to export indicator correlation heatmap")
    parser.add_argument(
        "--ind-cols",
        type=str,
        default="micro_price,imbalance_top_of_book,imbalance_multi_levels,normalized_spread,bid_depth,ask_depth,vamp,orderflow_imbalance,book_pressure_index,delta_midpoint,delta_vamp",
        help="Comma-separated list of indicator column names present in the data files",
    )
    parser.add_argument("--eval-csv", type=str, default=None, help="Path to eval_compare CSV (episode- or day-level)")
    parser.add_argument("--baselines", type=str, default="twap_pnl%,vwap_pnl%", help="Comma list of baseline columns present in eval CSV")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.progress)
    x_key = "time/total_timesteps"

    if x_key not in df.columns:
        _log("WARN", f"Column '{x_key}' not found. Available columns: {list(df.columns)[:10]}...")
        df[x_key] = np.arange(len(df))

    df = ensure_monotonic_time(df, x_key)

    def get_smoothed(col: str):
        if col not in df.columns:
            _log("INFO", f"Skipping missing column: {col}")
            return None
        return smooth_series(df[col].astype(float), method=args.smooth, window=args.window, span=args.span)

    x = df[x_key].astype(float)

    csv_target_kl = df["train/target_kl"].dropna().iloc[-1] if "train/target_kl" in df.columns and df["train/target_kl"].notna().any() else None
    csv_clip_range = df["train/clip_range"].dropna().iloc[-1] if "train/clip_range" in df.columns and df["train/clip_range"].notna().any() else None
    target_kl = args.target_kl if args.target_kl is not None else csv_target_kl
    clip_range = args.clip_range if args.clip_range is not None else csv_clip_range

    y = get_smoothed("rollout/ep_rew_mean")
    if y is not None:
        plot_line(x, y, "Learning Curve (Episode Reward, smoothed)",
                  "Total timesteps (millions)", "ep_rew_mean",
                  os.path.join(args.outdir, "fig_reward.png"))

    y = get_smoothed("train/explained_variance")
    if y is not None:
        plot_line(x, y, "Critic Fit (Explained Variance, smoothed)",
                  "Total timesteps (millions)", "explained_variance",
                  os.path.join(args.outdir, "fig_explained_variance.png"))

    y = get_smoothed("train/approx_kl")
    if y is not None:
        hlines = [target_kl] if target_kl is not None else None
        plot_line(x, y, "Policy Update Size (Approx KL, smoothed)",
                  "Total timesteps (millions)", "approx_kl",
                  os.path.join(args.outdir, "fig_approx_kl.png"),
                  hlines=hlines)

    y = get_smoothed("train/entropy_loss")
    if y is not None:
        plot_line(x, y, "Policy Entropy (smoothed)",
                  "Total timesteps (millions)", "entropy_loss",
                  os.path.join(args.outdir, "fig_entropy.png"))

    y = get_smoothed("train/value_loss")
    if y is not None:
        plot_line(x, y, "Value Loss (smoothed)",
                  "Total timesteps (millions)", "value_loss",
                  os.path.join(args.outdir, "fig_value_loss.png"),
                  logy=True)

    y = get_smoothed("train/clip_fraction")
    if y is not None:
        hlines = [clip_range] if clip_range is not None else None
        plot_line(x, y, "Clipping Fraction (smoothed)",
                  "Total timesteps (millions)", "clip_fraction",
                  os.path.join(args.outdir, "fig_clip_fraction.png"),
                  hlines=hlines)

    out_csv = os.path.join(args.outdir, "plotted_metrics.csv")
    cols = ["time/total_timesteps", "rollout/ep_rew_mean", "train/explained_variance",
            "train/approx_kl", "train/entropy_loss", "train/value_loss", "train/clip_fraction",
            "train/target_kl", "train/clip_range"]
    present = [c for c in cols if c in df.columns]
    df[present].to_csv(out_csv, index=False)
    _log("OK", f"Saved figures to: {args.outdir}")
    _log("OK", f"Saved compact CSV to: {out_csv}")

    # Export price plots and indicator heatmap from the same data source
    ind_cols = [c.strip() for c in (args.ind_cols or "").split(",") if c.strip()]
    if int(args.plot_prices):
        plot_midpoints_by_month(args.lob_glob, args.outdir)
    if int(args.plot_heatmap) and ind_cols:
        plot_indicator_heatmap(args.lob_glob, ind_cols, args.outdir)

    # Evaluation figures from eval_compare CSV
    if args.eval_csv:
        try:
            df_eval = pd.read_csv(args.eval_csv)
        except Exception as e:
            _log("WARN", f"Could not read eval CSV '{args.eval_csv}': {e}")
            df_eval = None
        if df_eval is not None:
            base_cols = [c.strip() for c in (args.baselines or "").split(",") if c.strip()]
            daily = _aggregate_eval_daily(df_eval, base_cols)
            # Core, paper-grade visuals:
            # 1. Sorted daily bar gaps per baseline
            for b in base_cols:
                if b in df_eval.columns:
                    _plot_gap_bars(daily, b, args.outdir)
                    _plot_scatter_rl_vs_base(daily, b, args.outdir)
                    _plot_cumsum_advantage(daily, b, args.outdir)
            # 2. Distributions (hist + ECDF) overlay across baselines
            _plot_gap_hist_ecdf(daily, [b for b in base_cols if f"diff_{b}_bps" in daily.columns], args.outdir)
            # 3. Export a compact summary CSV
            export_eval_summary(daily, base_cols, args.outdir)


if __name__ == "__main__":
    main()
