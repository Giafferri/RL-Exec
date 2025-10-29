# =============================================================
# results/stats_eval.py
#
# Statistical comparison of RL agent vs baselines from eval_compare CSV.
# =============================================================

from __future__ import annotations

import argparse
import os
import sys
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon, ttest_rel
    SCIPY_OK = True
except Exception as e:
    SCIPY_OK = False


def _log(level: str, message: str) -> None:
    """Lightweight, uniform console logging.
    
    Levels: "INFO", "WARN", "ERROR" are sent to stderr; anything else (e.g., "OK") goes to stdout.
    This keeps informational results (tables, summaries) on stdout and diagnostic messages on stderr.
    """
    lvl = (level or "").upper()
    stream = sys.stderr if lvl in {"INFO", "WARN", "ERROR"} else sys.stdout
    print(f"[{lvl}] {message}", file=stream)


def _read_results(csv_path: str) -> pd.DataFrame:
    """Load results CSV and normalize key columns (day as string, episode numeric)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize some common column names if present
    if "day" in df.columns:
        df["day"] = df["day"].astype(str)
    if "episode" in df.columns:
        df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    return df


def _aggregate(df: pd.DataFrame, rl_col: str, baseline_col: str, group: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return two aligned arrays (rl, base) according to the chosen grouping.
    - group='episode': use raw rows
    - group='day': average metrics per 'day'
    """
    if group == "day":
        if "day" not in df.columns:
            raise ValueError("Grouping by day requested, but column 'day' is missing in the CSV. "
                             "Re-run eval_compare with --write-csv from the modified version.")
        g = df.groupby("day", as_index=False)[[rl_col, baseline_col]].mean(numeric_only=True)
        rl_vals = g[rl_col].to_numpy(dtype=float)
        bs_vals = g[baseline_col].to_numpy(dtype=float)
        return rl_vals, bs_vals

    # episode-level (default)
    rl_vals = df[rl_col].to_numpy(dtype=float)
    bs_vals = df[baseline_col].to_numpy(dtype=float)
    return rl_vals, bs_vals


def _winsorize(x: np.ndarray, frac: float) -> np.ndarray:
    """Symmetric winsorization of array tails by fraction `frac` (e.g., 0.01)."""
    if frac is None or frac <= 0:
        return x
    lo = np.quantile(x, frac)
    hi = np.quantile(x, 1.0 - frac)
    return np.clip(x, lo, hi)


def _bootstrap_ci_mean(diff: np.ndarray, iters: int = 5000, alpha: float = 0.05, seed: int | None = None) -> Tuple[float, float]:
    """Bootstrap percentile CI for the **mean** of `diff` with `iters` resamples."""
    rng = np.random.default_rng(seed)
    n = diff.shape[0]
    if n <= 1:
        return (float("nan"), float("nan"))
    boots = np.empty(iters, dtype=float)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        boots[i] = diff[idx].mean()
    lo = np.quantile(boots, alpha / 2.0)
    hi = np.quantile(boots, 1.0 - alpha / 2.0)
    return (float(lo), float(hi))


def _bh_correction(pvals: List[float]) -> List[float]:
    """
    Benjamini–Hochberg FDR correction for a small list of p-values.
    Returns adjusted p-values in the original order.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = np.array(pvals, dtype=float)[order]
    adj = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * m / rank)
        adj[i] = val
        prev = val
    out = np.empty(m, dtype=float)
    out[order] = adj
    return out.tolist()


def _safe_wilcoxon(diff: np.ndarray, alternative: str = "greater"):
    """
    Wrapper that handles edge cases (all zeros, too small n) for Wilcoxon test.
    Returns (statistic, pvalue) or (nan, 1.0) on failure.
    """
    if not SCIPY_OK:
        return (float("nan"), float("nan"))
    # Remove exact zeros to comply with wilcoxon requirements
    d = diff[np.abs(diff) > 0.0]
    if d.size == 0:
        # No difference at all
        return (0.0, 1.0)
    try:
        res = wilcoxon(d, zero_method="wilcox", alternative=alternative, mode="auto")
        return (float(res.statistic), float(res.pvalue))
    except Exception:
        # Fallback: try without alternative (older scipy)
        try:
            res2 = wilcoxon(d, zero_method="wilcox")
            # Two-sided pvalue; approximate one-sided by halving if mean>0 and sign is right
            pv = float(res2.pvalue)
            if alternative in ("greater", "less"):
                m = float(np.mean(d))
                if (alternative == "greater" and m > 0) or (alternative == "less" and m < 0):
                    pv = pv / 2.0
                else:
                    pv = 1.0 - pv / 2.0
            return (float(res2.statistic), pv)
        except Exception:
            return (float("nan"), float("nan"))


def _paired_t(diff: np.ndarray, alternative: str = "greater"):
    """Paired t-test of mean(diff) against 0 with one-sided alternative.
    
    Falls back to two-sided SciPy and converts to one-sided if needed (older versions).
    Returns (t_statistic, pvalue).
    """
    if not SCIPY_OK:
        return (float("nan"), float("nan"))
    try:
        res = ttest_rel(diff, np.zeros_like(diff), alternative=alternative)
        return (float(res.statistic), float(res.pvalue))
    except Exception:
        # Older SciPy without 'alternative'
        try:
            res2 = ttest_rel(diff, np.zeros_like(diff))
            # Two-sided; convert roughly to one-sided
            pv = float(res2.pvalue) / 2.0
            if alternative == "greater":
                if np.mean(diff) < 0:
                    pv = 1.0 - pv
            elif alternative == "less":
                if np.mean(diff) > 0:
                    pv = 1.0 - pv
            return (float(res2.statistic), pv)
        except Exception:
            return (float("nan"), float("nan"))


def analyze(csv_path: str,
            rl_col: str,
            baseline_cols: List[str],
            group: str = "episode",
            test: str = "wilcoxon",
            one_sided: bool = True,
            alpha: float = 0.05,
            winsor_frac: float = 0.0,
            bootstrap_iters: int = 5000,
            seed: int | None = 123) -> pd.DataFrame:
    """
    Returns a DataFrame summary with one row per baseline.
    Columns include: n, rl_mean, base_mean, diff_mean, diff_std, diff_median,
    wilcoxon_stat, wilcoxon_p, t_stat, t_p, ci_lo, ci_hi, pct_rl_better, reject.
    """
    df = _read_results(csv_path)

    alt = "greater" if one_sided else "two-sided"
    rows = []

    for base in baseline_cols:
        if base not in df.columns:
            _log("WARN", f"Baseline column '{base}' not in CSV. Skipping.")
            continue
        if rl_col not in df.columns:
            raise ValueError(f"RL column '{rl_col}' not in CSV.")

        rl_vals, bs_vals = _aggregate(df, rl_col, base, group=group)

        # Drop NaNs pairwise
        mask = ~np.isnan(rl_vals) & ~np.isnan(bs_vals)
        rl_vals = rl_vals[mask]
        bs_vals = bs_vals[mask]

        if rl_vals.size == 0:
            _log("WARN", f"No data after NA drop for baseline '{base}'.")
            continue

        diff = rl_vals - bs_vals
        if winsor_frac and winsor_frac > 0:
            diff = _winsorize(diff, winsor_frac)

        n = int(diff.size)
        rl_mean = float(np.mean(rl_vals))
        bs_mean = float(np.mean(bs_vals))
        d_mean = float(np.mean(diff))
        d_std = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
        d_mdn = float(np.median(diff))

        # Tests
        if test == "wilcoxon":
            w_stat, w_p = _safe_wilcoxon(diff, alternative=alt if one_sided else "two-sided")
        else:
            w_stat, w_p = (float("nan"), float("nan"))

        t_stat, t_p = _paired_t(diff, alternative=alt if one_sided else "two-sided")

        # Effect sizes
        cohen_d = d_mean / d_std if (d_std and not math.isnan(d_std) and d_std != 0.0) else float("nan")

        # Bootstrap CI for mean difference
        ci_lo, ci_hi = _bootstrap_ci_mean(diff, iters=bootstrap_iters, alpha=alpha, seed=seed)

        # % of cases RL > baseline
        pct_better = float(np.mean(diff > 0.0)) * 100.0

        rows.append({
            "baseline": base,
            "group": group,
            "n": n,
            "rl_mean": rl_mean,
            "base_mean": bs_mean,
            "diff_mean": d_mean,
            "diff_std": d_std,
            "diff_median": d_mdn,
            "cohen_d": cohen_d,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
            "ttest_t": t_stat,
            "ttest_p": t_p,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "pct_rl_better": pct_better,
        })

    out = pd.DataFrame(rows)

    # BH correction across the set of baselines (Wilcoxon else t-test)
    if not out.empty:
        p_primary = out["wilcoxon_p"] if out["wilcoxon_p"].notna().any() else out["ttest_p"]
        # Replace NaNs with 1 for correction stability
        pvals = [float(p) if (p == p) else 1.0 for p in p_primary.tolist()]
        out["p_adj_bh"] = _bh_correction(pvals)
        out["reject@alpha"] = out["p_adj_bh"] < alpha

    return out


def _save_outputs(df: pd.DataFrame, out_csv: str | None, out_md: str | None):
    """Optionally write the summary DataFrame to CSV and Markdown files."""
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        _log("OK", f"Wrote CSV report: {out_csv}")
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("# Statistical Comparison: RL vs Baselines\n\n")
            f.write(df.to_markdown(index=False))
        _log("OK", f"Wrote Markdown report: {out_md}")


def main():
    ap = argparse.ArgumentParser(description="Statistical comparison of RL vs baselines from eval_compare CSV.")
    ap.add_argument("--csv", required=True, help="Path to CSV produced by RL.eval_compare --write-csv ...")
    ap.add_argument("--rl-col", default="rl_pnl%", help="Column name for RL metric (default: rl_pnl%)")
    ap.add_argument("--baseline-cols", default="twap_pnl%,vwap_pnl%",
                    help="Comma-separated baseline column names present in the CSV.")
    ap.add_argument("--group", choices=["episode", "day"], default="day",
                    help="Aggregation level before testing. 'day' is recommended.")
    ap.add_argument("--test", choices=["wilcoxon", "ttest"], default="wilcoxon",
                    help="Primary significance test (wilcoxon by default).")
    ap.add_argument("--two-sided", type=int, default=0,
                    help="Use two-sided tests instead of one-sided RL>baseline (default=0).")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level (default=0.05)")
    ap.add_argument("--winsor", type=float, default=0.0,
                    help="Winsorize tails of RL-baseline differences by this fraction (e.g., 0.01).")
    ap.add_argument("--bootstrap-iters", type=int, default=5000, help="Bootstrap iterations for CI (default=5000)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for bootstrap")
    ap.add_argument("--out-csv", type=str, default=None, help="Write summary to this CSV path")
    ap.add_argument("--out-md", type=str, default=None, help="Write summary to this Markdown path")

    args = ap.parse_args()

    baseline_cols = [c.strip() for c in args.baseline_cols.split(",") if c.strip()]
    one_sided = not bool(args.two_sided)

    summary = analyze(
        csv_path=args.csv,
        rl_col=args.rl_col,
        baseline_cols=baseline_cols,
        group=args.group,
        test=args.test,
        one_sided=one_sided,
        alpha=args.alpha,
        winsor_frac=args.winsor,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )

    # Pretty print
    if summary.empty:
        _log("ERROR", "No valid baselines to analyze or empty data.")
        sys.exit(2)

    pd.set_option("display.float_format", lambda v: f"{v:.6f}")
    print("\nStatistical Comparison (primary test: {test}, group: {group}):".format(
        test=args.test, group=args.group
    ))
    print(summary.to_string(index=False))

    # Explicit conclusion lines
    for _, r in summary.iterrows():
        base = r["baseline"]
        mean_gap = r["diff_mean"]
        ci = (r["ci_lo"], r["ci_hi"])
        p = r["p_adj_bh"]
        rej = bool(r["reject@alpha"])
        direction = ">" if mean_gap > 0 else "<="
        verdict = "REJECT H0: RL better than {b}".format(b=base) if rej else "Cannot reject H0"
        print(f"\n[baseline={base}] RL mean - {base} mean = {mean_gap:+.4f} (CI {ci[0]:+.4f}, {ci[1]:+.4f}) "
              f" | adj-p={p:.6g} | RL {direction} {base}? -> {verdict}")

    _save_outputs(summary, args.out_csv, args.out_md)


if __name__ == "__main__":
    main()
