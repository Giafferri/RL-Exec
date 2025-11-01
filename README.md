# RL-Exec — Reproducibility README (BTC-USD liquidation)

This README gives **exact, copy‑paste commands** to reproduce the **7 200 s** results reported in the paper, plus a few **safe knobs** you can tweak (e.g., 3 600 s / 1 800 s horizons, repeats per day).

## Environment

Python 3.10+ recommended.  
Install dependencies with:
```bash
pip install -r requirements.txt
```

**Repo layout (new):** source code lives under `src/` as two packages: `rl/` and `lob/`. Run commands with `PYTHONPATH=src` so Python can find these modules.

**Data layout (new):** depth‑20, 1‑second LOB snapshots are under `data/` directly:
- `data/202001*.feather` → **train** (Jan‑2020)
- `data/202002*.feather` → **test** (Feb‑2020; 28 days; one day may be missing)

> **Dataset acknowledgment** — BTC‑USD LOB data and preliminary analysis courtesy of **Jonathan Sadighian (SESAMm)** from *Extending Deep Reinforcement Learning Frameworks in Cryptocurrency Market Making*.
> Feather data (Google Drive): https://drive.google.com/drive/folders/1dttg68tt6yeo1DFY175uOfj3EsaXYflD?usp=sharing

---

## 0. One‑time shell setup

On macOS/Linux:
```bash
export PYTHONPATH=src:$PYTHONPATH
```
On Windows (PowerShell):
```powershell
$env:PYTHONPATH = "src;" + $env:PYTHONPATH
```

---

## 1. Train (Jan‑2020 → model & VecNormalize)

```bash
# Output folder
mkdir -p models/ppo_depth20_ind_sellonly_jan_d7200

python -m rl.train_sb3 \
  --data-glob "data/202001*.feather" \
  --preload-days 1 \
  --timesteps 10000000 \
  --n-envs 6 \
  --duration 7200 \
  --goal btc --target 0.10 \
  --trade-fraction 0.03 \
  --allow-opposite-side-trades 0 \
  --device cpu \
  --save-dir models/ppo_depth20_ind_sellonly_jan_d7200
```

**Outputs**
- `models/ppo_depth20_ind_sellonly_jan_d7200/ppo_lob_model.zip`
- `models/ppo_depth20_ind_sellonly_jan_d7200/vecnormalize.pkl`

---

## 2. Comparative evaluation (Feb‑2020, per‑day, 10 starts/day) → CSV

```bash
mkdir -p results

python -m rl.eval_compare \
  --data-glob "data/202002*.feather" \
  --preload-days 1 \
  --model models/ppo_depth20_ind_sellonly_jan_d7200/ppo_lob_model.zip \
  --stats models/ppo_depth20_ind_sellonly_jan_d7200/vecnormalize.pkl \
  --deterministic 1 \
  --randomize-start 1 \
  --randomize-day 0 \
  --duration 7200 \
  --goal btc --target 0.10 \
  --obs-kind depth20+ind \
  --allow-opposite-side-trades 0 \
  --baselines twap,vwap \
  --vwap-levels 20 \
  --per-day 1 \
  --episodes-per-day 10 \
  --write-csv results/eval_compare_2020-02_sellonly_rep10_d7200.csv
```

**Output**
- `results/eval_compare_2020-02_sellonly_rep10_d7200.csv`  
  (one row per episode: `day, episode, start_idx, duration, rl_pnl%, twap_pnl%, vwap_pnl%`)

---

## 3. Statistical tests (Wilcoxon 1‑sided + BH‑FDR + bootstrap CI)

```bash
python -m rl.stats_eval \
  --csv results/eval_compare_2020-02_sellonly_rep10_d7200.csv \
  --rl-col "rl_pnl%" \
  --baseline-cols "twap_pnl%,vwap_pnl%" \
  --group day \
  --test wilcoxon \
  --two-sided 0 \
  --alpha 0.05 \
  --bootstrap-iters 10000 \
  --out-md results/stats_2020-02_sellonly_rep10_d7200.md \
  --out-csv results/stats_2020-02_sellonly_rep10_d7200.csv
```

**Outputs**
- `results/stats_2020-02_sellonly_rep10_d7200.md` (human‑readable report)
- `results/stats_2020-02_sellonly_rep10_d7200.csv` (table: effect sizes, p‑values, CIs)

> If your `stats_eval.py` lives under a different package, run it as a script instead:  
> `python src/rl/stats_eval.py ...` (same arguments as above).

---

## 4. Regenerate Figures

```bash
python results/figures.py
```
This script reproduces the main learning curves and ablation plots shown in the paper (Figures 2–3).

---

## Safe knobs you can change (to explore variants)

- **Horizon**: set `--duration` to `3600` or `1800` in **both** training and evaluation; update folder/file names accordingly.
- **Repeats per day**: adjust `--episodes-per-day` (e.g., `5` or `1`). More repeats ↓ variance of daily means.
- **Baselines**: choose `--baselines twap`, `--baselines vwap`, or both; `--vwap-levels` (default `20`) sets order‑book depth used by the VWAP‑like proxy.
- **Strict sell‑only**: keep `--allow-opposite-side-trades 0` for apples‑to‑apples with the paper.
- **Training budget**: tune `--timesteps` (e.g., 6M–10M) and `--n-envs` to match your machine.
- **Statistics**: vary `--bootstrap-iters`, significance `--alpha`, or run two‑sided tests with `--two-sided 1` for sensitivity checks.

> We intentionally fix the observation spec to **`depth20+ind`** to match the paper.

---

## Tips & troubleshooting

- If you see `Cannot save file into a non-existent directory`, create it first:  
  `mkdir -p results`
- Evaluation is **per‑day** and **paired**: RL and baselines share the **exact same timestamps and costs**, enabling fair tests.
- Keep the generated CSV/MD artifacts — they are your audit trail for the paper.

---

## License and Citation

This project is released under the MIT License (see `LICENSE` file).

If you use this code or dataset, please cite:
> [Enzo Duflot, Stanislas Robineau]
