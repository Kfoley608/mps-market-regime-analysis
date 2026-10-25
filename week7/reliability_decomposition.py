"""Reviewer item 7 — reliability decomposition of the headline MPS features: how much of each
feature's variance is market signal (between windows) vs estimator noise (within-window refits)?

    reliability(feature, axis) = Var_between / (Var_between + mean Var_within(axis))

Var_between = variance across the M window-end dates of the BASE-config feature series;
Var_within  = per-window variance across that perturbation axis's refits, averaged over windows.
Also reports the base series' lag-1 autocorrelation (week-6 noise-proxy convention: committed
rolling P-functionals sit at ~0.96-0.98, a pure refit artifact near ~0.1).

Book: data/combined_gfc_2026.parquet. M=30 window ends evenly spaced across the usable sample,
WIN=252 (the parallel_rolling_v4.py window). Base config: d=8, chi=8, sweeps=8, seed=0, Fiedler
ordering and quantile bins from the window itself — these are DESCRIPTIVE window-level features
(no prediction is scored), so window-internal fitting is correct here. Note the committed GFC
rolling run used d=12; the week-7 validation spec sets the base at d=8 so the bin axis {6,8,10}
brackets it.

Perturbation axes (each config a full cold refit of the same window; base config deduped):
  seeds     {0,1,2,3,4}                        base + 4 refits
  bins d    {6,8,10}                           base + 2
  chi       {6,8,12}                           base + 2
  ordering  {Fiedler, reversed, random(7)}     base + 2
  bootstrap {3 stationary resamples of days}   3 (base is not a draw)
= 14 unique fits per window (17 grid cells minus the 3 base duplicates).

Features per refit (P-functionals only — signed-amplitude diagnostics withdrawn, BUG-LOG #14):
  tail_asym  features.tail_dependence_asymmetry (F12, q=0.10)
  block_mi   features.block_mutual_information (mid-chain cut, n_samples=20000, fixed seed)
  tail_lift  3-of-N bottom-bin joint-tail probability vs independence. No lift variant exists in
             features.py (tail_state_probability is the all-N corner, kway_corner_probability is
             unnormalized, tail_dependence_normalized is pairwise), so it is computed here:
             P_model from the fit's exact sampler, P_indep EXACTLY (Poisson-binomial over the
             one-site marginal tail masses).
  drift      features.distribution_drift vs the PREVIOUS window's BASE fit (first window NaN).
MS-Born p_stress reliability is SKIPPED (each refit would be a full EM run over the whole
history) — noted as future work.

Fits MPS models: RunPod pod ONLY, never the laptop.
Usage: python reliability_decomposition.py [--smoke] [--windows M] [--out-dir DIR]
  --smoke: M=4 windows, seeds {0,1}, bootstrap axis dropped.
Outputs: week7/results/reliability_long.csv (appended window by window),
         week7/results/reliability_summary.csv, week7/figures/reliability.png
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import sys, time, argparse, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
from mps_regime import binning, dmrg, features as F

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", help="M=4 windows, seeds {0,1}, no bootstrap axis")
ap.add_argument("--windows", type=int, default=30, help="number of window-end dates M")
ap.add_argument("--out-dir", default=None, help="override output root (default: week7/)")
args = ap.parse_args()

WIN, D0, CHI0, SWEEPS = 252, 8, 8, 8            # base mirrors parallel_rolling_v4 (sweeps=8)
SEEDS = [0, 1] if args.smoke else [0, 1, 2, 3, 4]
DGRID, CHIGRID = [6, 8, 10], [6, 8, 12]
NBOOT = 0 if args.smoke else 3
BOOT_BLOCK = 21                                  # stationary-bootstrap mean block length (days)
M = 4 if args.smoke else args.windows
NSAMP, K, Q = 20000, 3, 0.05

OUT = pathlib.Path(args.out_dir) if args.out_dir else HERE
RESULTS = OUT / "results"; FIGURES = OUT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
LONG = RESULTS / "reliability_long.csv"


def build_grid():
    """Explicit config grid; the base cell (seed0/d8/chi8/fiedler/no-boot) appears exactly once."""
    cfgs = [dict(label="base", axis="base", seed=0, d=D0, chi=CHI0, ordering="fiedler", boot=None)]
    for s in SEEDS[1:]:
        cfgs.append(dict(label=f"seed{s}", axis="seeds", seed=s, d=D0, chi=CHI0,
                         ordering="fiedler", boot=None))
    for d in DGRID:
        if d != D0:
            cfgs.append(dict(label=f"d{d}", axis="bins", seed=0, d=d, chi=CHI0,
                             ordering="fiedler", boot=None))
    for c in CHIGRID:
        if c != CHI0:
            cfgs.append(dict(label=f"chi{c}", axis="chi", seed=0, d=D0, chi=c,
                             ordering="fiedler", boot=None))
    for o in ["rev", "rand"]:
        cfgs.append(dict(label=f"ord_{o}", axis="ordering", seed=0, d=D0, chi=CHI0,
                         ordering=o, boot=None))
    for b in range(NBOOT):
        cfgs.append(dict(label=f"boot{b}", axis="bootstrap", seed=0, d=D0, chi=CHI0,
                         ordering="fiedler", boot=b))
    return cfgs


CFGS = build_grid()
AXIS_MEMBERS = {   # base participates in every PARAMETER axis; bootstrap = the draws only
    "seeds": ["base"] + [f"seed{s}" for s in SEEDS[1:]],
    "bins": ["base"] + [f"d{d}" for d in DGRID if d != D0],
    "chi": ["base"] + [f"chi{c}" for c in CHIGRID if c != CHI0],
    "ordering": ["base", "ord_rev", "ord_rand"],
}
if NBOOT:
    AXIS_MEMBERS["bootstrap"] = [f"boot{b}" for b in range(NBOOT)]


def sb_resample(w, seed):
    """Stationary bootstrap of the window's DAYS (mean block BOOT_BLOCK). Blocks keep short-range
    dependence but the resample still breaks the window's serial structure at block joins — this
    axis measures sampling noise under an (approximately) iid-blocks view of the window."""
    rng = np.random.default_rng(seed)
    n = len(w); idx = np.empty(n, dtype=int); t = int(rng.integers(n))
    for i in range(n):
        idx[i] = t
        t = int(rng.integers(n)) if rng.random() < 1.0 / BOOT_BLOCK else (t + 1) % n
    return w.iloc[idx]


def joint_tail_lift(fit, k=K, q=Q, n_samples=NSAMP, seed=0):
    """P_model(>=k of N assets in their bottom-m-bin tail set) / P_indep(same),
    m = max(1, round(q*d)) — matches the F14 tail-set convention. P_model from the exact perfect
    sampler; P_indep exact via Poisson-binomial over the one-site marginal tail masses (no
    independence-side sampling noise)."""
    m = max(1, int(round(q * fit.d)))
    X = fit.sample_bins(int(n_samples), seed=seed)
    p_model = float(((X < m).sum(axis=1) >= k).mean())
    p1 = fit.one_site_marginals()
    pj = [float(p1[j][:m].sum()) for j in range(fit.n_sites)]
    dp = np.zeros(fit.n_sites + 1); dp[0] = 1.0     # Poisson-binomial DP over asset count
    for p in pj:
        dp[1:] = dp[1:] * (1 - p) + dp[:-1] * p
        dp[0] *= (1 - p)
    p_indep = float(dp[k:].sum())
    return p_model / max(p_indep, 1e-12)


# ---- data and window schedule ----
px = pd.read_parquet(REPO / "data/combined_gfc_2026.parquet").astype(float)
r = np.log(px).diff().dropna()
N = r.shape[1]
ends = np.unique(np.linspace(WIN - 1, len(r) - 1, M).round().astype(int))
print(f"book N={N} days={len(r)} windows={len(ends)} configs/window={len(CFGS)} "
      f"total_fits={len(ends) * len(CFGS)} smoke={args.smoke}", flush=True)

LONG_COLS = ["window_date", "window_idx", "config", "axis", "seed", "d", "chi", "ordering",
             "boot", "feature", "value"]
LONG.write_text(",".join(LONG_COLS) + "\n")

FEATS = ["tail_asym", "block_mi", "tail_lift", "drift"]
prev_base = None
for wi, e in enumerate(ends):
    w = r.iloc[e - WIN + 1:e + 1]
    wdate = w.index[-1]
    fied = list(binning.discretize(w, n_bins=6)[2])     # v4's ordering convention (6-bin MI graph)
    ords = {"fiedler": fied, "rev": fied[::-1],
            "rand": list(np.random.default_rng(7).permutation(list(w.columns)))}
    base_fit = None
    for cfg in CFGS:
        t0 = time.time()
        if cfg["boot"] is None:
            data, order = w, ords[cfg["ordering"]]
        else:
            data = sb_resample(w, seed=10_000 * (wi + 1) + cfg["boot"])
            # a bootstrap draw recomputes its own ordering: the resample perturbs the FULL
            # estimator pipeline (bins + ordering + fit), which is the sampling noise we want
            order = list(binning.discretize(data, n_bins=6)[2])
        fit = dmrg.fit_dmrg(data[order], d=cfg["d"], chi=cfg["chi"], sweeps=SWEEPS,
                            seed=cfg["seed"])
        if cfg["label"] == "base":
            base_fit = fit
        feats = {
            "tail_asym": F.tail_dependence_asymmetry(fit, q=0.10),
            "block_mi": F.block_mutual_information(fit, n_samples=NSAMP, seed=0),
            "tail_lift": joint_tail_lift(fit),
            # drift scores the ORIGINAL window days under this fit vs the previous window's BASE
            # fit; on the bins axis the two fits use different d, so drift mixes bin-resolution
            # offset with distribution movement — low bins-axis reliability for drift is expected
            # and is part of the honest answer
            "drift": (F.distribution_drift(fit, prev_base, w)
                      if prev_base is not None else float("nan")),
        }
        with LONG.open("a") as fh:                       # appended incrementally, one fit at a time
            for fname, val in feats.items():
                fh.write(f"{wdate.date()},{wi},{cfg['label']},{cfg['axis']},{cfg['seed']},"
                         f"{cfg['d']},{cfg['chi']},{cfg['ordering']},{cfg['boot']},{fname},{val}\n")
        print(f"[{wi + 1}/{len(ends)}] {wdate.date()} {cfg['label']:8s} "
              f"{time.time() - t0:6.1f}s  "
              + "  ".join(f"{k_}={v:.4g}" for k_, v in feats.items()), flush=True)
    prev_base = base_fit                                 # drift reference for the NEXT window

# ---- reliability summary ----
L = pd.read_csv(LONG, parse_dates=["window_date"])
srows = []
for feat in FEATS:
    sub = L[L.feature == feat]
    base = sub[sub.config == "base"].sort_values("window_idx")["value"].to_numpy()
    x = base[np.isfinite(base)]
    vb = float(np.var(x, ddof=1)) if len(x) > 1 else np.nan
    ac = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 else np.nan
    axes_all = dict(AXIS_MEMBERS); axes_all["pooled"] = [c["label"] for c in CFGS]
    for axis, members in axes_all.items():
        wvars = []
        for wi in sorted(sub.window_idx.unique()):
            vals = sub[(sub.window_idx == wi) & (sub.config.isin(members))]["value"].to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals) > 1:
                wvars.append(float(np.var(vals, ddof=1)))
        vw = float(np.mean(wvars)) if wvars else np.nan
        rel = vb / (vb + vw) if np.isfinite(vb) and np.isfinite(vw) and (vb + vw) > 0 else np.nan
        srows.append(dict(feature=feat, axis=axis, reliability=rel, between_var=vb,
                          within_var=vw, base_lag1_ac=ac, n_windows=int(len(x))))
S = pd.DataFrame(srows)
S.to_csv(RESULTS / "reliability_summary.csv", index=False)
print(S.round(4).to_string(index=False), flush=True)

# ---- figure: grouped bars, reliability per feature per axis, 0.7 reference line ----
axes = [a for a in ["seeds", "bins", "chi", "ordering", "bootstrap", "pooled"]
        if a in S.axis.unique()]
xs = np.arange(len(FEATS)); wd = 0.8 / max(len(axes), 1)
fig, ax = plt.subplots(figsize=(10, 5))
for k_, a in enumerate(axes):
    vals = [float(S[(S.feature == f_) & (S.axis == a)].reliability.iloc[0]) for f_ in FEATS]
    ax.bar(xs + (k_ - (len(axes) - 1) / 2) * wd, vals, wd, label=a)
ax.axhline(0.7, color="k", ls=":", lw=1)
ax.set_xticks(xs); ax.set_xticklabels(FEATS)
ax.set_ylim(0, 1.05)
ax.set_ylabel("reliability = Var_between / (Var_between + Var_within)")
ax.set_title("MPS feature reliability by perturbation axis (reference: 0.7)")
ax.legend(fontsize=8)
for s_ in ["top", "right"]:
    ax.spines[s_].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES / "reliability.png", dpi=150)
print("saved", RESULTS / "reliability_summary.csv", "and", FIGURES / "reliability.png", flush=True)
