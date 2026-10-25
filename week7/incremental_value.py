"""Reviewer item 6 — incremental predictive value of the MPS gauges after classical indicators,
evaluated entirely OUT OF SAMPLE on the committed week-6 gauge panel.

Question: once a risk desk already has vol / correlation / PCA / absorption / turbulence /
VIX / MOVE, do the MPS gauges IMPROVE prediction of next-period risk outcomes? Three feature
sets (classical-only, mps-only, classical+mps) x ridge (continuous) / L2 logistic (binary),
scored on a chronological 70/30 split AND expanding-window one-step-ahead refits (60% burn-in),
with a 1-row purge (the label horizon) between train and test everywhere. The headline number
is the stationary-block-bootstrap CI on the (classical+mps MINUS classical-only) improvement
per metric — not the raw levels.

Targets (all strictly future: the value at the NEXT panel row; one row = 10 trading days at the
committed rolling cadence WIN=252/STEP=10 of analysis/parallel_rolling_v4.py):
  rv_next        next-row realized_vol                                   (continuous)
  dd_next        next-row drawdown                                       (continuous)
  rv_top         next-row realized_vol > causal expanding q90            (binary)
  varb_high      next-row var_breaches > causal expanding q90            (binary)
  tailfreq_next  next-row model-free 3-of-N 5%-tail 21d event frequency  (continuous)
  tailfreq_top   ... > causal expanding q90                              (binary)
The joint-tail target is recomputed MODEL-FREE from data/combined_gfc_2026.parquet (log
returns, EXPANDING causal per-asset 5% quantiles shifted one day, trailing 21d frequency of
the week-7 common event Y = 1{>=3 assets below their tail quantile}) — an MPS output
(G_jointcrash) is never used as a target. All decile thresholds are causal-expanding.

OOS R2 convention matches tier1_analysis_v3.py (BUG-LOG #16): 1 - SSE / TSS about the
TEST-sample mean. Standardization constants are TRAIN-only (BUG-LOG #15).

NO torch dependency — pure pandas/sklearn on the precomputed panel; allowed on the laptop.
Outputs: week7/results/incremental_value_metrics.csv
         week7/results/incremental_value_predictions.csv.gz
         week7/figures/incremental_value.png
"""
import pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, average_precision_score

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"; FIGURES = HERE / "figures"
RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)

B_BOOT, BLOCK = 1000, 6      # stationary bootstrap draws / mean block length (panel rows)
PURGE = 1                    # label horizon in rows — purged between train and test
MIN_HIST = 24                # rows before a causal expanding decile threshold is usable (~1y)
K, Q_TAIL, FREQ_D = 3, 0.05, 21   # week-7 common joint-tail event and its frequency window

CLS = ["realized_vol", "avg_corr", "pc1_share", "absorption_ratio", "turbulence",
       "gauss_tc", "VIX", "MOVE"]
MPS = ["G_asymmetry", "G_complexity_HO", "G_jointcrash", "G_taildep", "p_stress"]

P = pd.read_csv(REPO / "week6/results/gauge_panel.csv", parse_dates=["date"]).set_index("date")
CLS = [c for c in CLS if c in P.columns]
MPS = [c for c in MPS if c in P.columns]
SETS = {"classical": CLS, "mps": MPS, "both": CLS + MPS}
print(f"panel rows={len(P)} classical={CLS} mps={MPS}", flush=True)

# ---- model-free joint-tail frequency target (never an MPS output) ----
px = pd.read_parquet(REPO / "data/combined_gfc_2026.parquet").astype(float)
r = np.log(px).diff().dropna()
q = r.expanding(min_periods=252).quantile(Q_TAIL).shift(1)   # strictly-past quantile per day
Y = ((r < q).sum(axis=1) >= K).astype(float)
Y[q.isna().any(axis=1)] = np.nan
tail_freq = Y.rolling(FREQ_D).mean().reindex(P.index, method="ffill")  # trailing -> causal at row t


def causal_flag(series):
    """1{next-row value > expanding causal q90 of the series' own history through row t}."""
    thr = series.expanding(min_periods=MIN_HIST).quantile(0.90)
    nxt = series.shift(-1)
    out = (nxt > thr).astype(float)
    out[nxt.isna() | thr.isna()] = np.nan
    return out


TARGETS = {
    "rv_next":       (P["realized_vol"].shift(-1), "cont"),
    "rv_top":        (causal_flag(P["realized_vol"]), "bin"),
    "tailfreq_next": (tail_freq.shift(-1), "cont"),
    "tailfreq_top":  (causal_flag(tail_freq), "bin"),
}
if "drawdown" in P.columns:
    TARGETS["dd_next"] = (P["drawdown"].shift(-1), "cont")
if "var_breaches" in P.columns:
    TARGETS["varb_high"] = (causal_flag(P["var_breaches"]), "bin")


def fit_predict(Xtr, ytr, Xte, kind):
    """Ridge (cont) / L2 logistic (bin); TRAIN-only standardization (BUG-LOG #15)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if kind == "cont":
        return Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)
    if len(np.unique(ytr)) < 2:                      # degenerate early fold: predict prevalence
        return np.full(len(Xte), float(ytr.mean()))
    m = LogisticRegression(penalty="l2", C=1.0, max_iter=2000).fit(Xtr, ytr.astype(int))
    return m.predict_proba(Xte)[:, 1]


def score(y, p, kind):
    if kind == "cont":
        # OOS R2 as in tier1_analysis_v3.py (BUG-LOG #16): TSS about the TEST-sample mean
        return {"oos_r2": 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)}
    pc = np.clip(p, 1e-6, 1 - 1e-6); yi = y.astype(int)
    out = {"log_loss": log_loss(yi, pc, labels=[0, 1]), "brier": brier_score_loss(yi, pc)}
    two = len(np.unique(yi)) == 2                    # AUCs undefined on a one-class resample
    out["roc_auc"] = roc_auc_score(yi, pc) if two else np.nan
    out["pr_auc"] = average_precision_score(yi, pc) if two else np.nan
    return out


def sb_idx(n, rng):
    """Stationary bootstrap indices (Politis-Romano), mean block length BLOCK rows."""
    out = np.empty(n, dtype=int); t = int(rng.integers(n))
    for i in range(n):
        out[i] = t
        t = int(rng.integers(n)) if rng.random() < 1.0 / BLOCK else (t + 1) % n
    return out


rows, pred_rows = [], []
for tname, (ys, kind) in TARGETS.items():
    df = P[CLS + MPS].join(ys.rename("y")).dropna()
    y = df["y"].to_numpy(); dates = df.index; n = len(y)
    Xd = {s: df[c].to_numpy() for s, c in SETS.items()}
    print(f"{tname:14s} kind={kind} rows={n} "
          + (f"pos={int(y.sum())}" if kind == "bin" else ""), flush=True)

    splits = {}
    # chronological 70/30, 1-row purge between train and test
    cut = int(0.70 * n)
    splits["70_30"] = ({s: fit_predict(Xd[s][:cut - PURGE], y[:cut - PURGE], Xd[s][cut:], kind)
                        for s in SETS}, np.arange(cut, n))
    # expanding walk-forward: refit every step after a 60% burn-in, purge 1 row
    burn = int(0.60 * n)
    pw = {s: np.full(n, np.nan) for s in SETS}
    for i in range(burn, n):
        for s in SETS:
            pw[s][i] = fit_predict(Xd[s][:i - PURGE], y[:i - PURGE], Xd[s][i:i + 1], kind)[0]
    splits["walkforward"] = ({s: pw[s][burn:] for s in SETS}, np.arange(burn, n))

    for split, (preds, te) in splits.items():
        yte = y[te]
        base = {s: score(yte, preds[s], kind) for s in SETS}
        for s in SETS:
            for m, v in base[s].items():
                rows.append(dict(target=tname, split=split, feature_set=s, metric=m,
                                 value=v, ci_lo=np.nan, ci_hi=np.nan))
        # HEADLINE: classical+mps MINUS classical-only, stationary-block-bootstrap CI.
        # Sign convention: for oos_r2 / roc_auc / pr_auc positive = MPS adds value;
        # for log_loss / brier NEGATIVE = MPS adds value.
        rng = np.random.default_rng(7)
        diffs = {m: [] for m in base["both"]}
        for _ in range(B_BOOT):
            ix = sb_idx(len(yte), rng)
            mb = score(yte[ix], preds["both"][ix], kind)
            mc = score(yte[ix], preds["classical"][ix], kind)
            for m in diffs:
                d = mb[m] - mc[m]
                if np.isfinite(d):
                    diffs[m].append(d)
        for m in diffs:
            lo, hi = (np.percentile(diffs[m], [2.5, 97.5]) if diffs[m] else (np.nan, np.nan))
            rows.append(dict(target=tname, split=split, feature_set="both_minus_classical",
                             metric=m, value=base["both"][m] - base["classical"][m],
                             ci_lo=lo, ci_hi=hi))
        for s in SETS:
            for dt_, yt, pv in zip(dates[te], yte, preds[s]):
                pred_rows.append(dict(date=dt_, target=tname, split=split, feature_set=s,
                                      y_true=yt, y_pred=pv))
        print(f"  {split:12s} " + " | ".join(
            f"{m}: both-cls {base['both'][m] - base['classical'][m]:+.4f}"
            for m in base["both"]), flush=True)

MET = pd.DataFrame(rows)
MET.to_csv(RESULTS / "incremental_value_metrics.csv", index=False)
pd.DataFrame(pred_rows).to_csv(RESULTS / "incremental_value_predictions.csv.gz",
                               index=False, compression="gzip")
print("saved", RESULTS / "incremental_value_metrics.csv", MET.shape, flush=True)

# ---- figure: improvement (both - classical) with CI whiskers, headline metric per target ----
HEAD = {"cont": "oos_r2", "bin": "roc_auc"}      # up = MPS adds value for both
imp = MET[MET.feature_set == "both_minus_classical"]
tnames = list(TARGETS)
xs = np.arange(len(tnames)); wdt = 0.38
fig, ax = plt.subplots(figsize=(10, 5))
for split, off, col in [("70_30", -wdt / 2, "#4477aa"), ("walkforward", wdt / 2, "#ee6677")]:
    vals, lo_e, hi_e = [], [], []
    for t in tnames:
        hm = HEAD[TARGETS[t][1]]
        rw = imp[(imp.target == t) & (imp.split == split) & (imp.metric == hm)]
        v = float(rw.value.iloc[0]) if len(rw) else np.nan
        lo = float(rw.ci_lo.iloc[0]) if len(rw) else np.nan
        hi = float(rw.ci_hi.iloc[0]) if len(rw) else np.nan
        vals.append(v)
        lo_e.append(v - lo if np.isfinite(lo) else 0.0)
        hi_e.append(hi - v if np.isfinite(hi) else 0.0)
    ax.bar(xs + off, vals, wdt, yerr=[lo_e, hi_e], capsize=3, color=col, label=split)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([f"{t}\n(Δ{HEAD[TARGETS[t][1]]})" for t in tnames], fontsize=9)
ax.set_ylabel("classical+mps minus classical-only")
ax.set_title("Incremental OOS value of the MPS gauges (95% stationary-block-bootstrap CI)")
ax.legend()
for s_ in ["top", "right"]:
    ax.spines[s_].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES / "incremental_value.png", dpi=150)
print("saved", FIGURES / "incremental_value.png", flush=True)
