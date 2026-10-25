"""Tier-1 redundancy analysis v3 — on the causal v3 rolling features.
Fixes vs the old regime_analysis.py:
  1. dropna(subset=used columns) — an unused all-NaN column can no longer delete the GFC.
  2. Prints crisis-window coverage BEFORE/AFTER the join so silent sample loss is impossible.
  3. Reports in-sample R2 AND out-of-sample R2 (train first 60% of windows, score last 40%) —
     "adds info" now requires the classical set to fail out-of-sample too, not just in-sample.
Usage: python tier1_analysis_v3.py <rolling_v3_csv>
"""
import sys
import numpy as np, pandas as pd

CSV = sys.argv[1]
df = pd.read_csv(CSV, parse_dates=["date"]).set_index("date")

MPS = ["mps_entanglement", "mps_total_corr", "mps_mean_MI", "mps_tail_state",
       "mps_tail_asym", "mps_corr_len", "mps_bond_dim", "mps_surprise"]
CLS = ["realized_vol", "avg_corr", "pc1_share", "VIX", "MOVE", "drawdown", "var_breaches"]
CLS = [c for c in CLS if c in df.columns and df[c].notna().any()]

CRISES = {"GFC": ("2008-09-01", "2009-06-30"), "EuroCrisis": ("2011-07-01", "2011-12-31"),
          "COVID": ("2020-02-15", "2020-06-30"), "Hike2022": ("2022-01-01", "2022-10-31")}

n0 = len(df)
used = df.dropna(subset=[c for c in MPS + CLS if c in df.columns])
print(f"windows: {n0} -> {len(used)} after dropna(subset=used cols) | span {used.index.min().date()} -> {used.index.max().date()}")
for nm, (a, b) in CRISES.items():
    n_all = ((df.index >= a) & (df.index <= b)).sum()
    n_use = ((used.index >= a) & (used.index <= b)).sum()
    tag = "" if n_use == n_all else "  <-- LOST WINDOWS"
    print(f"  {nm:11s} windows in span: raw={n_all:3d} used={n_use:3d}{tag}")

def r2_ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    return 1 - resid.var() / y.var(), beta

# REVIEW FIX 2026-07-14 (BUG-LOG #16): standardize on the TRAIN split only (was full-sample —
# mild leakage through mean/sd), and use the standard OOS R2 (see r2 code below).
Xc_raw = used[CLS].to_numpy()
cut = int(0.6 * len(used))
mu, sd = Xc_raw[:cut].mean(0), Xc_raw[:cut].std(0) + 1e-12
Xc = (Xc_raw - mu) / sd

print(f"\nTier-1 redundancy: each MPS feature ~ classical set ({', '.join(CLS)})")
print(f"{'feature':18s} {'R2_in':>7} {'R2_oos':>7}  verdict (novel share = 1 - R2_oos)")
for f in MPS:
    if f not in used.columns or used[f].isna().all():
        continue
    y = used[f].to_numpy()
    y = (y - y[:cut].mean()) / (y[:cut].std() + 1e-12)   # train-only standardization (BUG-LOG #16)
    r2_in, _ = r2_ols(Xc, y)
    _, beta = r2_ols(Xc[:cut], y[:cut])
    X1te = np.column_stack([np.ones(len(used) - cut), Xc[cut:]])
    resid = y[cut:] - X1te @ beta
    # standard OOS R2: 1 - SSE / TSS about the TEST mean (was 1 - resid.var()/y.var(), which
    # silently removes mean-forecast bias and overstates OOS fit - BUG-LOG #16)
    r2_oos = 1 - (resid ** 2).sum() / (((y[cut:] - y[cut:].mean()) ** 2).sum() + 1e-12)
    verdict = ("REDUNDANT" if min(r2_in, r2_oos) > 0.75 else
               "ADDS INFO" if max(r2_in, r2_oos) < 0.60 else "partial")
    print(f"{f:18s} {r2_in:7.3f} {r2_oos:7.3f}  {verdict}")

print("\ncrisis-vs-calm ratios (mean in crisis windows / mean elsewhere):")
crisis_mask = np.zeros(len(used), dtype=bool)
for a, b in CRISES.values():
    crisis_mask |= np.asarray((used.index >= a) & (used.index <= b))
if crisis_mask.any() and (~crisis_mask).any():
    for f in MPS + CLS:
        if f in used.columns:
            c, k = used[f][crisis_mask].mean(), used[f][~crisis_mask].mean()
            print(f"  {f:18s} crisis={c:9.4f} calm={k:9.4f} ratio={c/k if k else float('nan'):6.2f}")
print("\nDONE")
