"""Rolling DMRG features v3 — the CAUSAL rerun (fixes the look-ahead in v2):
  1. Chain ordering (Fiedler-MI) frozen from the FIRST `ORD_DAYS` days only — never sees the future.
  2. Calm-reference MPS fit ONLY on low-vol days inside the same early period (threshold from that
     period too), so `mps_surprise` for later windows never scores against a future-informed reference.
  3. VaR-99 breach quantile shifted one day (no same-day self-inclusion).
  4. Worker exceptions are LOGGED with the window date (v2 swallowed them silently).
Everything else (feature battery, WIN/STEP defaults per the Tier-1 design) matches v2 so results are
comparable before/after the causality fix.
Usage: python parallel_rolling_v3.py <book_parquet> <vix_parquet> <out_csv> [nworkers] [win] [step]
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import warnings; warnings.filterwarnings("ignore")
import sys, pickle, traceback; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from multiprocessing import Pool
from mps_regime import data as D, binning, dmrg, features as F

BOOK = sys.argv[1]
VIXF = sys.argv[2] if len(sys.argv) > 2 else "data/vix_move_long.parquet"
OUT  = sys.argv[3] if len(sys.argv) > 3 else "data/rolling_v3.csv"
NW   = int(sys.argv[4]) if len(sys.argv) > 4 else 48
WIN  = int(sys.argv[5]) if len(sys.argv) > 5 else 252
STEP = int(sys.argv[6]) if len(sys.argv) > 6 else 10
D_BINS, CHI, SWEEPS, ORD_DAYS = 12, 8, 8, 250

rets = D.to_returns(pd.read_parquet(BOOK).astype(float), kind="log")
early = rets.iloc[:ORD_DAYS]
order = binning.discretize(early, n_bins=6)[2]              # ordering from EARLY window only
R = rets[order]; N = R.shape[1]; dates = R.index
vm = pd.read_parquet(VIXF).reindex(dates).ffill()
ew = R.mean(axis=1); cum = np.exp(ew.cumsum()); dd = cum / cum.cummax() - 1.0
var99 = ew.rolling(252).quantile(0.01).shift(1)             # yesterday's quantile (causal)
breach = (ew < var99).astype(float)

CALMPKL = OUT.replace(".csv", "_calmref.pkl")
if not os.path.exists(CALMPKL):
    # fit in a SEPARATE process (torch autograd in the parent breaks fork-based Pool — see v2)
    rc = os.system(f"python3 fit_calm_ref_v3.py {BOOK} {CALMPKL} {ORD_DAYS}")
    if rc != 0 or not os.path.exists(CALMPKL):
        sys.exit(f"calm-ref fit failed (rc={rc})")
CALM_REF = pickle.load(open(CALMPKL, "rb"))["fit"]
print(f"book={BOOK} N={N} WIN={WIN} STEP={STEP} | order(early-{ORD_DAYS}d): {list(order)}", flush=True)

# features only for windows that start AFTER the ordering/calm-ref period (fully causal series)
STARTS = [s for s in range(0, len(R) - WIN + 1, STEP) if s + WIN > ORD_DAYS]

def worker(si):
    w = R.iloc[si:si + WIN]; end = w.index[-1]
    try:
        fit = dmrg.fit_dmrg(w, d=D_BINS, chi=CHI, sweeps=SWEEPS)
        Cw = w.corr().to_numpy(); iu = np.triu_indices(N, 1)
        lam = np.sort(np.linalg.eigvalsh(Cw))[::-1]
        surprise = float(F.recent_nll(CALM_REF, CALM_REF.encode(w)))
    except Exception:
        print(f"WINDOW FAILED end={end.date()}:\n{traceback.format_exc()}", flush=True)
        return None
    return dict(date=end,
        mps_entanglement=F.entanglement_entropy(fit), mps_total_corr=F.total_correlation(fit, n_samples=2000),
        mps_mean_MI=F.mean_mutual_information(fit, adjacent=False), mps_tail_state=F.tail_state_probability(fit),
        mps_tail_asym=F.tail_dependence_asymmetry(fit), mps_corr_len=F.correlation_length(fit),
        mps_bond_dim=F.bond_dimension_at_quality(fit), mps_surprise=surprise,
        realized_vol=float(ew.iloc[si:si + WIN].std() * np.sqrt(252)), avg_corr=float(Cw[iu].mean()),
        pc1_share=float(lam[0] / lam.sum()), VIX=float(vm["VIX"].iloc[si:si + WIN].mean()),
        MOVE=float(vm["MOVE"].iloc[si:si + WIN].mean()), drawdown=float(dd.iloc[si + WIN - 1]),
        var_breaches=float(breach.iloc[si:si + WIN].sum()))

if __name__ == "__main__":
    print(f"windows={len(STARTS)} workers={NW} range={dates.min().date()}..{dates.max().date()}", flush=True)
    with Pool(NW) as p:
        rows = [r for r in p.map(worker, STARTS) if r is not None]
    print(f"completed {len(rows)}/{len(STARTS)} windows", flush=True)
    df = pd.DataFrame(rows).sort_values("date").set_index("date")
    df.to_csv(OUT)
    print(f"wrote {OUT} shape={df.shape}")
