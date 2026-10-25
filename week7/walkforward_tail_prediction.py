"""Week 7 item 1 (TOP): purged walk-forward out-of-sample joint-tail event prediction.

Protocol (per book, see week7/README.md shared conventions):
  - Log returns. Expanding training window (>= 1500 days), test block = 126 trading days,
    advance by 126, strictly chronological, no overlap. Train ends the day before the test
    block starts; the event Y_t is a same-day function of day-t returns, so a zero-day
    embargo is purge-correct (no label reaches back into the training window).
  - From TRAIN ONLY (BUG-LOG #15, frozen then applied to test): per-asset empirical q=0.05
    lower-tail thresholds, Fiedler chain ordering, and bin edges (inside fit_dmrg).
  - Event: Y_t = 1{ at least k=3 assets below their train q05 threshold on day t }.
  - Every model reports ONE P(Y=1) per block by simulation (N_SIM sampled days, same train
    thresholds applied to the samples):
      independent      per-asset i.i.d. bootstrap of train returns (columns resampled
                       independently -> cross-asset dependence destroyed)
      gaussian_copula  normal-scores (van der Waerden) correlation from train; MVN sample
                       -> Phi -> uniforms -> inverse empirical CDF of each train marginal
      t_copula         same correlation; dof nu by ML grid on the train pseudo-observations
      empirical        Laplace-smoothed train frequency of Y: (count + 0.5) / (n_train + 1)
      mps_chi{2,4,8}   fit_dmrg on the Fiedler-ordered train block (d=8, sweeps=10, seed=0),
                       sample_returns(within_bin_noise=True), same thresholds
  - Metrics pooled across all test days of all blocks: mean log score, Brier, ECE
    (10 equal-width buckets), PR-AUC, calibration table. Block bootstrap (resample test
    blocks with replacement) for CIs on log score / Brier and on each MPS variant minus
    gaussian_copula — the headline difference.

Run on the pod:  python week7/walkforward_tail_prediction.py [--books ...] [--smoke]
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import warnings; warnings.filterwarnings("ignore")

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln
from scipy.stats import norm, t as tdist

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mps_regime import binning, data as D, dmrg  # noqa: E402

BOOKS = {
    "primary": "primary_prices.parquet",
    "gfc": "combined_gfc_2026.parquet",
    "banks": "banks_crsp_prices.parquet",
    "crossasset": "combined_crossasset_2026_lagbtc.parquet",
    "futures": "futures_macro_prices.parquet",
    "semis": "semis_crsp_prices.parquet",
}

MIN_TRAIN, TEST_LEN = 1500, 126
Q_TAIL, K_ASSETS = 0.05, 3
D_BINS, SWEEPS = 8, 10
CHI_LIST = (2, 4, 8)
N_SIM, B_BOOT = 100_000, 2000
NU_GRID = (3, 4, 5, 6, 8, 10, 15, 20, 30)
ORDER_NBINS = 6          # MI/Fiedler binning (repo convention, cf. deep_tail_copula)
PCLIP = 1e-6
N_CAL_BUCKETS = 10


# --------------------------------------------------------------------------- #
# Data resolution                                                              #
# --------------------------------------------------------------------------- #
def resolve_book(fname):
    """repo data/ first (some books live there), then DATA_EXT, data_external, Documents."""
    cands = [ROOT / "data" / fname]
    ext = os.environ.get("DATA_EXT")
    if ext:
        cands.append(pathlib.Path(ext) / fname)
    cands.append(ROOT / "data_external" / fname)
    for c in cands:
        if c.exists():
            return c
    return None


# --------------------------------------------------------------------------- #
# Copula machinery (same approach as analysis/deep_tail_copula.py)              #
# --------------------------------------------------------------------------- #
def normal_scores_corr(R):
    T = R.shape[0]
    z = norm.ppf((np.argsort(np.argsort(R, axis=0), axis=0) + 0.5) / T)
    return np.corrcoef(z.T)


def pseudo_obs(R):
    T = R.shape[0]
    return (np.argsort(np.argsort(R, axis=0), axis=0) + 0.5) / T


def t_copula_loglik(U, Rho, nu):
    """t-copula log density summed over rows: log f_mvt(x;Rho,nu) - sum_i log f_t(x_i;nu)."""
    N = Rho.shape[0]
    x = tdist.ppf(U, df=nu)
    cf = cho_factor(Rho + 1e-10 * np.eye(N))
    logdet = 2.0 * np.log(np.diag(cf[0])).sum()
    q = (x * cho_solve(cf, x.T).T).sum(axis=1)
    lmv = (gammaln((nu + N) / 2) - gammaln(nu / 2) - 0.5 * N * np.log(nu * np.pi)
           - 0.5 * logdet - 0.5 * (nu + N) * np.log1p(q / nu))
    lmarg = tdist.logpdf(x, df=nu).sum(axis=1)
    return float((lmv - lmarg).sum())


def fit_t_nu(U, Rho):
    lls = [t_copula_loglik(U, Rho, nu) for nu in NU_GRID]
    return NU_GRID[int(np.argmax(lls))]


def inv_ecdf(train_arr, U):
    """Inverse empirical CDF per column: uniforms (n_sim, N) -> train-marginal returns."""
    n = train_arr.shape[0]
    out = np.empty_like(U)
    for j in range(train_arr.shape[1]):
        xs = np.sort(train_arr[:, j])
        out[:, j] = xs[np.clip((U[:, j] * n).astype(np.int64), 0, n - 1)]
    return out


def event_frac(sim, thr):
    """Fraction of simulated days with >= K_ASSETS assets below their (train) threshold."""
    return float(((sim < thr[None, :]).sum(axis=1) >= K_ASSETS).mean())


# --------------------------------------------------------------------------- #
# Per-block model probabilities                                                 #
# --------------------------------------------------------------------------- #
def p_independent(train_arr, thr, n_sim, seed):
    rng = np.random.default_rng(seed)
    T, N = train_arr.shape
    idx = rng.integers(0, T, size=(n_sim, N))          # independent per column
    sim = train_arr[idx, np.arange(N)[None, :]]
    return event_frac(sim, thr)


def p_gaussian_copula(train_arr, thr, Rho, n_sim, seed):
    rng = np.random.default_rng(seed)
    N = train_arr.shape[1]
    L = np.linalg.cholesky(Rho + 1e-10 * np.eye(N))
    Z = rng.standard_normal((n_sim, N)) @ L.T
    sim = inv_ecdf(train_arr, norm.cdf(Z))
    return event_frac(sim, thr)


def p_t_copula(train_arr, thr, Rho, nu, n_sim, seed):
    rng = np.random.default_rng(seed)
    N = train_arr.shape[1]
    L = np.linalg.cholesky(Rho + 1e-10 * np.eye(N))
    Z = rng.standard_normal((n_sim, N)) @ L.T
    W = rng.chisquare(nu, size=n_sim) / nu
    U = tdist.cdf(Z / np.sqrt(W)[:, None], df=nu)
    sim = inv_ecdf(train_arr, U)
    return event_frac(sim, thr)


def p_empirical(train_arr, thr):
    y = ((train_arr < thr[None, :]).sum(axis=1) >= K_ASSETS)
    return (y.sum() + 0.5) / (len(y) + 1.0)


def p_mps(train_ord, thr_ord, chi, n_sim, sample_seed):
    t0 = time.time()
    fit = dmrg.fit_dmrg(train_ord, d=D_BINS, chi=chi, sweeps=SWEEPS, seed=0)
    dt = time.time() - t0
    sim = fit.sample_returns(n_sim, seed=sample_seed, within_bin_noise=True)
    return event_frac(sim, thr_ord), dt


# --------------------------------------------------------------------------- #
# Metrics (daily p_hat is block-constant -> pooled scores from block stats)     #
# --------------------------------------------------------------------------- #
def pooled_scores(p, e, n):
    """Pooled mean log score and Brier from per-block (p_hat, n_events, n_days) arrays."""
    pc = np.clip(p, PCLIP, 1 - PCLIP)
    tot = n.sum()
    ls = (e * np.log(pc) + (n - e) * np.log(1 - pc)).sum() / tot
    br = (e * (1 - pc) ** 2 + (n - e) * pc ** 2).sum() / tot
    return ls, br


def block_bootstrap_ci(p_by_model, e, n, models, gc="gaussian_copula", n_boot=B_BOOT):
    """Resample test blocks with replacement; CIs on each model's pooled log score / Brier
    and on (model - gaussian_copula) for the MPS variants."""
    rng = np.random.default_rng(2026)
    nb = len(n)
    draws = {m: np.zeros((n_boot, 2)) for m in models}
    for b in range(n_boot):
        idx = rng.integers(0, nb, size=nb)             # paired across models
        for m in models:
            draws[m][b] = pooled_scores(p_by_model[m][idx], e[idx], n[idx])
    out = {}
    for m in models:
        ls, br = draws[m][:, 0], draws[m][:, 1]
        row = dict(log_score_lo=np.percentile(ls, 2.5), log_score_hi=np.percentile(ls, 97.5),
                   brier_lo=np.percentile(br, 2.5), brier_hi=np.percentile(br, 97.5))
        if m.startswith("mps_") and gc in draws:
            dls = ls - draws[gc][:, 0]
            dbr = br - draws[gc][:, 1]
            row.update(dlog_vs_gc_lo=np.percentile(dls, 2.5), dlog_vs_gc_hi=np.percentile(dls, 97.5),
                       dbrier_vs_gc_lo=np.percentile(dbr, 2.5), dbrier_vs_gc_hi=np.percentile(dbr, 97.5))
        out[m] = row
    return out


def ece_and_table(p_daily, y_daily):
    """10 equal-width probability buckets -> ECE and calibration table rows."""
    edges = np.linspace(0, 1, N_CAL_BUCKETS + 1)
    b = np.clip(np.digitize(p_daily, edges[1:-1]), 0, N_CAL_BUCKETS - 1)
    rows, ece = [], 0.0
    for k in range(N_CAL_BUCKETS):
        m = b == k
        if not m.any():
            continue
        mp, fr, nk = p_daily[m].mean(), y_daily[m].mean(), int(m.sum())
        ece += nk / len(p_daily) * abs(mp - fr)
        rows.append(dict(bucket=f"[{edges[k]:.1f},{edges[k+1]:.1f})", mean_p_hat=mp,
                         realized_freq=fr, n=nk))
    return ece, rows


def pr_auc(p_daily, y_daily):
    if y_daily.sum() == 0 or y_daily.sum() == len(y_daily):
        return float("nan")
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_daily, p_daily))


# --------------------------------------------------------------------------- #
# Per-book walk-forward                                                          #
# --------------------------------------------------------------------------- #
def run_book(book, path, chi_list, n_sim, n_boot, max_blocks, out_dir):
    rets = D.to_returns(pd.read_parquet(path).astype(float), kind="log")
    T, N = rets.shape
    starts = list(range(MIN_TRAIN, T - TEST_LEN + 1, TEST_LEN))   # full 126-day blocks only
    if max_blocks:
        starts = starts[:max_blocks]
    models = ["independent", "gaussian_copula", "t_copula", "empirical"] + \
             [f"mps_chi{c}" for c in chi_list]
    print(f"\n===== {book} | T={T} N={N} | {len(starts)} blocks x {len(chi_list)} chi =====",
          flush=True)

    block_rows, daily_rows = [], []
    for bi, s in enumerate(starts):
        train, test = rets.iloc[:s], rets.iloc[s:s + TEST_LEN]
        train_arr, test_arr = train.to_numpy(), test.to_numpy()
        # ---- train-only estimation, frozen (BUG-LOG #15) ----
        thr = np.quantile(train_arr, Q_TAIL, axis=0)
        y = ((test_arr < thr[None, :]).sum(axis=1) >= K_ASSETS).astype(int)
        order = binning.discretize(train, n_bins=ORDER_NBINS)[2]
        thr_s = pd.Series(thr, index=rets.columns)
        Rho = normal_scores_corr(train_arr)
        nu = fit_t_nu(pseudo_obs(train_arr), Rho)

        ph = {}
        ph["independent"] = p_independent(train_arr, thr, n_sim, seed=7000 + bi)
        ph["gaussian_copula"] = p_gaussian_copula(train_arr, thr, Rho, n_sim, seed=8000 + bi)
        ph["t_copula"] = p_t_copula(train_arr, thr, Rho, nu, n_sim, seed=9000 + bi)
        ph["empirical"] = p_empirical(train_arr, thr)
        print(f"[{book} b{bi:02d}] {test.index[0].date()}..{test.index[-1].date()} "
              f"| n_train={s} events={int(y.sum())}/{len(y)} | nu*={nu} "
              f"| ind={ph['independent']:.4f} gc={ph['gaussian_copula']:.4f} "
              f"t={ph['t_copula']:.4f} emp={ph['empirical']:.4f}", flush=True)
        for chi in chi_list:
            p, dt = p_mps(train[order], thr_s[order].to_numpy(), chi, n_sim,
                          sample_seed=10_000 + 100 * bi + chi)
            ph[f"mps_chi{chi}"] = p
            print(f"[{book} b{bi:02d}]   mps_chi{chi}: p_hat={p:.4f} (dmrg fit {dt:.1f}s)",
                  flush=True)

        for m in models:
            block_rows.append(dict(book=book, block_start=str(test.index[0].date()),
                                   block_end=str(test.index[-1].date()), model=m,
                                   p_hat=ph[m], realized_freq=float(y.mean()),
                                   n_days=len(y), n_events=int(y.sum())))
        for t_, yt in zip(test.index, y):
            daily_rows.append({"date": str(t_.date()), "Y": int(yt),
                               **{f"p_{m}": ph[m] for m in models}})

    # ---- write per-book tables ----
    out_dir.mkdir(parents=True, exist_ok=True)
    bl = pd.DataFrame(block_rows)
    bl.to_csv(out_dir / f"wf_tail_blocks_{book}.csv", index=False)
    dl = pd.DataFrame(daily_rows)
    dl.to_csv(out_dir / f"wf_tail_daily_{book}.csv.gz", index=False, compression="gzip")

    # ---- pooled metrics + block bootstrap ----
    n = bl[bl.model == models[0]].n_days.to_numpy(float)
    e = bl[bl.model == models[0]].n_events.to_numpy(float)
    p_by_model = {m: bl[bl.model == m].p_hat.to_numpy() for m in models}
    cis = block_bootstrap_ci(p_by_model, e, n, models, n_boot=n_boot)
    y_daily = dl.Y.to_numpy()
    met_rows, cal_rows = [], []
    gc_ls, gc_br = pooled_scores(p_by_model["gaussian_copula"], e, n)
    for m in models:
        ls, br = pooled_scores(p_by_model[m], e, n)
        p_daily = dl[f"p_{m}"].to_numpy()
        ece, cal = ece_and_table(p_daily, y_daily)
        row = dict(book=book, model=m, n_blocks=len(n), n_days=int(n.sum()),
                   n_events=int(e.sum()), log_score=ls, brier=br, ece=ece,
                   pr_auc=pr_auc(p_daily, y_daily),
                   dlog_vs_gc=ls - gc_ls if m.startswith("mps_") else np.nan,
                   dbrier_vs_gc=br - gc_br if m.startswith("mps_") else np.nan)
        row.update(cis[m])
        met_rows.append(row)
        print(f"\n[{book}] {m}: log_score={ls:.4f} "
              f"[{cis[m]['log_score_lo']:.4f},{cis[m]['log_score_hi']:.4f}] "
              f"brier={br:.4f} ece={ece:.4f} pr_auc={row['pr_auc']:.4f}", flush=True)
        print(f"  calibration (bucket, mean_p_hat, realized_freq, n):", flush=True)
        for r in cal:
            print(f"    {r['bucket']:>10} {r['mean_p_hat']:.4f} {r['realized_freq']:.4f} "
                  f"{r['n']:>5d}", flush=True)
        for r in cal:
            cal_rows.append(dict(book=book, model=m, **r))
    pd.DataFrame(cal_rows).to_csv(out_dir / f"wf_tail_calibration_{book}.csv", index=False)

    # ---- append/overwrite this book's rows in the stacked metrics file ----
    mpath = out_dir / "wf_tail_metrics.csv"
    mdf = pd.DataFrame(met_rows)
    if mpath.exists():
        old = pd.read_csv(mpath)
        mdf = pd.concat([old[old.book != book], mdf], ignore_index=True)
    mdf.to_csv(mpath, index=False)
    print(f"[{book}] saved blocks/daily/calibration/metrics -> {out_dir}", flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Walk-forward joint-tail prediction, MPS vs baselines")
    ap.add_argument("--books", default=",".join(BOOKS), help="comma list of book names")
    ap.add_argument("--smoke", action="store_true",
                    help="gfc only, 3 blocks, chi=2, N_SIM=20000, 200 bootstrap draws")
    ap.add_argument("--out-dir", default=str(ROOT / "week7" / "results"))
    args = ap.parse_args()

    books = [b.strip() for b in args.books.split(",") if b.strip()]
    chi_list, n_sim, n_boot, max_blocks = CHI_LIST, N_SIM, B_BOOT, None
    if args.smoke:
        books, chi_list, n_sim, n_boot, max_blocks = ["gfc"], (2,), 20_000, 200, 3
    out_dir = pathlib.Path(args.out_dir)

    t0 = time.time()
    for book in books:
        if book not in BOOKS:
            print(f"WARNING: unknown book '{book}' — skipping", flush=True)
            continue
        path = resolve_book(BOOKS[book])
        if path is None:
            print(f"WARNING: {BOOKS[book]} not found (DATA_EXT / data_external / repo data) "
                  f"— skipping {book}", flush=True)
            continue
        run_book(book, path, chi_list, n_sim, n_boot, max_blocks, out_dir)
    print(f"\nDONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
