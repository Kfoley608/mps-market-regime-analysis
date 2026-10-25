"""Week 7 item 9 — pairwise maximum-entropy (Ising) competitor under the walk-forward tail protocol.

The reviewer objection this answers: "you only beat covariance models." The clean triple is
  gaussian_copula  pairwise dependence, continuous margins (normal-scores corr, deep_tail_copula recipe)
  ising            pairwise max-ent over the BINARY tail indicators (matches first+second moments,
                   agnostic to higher-order interactions by construction)
  mps_chi8         flexible higher-order (DMRG Born machine)
  independent      product baseline (exact Poisson-binomial, no simulation)
If the MPS beats the Ising model too, the edge is specifically HIGHER-ORDER structure, not just
"better than a covariance matrix."

Protocol — numerically identical to week7/walkforward_tail_prediction.py (kept self-contained on
purpose; do not import from it):
  - expanding train (min 1500 days), 126-day test blocks, advance 126
  - event Y_t = 1{ >= 3 assets below their TRAIN q=0.05 quantiles } (README common definition)
  - every simulated model reports P(Y=1) from 100,000 simulated days through the SAME train
    thresholds (BUG-LOG #15: quantiles, orderings, bins, correlations all train-only)
  - P-functionals only (BUG-LOG #14): the MPS leg uses sample_returns / thresholds, no
    signed-amplitude diagnostics anywhere

Ising fit: pseudo-likelihood — per-site L2-regularized logistic regression of b_i on b_{-i}
(C=1.0), J symmetrized, fields from the intercepts; then Gibbs sampling for P(sum b >= 3).
NOTE: the tail indicators are RARE (train frequency ~5% per site), so pseudo-likelihood is
noisy in exactly the regime we care about — the L2 regularization is load-bearing, and the
train event frequency is recorded per block so this can be audited.

Outputs (incremental):
  week7/results/ising_blocks_<book>.csv   per-block probabilities + realized event counts
  week7/results/ising_metrics.csv         pooled metrics + bootstrap CIs, appended per book

Pod-only (fits MPS / sklearn / Gibbs).

Usage:
  python week7/ising_competitor.py                                  # default 3 books
  python week7/ising_competitor.py --books all                      # full six from week7/README
  python week7/ising_competitor.py --books primary_prices,banks_crsp_prices
  python week7/ising_competitor.py --smoke                          # 1 book, 3 blocks, 20k sims
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import argparse
import pathlib
import sys
import time
import warnings
import zlib

warnings.filterwarnings("ignore")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
from scipy.stats import norm

MIN_TRAIN, TEST_LEN = 1500, 126     # walkforward_tail_prediction.py protocol — keep identical
K_ASSETS, Q_TAIL = 3, 0.05          # event: >= 3 assets below train q05
N_SIMS = 100_000
D_BINS, CHI, SWEEPS = 8, 8, 10      # MPS leg
GIBBS_BURN, GIBBS_THIN, GIBBS_CHAINS = 2000, 5, 500
B_BOOT = 2000                       # block bootstrap replicates
EPS = 1e-6                          # probability clip for the log score

DEFAULT_BOOKS = ["combined_gfc_2026", "primary_prices", "banks_crsp_prices"]
ALL_BOOKS = ["primary_prices", "combined_gfc_2026", "banks_crsp_prices",
             "combined_crossasset_2026_lagbtc", "futures_macro_prices", "semis_crsp_prices"]
MODELS = ["gaussian_copula", "ising", "mps_chi8", "independent"]


def resolve_book(name):
    """Book file lookup: repo data/, then DATA_EXT, then repo data_external/, then Documents."""
    dirs = [REPO / "data",
            pathlib.Path(os.environ["DATA_EXT"]) if os.environ.get("DATA_EXT") else None,
            REPO / "data_external"]
    for d in dirs:
        if d is not None and (d / f"{name}.parquet").exists():
            return d / f"{name}.parquet"
    return None


# --------------------------------------------------------------------------- #
# Models — each returns a single P(Y=1) for the block                          #
# --------------------------------------------------------------------------- #
def normal_scores_corr(R):
    T = R.shape[0]
    z = norm.ppf((np.argsort(np.argsort(R, axis=0), axis=0) + 0.5) / T)
    return np.corrcoef(z.T)


def p_gaussian_copula(train, thr, n_sims, rng):
    """Gaussian copula + empirical margins (deep_tail_copula.py recipe): simulate the copula,
    threshold in u-space at each asset's ACTUAL train exceedance mass — identical to mapping
    through the inverse empirical CDF and applying the return-space thresholds."""
    R = train.to_numpy()
    N = R.shape[1]
    levels = (R < thr).mean(axis=0)                      # per-asset train tail mass (~q)
    Rho = normal_scores_corr(R) + 1e-10 * np.eye(N)
    Z = rng.standard_normal((n_sims, N)) @ np.linalg.cholesky(Rho).T
    lo = Z < norm.ppf(levels)[None, :]
    return float((lo.sum(axis=1) >= K_ASSETS).mean())


def fit_ising_pseudolikelihood(b):
    """Pairwise max-ent over b in {0,1}^N by pseudo-likelihood: per-site logistic regression of
    b_i on b_{-i} (L2, C=1.0), couplings symmetrized, fields from the intercepts. Rare-event
    binaries make this noisy — regularization matters (see module docstring)."""
    from sklearn.linear_model import LogisticRegression
    T, N = b.shape
    J = np.zeros((N, N))
    h = np.zeros(N)
    for i in range(N):
        y = b[:, i]
        others = [j for j in range(N) if j != i]
        if y.min() == y.max():                           # degenerate site: field-only fallback
            p = min(max(y.mean(), 1.0 / (T + 2)), 1 - 1.0 / (T + 2))
            h[i] = float(np.log(p / (1 - p)))
            continue
        lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000)
        lr.fit(b[:, others], y)
        J[i, others] = lr.coef_[0]
        h[i] = float(lr.intercept_[0])
    J = 0.5 * (J + J.T)                                  # symmetrize couplings
    np.fill_diagonal(J, 0.0)
    return h, J


def p_ising(train, thr, n_sims, rng):
    """Fit the pairwise max-ent model on the tail indicators, then Gibbs-sample P(sum b >= 3).
    GIBBS_CHAINS independent chains run in parallel (vectorized site updates), each burned in for
    GIBBS_BURN full sweeps; states are then kept every GIBBS_THIN sweeps until n_sims total —
    same budget as serial "2000 burn-in + 100k kept, thinning 5" but with far lower autocorrelation
    (independent chains) and numpy-vectorized inner loops."""
    b = (train.to_numpy() < thr).astype(np.float64)      # (T, N) tail indicators
    N = b.shape[1]
    h, J = fit_ising_pseudolikelihood(b)
    C = min(GIBBS_CHAINS, n_sims)
    keeps_per_chain = int(np.ceil(n_sims / C))
    state = (rng.random((C, N)) < b.mean(axis=0)[None, :]).astype(np.float64)
    hits = total = 0
    n_sweeps = GIBBS_BURN + GIBBS_THIN * keeps_per_chain
    for sweep in range(n_sweeps):
        for i in range(N):                               # sequential-scan Gibbs, chains vectorized
            field = h[i] + state @ J[i]
            state[:, i] = (rng.random(C) < 1.0 / (1.0 + np.exp(-field))).astype(np.float64)
        if sweep >= GIBBS_BURN and (sweep - GIBBS_BURN) % GIBBS_THIN == 0:
            hits += int((state.sum(axis=1) >= K_ASSETS).sum())
            total += C
    return float(hits / total)


def p_mps(train, thr, n_sims, seed):
    """DMRG Born machine on the Fiedler-ordered train (ordering + bins train-only), synthetic
    returns through the SAME train thresholds."""
    from mps_regime import binning, dmrg
    order = binning.discretize(train, n_bins=6)[2]       # train-only ordering (v4 convention)
    fit = dmrg.fit_dmrg(train[order], d=D_BINS, chi=CHI, sweeps=SWEEPS, seed=seed)
    sims = fit.sample_returns(n_sims, seed=seed + 1)     # (n_sims, N) in fit.columns order
    thr_o = np.array([thr[list(train.columns).index(c)] for c in fit.columns])
    return float(((sims < thr_o[None, :]).sum(axis=1) >= K_ASSETS).mean())


def p_independent(train, thr):
    """Exact product baseline: Poisson-binomial P(sum >= 3) with per-asset TRAIN exceedance
    rates (convolution, no simulation)."""
    rates = (train.to_numpy() < thr).mean(axis=0)
    pmf = np.array([1.0])
    for p in rates:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return float(pmf[K_ASSETS:].sum())


# --------------------------------------------------------------------------- #
# Metrics                                                                       #
# --------------------------------------------------------------------------- #
def log_score(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def ece10(p, y):
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    e = 0.0
    for b in range(10):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def pr_auc(p, y):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


def bootstrap_diffs(blocks_p, blocks_y, m1, m2, rng):
    """Block bootstrap (resample whole test blocks) CI on pooled Brier and log-score differences
    m1 - m2. Returns (brier_diff, blo, bhi, ls_diff, llo, lhi)."""
    nb = len(blocks_y)
    p1 = np.concatenate([blocks_p[m1][j] for j in range(nb)])
    p2 = np.concatenate([blocks_p[m2][j] for j in range(nb)])
    y = np.concatenate(blocks_y)
    bd, ld = brier(p1, y) - brier(p2, y), log_score(p1, y) - log_score(p2, y)
    bs_b, bs_l = np.empty(B_BOOT), np.empty(B_BOOT)
    for r in range(B_BOOT):
        pick = rng.integers(nb, size=nb)
        yb = np.concatenate([blocks_y[j] for j in pick])
        p1b = np.concatenate([blocks_p[m1][j] for j in pick])
        p2b = np.concatenate([blocks_p[m2][j] for j in pick])
        bs_b[r] = brier(p1b, yb) - brier(p2b, yb)
        bs_l[r] = log_score(p1b, yb) - log_score(p2b, yb)
    blo, bhi = np.percentile(bs_b, [2.5, 97.5])
    llo, lhi = np.percentile(bs_l, [2.5, 97.5])
    return bd, float(blo), float(bhi), ld, float(llo), float(lhi)


# --------------------------------------------------------------------------- #
# Per-book walk-forward                                                          #
# --------------------------------------------------------------------------- #
def run_book(book, path, args, out_dir):
    from mps_regime import data as D
    rets = D.to_returns(pd.read_parquet(path).astype(float), kind="log")
    T, N = rets.shape
    n_blocks = (T - MIN_TRAIN) // TEST_LEN
    if n_blocks < 1:
        print(f"{book}: only {T} days (< {MIN_TRAIN + TEST_LEN}) — skipped", flush=True)
        return
    if args.smoke:
        n_blocks = min(n_blocks, 3)
    n_sims = 20_000 if args.smoke else N_SIMS
    base_seed = zlib.crc32(book.encode()) % 1_000_000
    blk_csv = out_dir / f"ising_blocks_{book}.csv"
    print(f"\n===== {book} | T={T} N={N} | blocks={n_blocks} sims={n_sims} "
          f"| {rets.index[0].date()}..{rets.index[-1].date()} =====", flush=True)

    done = set()
    if blk_csv.exists():                                 # resume completed blocks
        done = set(pd.read_csv(blk_csv)["block"].astype(int))
        print(f"resume: blocks {sorted(done)} already done", flush=True)

    t0 = time.time()
    for bi in range(n_blocks):
        split = MIN_TRAIN + bi * TEST_LEN
        train, test = rets.iloc[:split], rets.iloc[split:split + TEST_LEN]
        if bi in done:
            continue
        seed = base_seed + 977 * bi                      # deterministic per (book, block)
        thr = np.quantile(train.to_numpy(), Q_TAIL, axis=0)          # train-only thresholds
        y = ((test.to_numpy() < thr[None, :]).sum(axis=1) >= K_ASSETS).astype(float)
        train_freq = float(((train.to_numpy() < thr[None, :]).sum(axis=1)
                            >= K_ASSETS).mean())         # audit: rare-event base rate
        rng = np.random.default_rng(seed)
        pg = p_gaussian_copula(train, thr, n_sims, rng)
        t1 = time.time()
        pi_ = p_ising(train, thr, n_sims, np.random.default_rng(seed + 1))
        t2 = time.time()
        pm = p_mps(train, thr, n_sims, seed=seed + 2)
        t3 = time.time()
        pind = p_independent(train, thr)
        row = pd.DataFrame([dict(
            book=book, block=bi, train_end=str(train.index[-1].date()),
            test_start=str(test.index[0].date()), test_end=str(test.index[-1].date()),
            n_test=len(test), n_events=int(y.sum()), train_event_freq=train_freq,
            gaussian_copula=pg, ising=pi_, mps_chi8=pm, independent=pind, seed=seed)])
        row.to_csv(blk_csv, mode="a", header=not blk_csv.exists(), index=False)
        print(f"[{time.time() - t0:7.0f}s] block {bi:2d} {row.iloc[0]['test_start']}"
              f"..{row.iloc[0]['test_end']}  events={int(y.sum()):3d}/{len(test)}"
              f"  train_freq={train_freq:.4f}  gauss={pg:.4f} ising={pi_:.4f}({t2 - t1:.0f}s)"
              f" mps={pm:.4f}({t3 - t2:.0f}s) indep={pind:.4f}", flush=True)

    # ---- pooled metrics + bootstrap CIs from the (possibly resumed) block CSV ----
    df = pd.read_csv(blk_csv).sort_values("block")
    blocks_y, blocks_p = [], {m: [] for m in MODELS}
    for _, r in df.iterrows():
        split = MIN_TRAIN + int(r["block"]) * TEST_LEN
        train, test = rets.iloc[:split], rets.iloc[split:split + TEST_LEN]
        thr = np.quantile(train.to_numpy(), Q_TAIL, axis=0)
        y = ((test.to_numpy() < thr[None, :]).sum(axis=1) >= K_ASSETS).astype(float)
        blocks_y.append(y)
        for m in MODELS:
            blocks_p[m].append(np.full(len(y), float(r[m])))
    y_all = np.concatenate(blocks_y)
    met_csv = out_dir / "ising_metrics.csv"
    rows = []
    for m in MODELS:
        p_all = np.concatenate(blocks_p[m])
        rows.append(dict(book=book, kind="model", name=m, n_days=len(y_all),
                         n_events=int(y_all.sum()), event_rate=float(y_all.mean()),
                         log_score=log_score(p_all, y_all), brier=brier(p_all, y_all),
                         ece10=ece10(p_all, y_all), pr_auc=pr_auc(p_all, y_all)))
    brng = np.random.default_rng(base_seed + 555)
    for m1, m2 in [("mps_chi8", "ising"), ("mps_chi8", "gaussian_copula"),
                   ("ising", "gaussian_copula")]:
        bd, blo, bhi, ld, llo, lhi = bootstrap_diffs(blocks_p, blocks_y, m1, m2, brng)
        rows.append(dict(book=book, kind="diff", name=f"{m1}_minus_{m2}", n_days=len(y_all),
                         n_events=int(y_all.sum()), brier=bd, brier_lo=blo, brier_hi=bhi,
                         log_score=ld, log_score_lo=llo, log_score_hi=lhi))
    mdf = pd.DataFrame(rows)
    mdf.to_csv(met_csv, mode="a", header=not met_csv.exists(), index=False)
    with pd.option_context("display.width", 200):
        print(mdf.to_string(index=False), flush=True)
    print(f"metrics appended -> {met_csv}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--books", default=",".join(DEFAULT_BOOKS),
                    help='comma list, or "all" for the full six (week7/README)')
    ap.add_argument("--smoke", action="store_true", help="1 book, 3 blocks, 20000 sims")
    ap.add_argument("--out-dir", default=None, help="results dir (default week7/results)")
    ap.add_argument("--allow-local", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if sys.platform == "win32" and not args.allow_local:
        sys.exit("ising_competitor.py fits MPS / Ising models — run on the RunPod server, not "
                 "the laptop (--allow-local to override).")
    books = ALL_BOOKS if args.books.strip() == "all" else [b.strip() for b in args.books.split(",")]
    if args.smoke:
        books = books[:1]
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else REPO / "week7" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    for book in books:
        path = resolve_book(book)
        if path is None:
            print(f"{book}: parquet not found in data/, DATA_EXT, data_external, Documents "
                  f"backup — skipped", flush=True)
            continue
        run_book(book, path, args, out_dir)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
