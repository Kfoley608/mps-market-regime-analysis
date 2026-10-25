"""Week-7 items 4+5: MS-Born stress-PROBABILITY calibration against a precisely defined
future joint-tail stress regime, with Brier/Murphy decomposition, vs three baselines.

Target (shared conventions, week7/README.md)
--------------------------------------------
Y_t = 1{ >= k=3 assets below their train-window (data up to the current refit date)
         empirical q=0.05 quantile on day t }.
S_t = 1{ mean(Y_{t+1..t+21}) > trailing 90th percentile of the 21-day rolling mean of Y
         computed on data up to t } — expanding causal threshold, >= 500 valid prior days
         of the rolling mean before any day is scored.
Y itself is causal: tail thresholds refresh on the SAME 63-day refit grid as the models
(pre-model burn-in uses the same stride starting once 250 days of history exist), always
from data strictly before the segment they label.

Causal probability protocol (walk-forward refit)
------------------------------------------------
Expanding train window, min 1500 days, refit every 63 trading days. At each refit date s:
fit on rets[:s] (train-only bins/ordering, BUG-LOG #15), freeze parameters, run the causal
HMM FILTER over the full history through the block end e, keep the filtered stress
probability for the 63 out-of-sample days [s, e) only. NOTE: MarkovSwitchingBorn.fit()
has no cross-refit warm-start API (init_cores is internal to one fit's EM loop), so
refits are cold and run in a process pool, mirroring analysis/run_msborn_books.py.

Models
------
  msborn              MarkovSwitchingBorn K=2 emission="born", production config
                      (d=8 chi=8 sticky=120 lag_sites=0 em_iters=6 n_init=2 seed=0).
  gauss_hmm           identical but emission="gaussian" (covariance-only ablation twin).
  vol_threshold       trailing 21d realized vol of the equal-weight book return, mapped
                      to a probability as its expanding-window percentile rank. No fit.
  logistic_classical  logistic regression on causal classical indicators (21d realized
                      vol, 252d avg pairwise correlation, PC1 variance share, week-6
                      causal turbulence), refit at each refit date on train S labels
                      PURGED of the last 21 train days (their S is not yet observable).

p_stress convention: state 0 is canonically the calmest, so p_stress = regime_{K-1}
(most-stressed state; identical to 1 - regime_0 for K=2 — BUG-LOG #18d).

Evaluation (out-of-sample days only, common valid-day mask across models per outcome)
-------------------------------------------------------------------------------------
Reliability table (10 probability buckets: n, mean predicted p, realized freq of S);
Brier with the Murphy decomposition Brier = reliability - resolution + uncertainty
(terms from the bucket table; raw Brier reported from per-day losses); log loss;
ROC AUC; PR-AUC. Same probabilities also scored against four secondary outcomes
(top-decile future 21d realized vol; any Y in next 21d; future 21d max drawdown
beyond its causal 90th pct; 1% one-day VaR exceedance count in next 21d beyond its
causal 90th pct). Block bootstrap (63d blocks, 2000 draws) CIs on Brier and log loss
per model and on msborn-minus-each-baseline paired differences.

Outputs (written incrementally so partial runs survive; HMM block results checkpoint
to calib_blocks_<book>_<model>.csv and reruns resume from finished blocks)
--------------------------------------------------------------------------
  week7/results/calib_daily_<book>.csv        date, p_<model>..., Y, S, secondary outcomes
  week7/results/calib_metrics.csv             book x model x outcome metric rows + CIs
  week7/results/calib_reliability_<book>.csv  bucket tables (model x outcome x bucket)
  week7/results/calib_meta_<book>.json        params, grid, per-block fit seconds

Run on the RunPod server, never the laptop.

Usage:
  python msborn_calibration.py [--books combined_gfc_2026 ...] [--smoke]
      [--out-dir week7/results] [--min-train 1500] [--refit-stride 63]
      [--nworkers 0] [--boot-draws 2000] [--seed 0]
"""
from __future__ import annotations

import os

# 4 BLAS threads per process (task spec). Block fits also parallelize across processes;
# default worker count divides cpu_count by 4 to avoid oversubscription.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import pathlib
import sys
import time
import traceback
import zlib
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mps_regime.data import to_returns  # noqa: E402
from mps_regime.msborn import MarkovSwitchingBorn  # noqa: E402

# ---------------------------------------------------------------------------- #
# constants                                                                     #
# ---------------------------------------------------------------------------- #
K_JOINT = 3          # joint-tail event: >= 3 assets in their 5% tail
Q_TAIL = 0.05
HORIZON = 21         # forward window (trading days) for the stress regime
PCTL = 0.90          # causal percentile threshold for S and the secondary outcomes
MIN_PRIOR = 500      # valid prior rolling-mean days required before S is scored
Y_MIN_HIST = 250     # days of history before the first Y tail thresholds exist
CORR_WIN = 252       # trailing window for correlation / PC1 / turbulence features
BLOCK_LEN = 63       # bootstrap block length (= refit stride)
CLIP = 1e-6          # log-loss probability clip

KNOWN_BOOKS = ["combined_gfc_2026", "primary_prices", "banks_crsp_prices",
               "combined_crossasset_2026_lagbtc", "futures_macro_prices",
               "semis_crsp_prices"]
DEFAULT_BOOKS = ["combined_gfc_2026", "primary_prices", "banks_crsp_prices"]

# Production MS-Born config (defaults of analysis/run_msborn_books.py).
PROD = dict(n_states=2, d=8, chi=8, lag_sites=0, sticky=120.0, em_iters=6, n_init=2)

HMM_MODELS = {"msborn": "born", "gauss_hmm": "gaussian"}
ALL_MODELS = ["msborn", "gauss_hmm", "vol_threshold", "logistic_classical"]
SMOKE_MODELS = ["msborn", "vol_threshold"]
OUTCOME_COLS = ["S", "fut_vol_top", "y_any21", "fut_dd_top", "var_exc_top"]


def resolve_book(name: str) -> pathlib.Path:
    """repo data/ -> $DATA_EXT -> <repo>/data_external -> laptop Documents backup."""
    dirs = [REPO / "data"]
    if os.environ.get("DATA_EXT"):
        dirs.append(pathlib.Path(os.environ["DATA_EXT"]))
    dirs += [REPO / "data_external"]
    for d in dirs:
        p = d / f"{name}.parquet"
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}.parquet not found in {[str(d) for d in dirs]}")


# ---------------------------------------------------------------------------- #
# targets (all causal)                                                          #
# ---------------------------------------------------------------------------- #
def causal_Y(rets: pd.DataFrame, min_train: int, stride: int) -> pd.Series:
    """Joint-tail event series with walk-forward tail thresholds.

    Threshold refresh points = model refit grid (min_train + k*stride) plus a burn-in
    grid at the same stride from Y_MIN_HIST, so every labeled day uses per-asset 5%
    quantiles from data STRICTLY before its segment. NaN before Y_MIN_HIST.
    """
    T = len(rets)
    R = rets.to_numpy()
    pts = sorted(set(list(range(Y_MIN_HIST, min_train, stride))
                     + list(range(min_train, T, stride))))
    Y = np.full(T, np.nan)
    for i, a in enumerate(pts):
        b = pts[i + 1] if i + 1 < len(pts) else T
        thr = np.quantile(R[:a], Q_TAIL, axis=0)          # train-only, frozen
        Y[a:b] = ((R[a:b] < thr[None, :]).sum(axis=1) >= K_JOINT).astype(float)
    return pd.Series(Y, index=rets.index, name="Y")


def _exceeds_causal_q(fwd: pd.Series, q: float = PCTL,
                      min_prior: int = MIN_PRIOR) -> pd.Series:
    """1{forward-window stat > expanding q-quantile of its own PAST realized values}.

    fwd_t summarizes days t+1..t+HORIZON, so fwd_{t-HORIZON} is fully observable at t;
    the reference distribution is fwd shifted by HORIZON, expanding — strictly causal.
    """
    ref = fwd.shift(HORIZON)
    thr = ref.expanding(min_periods=min_prior).quantile(q)
    out = (fwd > thr).astype(float)
    out[fwd.isna() | thr.isna()] = np.nan
    return out


def _forward_max_drawdown(ew: pd.Series) -> pd.Series:
    """Max drawdown of the cumulative EW log-return path over days t+1..t+HORIZON."""
    v = ew.to_numpy()
    T = len(v)
    out = np.full(T, np.nan)
    for t in range(T - HORIZON):
        path = np.concatenate([[0.0], np.cumsum(v[t + 1:t + 1 + HORIZON])])
        out[t] = float(np.max(np.maximum.accumulate(path) - path))
    return pd.Series(out, index=ew.index)


def build_outcomes(rets: pd.DataFrame, Y: pd.Series) -> pd.DataFrame:
    """S plus the four secondary outcome columns (NaN where not yet defined/observable)."""
    ew = rets.mean(axis=1)
    T = len(rets)

    # primary stress regime S (spec-exact: threshold = trailing rolling-mean percentile)
    m = Y.rolling(HORIZON, min_periods=HORIZON).mean()          # mean over t-20..t
    fwd_freq = m.shift(-HORIZON)                                # mean over t+1..t+21
    thr_s = m.expanding(min_periods=MIN_PRIOR).quantile(PCTL)   # data up to t only
    S = (fwd_freq > thr_s).astype(float)
    S[fwd_freq.isna() | thr_s.isna()] = np.nan

    # secondary 1: top-decile future 21d realized vol (expanding causal decile)
    fwd_vol = ew.rolling(HORIZON, min_periods=HORIZON).std().shift(-HORIZON)
    fut_vol_top = _exceeds_causal_q(fwd_vol)

    # secondary 2: at least one joint-tail event in the next 21 days
    fwd_cnt = Y.rolling(HORIZON, min_periods=HORIZON).sum().shift(-HORIZON)
    y_any21 = (fwd_cnt >= 1).astype(float)
    y_any21[fwd_cnt.isna()] = np.nan

    # secondary 3: future 21d max drawdown beyond its causal 90th percentile
    fut_dd_top = _exceeds_causal_q(_forward_max_drawdown(ew))

    # secondary 4: count of 1% one-day VaR exceedances in the next 21 days beyond its
    # causal 90th percentile. VaR_t = expanding 1% quantile of the EW return up to t,
    # frozen at t and applied to days t+1..t+21.
    var1 = ew.expanding(min_periods=Y_MIN_HIST).quantile(0.01)
    ev = ew.to_numpy()
    vv = var1.to_numpy()
    cnt = np.full(T, np.nan)
    for t in range(T - HORIZON):
        if not np.isnan(vv[t]):
            cnt[t] = float((ev[t + 1:t + 1 + HORIZON] < vv[t]).sum())
    var_exc_top = _exceeds_causal_q(pd.Series(cnt, index=rets.index))

    return pd.DataFrame({"S": S, "fut_vol_top": fut_vol_top, "y_any21": y_any21,
                         "fut_dd_top": fut_dd_top, "var_exc_top": var_exc_top})


# ---------------------------------------------------------------------------- #
# baseline probabilities                                                        #
# ---------------------------------------------------------------------------- #
def vol_threshold_prob(rets: pd.DataFrame) -> pd.Series:
    """Trailing 21d EW realized vol -> expanding percentile rank. Causal, no fitting."""
    rv = rets.mean(axis=1).rolling(HORIZON, min_periods=HORIZON).std().to_numpy()
    out = np.full(len(rv), np.nan)
    for t in range(len(rv)):
        if np.isnan(rv[t]):
            continue
        past = rv[:t + 1]
        past = past[~np.isnan(past)]
        out[t] = float((past <= rv[t]).mean())
    return pd.Series(out, index=rets.index, name="p_vol_threshold")


def classical_features(rets: pd.DataFrame) -> pd.DataFrame:
    """Causal classical indicators: 21d realized vol, 252d avg pairwise correlation,
    PC1 variance share (252d correlation eigenvalues), and week-6 causal turbulence
    (reference mean/cov from the 252 days PRECEDING the 21 scored days; scored days
    excluded from their own reference — BUG-LOG #18 review item i)."""
    R = rets.to_numpy()
    T, N = R.shape
    ew = rets.mean(axis=1)
    rv21 = ew.rolling(HORIZON, min_periods=HORIZON).std().to_numpy()
    avg_corr = np.full(T, np.nan)
    pc1 = np.full(T, np.nan)
    turb = np.full(T, np.nan)
    iu = np.triu_indices(N, 1)
    for t in range(CORR_WIN - 1, T):
        C = np.corrcoef(R[t - CORR_WIN + 1:t + 1].T)
        avg_corr[t] = float(C[iu].mean())
        evals = np.linalg.eigvalsh(C)
        pc1[t] = float(evals[-1] / evals.sum())
    for t in range(CORR_WIN + HORIZON - 1, T):
        w = R[t - CORR_WIN - HORIZON + 1:t + 1]
        ref, scored = w[:-HORIZON], w[-HORIZON:]
        C = np.cov(ref.T)
        mu = ref.mean(axis=0)
        try:
            P = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            continue
        d = scored - mu
        turb[t] = float(np.einsum("ti,ij,tj->t", d, P, d).mean())
    return pd.DataFrame({"rv21": rv21, "avg_corr": avg_corr, "pc1_share": pc1,
                         "turbulence": turb}, index=rets.index)


def logistic_probs(feats: pd.DataFrame, S: pd.Series, grid: list) -> pd.Series:
    """Walk-forward logistic regression on the classical features.

    PURGE: S_t needs Y through t+21, so at refit date s only labels with
    t <= s - HORIZON - 1 are observable — the last 21 train days are excluded.
    Features standardized on train only. Degenerate train labels -> constant base rate.
    """
    from sklearn.linear_model import LogisticRegression

    F = feats.to_numpy()
    y = S.to_numpy()
    feat_ok = ~np.isnan(F).any(axis=1)
    out = np.full(len(F), np.nan)
    for _b, s, e in grid:
        tr = np.zeros(len(F), dtype=bool)
        tr[:max(s - HORIZON, 0)] = True                 # t <= s - HORIZON - 1
        tr &= feat_ok & ~np.isnan(y)
        blk = np.zeros(len(F), dtype=bool)
        blk[s:e] = True
        blk &= feat_ok
        if tr.sum() < 100 or not blk.any():
            continue
        ytr = y[tr]
        if len(np.unique(ytr)) < 2:
            out[blk] = float(np.clip(ytr.mean(), 1e-3, 1 - 1e-3))
            continue
        mu = F[tr].mean(axis=0)
        sd = F[tr].std(axis=0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit((F[tr] - mu) / sd, ytr.astype(int))
        out[blk] = clf.predict_proba((F[blk] - mu) / sd)[:, 1]
    return pd.Series(out, index=feats.index, name="p_logistic_classical")


# ---------------------------------------------------------------------------- #
# HMM walk-forward (process pool, checkpointed per block)                       #
# ---------------------------------------------------------------------------- #
_G: dict = {}


def _init_worker(book_path: str, seed: int) -> None:
    _G["rets"] = to_returns(pd.read_parquet(book_path).astype(float), kind="log")
    _G["seed"] = seed


def _fit_block(task: tuple) -> dict:
    """Fit one (model, block): train on [0, s), causal-filter [0, e), keep [s, e)."""
    model_name, emission, b, s, e = task
    rets: pd.DataFrame = _G["rets"]
    try:
        t0 = time.time()
        model = MarkovSwitchingBorn(emission=emission, seed=_G["seed"], **PROD)
        model.fit(rets.iloc[:s])
        probs_full = model.filter(rets.iloc[:e])          # forward-only, full history
        block_dates = rets.index[s:e]
        probs = probs_full.loc[probs_full.index.intersection(block_dates)]
        # p_stress: most-stressed state (canonical order; = 1 - regime_0 for K=2).
        p = probs[f"regime_{model.K - 1}"]
        return dict(ok=True, model=model_name, block=b,
                    dates=[str(d.date()) for d in p.index],
                    p=[float(x) for x in p.to_numpy()],
                    train_loglik=float(model.train_loglik_),
                    fit_seconds=round(time.time() - t0, 1))
    except Exception:
        return dict(ok=False, model=model_name, block=b, error=traceback.format_exc())


def hmm_probs(book: str, book_path: pathlib.Path, rets: pd.DataFrame, grid: list,
              model_names: list[str], out_dir: pathlib.Path, nworkers: int,
              seed: int, shard: tuple | None = None) -> dict[str, pd.Series] | None:
    """Run the walk-forward refits for the HMM models, checkpointing each finished
    block to calib_blocks_<book>_<model>.csv so interrupted runs resume.
    shard=(r, n): compute only blocks with b % n == r and return None (no assembly);
    merge the checkpoint CSVs from all shards, then rerun without --shard to assemble."""
    part_path = {m: out_dir / f"calib_blocks_{book}_{m}.csv" for m in model_names}
    done: dict[str, set] = {}
    for m in model_names:
        done[m] = (set(pd.read_csv(part_path[m])["block"].unique())
                   if part_path[m].exists() else set())
    tasks = [(m, HMM_MODELS[m], b, s, e)
             for m in model_names for (b, s, e) in grid if b not in done[m]]
    if shard is not None:
        r, n = shard
        tasks = [t for t in tasks if t[2] % n == r]
    if tasks:
        nw = nworkers or max(1, min(len(tasks), cpu_count() // 4))
        print(f"[{book}] {len(tasks)} HMM block fits to run "
              f"({len(grid)} blocks x {len(model_names)} models, "
              f"{sum(len(d) for d in done.values())} already done), workers={nw}",
              flush=True)
        t0 = time.time()
        ndone = 0
        with Pool(nw, initializer=_init_worker,
                  initargs=(str(book_path), seed)) as pool:
            for r in pool.imap_unordered(_fit_block, tasks):
                ndone += 1
                if not r["ok"]:
                    print(f"[{book}/{r['model']}] block {r['block']} FAILED:\n"
                          f"{r['error']}", file=sys.stderr, flush=True)
                    continue
                df = pd.DataFrame(dict(block=r["block"], date=r["dates"], p=r["p"],
                                       train_loglik=r["train_loglik"],
                                       fit_seconds=r["fit_seconds"]))
                pp = part_path[r["model"]]
                df.to_csv(pp, mode="a", header=not pp.exists(), index=False)
                print(f"[{book}/{r['model']}] block {r['block']}/{len(grid) - 1} "
                      f"done in {r['fit_seconds']}s "
                      f"({ndone}/{len(tasks)}, {time.time() - t0:.0f}s elapsed)",
                      flush=True)

    if shard is not None:
        print(f"[{book}] shard {shard[0]}/{shard[1]} complete ({len(tasks)} fits) — "
              "merge checkpoint CSVs and rerun without --shard to assemble", flush=True)
        return None

    out: dict[str, pd.Series] = {}
    expected = len(rets) - grid[0][1]                     # every day after min_train
    for m in model_names:
        if not part_path[m].exists():
            raise RuntimeError(f"[{book}/{m}] no block results — all fits failed?")
        df = pd.read_csv(part_path[m], parse_dates=["date"])
        df = df.drop_duplicates(subset="date").sort_values("date")
        if len(df) != expected:
            raise RuntimeError(
                f"[{book}/{m}] stitched series has {len(df)} rows, expected {expected}"
                " — failed blocks above; rerun to resume from the checkpoints")
        out[m] = df.set_index("date")["p"].rename(f"p_{m}")
    return out


# ---------------------------------------------------------------------------- #
# evaluation                                                                    #
# ---------------------------------------------------------------------------- #
def bucket_table(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Reliability table: 10 equal-width probability buckets."""
    idx = np.clip((p * 10).astype(int), 0, 9)
    rows = []
    for b in range(10):
        m = idx == b
        rows.append(dict(bucket=f"{b / 10:.1f}-{(b + 1) / 10:.1f}", n=int(m.sum()),
                         mean_p=float(p[m].mean()) if m.any() else np.nan,
                         freq=float(y[m].mean()) if m.any() else np.nan))
    return pd.DataFrame(rows)


def murphy_terms(tab: pd.DataFrame, y: np.ndarray) -> tuple[float, float, float]:
    """Murphy decomposition from the bucket table: Brier = REL - RES + UNC."""
    ybar = float(y.mean())
    n = float(len(y))
    t = tab[tab["n"] > 0]
    rel = float((t["n"] * (t["mean_p"] - t["freq"]) ** 2).sum() / n)
    res = float((t["n"] * (t["freq"] - ybar) ** 2).sum() / n)
    unc = ybar * (1.0 - ybar)
    return rel, res, unc


def boot_ci(arr: np.ndarray, rng: np.random.Generator, draws: int,
            L: int = BLOCK_LEN) -> tuple[float, float]:
    """Moving-block bootstrap 95% CI on the mean of a per-day loss series."""
    n = arr.size
    if n <= L:
        return np.nan, np.nan
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(draws, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(draws, nb * L)[:, :n]
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(daily: pd.DataFrame, models: list[str], book: str, draws: int,
             seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics + reliability rows for every (model, outcome), on the common valid mask."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    met_rows, rel_rows = [], []
    for outcome in OUTCOME_COLS:
        mask = daily[outcome].notna()
        for m in models:
            mask &= daily[f"p_{m}"].notna()
        sub = daily[mask]
        y = sub[outcome].to_numpy(dtype=float)
        if len(y) < 100 or y.min() == y.max():
            print(f"[{book}] outcome {outcome}: skipped (n={len(y)}, degenerate)",
                  flush=True)
            continue
        rng = np.random.default_rng(seed + zlib.crc32(f"{book}|{outcome}".encode()) % 10 ** 6)
        bl, ll = {}, {}
        for m in models:
            p = np.clip(sub[f"p_{m}"].to_numpy(dtype=float), 0.0, 1.0)
            pc = np.clip(p, CLIP, 1 - CLIP)
            bl[m] = (p - y) ** 2
            ll[m] = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
            tab = bucket_table(p, y)
            rel, res, unc = murphy_terms(tab, y)
            b_lo, b_hi = boot_ci(bl[m], rng, draws)
            l_lo, l_hi = boot_ci(ll[m], rng, draws)
            met_rows.append(dict(
                book=book, model=m, outcome=outcome, kind="prob", n=len(y),
                base_rate=float(y.mean()),
                brier=float(bl[m].mean()), brier_lo=b_lo, brier_hi=b_hi,
                reliability=rel, resolution=res, uncertainty=unc,
                logloss=float(ll[m].mean()), logloss_lo=l_lo, logloss_hi=l_hi,
                auc=float(roc_auc_score(y, p)),
                pr_auc=float(average_precision_score(y, p))))
            tab.insert(0, "outcome", outcome)
            tab.insert(0, "model", m)
            tab.insert(0, "book", book)
            rel_rows.append(tab)
        # paired msborn-minus-baseline differences (same days, differenced per day)
        if "msborn" in models:
            for m in models:
                if m == "msborn":
                    continue
                db, dl = bl["msborn"] - bl[m], ll["msborn"] - ll[m]
                b_lo, b_hi = boot_ci(db, rng, draws)
                l_lo, l_hi = boot_ci(dl, rng, draws)
                met_rows.append(dict(
                    book=book, model=f"msborn_minus_{m}", outcome=outcome,
                    kind="diff", n=len(y), base_rate=float(y.mean()),
                    brier=float(db.mean()), brier_lo=b_lo, brier_hi=b_hi,
                    reliability=np.nan, resolution=np.nan, uncertainty=np.nan,
                    logloss=float(dl.mean()), logloss_lo=l_lo, logloss_hi=l_hi,
                    auc=np.nan, pr_auc=np.nan))
    return pd.DataFrame(met_rows), pd.concat(rel_rows, ignore_index=True)


def merge_metrics(out_dir: pathlib.Path, book: str, met: pd.DataFrame) -> None:
    """Incremental calib_metrics.csv update: replace this book's rows, keep the rest."""
    path = out_dir / "calib_metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        met = pd.concat([old[old["book"] != book], met], ignore_index=True)
    met.to_csv(path, index=False)


# ---------------------------------------------------------------------------- #
# driver                                                                        #
# ---------------------------------------------------------------------------- #
def run_book(book: str, models: list[str], args) -> None:
    t0 = time.time()
    book_path = resolve_book(book)
    rets = to_returns(pd.read_parquet(book_path).astype(float), kind="log")
    T = len(rets)
    if T <= args.min_train + HORIZON:
        print(f"[{book}] T={T} too short for min-train {args.min_train} — skipped",
              flush=True)
        return
    starts = list(range(args.min_train, T, args.refit_stride))
    grid = [(b, s, min(s + args.refit_stride, T)) for b, s in enumerate(starts)]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{book}] T={T} N={rets.shape[1]} span={rets.index[0].date()}.."
          f"{rets.index[-1].date()} refits={len(grid)} models={models}", flush=True)

    Y = causal_Y(rets, args.min_train, args.refit_stride)
    outcomes = build_outcomes(rets, Y)

    probs: dict[str, pd.Series] = {}
    hmm = [m for m in models if m in HMM_MODELS]
    if hmm:
        hp = hmm_probs(book, book_path, rets, grid, hmm, out_dir,
                       args.nworkers, args.seed, shard=args.shard_rn)
        if hp is None:
            return                                        # shard mode: fits only
        probs.update(hp)
    if "vol_threshold" in models:
        probs["vol_threshold"] = vol_threshold_prob(rets)
    if "logistic_classical" in models:
        probs["logistic_classical"] = logistic_probs(
            classical_features(rets), outcomes["S"], grid)

    # daily panel: out-of-sample days only
    oos = rets.index[args.min_train:]
    daily = pd.DataFrame(index=oos)
    daily.index.name = "date"
    for m in models:
        daily[f"p_{m}"] = probs[m].reindex(oos)
    daily["Y"] = Y.reindex(oos)
    for c in OUTCOME_COLS:
        daily[c] = outcomes[c].reindex(oos)
    daily_path = out_dir / f"calib_daily_{book}.csv"
    daily.to_csv(daily_path)
    print(f"[{book}] wrote {daily_path} shape={daily.shape}", flush=True)

    met, rel = evaluate(daily, models, book, args.boot_draws, args.seed)
    rel.to_csv(out_dir / f"calib_reliability_{book}.csv", index=False)
    merge_metrics(out_dir, book, met)
    with open(out_dir / f"calib_meta_{book}.json", "w") as fh:
        json.dump(dict(book=str(book_path), models=models, prod_config=PROD,
                       min_train=args.min_train, refit_stride=args.refit_stride,
                       n_refits=len(grid), n_days=T, n_assets=int(rets.shape[1]),
                       n_oos=len(daily), boot_draws=args.boot_draws, seed=args.seed,
                       target=dict(k=K_JOINT, q=Q_TAIL, horizon=HORIZON, pctl=PCTL,
                                   min_prior=MIN_PRIOR, y_min_hist=Y_MIN_HIST),
                       elapsed_seconds=round(time.time() - t0, 1)), fh, indent=2)
    print(f"[{book}] done in {time.time() - t0:.0f}s — metrics for S:", flush=True)
    show = met[(met["outcome"] == "S")][["model", "kind", "brier", "brier_lo",
                                         "brier_hi", "reliability", "resolution",
                                         "logloss", "auc", "pr_auc"]]
    print(show.to_string(index=False), flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--books", nargs="+", choices=KNOWN_BOOKS, default=DEFAULT_BOOKS)
    ap.add_argument("--smoke", action="store_true",
                    help="one book, refit stride 252, msborn + vol_threshold only, "
                         "200 bootstrap draws")
    ap.add_argument("--out-dir", default=str(REPO / "week7" / "results"))
    ap.add_argument("--min-train", type=int, default=1500)
    ap.add_argument("--refit-stride", type=int, default=BLOCK_LEN)
    ap.add_argument("--nworkers", type=int, default=0,
                    help="0 = cpu_count // 4 (4 BLAS threads per worker)")
    ap.add_argument("--boot-draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default=None, metavar="R,N",
                    help="compute only blocks with block %% N == R (fits only, no "
                         "assembly); merge checkpoints and rerun without --shard")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.shard_rn = None
    if args.shard:
        r, n = (int(x) for x in args.shard.split(","))
        assert 0 <= r < n, f"bad --shard {args.shard}"
        args.shard_rn = (r, n)
    models = list(ALL_MODELS)
    books = list(args.books)
    if args.smoke:
        books = books[:1]
        models = list(SMOKE_MODELS)
        args.refit_stride = 252
        args.boot_draws = 200
        print(f"SMOKE: book={books[0]} stride=252 models={models}", flush=True)
    t0 = time.time()
    for book in books:
        run_book(book, models, args)
    print(f"all books done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
