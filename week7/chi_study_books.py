"""Week 7 item 3: bond-dimension study across all six books, chronological folds x seeds.

Reviewer question (measured, not assumed): do highly correlated sector books need LESS
tensor capacity than heterogeneous cross-asset books, or MORE?

Protocol per book: log returns; 6 equal contiguous segments; fold i (i=1..5) trains on
segments 1..i, tests on segment i+1 (expanding chronological folds — the week-6
binmismatch_impact.py predecessor used a random split, corrected here). Per fold, the
Fiedler ordering, the d=12 quantile bin grid, and the independent per-site baseline all
come from the TRAIN slice only (BUG-LOG #15), then are frozen and applied to the test
slice. Metric per (book, fold, chi, seed): held-out dependence gain = independent
per-site NLL (train frequencies, +1 smoothing) minus MPS NLL, both on the test codes —
the exact "fixed"-mode metric of week6/review_check/binmismatch_impact.py.

Fitting runs on the RunPod server. `--plot` renders from the CSV with no torch and no
fitting, so it runs on the laptop. Long CSV rows are appended + flushed per fit and
already-done (book, fold, chi, seed) keys are skipped on restart, so partial runs survive.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"; os.environ["MKL_NUM_THREADS"] = "4"
import argparse
import csv
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Books table from week7/README.md: name -> parquet file name.
BOOKS = {
    "primary24":     "primary_prices.parquet",
    "gfc":           "combined_gfc_2026.parquet",
    "banks":         "banks_crsp_prices.parquet",
    "crossasset":    "combined_crossasset_2026_lagbtc.parquet",
    "futures_macro": "futures_macro_prices.parquet",
    "semis":         "semis_crsp_prices.parquet",
}

CHI_GRID = [1, 2, 4, 8, 12, 16, 24]
SEEDS = [0, 1, 2]
D_PHYS = 12          # physical dim (bins/asset), matches the predecessor's selection run
SWEEPS = 10          # DMRG sweeps, matches the predecessor
ORDER_BINS = 6       # coarse bins for the MI/Fiedler ordering, matches the predecessor
N_SEG = 6            # contiguous segments -> 5 expanding folds

LONG_COLS = ["book", "fold", "chi", "seed", "n_assets", "n_train", "n_test",
             "gain", "ind_nll", "test_nll", "train_nll", "seconds", "n_params"]


def data_dirs():
    """Candidate directories for book parquets, in priority order."""
    dirs = [ROOT / "data"]
    ext = os.environ.get("DATA_EXT")
    if ext:
        dirs.append(pathlib.Path(ext))
    dirs.append(ROOT / "data_external")
    return dirs


def resolve_book(fname):
    for d in data_dirs():
        p = d / fname
        if p.exists():
            return p
    return None


def independent_nll(codes_tr, codes_te, d):
    """Per-site independent baseline: train frequencies with +1 smoothing, scored on test."""
    ind = 0.0
    for k in range(codes_tr.shape[1]):
        p = np.bincount(codes_tr[:, k], minlength=d) + 1.0
        p = p / p.sum()
        ind += -np.log(p[codes_te[:, k]]).mean()
    return float(ind)


def run_study(args):
    import torch
    torch.set_num_threads(4)
    from mps_regime import binning, dmrg, mps
    from mps_regime import data as D

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "chi_study_long.csv"

    # resume: skip (book, fold, chi, seed) rows already in the long CSV
    done = set()
    if long_path.exists():
        prev = pd.read_csv(long_path)
        done = {(r.book, int(r.fold), int(r.chi), int(r.seed)) for r in prev.itertuples()}
        print(f"resuming: {len(done)} rows already in {long_path}", flush=True)

    books = list(args.books.split(",")) if args.books else list(BOOKS)
    chi_grid, seeds, folds = CHI_GRID, SEEDS, list(range(1, N_SEG))
    if args.smoke:
        books, chi_grid, seeds, folds = books[:1], [2, 8], [0], [4, 5]
        print("SMOKE run: " + ", ".join(books), flush=True)

    fh = open(long_path, "a", newline="")
    writer = csv.writer(fh)
    if fh.tell() == 0:
        writer.writerow(LONG_COLS); fh.flush()

    for book in books:
        fname = BOOKS.get(book)
        if fname is None:
            print(f"WARNING: unknown book '{book}' — skipping", flush=True); continue
        path = resolve_book(fname)
        if path is None:
            print(f"WARNING: {fname} not found in {[str(d) for d in data_dirs()]} — skipping {book}", flush=True)
            continue
        rets_full = D.to_returns(pd.read_parquet(path).astype(float), kind="log")
        T, N = len(rets_full), rets_full.shape[1]
        bounds = [round(i * T / N_SEG) for i in range(N_SEG + 1)]
        print(f"[{book}] N={N} T={T} segments={bounds}", flush=True)

        for fold in folds:
            train = rets_full.iloc[bounds[0]:bounds[fold]]
            test = rets_full.iloc[bounds[fold]:bounds[fold + 1]]
            # train-only Fiedler ordering + train-only d=12 bin grid (fixed-mode pattern)
            _, _, order = binning.discretize(train, n_bins=ORDER_BINS)
            tr, te = train[order], test[order]
            codes_tr, edges_tr, _, _ = mps._discretize(tr, D_PHYS)
            arr_te = te.to_numpy()
            codes_te = np.column_stack([np.digitize(arr_te[:, k], edges_tr[k]) for k in range(N)])
            ind = independent_nll(codes_tr, codes_te, D_PHYS)
            print(f"[{book}] fold {fold}: train={len(tr)} test={len(te)} ind_nll={ind:.4f}", flush=True)

            for chi in chi_grid:
                for seed in seeds:
                    if (book, fold, chi, seed) in done:
                        print(f"[{book}] fold {fold} chi={chi:2d} seed={seed} already done — skip", flush=True)
                        continue
                    t0 = time.perf_counter()
                    fit = dmrg.fit_dmrg(tr, d=D_PHYS, chi=chi, sweeps=SWEEPS, seed=seed)
                    secs = time.perf_counter() - t0
                    tc = [torch.tensor(c) for c in fit.cores]
                    test_nll = float(mps._nll(tc, torch.tensor(codes_te, dtype=torch.long)))
                    train_nll = float(fit.nll_history[-1])   # exact transfer-matrix NLL on train
                    n_params = int(sum(c.size for c in fit.cores))  # packaged cores (bonds padded to chi)
                    gain = ind - test_nll
                    writer.writerow([book, fold, chi, seed, N, len(tr), len(te),
                                     f"{gain:.6f}", f"{ind:.6f}", f"{test_nll:.6f}",
                                     f"{train_nll:.6f}", f"{secs:.2f}", n_params])
                    fh.flush()
                    print(f"[{book}] fold {fold} chi={chi:2d} seed={seed} "
                          f"gain={gain:+.4f} train_nll={train_nll:.4f} {secs:.1f}s", flush=True)
    fh.close()
    write_summary(out_dir)


def write_summary(out_dir):
    long_path = out_dir / "chi_study_long.csv"
    df = pd.read_csv(long_path)
    g = df.groupby(["book", "chi"])
    summ = g.agg(gain_mean=("gain", "mean"), gain_std=("gain", "std"),
                 n_runs=("gain", "size"), seconds_mean=("seconds", "mean"),
                 n_params=("n_params", "first")).reset_index()
    best = summ.loc[summ.groupby("book")["gain_mean"].idxmax(), ["book", "chi"]]
    summ = summ.merge(best.rename(columns={"chi": "best_chi"}), on="book")
    out = out_dir / "chi_study_summary.csv"
    summ.to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out}", flush=True)
    for book, chi in best.itertuples(index=False):
        print(f"  {book}: argmax chi = {chi}", flush=True)


def make_plot(args):
    # laptop path: reads the CSV only — no torch, no mps_regime, no fitting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    long_path = pathlib.Path(args.out_dir) / "chi_study_long.csv"
    df = pd.read_csv(long_path)
    fig_dir = ROOT / "week7" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    # fixed color per book (assigned by BOOKS order, never cycled by presence)
    palette = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
    colors = {b: palette[i % len(palette)] for i, b in enumerate(BOOKS)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for book in BOOKS:
        sub = df[df["book"] == book]
        if sub.empty:
            continue
        g = sub.groupby("chi")["gain"]
        mu, sd = g.mean(), g.std().fillna(0.0)
        ax.plot(mu.index, mu.values, marker="o", markersize=4, linewidth=2,
                color=colors[book], label=book)
        ax.fill_between(mu.index, (mu - sd).values, (mu + sd).values,
                        color=colors[book], alpha=0.15, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(CHI_GRID)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("bond dimension chi (log2 scale)")
    ax.set_ylabel("held-out dependence gain (nats/day)")
    ax.set_title("Bond-dimension study: held-out dependence gain by book\n"
                 "(mean +- 1 std across chronological folds x seeds)")
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = fig_dir / "chi_study.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Bond-dimension study across books (week 7 item 3)")
    ap.add_argument("--books", default=None, help=f"comma list from {list(BOOKS)} (default all)")
    ap.add_argument("--smoke", action="store_true", help="one book, folds 4-5, chi 2/8, seed 0")
    ap.add_argument("--plot", action="store_true", help="render figure from the long CSV (no fitting)")
    ap.add_argument("--out-dir", default=str(ROOT / "week7" / "results"))
    args = ap.parse_args()
    if args.plot:
        make_plot(args)
    else:
        run_study(args)


if __name__ == "__main__":
    main()
