"""Reviewer item 2 — the parity counterexample as a MODEL demonstration.

The identifiability tests (tests/test_block_mi.py) use the reviewer's XOR/parity
construction at the distribution level. Here the same construction becomes a model
bake-off: two 3-asset worlds, EVEN parity (x1*x2*x3 = +1, four configurations
equally likely) and ODD parity (product = -1), with IDENTICAL marginals, IDENTICAL
(zero) pairwise correlations, and identical two-asset joints — the two worlds differ
ONLY in three-way structure. Any model whose likelihood factorizes over <=2-body
statistics (Gaussian, independent-bins, MPS chi=1 product state) is mathematically
blind to the difference; the parity state is exactly a chi=2 MPS (Hadamard-rotated
GHZ), so MPS chi=2 can and does separate the worlds. Target table:

    | Model                  | Detects pairwise difference? | Detects three-way difference? |
    | correlation / Gaussian | No                           | No                            |
    | independent            | No                           | No                            |
    | MPS chi=1              | No                           | No                            |
    | MPS chi=2              | No (pairwise)                | Yes                           |

Protocol: fit every model on one world's train set (bins/edges/moments train-only,
BUG-LOG #15), score mean held-out log-likelihood per observation on BOTH test sets;
"detects the three-way difference" = train-world test LL exceeds other-world test LL
by > 3 combined standard errors, in BOTH train directions (even->odd and odd->even).
The "pairwise" column is a DATA check, not a model check: chi-squared homogeneity
tests confirm the even/odd pairwise joint bin tables are indistinguishable, so there
is no pairwise difference for any model to detect.

Outputs: results/parity_demo.csv, results/parity_demo_ptable.csv,
figures/parity_demo.png. Tiny job (seconds) but runs on the RunPod server like
everything else. Usage: python week7/parity_model_demo.py [--smoke] [--out-dir D]
"""
import argparse
import itertools
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2_contingency

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mps_regime.dmrg import fit_dmrg
from mps_regime.mps import _discretize

ASSETS = ["A1", "A2", "A3"]
MODELS = ["gaussian", "independent", "mps_chi1", "mps_chi2"]
DISPLAY = {"gaussian": "correlation / Gaussian", "independent": "independent",
           "mps_chi1": "MPS chi=1", "mps_chi2": "MPS chi=2"}
GRID8 = np.array(list(itertools.product([0, 1], repeat=3)), dtype=np.int64)  # bin configs


# --------------------------------------------------------------------------- #
# Synthetic worlds                                                              #
# --------------------------------------------------------------------------- #
def world_returns(parity, n, seed, move=0.01, noise=0.001):
    """3 assets, each day one of the four sign configurations with x1*x2*x3 = +1
    (even) or -1 (odd), equally likely; returns = +-1% + N(0, 0.1%) jitter so the
    standard d=2 quantile binning (median split) recovers the signs exactly."""
    rng = np.random.default_rng(seed)
    signs = np.array([s for s in itertools.product([-1, 1], repeat=3)
                      if np.prod(s) == (+1 if parity == "even" else -1)])   # (4, 3)
    r = move * signs[rng.integers(0, 4, size=n)] + noise * rng.standard_normal((n, 3))
    return pd.DataFrame(r, columns=ASSETS)


def config_label(cfg):
    return "".join("+" if b == 1 else "-" for b in cfg)


def true_probs(parity):
    """Ground-truth P over the 8 bin configs (bin 1 = up = +1)."""
    p = np.zeros(len(GRID8))
    for i, cfg in enumerate(GRID8):
        prod = np.prod(np.where(cfg == 1, 1, -1))
        p[i] = 0.25 if prod == (+1 if parity == "even" else -1) else 0.0
    return p


# --------------------------------------------------------------------------- #
# Models (all statistics train-only; test encoded with train edges)             #
# --------------------------------------------------------------------------- #
def fit_all(train_df, sweeps, seed=0):
    """Fit the four models on one training world. Returns dict of scorers +
    the shared train-only bin edges (for the independent model / data checks)."""
    arr = train_df.to_numpy()
    mu = arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False, ddof=0)

    codes, edges, _, _ = _discretize(train_df, 2)           # train-only median edges
    # per-asset bin frequencies, 0.5 pseudo-count guard against empty bins
    freq = np.zeros((3, 2))
    for j in range(3):
        cnt = np.bincount(codes[:, j], minlength=2).astype(float) + 0.5
        freq[j] = cnt / cnt.sum()

    torch.manual_seed(seed)
    mps1 = fit_dmrg(train_df, d=2, chi=1, sweeps=sweeps, seed=seed)
    torch.manual_seed(seed)
    mps2 = fit_dmrg(train_df, d=2, chi=2, sweeps=sweeps, seed=seed)

    def gauss_ll(test_df):
        x = test_df.to_numpy() - mu
        L = np.linalg.cholesky(cov)
        z = np.linalg.solve(L, x.T)
        maha = (z ** 2).sum(axis=0)
        logdet = 2.0 * np.log(np.diag(L)).sum()
        return -0.5 * (3 * np.log(2 * np.pi) + logdet + maha)   # continuous log-DENSITY

    def encode(test_df):
        t = test_df[ASSETS].to_numpy()
        return np.column_stack([np.digitize(t[:, j], edges[j]) for j in range(3)])

    def indep_ll(test_df):
        c = encode(test_df)
        return np.log(freq[np.arange(3)[None, :], c]).sum(axis=1)

    scorers = {
        "gaussian": gauss_ll,
        "independent": indep_ll,
        "mps_chi1": lambda t: mps1.log_prob(mps1.encode(t)),
        "mps_chi2": lambda t: mps2.log_prob(mps2.encode(t)),
    }

    def indep_ptable():
        return np.array([np.prod(freq[np.arange(3), cfg]) for cfg in GRID8])

    def mps_ptable(fit):
        p = np.exp(fit.log_prob(GRID8))
        return p / p.sum()

    def gauss_ptable(n_mc, seed_mc):
        rng = np.random.default_rng(seed_mc)
        s = rng.multivariate_normal(mu, cov, size=n_mc)
        c = np.column_stack([np.digitize(s[:, j], edges[j]) for j in range(3)])
        idx = c @ np.array([4, 2, 1])
        return np.bincount(idx, minlength=8) / n_mc

    ptables = {"gaussian_mc": gauss_ptable, "independent": indep_ptable,
               "mps_chi1": lambda: mps_ptable(mps1), "mps_chi2": lambda: mps_ptable(mps2)}
    return scorers, ptables, edges, encode


# --------------------------------------------------------------------------- #
# Data check: pairwise joints of even vs odd test are indistinguishable         #
# --------------------------------------------------------------------------- #
def pairwise_check(codes_even, codes_odd):
    """Chi-squared homogeneity per asset pair on the 2x2 joint bin cells,
    even-test vs odd-test. High p-values everywhere = no pairwise difference."""
    out = []
    for i, j in itertools.combinations(range(3), 2):
        rows = []
        for c in (codes_even, codes_odd):
            cell = c[:, i] * 2 + c[:, j]
            rows.append(np.bincount(cell, minlength=4))
        stat, p, dof, _ = chi2_contingency(np.array(rows))
        out.append({"pair": f"{ASSETS[i]}-{ASSETS[j]}", "chi2": stat, "dof": dof, "p": p})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="halve sample sizes")
    ap.add_argument("--out-dir", default=str(pathlib.Path(__file__).parent / "results"),
                    help="CSV output dir; figure goes to <out-dir>/../figures")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    fig_dir = out_dir.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_train = 2000 if args.smoke else 4000
    n_test = 500 if args.smoke else 1000
    sweeps = 12
    n_mc = 100_000 if args.smoke else 200_000

    print(f"[parity_model_demo] n_train={n_train} n_test={n_test} sweeps={sweeps}", flush=True)

    data = {
        ("even", "train"): world_returns("even", n_train, seed=11),
        ("even", "test"):  world_returns("even", n_test,  seed=12),
        ("odd",  "train"): world_returns("odd",  n_train, seed=13),
        ("odd",  "test"):  world_returns("odd",  n_test,  seed=14),
    }

    # ---- fit on each train world, score both test worlds ----
    rows = []
    ptable_even = None
    encode_even = None
    for tw in ("even", "odd"):
        print(f"[fit] train world = {tw}", flush=True)
        scorers, ptables, edges, encode = fit_all(data[(tw, "train")], sweeps=sweeps)
        if tw == "even":
            encode_even = encode
            ptable_even = {"true_even": true_probs("even"), "true_odd": true_probs("odd"),
                           "gaussian": ptables["gaussian_mc"](n_mc, seed_mc=99),
                           "independent": ptables["independent"](),
                           "mps_chi1": ptables["mps_chi1"](),
                           "mps_chi2": ptables["mps_chi2"]()}
        for m in MODELS:
            per = {}
            for sw in ("even", "odd"):
                ll = scorers[m](data[(sw, "test")])
                per[sw] = (float(ll.mean()), float(ll.std(ddof=1) / np.sqrt(len(ll))), len(ll))
            gap = per["even"][0] - per["odd"][0]                       # even minus odd
            gap_se = float(np.hypot(per["even"][1], per["odd"][1]))
            toward_train = gap if tw == "even" else -gap
            detects = bool(toward_train > 3.0 * gap_se)
            for sw in ("even", "odd"):
                rows.append({"model": m, "train_world": tw, "test_world": sw,
                             "n_test": per[sw][2], "loglik_mean": per[sw][0],
                             "loglik_se": per[sw][1],
                             "is_density": m == "gaussian",
                             "gap_even_minus_odd": gap, "gap_se": gap_se,
                             "detects_threeway": detects})
            print(f"  {m:12s} LL(even-test)={per['even'][0]:+10.4f}  "
                  f"LL(odd-test)={per['odd'][0]:+10.4f}  gap={gap:+10.4f} "
                  f"(3SE={3*gap_se:.4f})  detect={detects}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "parity_demo.csv", index=False)

    # ---- data check: pairwise tables even vs odd test (train-even edges) ----
    ce = encode_even(data[("even", "test")])
    co = encode_even(data[("odd", "test")])
    pw = pairwise_check(ce, co)
    pairwise_clean = bool((pw["p"] > 0.01).all())
    print("\n[data check] pairwise joint bin tables, even-test vs odd-test "
          "(chi-squared homogeneity; high p = indistinguishable):", flush=True)
    print(pw.to_string(index=False), flush=True)
    print(f"pairwise difference present: {'NO' if pairwise_clean else 'YES (check data gen!)'}",
          flush=True)

    # ---- P(x) table over the 8 configurations (models trained on EVEN) ----
    pt = pd.DataFrame(ptable_even, index=[config_label(c) for c in GRID8])
    pt.index.name = "config"
    pt.to_csv(out_dir / "parity_demo_ptable.csv")
    print("\n[P(x) over the 8 configurations | models trained on EVEN world]", flush=True)
    print("(gaussian = fitted MVN mass per bin cell, Monte Carlo; chi=1 mathematically "
          "cannot deviate from the product form)", flush=True)
    print(pt.round(4).to_string(), flush=True)

    # ---- headline table ----
    det = {m: {} for m in MODELS}
    for m in MODELS:
        sub = res[res.model == m].drop_duplicates(["train_world"])
        det[m] = {tw: bool(sub[sub.train_world == tw]["detects_threeway"].iloc[0])
                  for tw in ("even", "odd")}
    print("\n| Model | Detects pairwise difference? | Detects three-way difference? |", flush=True)
    print("|---|---|---|", flush=True)
    for m in MODELS:
        pw_ans = "No" if pairwise_clean else "DATA CHECK FAILED"
        if m == "mps_chi2" and pairwise_clean:
            pw_ans = "No (pairwise)"
        both = det[m]["even"] and det[m]["odd"]
        print(f"| {DISPLAY[m]} | {pw_ans} | {'Yes' if both else 'No'} |", flush=True)
    print("\nNotes: gaussian scores are continuous log-DENSITIES — only its even-vs-odd "
          "GAP is comparable to the discrete models. Symmetry: detection flags per train "
          "direction:", flush=True)
    for m in MODELS:
        print(f"  {DISPLAY[m]:24s} train-even: {det[m]['even']}   train-odd: {det[m]['odd']}",
              flush=True)

    # ---- figure ----
    make_figure(res, pt, fig_dir / "parity_demo.png")
    print(f"\n[done] wrote {out_dir / 'parity_demo.csv'}, "
          f"{out_dir / 'parity_demo_ptable.csv'}, {fig_dir / 'parity_demo.png'}", flush=True)


def make_figure(res, pt, path):
    """Left: held-out LL gap (even-test minus odd-test) per model, both train
    directions, 3-SE error bars, symlog (the chi=2 gap is orders of magnitude
    larger). Right: P(x) heatmap (single-hue sequential), truth on top."""
    c_even, c_odd = "#4269d0", "#efb118"        # CVD-safe blue/orange pair
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    xs = np.arange(len(MODELS))
    for off, tw, col in ((-0.18, "even", c_even), (+0.18, "odd", c_odd)):
        sub = res[(res.train_world == tw)].drop_duplicates(["model"]).set_index("model")
        g = sub.loc[MODELS, "gap_even_minus_odd"].to_numpy()
        e = 3.0 * sub.loc[MODELS, "gap_se"].to_numpy()
        ax1.bar(xs + off, g, width=0.32, color=col, label=f"trained on {tw}",
                yerr=e, capsize=3, error_kw={"lw": 1})
        for x, v in zip(xs + off, g):
            ax1.annotate(f"{v:+.2f}", (x, v), ha="center",
                         va="bottom" if v >= 0 else "top", fontsize=8)
    ax1.axhline(0, color="0.4", lw=0.8)
    ax1.set_yscale("symlog", linthresh=0.5)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([DISPLAY[m] for m in MODELS], fontsize=9)
    ax1.set_ylabel("held-out log-lik per obs: even-test minus odd-test")
    ax1.set_title("Only MPS chi=2 separates the worlds\n(error bars = 3 SE; symlog scale)")
    ax1.legend(frameon=False)
    ax1.grid(axis="y", color="0.9", lw=0.6)
    ax1.set_axisbelow(True)

    rows = ["true_even", "gaussian", "independent", "mps_chi1", "mps_chi2"]
    labels = ["truth (even world)", "Gaussian (binned, MC)", "independent",
              "MPS chi=1", "MPS chi=2"]
    M = pt[rows].to_numpy().T                                  # (5, 8)
    im = ax2.imshow(M, cmap="Blues", vmin=0.0, vmax=0.30, aspect="auto")
    ax2.set_xticks(range(8))
    ax2.set_xticklabels(pt.index, fontsize=9)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels(labels, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax2.annotate(f"{M[i, j]:.2f}", (j, i), ha="center", va="center", fontsize=8,
                         color="white" if M[i, j] > 0.18 else "#1a1a2e")
    ax2.set_title("P(x) over the 8 sign configurations (trained on EVEN)\n"
                  "chi=2 recovers the parity table; chi=1 mathematically cannot")
    fig.colorbar(im, ax=ax2, shrink=0.8, label="probability")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
