"""Week 7 item 8 — block mutual information between NAMED asset groups, time-varying heatmap.

For each economically named group pair (equities|Treasuries, equities|credit, risk|defensive, ...)
fit ONE MPS per rolling window with the chain arranged as
    [group A sites (Fiedler-ordered within) | group B sites (Fiedler-ordered within)]
and read the EXACT probability-identified between-group dependence with
features.block_mutual_information(fit, cut=len(A)) — sampler expectation + exact block-marginal
log-probs (P-functional, BUG-LOG #14 compliant). Stacked over windows this yields statements like
"during March 2020 dependence rose not only within equities but between risk assets, Treasuries,
credit, and commodities."

Protocol (matches analysis/parallel_rolling_v4.py cadence):
  - book: data/combined_gfc_2026.parquet (12 cross-asset tickers, 2007-2026)
  - 252-day windows advancing 21 days; d=8, chi=8, sweeps=8 DMRG per (window, pair)
  - within-group Fiedler orderings FROZEN from the first ORD_DAYS days only (BUG-LOG #15:
    ordering never sees the future; windows start at/after ORD_DAYS, v4 discipline)
  - block MI estimated with n_samples=20000, seed derived from (pair, window) index
  - normalization: raw nats AND a causal z-score vs the pair's own EXPANDING history
    (strictly prior windows only), so the heatmap reads "unusually coupled vs its own baseline"

Outputs (incremental — safe to kill and resume; completed (pair, date) rows are skipped):
  week7/results/blockmi_groups.csv           date x pair x mi_nats x z
  week7/results/blockmi_groups_mapping.json  group -> tickers, pair list, frozen orderings, params
  week7/figures/blockmi_heatmap.png          x=time, y=pair, color=z, crisis shading

Pod-only (fits MPS — RunPod, never the laptop). --plot-only regenerates the figure anywhere.

Usage:
  python week7/blockmi_groups.py                    # full run (8 pairs, ~200 windows each)
  python week7/blockmi_groups.py --smoke            # 3 pairs, every 126 days
  python week7/blockmi_groups.py --pairs "us_equities|treasuries,risk|defensive"
  python week7/blockmi_groups.py --plot-only        # rebuild the heatmap from the CSV
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import argparse
import json
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

BOOK = REPO / "data" / "combined_gfc_2026.parquet"
WIN, STEP = 252, 21          # v4 window; 21-day (monthly) advance
ORD_DAYS = 252               # within-group orderings frozen from this early span only
D_BINS, CHI, SWEEPS = 8, 8, 8
N_SAMPLES = 20000            # block-MI estimator draws
MIN_HIST = 8                 # windows of history required before a z-score is emitted

# --- named groups over the combined_gfc_2026 tickers (all 12 covered) ---------------------------
GROUPS = {
    "us_equities":   ["USMKT", "AAPL", "NVDA", "IWM"],
    "intl_equities": ["EFA", "EEM"],
    "treasuries":    ["TLT"],
    "credit":        ["LQD", "HYG"],
    "commodities":   ["GLD", "DBC"],
    "real_estate":   ["VNQ"],
    # composites for the risk-on/risk-off cut (no financials sleeve in this book)
    "risk":          ["USMKT", "AAPL", "NVDA", "IWM", "EFA", "EEM", "HYG", "VNQ", "DBC"],
    "defensive":     ["TLT", "LQD", "GLD"],
}
DEFAULT_PAIRS = [
    ("us_equities", "treasuries"),
    ("us_equities", "credit"),
    ("us_equities", "commodities"),
    ("us_equities", "intl_equities"),
    ("treasuries", "credit"),
    ("credit", "commodities"),
    ("us_equities", "real_estate"),
    ("risk", "defensive"),
]
CRISES = [  # (start, end, label) — vertical spans on the heatmap
    ("2008-01-01", "2009-06-30", "GFC"),
    ("2011-05-01", "2012-07-31", "euro"),
    ("2013-05-01", "2013-09-30", "taper"),
    ("2015-08-01", "2016-02-29", "China/oil"),
    ("2018-01-26", "2018-02-28", "Volmageddon"),
    ("2020-02-15", "2020-04-30", "COVID"),
    ("2022-01-01", "2022-10-31", "hiking"),
]


def parse_pairs(spec):
    out = []
    for tok in spec.split(","):
        a, b = tok.strip().split("|")
        if a not in GROUPS or b not in GROUPS:
            sys.exit(f"unknown group in pair {tok!r}; known: {sorted(GROUPS)}")
        if set(GROUPS[a]) & set(GROUPS[b]):
            sys.exit(f"pair {tok!r} has overlapping members — groups in a pair must be disjoint")
        out.append((a, b))
    return out


def frozen_orderings(rets_early, pairs):
    """Fiedler ordering WITHIN each group, from the early span only (frozen for every window).
    Groups of size <= 2 keep their listed order (seriation is trivial/undefined there)."""
    from mps_regime import binning
    need = sorted({g for p in pairs for g in p})
    out = {}
    for g in need:
        cols = GROUPS[g]
        if len(cols) <= 2:
            out[g] = list(cols)
            continue
        codes = binning.quantile_bins(rets_early[cols], n_bins=6)[0]
        out[g] = binning.fiedler_ordering(binning.mutual_information_matrix(codes))
    return out


def run_fits(args, pairs, out_csv, mapping_path):
    from mps_regime import data as D, dmrg
    from mps_regime.features import block_mutual_information

    rets = D.to_returns(pd.read_parquet(BOOK).astype(float), kind="log")
    step = 126 if args.smoke else STEP
    orderings = frozen_orderings(rets.iloc[:ORD_DAYS], pairs)
    mapping = dict(book=str(BOOK.name), groups=GROUPS,
                   pairs=[f"{a}|{b}" for a, b in pairs],
                   orderings=orderings, window=WIN, step=step, ord_days=ORD_DAYS,
                   d=D_BINS, chi=CHI, sweeps=SWEEPS, n_samples=N_SAMPLES,
                   crises=CRISES)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping, indent=2))
    print(f"mapping -> {mapping_path}", flush=True)
    for g in sorted(orderings):
        print(f"  {g:<14} {orderings[g]}", flush=True)

    # windows START at/after ORD_DAYS so no window overlaps the ordering span (v4 discipline)
    starts = [s for s in range(0, len(rets) - WIN + 1, step) if s >= ORD_DAYS]
    print(f"windows={len(starts)} pairs={len(pairs)} fits={len(starts) * len(pairs)}", flush=True)

    done = set()
    hist = {}                                  # pair -> list of prior mi values (chronological)
    if out_csv.exists():                       # resume: skip completed rows, rebuild z history
        prev = pd.read_csv(out_csv)
        for _, r in prev.sort_values("date").iterrows():
            done.add((r["pair"], r["date"]))
            hist.setdefault(r["pair"], []).append(float(r["mi_nats"]))
        print(f"resume: {len(done)} rows already in {out_csv}", flush=True)

    t0 = time.time()
    n_fit = 0
    for pi, (ga, gb) in enumerate(pairs):
        pair = f"{ga}|{gb}"
        cols_a, cols_b = orderings[ga], orderings[gb]
        cols = cols_a + cols_b
        cut = len(cols_a)
        h = hist.setdefault(pair, [])
        for wi, s in enumerate(starts):
            w = rets.iloc[s:s + WIN]
            date = str(w.index[-1].date())
            if (pair, date) in done:
                continue
            seed = 1000 * pi + wi              # deterministic per (pair, window)
            fit = dmrg.fit_dmrg(w[cols], d=D_BINS, chi=CHI, sweeps=SWEEPS, seed=seed)
            mi = block_mutual_information(fit, cut=cut, n_samples=N_SAMPLES, seed=seed)
            # causal z: this pair's own EXPANDING history, strictly prior windows only
            z = float("nan")
            if len(h) >= MIN_HIST:
                mu, sd = float(np.mean(h)), float(np.std(h))
                z = (mi - mu) / sd if sd > 1e-12 else float("nan")
            h.append(mi)
            row = pd.DataFrame([dict(date=date, pair=pair, cut=cut, n_left=len(cols_a),
                                     n_right=len(cols_b), mi_nats=mi, z=z, seed=seed)])
            row.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            n_fit += 1
            print(f"[{time.time() - t0:7.0f}s] {pair:<28} {date}  mi={mi:.4f}  z={z:+.2f}"
                  f"  ({n_fit} fits)", flush=True)
    print(f"done: {n_fit} new fits in {time.time() - t0:.0f}s -> {out_csv}", flush=True)


def make_heatmap(out_csv, fig_path, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    df = pd.read_csv(out_csv, parse_dates=["date"])
    order = [f"{a}|{b}" for a, b in pairs if f"{a}|{b}" in set(df["pair"])]
    piv = (df.drop_duplicates(["pair", "date"], keep="last")
             .pivot(index="pair", columns="date", values="z").reindex(order))
    x = mdates.date2num(piv.columns.to_pydatetime())
    fig, (ax_c, ax) = plt.subplots(
        2, 1, figsize=(14, 1.0 + 0.55 * len(order)), sharex=True,
        gridspec_kw=dict(height_ratios=[0.12, 1.0], hspace=0.05))
    # crisis strip (separate axis so the shading never obscures the heatmap colors)
    for s, e, lab in CRISES:
        s, e = mdates.date2num(pd.Timestamp(s)), mdates.date2num(pd.Timestamp(e))
        ax_c.axvspan(s, e, color="0.55", alpha=0.6)
        ax.axvline(s, color="0.3", lw=0.6, ls="--", alpha=0.6)
        ax.axvline(e, color="0.3", lw=0.6, ls="--", alpha=0.6)
        ax_c.text((s + e) / 2, 0.5, lab, ha="center", va="center", fontsize=7, rotation=0)
    ax_c.set_yticks([]); ax_c.set_ylim(0, 1)
    for sp in ("top", "left", "right"):
        ax_c.spines[sp].set_visible(False)
    # z heatmap, diverging, centered on the pair's own baseline
    xe = np.concatenate([x, [x[-1] + (x[-1] - x[-2] if len(x) > 1 else 21)]])
    ye = np.arange(len(order) + 1)
    pc = ax.pcolormesh(xe, ye, piv.to_numpy(), cmap="RdBu_r", vmin=-3, vmax=3, shading="flat")
    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_yticklabels([p.replace("|", " | ") for p in order], fontsize=8)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.colorbar(pc, ax=ax, pad=0.01, label="block MI z-score vs own causal baseline")
    ax_c.set_title("Between-group block mutual information (MPS, exact block marginals) — "
                   f"{WIN}d windows / {D_BINS} bins / chi={CHI}", fontsize=10)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"figure -> {fig_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="3 pairs, every 126 days")
    ap.add_argument("--pairs", default=None, help='e.g. "us_equities|treasuries,risk|defensive"')
    ap.add_argument("--out-dir", default=None, help="results dir (default week7/results)")
    ap.add_argument("--plot-only", action="store_true", help="rebuild the heatmap from the CSV")
    ap.add_argument("--allow-local", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    pairs = parse_pairs(args.pairs) if args.pairs else list(DEFAULT_PAIRS)
    if args.smoke:
        pairs = pairs[:3]
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else REPO / "week7" / "results"
    out_csv = out_dir / "blockmi_groups.csv"
    mapping_path = out_dir / "blockmi_groups_mapping.json"
    fig_path = REPO / "week7" / "figures" / "blockmi_heatmap.png"

    if not args.plot_only:
        if sys.platform == "win32" and not args.allow_local:
            sys.exit("blockmi_groups.py fits MPS models — run on the RunPod server, not the laptop "
                     "(--allow-local to override).")
        out_dir.mkdir(parents=True, exist_ok=True)
        run_fits(args, pairs, out_csv, mapping_path)
    if not out_csv.exists():
        sys.exit(f"no results at {out_csv} — nothing to plot")
    make_heatmap(out_csv, fig_path, pairs)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
