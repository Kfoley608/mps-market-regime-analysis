"""Discretize per-asset returns into bins (the MPS models a distribution over bin states),
and order the assets along the 1-D MPS chain.

Defaults (see the Week-2 worklog for the rationale):
  - 3 quantile bins per asset (terciles: down / flat / up). Coarsest scheme that still captures
    directional dependence, most estimable from a ~250-day window, and keeps the per-site
    dimension d=3 small so the bond dimension stays low.
  - Quantile (equal-mass) cuts, not fixed-width — returns are fat-tailed, so equal-mass bins
    don't waste cells on the sparse extremes.
  - Asset ordering fixed once via the Fiedler vector of the pairwise mutual-information graph
    (highly-dependent assets placed adjacent → lower required bond dimension). Ordering
    *sensitivity* is itself a feature (F8 in features.py).

Look-ahead discipline (see worklog checklist):
  - Bin EDGES must be computed inside each training window (`bin_edges` on that window), never
    over the full sample.
  - The chain ORDERING should be computed ONCE (e.g. on an early/expanding window) and reused
    for every window — both to avoid look-ahead and to keep features comparable across time.
    Do NOT recompute the ordering per window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Binning                                                                      #
# --------------------------------------------------------------------------- #
def bin_edges(returns: pd.DataFrame, n_bins: int = 3) -> dict[str, np.ndarray]:
    """Interior quantile cut points per asset (n_bins-1 edges each).

    Guards (2026-07-14, review item l): raises on NaNs (np.quantile propagates them to every edge
    and np.digitize(NaN) lands silently in the TOP bin) and warns on constant columns / duplicate
    edges (degenerate bins).

    Compute these on the TRAINING window only, then apply to the days you want to label.
    """
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]  # drop 0 and 1 -> interior cuts
    out = {}
    for c in returns.columns:
        x = returns[c].to_numpy()
        if np.isnan(x).any():
            raise ValueError(f"bin_edges: NaNs in column {c!r} — clean upstream (silent top-bin coding otherwise)")
        e = np.quantile(x, qs)
        if x.std() == 0 or (len(e) > 1 and (np.diff(e) == 0).any()):
            import warnings
            warnings.warn(f"bin_edges: degenerate/duplicate quantile edges for column {c!r}")
        out[c] = e
    return out


def apply_bins(returns: pd.DataFrame, edges: dict[str, np.ndarray]) -> pd.DataFrame:
    """Apply precomputed edges -> integer bin labels in 0 .. n_bins-1."""
    out = {c: np.digitize(returns[c].to_numpy(), edges[c]) for c in returns.columns}
    return pd.DataFrame(out, index=returns.index, columns=returns.columns)


def quantile_bins(returns: pd.DataFrame, n_bins: int = 3,
                  edges: dict[str, np.ndarray] | None = None
                  ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Map each asset's returns to integer bin labels via per-asset quantiles.

    Returns (binned_labels, edges). If `edges` is None they are computed from `returns`;
    pass precomputed edges (from the training window) to label out-of-window days causally.
    """
    if edges is None:
        edges = bin_edges(returns, n_bins)
    return apply_bins(returns, edges), edges


# --------------------------------------------------------------------------- #
# Asset ordering along the chain (mutual-information / Fiedler)                 #
# --------------------------------------------------------------------------- #
def mutual_information_matrix(binned: pd.DataFrame) -> pd.DataFrame:
    """Symmetric pairwise mutual information (nats) between the discretized assets.

    MI captures *all* statistical dependence (including nonlinear), not just correlation —
    the right notion for placing dependent assets adjacent on the MPS chain.
    """
    from sklearn.metrics import mutual_info_score

    cols = list(binned.columns)
    n = len(cols)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mi = mutual_info_score(binned.iloc[:, i], binned.iloc[:, j])
            m[i, j] = m[j, i] = mi
    return pd.DataFrame(m, index=cols, columns=cols)


def fiedler_ordering(mi_matrix: pd.DataFrame) -> list[str]:
    """Order assets by the Fiedler vector (2nd-smallest eigenvector) of the MI-graph Laplacian.

    This spectral seriation places strongly-dependent assets next to each other, which
    minimizes the long-range entanglement the MPS must carry and keeps the bond dimension low.
    """
    w = mi_matrix.to_numpy().copy()
    np.fill_diagonal(w, 0.0)
    lap = np.diag(w.sum(axis=1)) - w           # graph Laplacian L = D - W
    _, vecs = np.linalg.eigh(lap)              # ascending eigenvalues
    fiedler = vecs[:, 1]                        # 2nd-smallest -> the seriation axis
    order = np.argsort(fiedler)
    return [mi_matrix.columns[k] for k in order]


# kept for backwards-compatibility with the original stub name
def mi_ordering(binned: pd.DataFrame) -> list[str]:
    """Convenience: MI matrix -> Fiedler ordering of the chain."""
    return fiedler_ordering(mutual_information_matrix(binned))


# --------------------------------------------------------------------------- #
# One-call entry point                                                         #
# --------------------------------------------------------------------------- #
def discretize(returns: pd.DataFrame, n_bins: int = 3,
               ordering: list[str] | None = None,
               edges: dict[str, np.ndarray] | None = None
               ) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str]]:
    """returns -> (binned labels in Fiedler-chain order, edges, ordering).

    If `ordering` is None it is derived from this data's MI matrix; in the rolling pipeline,
    compute it once and pass it back in for every window (see the look-ahead note above).
    """
    binned, edges = quantile_bins(returns, n_bins, edges)
    if ordering is None:
        ordering = fiedler_ordering(mutual_information_matrix(binned))
    return binned[ordering], edges, ordering
