"""Classical covariance-level indicators — the comparison set AND the Tier-1 redundancy
regressors. Every one of these is a function of second moments only; the project's whole
claim is that MPS features add information beyond this set.

Each `*_series` function returns a rolling time-series aligned to the returns index (NaN
until the first full window). Point-in-time functions take a single window's returns block.

References:
  - Absorption ratio: Kritzman, Li, Page, Rigobon (2011), J. Portfolio Management 37(4).
  - Turbulence:       Kritzman & Li (2010), Financial Analysts Journal 66(5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Point-in-time estimators (operate on one window block of returns)           #
# --------------------------------------------------------------------------- #
def realized_vol(returns: pd.DataFrame, annualize: bool = True) -> float:
    """Average per-asset realized volatility over the window."""
    vol = returns.std(ddof=1).mean()
    return float(vol * np.sqrt(252)) if annualize else float(vol)


def average_correlation(returns: pd.DataFrame) -> float:
    """Mean off-diagonal pairwise correlation."""
    corr = returns.corr().to_numpy()
    n = corr.shape[0]
    off = corr[~np.eye(n, dtype=bool)]
    return float(np.nanmean(off))


def pca_first_explained(returns: pd.DataFrame) -> float:
    """Fraction of variance explained by the first principal component."""
    eig = _corr_eigenvalues(returns)
    return float(eig[0] / eig.sum())


def absorption_ratio(returns: pd.DataFrame, n_eigen: int | None = None) -> float:
    """Kritzman-Li-Page-Rigobon absorption ratio: variance share of the top eigenvectors.

    Default n_eigen = ceil(n_assets / 5) per the original paper.
    """
    eig = _corr_eigenvalues(returns)
    n = len(eig)
    k = n_eigen if n_eigen is not None else int(np.ceil(n / 5))
    k = max(1, min(k, n))
    return float(eig[:k].sum() / eig.sum())


def spectral_entropy(returns: pd.DataFrame) -> float:
    """Shannon entropy of the normalized correlation-matrix eigenvalue spectrum.

    Low entropy = variance concentrated in few modes (tightly coupled market). This is the
    classical measure closest to an MPS entanglement diagnostic, so it MUST stay in the
    control set (a reviewer will ask why an MPS entropy beats the spectral entropy).
    """
    eig = _corr_eigenvalues(returns)
    p = eig / eig.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def max_drawdown(returns: pd.DataFrame | pd.Series, weights: np.ndarray | None = None,
                 log_returns: bool = True) -> float:
    """Worst peak-to-trough drawdown of the (portfolio) wealth path over the window.

    the sponsor-listed comparison indicator. `returns` may be a single series or a panel; a panel is
    aggregated to a portfolio (equal-weight by default).
    """
    dd = drawdown_series(returns, weights=weights, log_returns=log_returns)
    return float(dd.min())


def turbulence(returns: pd.DataFrame, ref_mean: np.ndarray, ref_cov: np.ndarray) -> pd.Series:
    """Mahalanobis financial turbulence of each day vs a reference distribution.

    d_t = (r_t - mu)' Sigma^{-1} (r_t - mu), using a robust pseudo-inverse of the reference
    covariance. `ref_mean`/`ref_cov` are estimated from a trailing reference window.
    """
    inv = np.linalg.pinv(ref_cov)
    centered = returns.to_numpy() - ref_mean
    d = np.einsum("ij,jk,ik->i", centered, inv, centered)
    return pd.Series(d, index=returns.index, name="turbulence")


# --------------------------------------------------------------------------- #
# Rolling-series wrappers (the actual Week-1 deliverable)                      #
# --------------------------------------------------------------------------- #
def rolling_indicators(returns: pd.DataFrame, window: int = 250, step: int = 5) -> pd.DataFrame:
    """Compute the full second-moment indicator panel on rolling windows.

    Returns a DataFrame indexed by the window's END date, one column per indicator. `step`
    controls how often we re-estimate (matches the MPS refit cadence in universe.yaml).
    """
    idx = returns.index
    rows, dates = [], []
    for end in range(window, len(idx) + 1, step):
        block = returns.iloc[end - window:end]
        rows.append(
            {
                "realized_vol": realized_vol(block),
                "avg_corr": average_correlation(block),
                "pca1_explained": pca_first_explained(block),
                "absorption_ratio": absorption_ratio(block),
                "spectral_entropy": spectral_entropy(block),
                "max_drawdown": max_drawdown(block),
            }
        )
        dates.append(idx[end - 1])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))


def drawdown_series(returns: pd.DataFrame | pd.Series, weights: np.ndarray | None = None,
                    log_returns: bool = True) -> pd.Series:
    """Running drawdown of the (portfolio) wealth path: wealth/peak - 1, <= 0.

    the sponsor-listed comparison indicator. A return panel is collapsed to a portfolio first
    (equal-weight unless `weights` given). Assumes log returns by default.
    """
    if isinstance(returns, pd.DataFrame):
        if weights is None:
            weights = np.full(returns.shape[1], 1.0 / returns.shape[1])
        port = pd.Series(returns.to_numpy() @ weights, index=returns.index)
    else:
        port = returns
    wealth = np.exp(port.cumsum()) if log_returns else (1.0 + port).cumprod()
    peak = wealth.cummax()
    return (wealth / peak - 1.0).rename("drawdown")


def var_breaches(returns: pd.Series, window: int = 250, alpha: float = 0.99) -> pd.Series:
    """Historical-VaR exceedance flags (1 where loss exceeds the trailing VaR_alpha).

    the sponsor-listed comparison; also the bridge to the VaR/CVaR spec variant. Causal: VaR for day
    t is estimated from the trailing `window` of returns ending at t-1.
    """
    r = returns.to_numpy()
    out = np.full(len(returns), np.nan)
    for t in range(window, len(returns)):
        var = np.quantile(r[t - window:t], 1.0 - alpha)  # left-tail quantile (negative)
        out[t] = 1.0 if r[t] < var else 0.0
    return pd.Series(out, index=returns.index, name=f"var_breach_{int(alpha*100)}")


def turbulence_series(returns: pd.DataFrame, window: int = 250) -> pd.Series:
    """Daily turbulence using a trailing-window reference distribution (causal)."""
    arr = returns.to_numpy()
    out = np.full(len(returns), np.nan)
    for t in range(window, len(returns)):
        ref = arr[t - window:t]
        mu = ref.mean(axis=0)
        cov = np.cov(ref, rowvar=False)
        inv = np.linalg.pinv(cov)
        c = arr[t] - mu
        out[t] = c @ inv @ c
    return pd.Series(out, index=returns.index, name="turbulence")


# --------------------------------------------------------------------------- #
# Crisis / stress periods (the sponsor-listed comparison)                             #
# --------------------------------------------------------------------------- #
# Well-known US-market stress windows, for shading plots and checking whether an
# indicator actually spikes when dependence breaks down. These are descriptive overlays
# of KNOWN history — not predictive labels — so using them introduces no look-ahead.
CRISIS_PERIODS: dict[str, tuple[str, str]] = {
    "GFC (2008)":              ("2008-09-01", "2009-06-30"),
    "Euro/US downgrade (2011)":("2011-07-01", "2011-10-31"),
    "China & oil (2015-16)":   ("2015-08-01", "2016-02-29"),
    "Q4 selloff (2018)":       ("2018-10-01", "2018-12-31"),
    "COVID crash (2020)":      ("2020-02-20", "2020-04-30"),
    "Rate-hike bear (2022)":   ("2022-01-01", "2022-10-31"),
    "Tariff repricing (2025)": ("2025-04-01", "2025-05-31"),
}


def crisis_periods() -> dict[str, tuple[str, str]]:
    """The named (start, end) stress windows used for overlays and stress-period checks."""
    return dict(CRISIS_PERIODS)


def crisis_mask(index: pd.DatetimeIndex, periods: dict | None = None) -> pd.Series:
    """Boolean Series over `index`, True on days inside any crisis window."""
    periods = periods or CRISIS_PERIODS
    mask = pd.Series(False, index=index)
    for start, end in periods.values():
        mask |= (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    return mask.rename("crisis")


# --------------------------------------------------------------------------- #
# Assembled comparison panel — every the sponsor-listed classical indicator, aligned   #
# --------------------------------------------------------------------------- #
def build_classical_panel(
    returns: pd.DataFrame,
    window: int = 250,
    step: int = 5,
    weights: np.ndarray | None = None,
    vix: pd.Series | None = None,
    fetch_overlays: bool = True,
) -> pd.DataFrame:
    """Assemble all classical comparison indicators into one date-aligned DataFrame.

    Columns (the sponsor "compare against classical indicators" list + our stronger extras):
      realized_vol, avg_corr, pca1_explained        <- the sponsor-listed
      vix                                            <- the sponsor-listed (external overlay)
      drawdown, max_drawdown                         <- the sponsor-listed
      var_breach_rate                                <- the sponsor-listed (rolling 99% breach frequency)
      crisis                                         <- the sponsor-listed (stress-period flag)
      absorption_ratio, turbulence, spectral_entropy <- extras (stronger second-moment regime gauges)

    Indexed by rolling-window END dates (every `step` days). Daily path-dependent measures
    (drawdown, turbulence, VaR-breach rate) and the VIX overlay are sampled onto those dates.
    `vix` may be passed in; otherwise it is fetched from Yahoo when `fetch_overlays=True`.
    """
    panel = rolling_indicators(returns, window=window, step=step)  # vol/corr/pca1/absorption/spectral/max_dd
    idx = panel.index

    # Equal-weight (or supplied) portfolio for the path-dependent measures.
    if weights is None:
        weights = np.full(returns.shape[1], 1.0 / returns.shape[1])
    port = pd.Series(returns.to_numpy() @ weights, index=returns.index)

    panel["turbulence"] = turbulence_series(returns, window=window).reindex(idx)
    panel["drawdown"] = drawdown_series(returns, weights=weights).reindex(idx)
    breach_rate = var_breaches(port, window=window, alpha=0.99).rolling(window).mean()
    panel["var_breach_rate"] = breach_rate.reindex(idx)

    # VIX overlay (external series on the same calendar; forward-fill onto panel dates).
    if vix is None and fetch_overlays:
        try:
            from .data import load_vix
            vix = load_vix(str(returns.index.min().date()))
        except Exception:
            vix = None
    if vix is not None and not vix.empty:
        panel["vix"] = vix.reindex(idx, method="ffill")

    panel["crisis"] = crisis_mask(idx).astype(int)
    return panel


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _corr_eigenvalues(returns: pd.DataFrame) -> np.ndarray:
    """Descending eigenvalues of the correlation matrix (real, clipped at 0)."""
    corr = returns.corr().to_numpy()
    eig = np.linalg.eigvalsh(corr)          # ascending, real (symmetric)
    eig = np.clip(eig[::-1], 0.0, None)      # descending, non-negative
    return eig
