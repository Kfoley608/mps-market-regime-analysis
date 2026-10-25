"""Glasgow higher-order comoments replicated with an MPS Born machine.

This is a project-specific extension of:

    Cerrato, Crosby, Kim & Zhao (2017), "Relation between higher order
    comoments and dependence structure of equity portfolio", Journal of
    Empirical Finance 40, 101-120, doi:10.1016/j.jempfin.2016.11.007.

The paper sorts assets into portfolios using beta, coskewness and cokurtosis,
then evaluates asymmetric dependence and one-day-ahead VaR/ES. This module keeps
that economic question but replaces the paper's bivariate dynamic copula with
the repository's multivariate MPS Born machine.

Protocol v2 (review response)
-----------------------------
The first pass compared MPS samples against a covariance-only Gaussian with
*unmatched marginals*, so joint-tail differences confounded marginal tails with
dependence. This version adds:

* gauss_copula: a matched-marginal Gaussian copula benchmark (normal-scores
  correlation + empirical train marginals) so dependence is isolated;
* marginal diagnostics per model (train-threshold tail rate, volatility ratio);
* a rank/PIT joint-tail probability using each model's own sample quantiles,
  which is invariant to marginals and measures dependence only;
* Kupiec and Christoffersen VaR backtests and the Fissler-Ziegel FZ0 joint
  VaR-ES score evaluated on every day (not only on each model's own hit days);
* moving-block bootstrap confidence intervals for hit rates and FZ0 deltas;
* a leave-one-out market for beta/coskewness/cokurtosis (each asset is excluded
  from its own market benchmark);
* MPS chain ordering bins default to the sampling bin dimension d.

Joint-tail and marginal statistics are computed once per window on the full
asset vector and reported separately (glasgow_jointtail_*), because they do not
depend on the characteristic sort. Characteristic-specific tables carry the
portfolio-level tail dependence and VaR/ES results.

Causal rolling protocol
-----------------------
For a forecast date t, only the preceding window observations are used. Asset
characteristics, quintile membership, MPS chain ordering, MPS binning, Gaussian
moments, copula correlation, empirical marginals, and empirical scenarios are
estimated on that window. A scenario model is then used to forecast the next
day's high-minus-low portfolio return. The default step is 21 trading days for
a practical run; --step 1 reproduces the paper's daily rolling cadence.

Models
------
* gaussian: multivariate Gaussian using the training mean and covariance
  (the covariance-only reference with Gaussian marginals).
* gauss_copula: Gaussian copula with empirical train marginals (the
  matched-marginal dependence reference).
* empirical: row bootstrap retaining all within-day dependence.
* mps_chiX: MPS Born machine fit with bond dimension X and direct sampling.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd


BOOKS = {
    "primary": "primary_prices.parquet",
    "gfc": "combined_gfc_2026.parquet",
    "banks": "banks_crsp_prices.parquet",
    "crossasset": "combined_crossasset_2026_lagbtc.parquet",
    "futures": "futures_macro_prices.parquet",
    "semis": "semis_crsp_prices.parquet",
}

CHARACTERISTICS = ("beta", "coskewness", "cokurtosis")
COPULA_MODEL = "gauss_copula"


def higher_order_comoments(returns: pd.DataFrame, market: str = "loo") -> pd.DataFrame:
    """Estimate the paper's beta, coskewness and cokurtosis for every asset.

    market="loo" (default) measures each asset against the equal-weighted
    average of the *other* assets, removing the mechanical own-asset component.
    market="ew" reproduces the v1 behavior (asset included in the market).
    The normalizations follow the standard coskewness/cokurtosis convention:
    sigma_i * sigma_m**2 and sigma_i * sigma_m**3 respectively.
    """
    if returns.shape[1] < 2:
        raise ValueError("At least two assets are required.")
    if market not in ("loo", "ew"):
        raise ValueError("market must be 'loo' or 'ew'.")
    x = returns.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Returns contain NaN or infinite values.")
    n_assets = x.shape[1]
    ri = x - x.mean(axis=0, keepdims=True)
    if market == "ew":
        m = np.repeat(x.mean(axis=1, keepdims=True), n_assets, axis=1)
    else:
        total = x.sum(axis=1, keepdims=True)
        m = (total - x) / (n_assets - 1)
    mc = m - m.mean(axis=0, keepdims=True)
    var_m = np.mean(mc**2, axis=0)
    sd_m = np.sqrt(np.maximum(var_m, 1e-30))
    sd_i = np.sqrt(np.mean(ri**2, axis=0)).clip(1e-15)
    beta = np.mean(ri * mc, axis=0) / np.maximum(var_m, 1e-30)
    cosk = np.mean(ri * mc**2, axis=0) / (sd_i * sd_m**2)
    cokt = np.mean(ri * mc**3, axis=0) / (sd_i * sd_m**3)
    return pd.DataFrame(
        {"beta": beta, "coskewness": cosk, "cokurtosis": cokt},
        index=returns.columns,
    )


def quintile_labels(values: Iterable[float], n_groups: int = 5) -> np.ndarray:
    """Stable low-to-high group labels with approximately equal membership."""
    values = np.asarray(list(values), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Sorting characteristic contains NaN or infinite values.")
    n = len(values)
    g = min(max(2, int(n_groups)), n)
    ordered = np.argsort(values, kind="mergesort")
    labels = np.empty(n, dtype=int)
    for k, idx in enumerate(np.array_split(ordered, g)):
        labels[idx] = k
    return labels


def portfolio_weights(values: Iterable[float], n_groups: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-weight portfolio weights and low-to-high group labels."""
    labels = quintile_labels(values, n_groups)
    n = len(labels)
    groups = int(labels.max()) + 1
    w = np.zeros((n, groups), dtype=float)
    for k in range(groups):
        idx = labels == k
        w[idx, k] = 1.0 / max(int(idx.sum()), 1)
    return w, labels


def nearest_psd(cov: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Symmetrize and ridge a covariance matrix for stable Gaussian sampling."""
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0
    scale = max(float(np.trace(cov)) / max(cov.shape[0], 1), 1e-12)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, ridge * scale)
    return (vecs * vals) @ vecs.T


def gaussian_samples(train: pd.DataFrame, n_sim: int, seed: int) -> np.ndarray:
    """Sample continuous returns from the covariance-only Gaussian benchmark."""
    rng = np.random.default_rng(seed)
    x = train.to_numpy(dtype=float)
    cov = nearest_psd(np.cov(x, rowvar=False, ddof=0))
    return rng.multivariate_normal(x.mean(axis=0), cov, size=n_sim)


def gaussian_copula_samples(train: pd.DataFrame, n_sim: int, seed: int) -> np.ndarray:
    """Matched-marginal Gaussian copula benchmark.

    Dependence: normal-scores (van der Waerden) correlation of the training
    window. Marginals: empirical training quantiles. Because marginals are held
    at the training distribution, any joint-tail difference against this model
    reflects dependence structure rather than marginal tail width.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    x = train.to_numpy(dtype=float)
    n, m = x.shape
    ranks = x.argsort(axis=0).argsort(axis=0).astype(float) + 1.0
    z = norm.ppf(ranks / (n + 1.0))
    corr = np.corrcoef(z, rowvar=False)
    corr = nearest_psd(corr)
    d = np.sqrt(np.diag(corr)).clip(1e-12)
    corr = corr / np.outer(d, d)
    zs = rng.multivariate_normal(np.zeros(m), corr, size=n_sim)
    us = norm.cdf(zs)
    out = np.empty((n_sim, m), dtype=float)
    for j in range(m):
        out[:, j] = np.quantile(x[:, j], us[:, j])
    return out


def empirical_samples(train: pd.DataFrame, n_sim: int, seed: int) -> np.ndarray:
    """Bootstrap whole return rows, preserving empirical cross-asset dependence."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(train), size=n_sim)
    return train.to_numpy(dtype=float)[idx]


def tail_dependence_pair(x: np.ndarray, y: np.ndarray, q: float = 0.05) -> tuple[float, float, float]:
    """Empirical lower, upper, and lower-minus-upper tail dependence."""
    if not 0 < q < 0.5:
        raise ValueError("q must lie strictly between 0 and 0.5.")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xl, yl = np.quantile(x, q), np.quantile(y, q)
    xu, yu = np.quantile(x, 1.0 - q), np.quantile(y, 1.0 - q)
    lower = float(np.mean((x <= xl) & (y <= yl)) / q)
    upper = float(np.mean((x >= xu) & (y >= yu)) / q)
    return lower, upper, lower - upper


def joint_tail_probability(
    samples: np.ndarray,
    q: float = 0.05,
    k: int = 3,
    thresholds: np.ndarray | None = None,
) -> float:
    """Probability of a k-way event using the supplied per-asset thresholds.

    With train-window thresholds this mixes marginal and dependence effects;
    with the model's own sample quantiles (thresholds=None) it is a rank/PIT
    statistic that measures dependence only.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must be a two-dimensional array.")
    if k < 1 or k > samples.shape[1]:
        raise ValueError("k must be between 1 and the number of assets.")
    if thresholds is None:
        thresholds = np.quantile(samples, q, axis=0)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape != (samples.shape[1],):
        raise ValueError("thresholds must have one value per sampled asset.")
    return float(np.mean((samples < thresholds[None, :]).sum(axis=1) >= k))


def marginal_diagnostics(
    samples: np.ndarray, train: np.ndarray, q: float
) -> tuple[float, float]:
    """Marginal calibration of a scenario model against its training window.

    Returns (tail_rate, vol_ratio): the average across assets of the model's
    probability mass below the train q-quantile (target: q if marginals match)
    and of the model/train standard-deviation ratio (target: 1).
    """
    thresholds = np.quantile(train, q, axis=0)
    tail_rate = float(np.mean(np.mean(samples < thresholds[None, :], axis=0)))
    train_sd = train.std(axis=0).clip(1e-15)
    vol_ratio = float(np.mean(samples.std(axis=0) / train_sd))
    return tail_rate, vol_ratio


def var_es(samples: np.ndarray, alpha: float = 0.01) -> tuple[float, float]:
    """Lower-tail VaR quantile and ES for a return distribution."""
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must lie strictly between 0 and 0.5.")
    x = np.asarray(samples, dtype=float)
    var = float(np.quantile(x, alpha))
    tail = x[x <= var]
    es = float(tail.mean()) if len(tail) else var
    return var, es


def fz0_loss(actual: float, var: float, es: float, alpha: float) -> float:
    """Fissler-Ziegel FZ0 joint VaR-ES scoring rule (Patton et al., 2019).

    Strictly consistent for the (VaR, ES) pair, evaluated on every day, which
    avoids comparing models on their own (different) hit-day subsets. Requires
    es < 0; degenerate forecasts score NaN.
    """
    if not np.isfinite(actual) or not np.isfinite(var) or not np.isfinite(es):
        return float("nan")
    if es >= -1e-12:
        return float("nan")
    hit = 1.0 if actual <= var else 0.0
    return float(
        -hit * (var - actual) / (alpha * es) + var / es + np.log(-es) - 1.0
    )


def kupiec_pvalue(hits: np.ndarray, alpha: float) -> float:
    """Kupiec (1995) proportion-of-failures likelihood-ratio p-value."""
    from scipy.stats import chi2

    hits = np.asarray(hits, dtype=float)
    n = len(hits)
    if n == 0:
        return float("nan")
    x = float(hits.sum())
    p_hat = min(max(x / n, 1e-10), 1 - 1e-10)
    p0 = min(max(alpha, 1e-10), 1 - 1e-10)

    def loglik(p: float) -> float:
        return (n - x) * np.log(1 - p) + x * np.log(p)

    lr = -2.0 * (loglik(p0) - loglik(p_hat))
    return float(chi2.sf(max(lr, 0.0), 1))


def christoffersen_pvalues(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Christoffersen (1998) independence and conditional-coverage p-values.

    Returns (p_independence, p_conditional_coverage). Requires the hit series
    in chronological order. Degenerate transition tables give NaN.
    """
    from scipy.stats import chi2

    hits = np.asarray(hits, dtype=int)
    n = len(hits)
    if n < 2:
        return float("nan"), float("nan")
    prev, curr = hits[:-1], hits[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    if (n01 + n11) == 0 or (n00 + n10) == 0:
        return float("nan"), float("nan")

    def safe_p(num: int, den: int) -> float:
        return min(max(num / den if den else 0.0, 1e-10), 1 - 1e-10)

    pi01 = safe_p(n01, n00 + n01)
    pi11 = safe_p(n11, n10 + n11)
    pi = safe_p(n01 + n11, n - 1)
    ll_null = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll_alt = (
        n00 * np.log(1 - pi01)
        + n01 * np.log(pi01)
        + n10 * np.log(1 - pi11)
        + n11 * np.log(pi11)
    )
    lr_ind = max(-2.0 * (ll_null - ll_alt), 0.0)
    p_ind = float(chi2.sf(lr_ind, 1))

    x = float(hits.sum())
    p0 = min(max(alpha, 1e-10), 1 - 1e-10)
    p_hat = min(max(x / n, 1e-10), 1 - 1e-10)
    lr_pof = max(
        -2.0
        * (
            ((n - x) * np.log(1 - p0) + x * np.log(p0))
            - ((n - x) * np.log(1 - p_hat) + x * np.log(p_hat))
        ),
        0.0,
    )
    p_cc = float(chi2.sf(lr_pof + lr_ind, 2))
    return p_ind, p_cc


def block_bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    block: int | None = None,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Moving-block bootstrap CI for the mean of a (serially dependent) series."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 5:
        return float("nan"), float("nan")
    if block is None:
        block = max(1, min(21, n // 10))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    max_start = n - block
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([values[s : s + block] for s in starts])[:n]
        means[b] = sample.mean()
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return lo, hi


def fiedler_order(train: pd.DataFrame, n_bins: int) -> list[str]:
    """Compute the chain order once from the first causal training window."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from mps_regime import binning

    _, _, order = binning.discretize(train, n_bins=n_bins)
    return order


def mps_samples(
    train: pd.DataFrame,
    order: list[str],
    *,
    d: int,
    chi: int,
    sweeps: int,
    n_sim: int,
    seed: int,
) -> np.ndarray:
    """Fit the repository DMRG MPS Born machine and return samples in original order."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from mps_regime import dmrg

    ordered = train[order]
    fit = dmrg.fit_dmrg(
        ordered,
        d=d,
        chi=chi,
        sweeps=sweeps,
        seed=seed,
    )
    sampled = fit.sample_returns(n_sim, seed=seed + 17, within_bin_noise=True)
    original = np.empty_like(sampled)
    positions = [list(train.columns).index(c) for c in order]
    original[:, positions] = sampled
    return original


def _model_name(chi: int) -> str:
    return f"mps_chi{int(chi)}"


def _jointtail_rows(
    *,
    book: str,
    date: pd.Timestamp,
    train_arr: np.ndarray,
    samples_by_model: dict[str, np.ndarray],
    tail_thresholds: np.ndarray,
    q_tail: float,
    k_joint: int,
) -> list[dict]:
    """Global (characteristic-independent) joint-tail and marginal diagnostics."""
    probs = {
        m: joint_tail_probability(s, q_tail, k_joint, thresholds=tail_thresholds)
        for m, s in samples_by_model.items()
    }
    rank_probs = {
        m: joint_tail_probability(s, q_tail, k_joint, thresholds=None)
        for m, s in samples_by_model.items()
    }
    floor = 1e-12
    rows = []
    for model, samples in samples_by_model.items():
        tail_rate, vol_ratio = marginal_diagnostics(samples, train_arr, q_tail)
        rows.append(
            {
                "book": book,
                "date": str(pd.Timestamp(date).date()),
                "model": model,
                "marginal_tail_rate": tail_rate,
                "marginal_vol_ratio": vol_ratio,
                "joint_tail_prob": probs[model],
                "joint_tail_prob_rank": rank_probs[model],
                "joint_tail_lift_vs_gaussian": probs[model] / max(probs["gaussian"], floor),
                "joint_tail_lift_vs_gauss_copula": probs[model] / max(probs[COPULA_MODEL], floor),
                "rank_joint_tail_lift_vs_gauss_copula": rank_probs[model]
                / max(rank_probs[COPULA_MODEL], floor),
            }
        )
    return rows


def _forecast_rows(
    *,
    book: str,
    date: pd.Timestamp,
    train: pd.DataFrame,
    actual: pd.Series,
    stats: pd.DataFrame,
    samples_by_model: dict[str, np.ndarray],
    groups: int,
    q_tail: float,
    alpha: float,
) -> list[dict]:
    rows: list[dict] = []
    actual_arr = actual.to_numpy(dtype=float)

    for char in CHARACTERISTICS:
        weights, labels = portfolio_weights(stats[char].to_numpy(), groups)
        low = weights[:, 0]
        high = weights[:, -1]
        bms = high - low
        actual_bms = float(actual_arr @ bms)
        for model, samples in samples_by_model.items():
            low_sim = samples @ low
            high_sim = samples @ high
            bms_sim = high_sim - low_sim
            var, es = var_es(bms_sim, alpha)
            ltd, utd, asym = tail_dependence_pair(low_sim, high_sim, q_tail)
            rows.append(
                {
                    "book": book,
                    "date": str(pd.Timestamp(date).date()),
                    "characteristic": char,
                    "model": model,
                    "n_train": len(train),
                    "n_assets": train.shape[1],
                    "groups": int(labels.max()) + 1,
                    "low_group_size": int((labels == labels.min()).sum()),
                    "high_group_size": int((labels == labels.max()).sum()),
                    "actual_bms_return": actual_bms,
                    "var": var,
                    "es": es,
                    "var_hit": int(actual_bms < var),
                    "fz0_loss": fz0_loss(actual_bms, var, es, alpha),
                    "lower_tail_dep": ltd,
                    "upper_tail_dep": utd,
                    "asym_tail_dep": asym,
                }
            )
    return rows


def summarize_forecasts(forecasts: pd.DataFrame, alpha: float = 0.01) -> pd.DataFrame:
    """Backtest summary per characteristic/model.

    VaR calibration: hit rate with moving-block bootstrap CI, Kupiec POF and
    Christoffersen independence/conditional-coverage p-values. Joint VaR-ES
    accuracy: mean FZ0 score on all days, and the paired daily FZ0 differential
    against the matched-marginal Gaussian copula with a block-bootstrap CI.
    """
    required = {"date", "characteristic", "model", "actual_bms_return", "var", "es", "var_hit"}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecast table missing columns: {sorted(missing)}")
    forecasts = forecasts.sort_values("date", kind="mergesort")
    if "fz0_loss" not in forecasts.columns:
        forecasts = forecasts.assign(
            fz0_loss=[
                fz0_loss(a, v, e, alpha)
                for a, v, e in zip(
                    forecasts["actual_bms_return"], forecasts["var"], forecasts["es"]
                )
            ]
        )
    out: list[dict] = []
    for (char, model), g in forecasts.groupby(["characteristic", "model"], sort=True):
        hits = g["var_hit"].to_numpy(dtype=float)
        hit_rate = float(hits.mean())
        hit_lo, hit_hi = block_bootstrap_ci(hits)
        p_ind, p_cc = christoffersen_pvalues(hits, alpha)
        row = {
            "characteristic": char,
            "model": model,
            "n_forecasts": int(len(g)),
            "var_hit_rate": hit_rate,
            "var_hit_ci_lo": hit_lo,
            "var_hit_ci_hi": hit_hi,
            "var_target_rate": alpha,
            "kupiec_p": kupiec_pvalue(hits, alpha),
            "christoffersen_ind_p": p_ind,
            "christoffersen_cc_p": p_cc,
            "mean_var": float(g["var"].mean()),
            "mean_es": float(g["es"].mean()),
            "fz0_mean": float(np.nanmean(g["fz0_loss"].to_numpy(dtype=float))),
            "mean_asym_tail_dep": float(g["asym_tail_dep"].mean())
            if "asym_tail_dep" in g
            else np.nan,
        }
        out.append(row)
    result = pd.DataFrame(out)
    if result.empty:
        return result

    # paired FZ0 differential vs the matched-marginal copula benchmark
    result["fz0_delta_vs_gauss_copula"] = np.nan
    result["fz0_delta_ci_lo"] = np.nan
    result["fz0_delta_ci_hi"] = np.nan
    for char in result["characteristic"].unique():
        sub = forecasts[forecasts["characteristic"] == char]
        pivot = sub.pivot_table(index="date", columns="model", values="fz0_loss", aggfunc="first")
        if COPULA_MODEL not in pivot.columns:
            continue
        for model in pivot.columns:
            if model == COPULA_MODEL:
                continue
            diff = (pivot[model] - pivot[COPULA_MODEL]).to_numpy(dtype=float)
            mask = (result["characteristic"] == char) & (result["model"] == model)
            result.loc[mask, "fz0_delta_vs_gauss_copula"] = float(np.nanmean(diff))
            lo, hi = block_bootstrap_ci(diff)
            result.loc[mask, "fz0_delta_ci_lo"] = lo
            result.loc[mask, "fz0_delta_ci_hi"] = hi
    return result


def summarize_jointtail(jointtail: pd.DataFrame) -> pd.DataFrame:
    """Global per-model joint-tail and marginal-calibration summary."""
    required = {"date", "model", "joint_tail_prob", "joint_tail_prob_rank"}
    missing = required - set(jointtail.columns)
    if missing:
        raise ValueError(f"jointtail table missing columns: {sorted(missing)}")
    out: list[dict] = []
    for model, g in jointtail.groupby("model", sort=True):
        row = {
            "model": model,
            "n_windows": int(len(g)),
            "mean_marginal_tail_rate": float(g["marginal_tail_rate"].mean()),
            "mean_marginal_vol_ratio": float(g["marginal_vol_ratio"].mean()),
            "mean_joint_tail_prob": float(g["joint_tail_prob"].mean()),
            "mean_joint_tail_prob_rank": float(g["joint_tail_prob_rank"].mean()),
            "mean_joint_tail_lift_vs_gaussian": float(g["joint_tail_lift_vs_gaussian"].mean()),
            "mean_joint_tail_lift_vs_gauss_copula": float(
                g["joint_tail_lift_vs_gauss_copula"].mean()
            ),
            "mean_rank_lift_vs_gauss_copula": float(
                g["rank_joint_tail_lift_vs_gauss_copula"].mean()
            ),
        }
        log_lift = np.log(
            g["rank_joint_tail_lift_vs_gauss_copula"].to_numpy(dtype=float).clip(1e-12)
        )
        lo, hi = block_bootstrap_ci(log_lift)
        row["rank_lift_log_ci_lo"] = lo
        row["rank_lift_log_ci_hi"] = hi
        out.append(row)
    return pd.DataFrame(out)


def _write_report(
    path: pathlib.Path,
    book: str,
    summary: pd.DataFrame,
    args: argparse.Namespace,
    jointtail_summary: pd.DataFrame | None = None,
) -> None:
    """Write a concise, presentation-ready interpretation scaffold."""
    lines = [
        f"# Glasgow MPS extension — {book}",
        "",
        "This output applies the higher-order comoment portfolio design of "
        "Cerrato et al. (2017) to the repository's MPS Born machine, protocol v2 "
        "(matched-marginal copula benchmark, rank/PIT joint tails, proper backtests).",
        "",
        "## Causal protocol",
        "",
        f"- trailing window: {args.window} observations",
        f"- forecast step: every {args.step} observations",
        f"- VaR level: {args.alpha:.2%}",
        f"- joint lower-tail level: {args.q_tail:.2%}; k={args.k_joint}",
        f"- MPS bins: d={args.d}; chi={','.join(map(str, args.chi_list))}",
        "- market for comoments: leave-one-out equal weight"
        if getattr(args, "market", "loo") == "loo"
        else "- market for comoments: equal weight incl. own asset (v1)",
        "- all sorting, covariance, copula, binning, and model fitting uses the trailing window only",
        "",
        "## How to read the output",
        "",
        "- gauss_copula holds train marginals fixed: differences against it are dependence, not marginal width.",
        "- marginal_tail_rate should equal the tail level if a model's marginals are calibrated; "
        "marginal_vol_ratio should be 1.",
        "- joint_tail_prob_rank uses each model's own quantiles (PIT scale) and is invariant to marginals.",
        "- rank_lift_vs_gauss_copula > 1 with a CI excluding 0 in logs is evidence of higher-order "
        "lower-tail dependence beyond the Gaussian copula.",
        "- fz0_mean is the strictly consistent joint VaR-ES score on all days (lower is better); "
        "fz0_delta_vs_gauss_copula < 0 with CI below 0 means better risk forecasts than the copula.",
        "- kupiec_p / christoffersen_cc_p below 0.05 reject correct VaR coverage/independence.",
        "",
        "## Joint-tail and marginal calibration (global, characteristic-independent)",
        "",
    ]
    if jointtail_summary is not None and not jointtail_summary.empty:
        lines.append(jointtail_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    else:
        lines.append("Not available.")
    lines += [
        "",
        "## VaR/ES backtests by characteristic",
        "",
    ]
    if summary.empty:
        lines.append("No forecasts were generated.")
    else:
        lines.append(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_book(filename: str) -> pathlib.Path | None:
    """Resolve a parquet book from the repository or DATA_EXT."""
    candidates: list[pathlib.Path] = []
    data_ext = os.environ.get("DATA_EXT")
    if data_ext:
        candidates.append(pathlib.Path(data_ext) / filename)
    for root in (
        pathlib.Path("data_external"),
        pathlib.Path("data"),
        pathlib.Path("data/raw"),
        pathlib.Path("."),
    ):
        candidates.append(root / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def run_book(book: str, path: pathlib.Path, args: argparse.Namespace) -> pd.DataFrame:
    """Run the causal rolling experiment for one repository asset book."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from mps_regime.data import to_returns

    prices = pd.read_parquet(path).astype(float)
    returns = to_returns(prices, kind="log")
    if returns.shape[1] < 2:
        raise ValueError(f"{book}: fewer than two usable assets after cleaning.")
    if len(returns) <= args.window + 1:
        raise ValueError(f"{book}: not enough observations for window={args.window}.")
    order_bins = args.order_bins if args.order_bins else args.d
    order = fiedler_order(returns.iloc[: args.window], order_bins)
    starts = list(range(args.window, len(returns) - 1, args.step))
    if args.max_windows:
        starts = starts[: args.max_windows]
    if args.shard_count > 1:
        if not 0 <= args.shard_index < args.shard_count:
            raise ValueError("shard-index must lie in [0, shard-count).")
        starts = starts[args.shard_index :: args.shard_count]
    if not starts:
        raise ValueError(f"{book}: no forecast dates available.")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    forecast_rows: list[dict] = []
    jointtail_rows: list[dict] = []
    comoment_rows: list[pd.DataFrame] = []
    t0 = time.time()
    for wi, start in enumerate(starts):
        train = returns.iloc[start - args.window : start]
        actual = returns.iloc[start]
        date = returns.index[start]
        stats = higher_order_comoments(train, market=args.market)
        train_arr = train.to_numpy(dtype=float)
        tail_thresholds = np.quantile(train_arr, args.q_tail, axis=0)
        cm = stats.reset_index().rename(columns={"index": "asset"})
        cm.insert(0, "date", str(pd.Timestamp(date).date()))
        cm.insert(0, "book", book)
        comoment_rows.append(cm)

        samples: dict[str, np.ndarray] = {
            "gaussian": gaussian_samples(train, args.n_sim, args.seed + 1000 + wi),
            COPULA_MODEL: gaussian_copula_samples(train, args.n_sim, args.seed + 3000 + wi),
            "empirical": empirical_samples(train, args.n_sim, args.seed + 2000 + wi),
        }
        for chi in args.chi_list:
            name = _model_name(chi)
            print(
                f"[{book}] {wi + 1}/{len(starts)} {date.date()} {name}",
                flush=True,
            )
            samples[name] = mps_samples(
                train,
                order,
                d=args.d,
                chi=chi,
                sweeps=args.sweeps,
                n_sim=args.n_sim,
                seed=args.seed + 10_000 + 101 * wi + chi,
            )
        jointtail_rows.extend(
            _jointtail_rows(
                book=book,
                date=date,
                train_arr=train_arr,
                samples_by_model=samples,
                tail_thresholds=tail_thresholds,
                q_tail=args.q_tail,
                k_joint=args.k_joint,
            )
        )
        forecast_rows.extend(
            _forecast_rows(
                book=book,
                date=date,
                train=train,
                actual=actual,
                stats=stats,
                samples_by_model=samples,
                groups=args.groups,
                q_tail=args.q_tail,
                alpha=args.alpha,
            )
        )
    forecasts = pd.DataFrame(forecast_rows)
    jointtail = pd.DataFrame(jointtail_rows)
    summary = summarize_forecasts(forecasts, alpha=args.alpha)
    jointtail_summary = summarize_jointtail(jointtail)
    suffix = pathlib.Path(BOOKS[book]).stem if book in BOOKS else book
    forecasts.to_csv(
        out_dir / f"glasgow_forecasts_{suffix}.csv.gz",
        index=False,
        compression="gzip",
    )
    jointtail.to_csv(
        out_dir / f"glasgow_jointtail_{suffix}.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.concat(comoment_rows, ignore_index=True).to_csv(
        out_dir / f"glasgow_comoments_{suffix}.csv",
        index=False,
    )
    summary.insert(0, "book", book)
    summary.to_csv(out_dir / f"glasgow_summary_{book}.csv", index=False)
    jointtail_summary.insert(0, "book", book)
    jointtail_summary.to_csv(out_dir / f"glasgow_jointtail_summary_{book}.csv", index=False)
    _write_report(out_dir / f"glasgow_report_{book}.md", book, summary, args, jointtail_summary)
    with open(out_dir / f"glasgow_metadata_{book}.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "paper": "Cerrato, Crosby, Kim & Zhao (2017), JEF 40, 101-120",
                "doi": "10.1016/j.jempfin.2016.11.007",
                "protocol": "v2",
                "book": book,
                "window": args.window,
                "step": args.step,
                "n_sim": args.n_sim,
                "d": args.d,
                "chi_list": args.chi_list,
                "groups": args.groups,
                "q_tail": args.q_tail,
                "k_joint": args.k_joint,
                "alpha": args.alpha,
                "order_bins": order_bins,
                "market": args.market,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "n_observations": len(returns),
                "n_assets": returns.shape[1],
                "n_forecasts": len(starts),
                "elapsed_seconds": round(time.time() - t0, 1),
            },
            fh,
            indent=2,
        )
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--books", nargs="+", choices=sorted(BOOKS), default=["gfc"])
    parser.add_argument("--out-dir", default="week8/results")
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--step", type=int, default=21)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--n-sim", type=int, default=20_000)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--chi-list", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument(
        "--order-bins",
        type=int,
        default=0,
        help="bins for the Fiedler chain ordering; 0 = use --d (default)",
    )
    parser.add_argument("--market", choices=["loo", "ew"], default="loo")
    parser.add_argument("--q-tail", type=float, default=0.05)
    parser.add_argument("--k-joint", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.smoke:
        args.books = args.books[:1]
        args.step = max(args.step, 63)
        args.max_windows = args.max_windows or 3
        args.n_sim = min(args.n_sim, 5_000)
        args.sweeps = min(args.sweeps, 2)
        args.chi_list = [1, 2]
        print(
            f"SMOKE: books={args.books} windows={args.max_windows} "
            f"step={args.step} n_sim={args.n_sim} chi={args.chi_list}",
            flush=True,
        )
    summaries = []
    for book in args.books:
        path = resolve_book(BOOKS[book])
        if path is None:
            raise FileNotFoundError(
                f"{BOOKS[book]} not found. Set DATA_EXT to the directory containing the parquet books."
            )
        summaries.append(run_book(book, path, args))
    if summaries:
        all_summary = pd.concat(summaries, ignore_index=True)
        out_dir = pathlib.Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        all_summary.to_csv(out_dir / "glasgow_summary.csv", index=False)
        print(f"saved {out_dir / 'glasgow_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
