"""Markov-switching Born machine (MS-Born): a K-state HMM whose emissions are MPS Born machines.

The regime-detection engine of the module. Each hidden state k holds its own Born machine
P_k(x) = |psi_k(x)|^2 / Z_k over the day's quantile-binned return vector; a sticky Markov chain
switches between them. Estimated by EM:

    E-step : exact per-day log-likelihoods under every emission (analytic, no sampling) ->
             forward-backward posteriors (log-space, numerically safe).
    M-step : each Born machine refit by POSTERIOR-WEIGHTED two-site DMRG
             (`fit_dmrg(..., weights=gamma[:, k])`), warm-started from its previous cores;
             sticky-MAP transition update (Dirichlet pseudo-count kappa on the diagonal).

Design decisions (each is a causality or comparability requirement, not a convenience):
  * ONE shared discretization, fit on the TRAINING period only. All K emissions must live on the
    same grid or their likelihoods are not comparable; freezing edges at train time keeps every
    out-of-sample filtered probability causal.
  * `filter()` is strictly causal (forward recursion only) — use it for anything evaluated
    out-of-sample. `smooth()`/`viterbi()` use the full sample — descriptive plots only.
  * Optional TEMPORAL site (`lag_sites=1`): appends yesterday's equal-weight portfolio return,
    binned on train-period edges, as an extra chain site. This gives emissions a channel for
    serial dependence, so states can differentiate momentum-like (today's joint distribution
    shifted in the direction of yesterday's move) from mean-reverting regimes — dependence the
    contemporaneous cross-section cannot express. The Gaussian twin receives the same column, so
    the Born-vs-Gaussian ablation stays apples-to-apples.
  * `emission="gaussian"` swaps the emission family for a weighted-MLE multivariate normal on the
    SAME columns with identical transition machinery — the ablation isolating what higher-order
    (beyond-covariance) structure contributes to regime detection. Cross-family NLLs are not
    comparable (discrete vs continuous measure); compare DETECTION metrics, never likelihoods.

Regime identification beyond "crisis": `regime_cards()` characterizes each state by annualized
vol, mean return, within-regime AR(1) of the equal-weight portfolio (momentum vs mean-reversion),
average pairwise correlation, and the emission's own tail diagnostics (joint-tail-state
probability, k-way deep-corner mass, entanglement entropy) — the states are discovered by the
data and *named* afterward from the card, never imposed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import dmrg as _dmrg
from .mps import MPSFit, _discretize

LOGEPS = -745.0  # log of smallest positive double; floor for log-space recursions


# --------------------------------------------------------------------------- #
# log-space HMM machinery                                                       #
# --------------------------------------------------------------------------- #
def _logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return float(out.ravel()[0]) if axis is None else np.squeeze(out, axis=axis)


def forward_backward(logB, logA, logpi):
    """Exact posteriors for an HMM. logB (T,K) emission log-liks; logA (K,K); logpi (K,).
    Returns (gamma (T,K), xi_sum (K,K), loglik) — xi_sum = sum_t P(s_t=i, s_t+1=j | data)."""
    T, K = logB.shape
    la = np.zeros((T, K))
    la[0] = logpi + logB[0]
    for t in range(1, T):
        la[t] = logB[t] + _logsumexp(la[t - 1][:, None] + logA, axis=0)
    loglik = float(_logsumexp(la[-1]))
    lb = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        lb[t] = _logsumexp(logA + (logB[t + 1] + lb[t + 1])[None, :], axis=1)
    lg = la + lb - loglik
    gamma = np.exp(np.clip(lg, LOGEPS, 0.0))
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi_sum = np.zeros((K, K))
    for t in range(T - 1):
        lxi = la[t][:, None] + logA + (logB[t + 1] + lb[t + 1])[None, :] - loglik
        xi_sum += np.exp(np.clip(lxi, LOGEPS, 0.0))
    return gamma, xi_sum, loglik


def forward_filter(logB, logA, logpi):
    """STRICTLY CAUSAL filtered probabilities P(s_t | x_1..x_t). Returns (T,K)."""
    T, K = logB.shape
    out = np.zeros((T, K))
    la = logpi + logB[0]
    out[0] = np.exp(la - _logsumexp(la))
    for t in range(1, T):
        la = logB[t] + _logsumexp(la[:, None] + logA, axis=0)
        la -= _logsumexp(la)          # renormalize in log space (prevents drift)
        out[t] = np.exp(la)
    return out


def viterbi(logB, logA, logpi):
    T, K = logB.shape
    delta = logpi + logB[0]
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        cand = delta[:, None] + logA
        back[t] = np.argmax(cand, axis=0)
        delta = logB[t] + np.max(cand, axis=0)
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1][path[t + 1]]
    return path


# --------------------------------------------------------------------------- #
# Emission families                                                             #
# --------------------------------------------------------------------------- #
class _GaussianEmission:
    """Weighted-MLE multivariate normal emission (the covariance-only ablation twin).
    diag=True zeroes the off-diagonal covariance — the INDEPENDENCE reference within the
    continuous measure, used by the I - I_Gauss daily index (the full-vs-diag likelihood gap
    is the Gaussian dependence information; densities' scale cancels in the difference)."""

    def __init__(self, ridge: float = 1e-8, diag: bool = False):
        self.mu = None
        self.cov = None
        self.ridge = ridge
        self.diag = diag

    def fit(self, R: np.ndarray, w: np.ndarray):
        w = np.asarray(w, dtype=float)
        w = w / w.sum().clip(1e-300)
        self.mu = w @ R
        Xc = R - self.mu
        self.cov = (Xc * w[:, None]).T @ Xc
        if self.diag:
            self.cov = np.diag(np.diag(self.cov))
        # absolute floor added 2026-07-14 (review item e, BUG-LOG #18): trace-scaled ridge alone
        # vanishes if trace==0 (degenerate weights/constant returns) -> -inf logdet.
        scale = max(float(np.trace(self.cov)) / R.shape[1], 1e-12)
        self.cov += self.ridge * scale * np.eye(R.shape[1])
        return self

    def logpdf(self, R: np.ndarray) -> np.ndarray:
        n = R.shape[1]
        sign, logdet = np.linalg.slogdet(self.cov)
        prec = np.linalg.pinv(self.cov)
        Xc = R - self.mu
        maha = np.einsum("ti,ij,tj->t", Xc, prec, Xc)
        return -0.5 * (n * np.log(2 * np.pi) + logdet + maha)


# --------------------------------------------------------------------------- #
# The model                                                                     #
# --------------------------------------------------------------------------- #
class MarkovSwitchingBorn:
    """K-state HMM with Born-machine (or Gaussian) emissions on daily multi-asset returns.

    Parameters
    ----------
    n_states  : number of hidden regimes K (use `select_k` for a data-driven choice).
    d, chi    : Born-machine bins per site and bond dimension.
    emission  : "born" (MPS Born machine) or "gaussian" (weighted-MLE MVN ablation twin).
    lag_sites : 0 or 1 — append yesterday's binned equal-weight return as an extra site
                (temporal channel: momentum vs mean-reversion separation).
    sticky    : Dirichlet pseudo-count added to the transition-matrix DIAGONAL in the M-step.
                Larger = more persistent regimes, less chatter. In units of observations.
    em_iters  : EM iterations. em_sweeps: DMRG sweeps per M-step refit (warm-started, so small).
    init      : "vol" (structured: trailing-vol quantile split — deterministic) or "random".
    n_init    : extra random restarts (best train log-lik wins). Ignored for init="vol" alone.
    seed      : RNG seed for restarts and any resampling.
    """

    def __init__(self, n_states=2, d=8, chi=8, emission="born", lag_sites=0,
                 sticky=20.0, em_iters=6, em_sweeps=4, init_sweeps=10,
                 init="vol", n_init=2, seed=0, min_state_weight=20.0, verbose=False):
        assert emission in ("born", "gaussian", "gaussian_diag")
        assert lag_sites in (0, 1)
        self.K = int(n_states)
        self.d, self.chi = int(d), int(chi)
        self.emission = emission
        self.lag_sites = int(lag_sites)
        self.sticky = float(sticky)
        self.em_iters, self.em_sweeps, self.init_sweeps = em_iters, em_sweeps, init_sweeps
        self.init, self.n_init, self.seed = init, int(n_init), int(seed)
        self.min_state_weight = float(min_state_weight)
        self.verbose = verbose
        # learned state
        self.emissions_: list = []
        self.logA_ = None
        self.logpi_ = None
        self.edges_ = None          # per-site bin edges (train-period, frozen)
        self.columns_ = None
        self.train_index_ = None
        self.train_loglik_ = None
        self.history_ = []

    # ---- data plumbing -------------------------------------------------------
    def _extend(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Append the lag site column (yesterday's EW return) if configured; drops first row."""
        if not self.lag_sites:
            return returns
        if self.columns_ is not None:                       # inference: freeze the EW basket
            assets = [c for c in self.columns_ if c != "_LAG_EW"]
            returns = returns[assets]
        ew = returns.mean(axis=1)
        ext = returns.iloc[1:].copy()
        ext["_LAG_EW"] = ew.shift(1).iloc[1:]
        return ext

    def _fit_edges(self, ext_train: pd.DataFrame):
        """Quantile bin edges per site from TRAIN data only. Also stores the train support
        (per-site min/max) so end-bin widths are defined for the PMF->density conversion."""
        _, edges, bin_values, bin_std = _discretize(ext_train, self.d)
        self.edges_, self._bin_values, self._bin_std = edges, bin_values, bin_std
        self._train_lo = ext_train.min(axis=0).to_numpy()
        self._train_hi = ext_train.max(axis=0).to_numpy()

    def _encode(self, ext: pd.DataFrame) -> np.ndarray:
        """Bin with FROZEN train edges (out-of-range days land in the end bins — causal)."""
        vals = ext.to_numpy()
        if np.isnan(vals).any():
            raise ValueError(f"NaN returns passed to encode() ({int(np.isnan(vals).sum())} cells)"
                             " — clean the panel first; NaN would silently bin as extreme-up")
        codes = np.zeros(ext.shape, dtype=np.int64)
        for j in range(ext.shape[1]):
            codes[:, j] = np.clip(np.digitize(vals[:, j], self.edges_[j]), 0, self.d - 1)
        return codes

    def _emission_loglik(self, ext: pd.DataFrame, codes: np.ndarray) -> np.ndarray:
        logB = np.zeros((len(ext), self.K))
        for k, em in enumerate(self.emissions_):
            if self.emission == "born":
                logB[:, k] = em.log_prob(codes)
            else:
                logB[:, k] = em.logpdf(ext.to_numpy())
        return np.clip(logB, LOGEPS, -LOGEPS)   # ceil guards singular-cov +inf

    # ---- init ----------------------------------------------------------------
    def _init_gamma(self, ext: pd.DataFrame, rng, mode) -> np.ndarray:
        T = len(ext)
        g = np.full((T, self.K), 1.0 / self.K)
        if mode == "vol":
            rv = ext.drop(columns=["_LAG_EW"], errors="ignore").mean(axis=1) \
                    .rolling(21, min_periods=5).std().bfill()
            qs = pd.qcut(rv, self.K, labels=False, duplicates="drop").to_numpy()
            qs = np.nan_to_num(qs, nan=0).astype(int).clip(0, self.K - 1)
            g[:] = 0.05 / max(self.K - 1, 1)
            g[np.arange(T), qs] = 0.95
        else:
            raw = rng.random((T, self.K)) + 0.5
            g = raw / raw.sum(axis=1, keepdims=True)
        return g

    def _mstep_emissions(self, ext, codes, gamma, sweeps, prev=None):
        ems = []
        for k in range(self.K):
            w = gamma[:, k]
            if w.sum() < self.min_state_weight:      # collapsed state: reseed on worst-fit days
                if self.verbose:
                    print(f"  state {k} collapsed (w={w.sum():.1f}) — reseeding")
                rng_k = np.random.default_rng(self.seed + 7919 * (k + 1))
                w = np.full(len(ext), 1e-3) + 1e-4 * rng_k.random(len(ext))
                if self.emissions_:
                    cur = self._emission_loglik(ext, codes)
                    worst = np.argsort(cur.max(axis=1))[: max(len(ext) // 10, 30)]
                    w[worst] = 1.0 + 0.1 * rng_k.random(worst.size)
            if self.emission == "born":
                init_cores = prev[k].cores if prev is not None else None
                fit = _dmrg.fit_dmrg(
                    ext, d=self.d, chi=self.chi, sweeps=sweeps, seed=self.seed + k,
                    weights=w, init_cores=init_cores,
                    discretization=(codes, self.edges_, self._bin_values, self._bin_std))
                ems.append(fit)
            else:
                ems.append(_GaussianEmission(diag=(self.emission == "gaussian_diag"))
                           .fit(ext.to_numpy(), w))
        return ems

    # ---- fit -----------------------------------------------------------------
    def fit(self, returns_train: pd.DataFrame):
        ext = self._extend(returns_train)
        self.columns_ = list(ext.columns)
        self.train_index_ = ext.index
        self._fit_edges(ext)
        codes = self._encode(ext)

        inits = ([("vol", None)] if self.init == "vol" else []) + \
            [("random", i) for i in range(max(self.n_init, 0 if self.init == "vol" else 1))]
        best = None
        for mode, ridx in inits:
            rng = np.random.default_rng(self.seed if ridx is None else self.seed + 1000 + ridx)
            gamma = self._init_gamma(ext, rng, mode)
            logA = np.log(np.full((self.K, self.K), 1.0 / self.K))
            logpi = np.log(np.full(self.K, 1.0 / self.K))
            self.emissions_ = []
            hist = []
            for it in range(self.em_iters):
                sweeps = self.init_sweeps if it == 0 else self.em_sweeps
                self.emissions_ = self._mstep_emissions(
                    ext, codes, gamma, sweeps,
                    prev=self.emissions_ if (it > 0 and self.emission == "born") else None)
                logB = self._emission_loglik(ext, codes)
                gamma, xi_sum, ll = forward_backward(logB, logA, logpi)
                # sticky MAP transition update
                A = xi_sum + self.sticky * np.eye(self.K) + 1e-3
                A /= A.sum(axis=1, keepdims=True)
                logA = np.log(A)
                pi = gamma[0] + 1e-3
                logpi = np.log(pi / pi.sum())
                hist.append(ll)
                if self.verbose:
                    print(f"[{mode}{'' if ridx is None else ridx}] EM iter {it}: loglik {ll:.2f}")
                if it > 0 and abs(hist[-1] - hist[-2]) < 1e-4 * abs(hist[-2]):
                    break
            if best is None or hist[-1] > best[0]:
                best = (hist[-1], [e for e in self.emissions_], logA.copy(), logpi.copy(),
                        gamma.copy(), list(hist))
        self.train_loglik_, self.emissions_, self.logA_, self.logpi_, self._gamma_train, \
            self.history_ = best
        # canonical state order: ascending emission "stress" (tail-state mass for born,
        # generalized variance for gaussian) so state 0 is always the calmest — reproducible labels
        order = np.argsort(self._state_stress())
        self._reorder(order)
        return self

    def _state_stress(self):
        out = []
        for em in self.emissions_:
            if self.emission == "born":
                from .features import tail_state_probability
                out.append(tail_state_probability(em))
            else:
                out.append(float(np.linalg.slogdet(em.cov)[1]))
        return np.asarray(out)

    def _reorder(self, order):
        self.emissions_ = [self.emissions_[i] for i in order]
        P = np.eye(self.K)[order]
        self.logA_ = np.log(np.clip(P @ np.exp(self.logA_) @ P.T, 1e-300, None))
        self.logpi_ = np.log(np.clip(P @ np.exp(self.logpi_), 1e-300, None))
        self._gamma_train = self._gamma_train[:, order]

    # ---- inference -----------------------------------------------------------
    def _prep(self, returns: pd.DataFrame):
        ext = self._extend(returns)[self.columns_]     # enforce training column order
        return ext, self._encode(ext)

    def filter(self, returns: pd.DataFrame) -> pd.DataFrame:
        """CAUSAL filtered regime probabilities for each day (out-of-sample safe)."""
        ext, codes = self._prep(returns)
        probs = forward_filter(self._emission_loglik(ext, codes), self.logA_, self.logpi_)
        return pd.DataFrame(probs, index=ext.index,
                            columns=[f"regime_{k}" for k in range(self.K)])

    def smooth(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Two-sided smoothed probabilities (descriptive use ONLY — sees the future)."""
        ext, codes = self._prep(returns)
        gamma, _, _ = forward_backward(self._emission_loglik(ext, codes),
                                       self.logA_, self.logpi_)
        return pd.DataFrame(gamma, index=ext.index,
                            columns=[f"regime_{k}" for k in range(self.K)])

    def viterbi_path(self, returns: pd.DataFrame) -> pd.Series:
        ext, codes = self._prep(returns)
        path = viterbi(self._emission_loglik(ext, codes), self.logA_, self.logpi_)
        return pd.Series(path, index=ext.index, name="regime")

    def predictive_daily_loglik(self, returns: pd.DataFrame) -> pd.Series:
        """STRICTLY CAUSAL per-day one-step predictive log-likelihood log p(x_t | x_1..t-1)
        under the fitted HMM (forward recursion normalizers). This is the per-day quantity
        behind the Delta_HO higher-order index (Safaai & Marin Vargas 2026, Eq. 8): the
        walk-forward difference between a Born-emission model's series and its Gaussian
        twin's estimates the daily information excess I - I_Gauss beyond covariance
        (Calsaverini & Vicente 2009). NOTE: cross-emission-family differences are meaningful
        only when both models share the same conditioning protocol — use the same walk-forward
        refits and identical columns/lag settings for both."""
        ext, codes = self._prep(returns)
        logB = self._emission_loglik(ext, codes)
        T, K = logB.shape
        out = np.zeros(T)
        la = self.logpi_ + logB[0]
        out[0] = _logsumexp(la)
        for t in range(1, T):
            la_norm = la - _logsumexp(la)                 # log filtered posterior at t-1
            la = logB[t] + _logsumexp(la_norm[:, None] + self.logA_, axis=0)
            out[t] = _logsumexp(la)                       # log p(x_t | x_1..t-1)
        if self.emission == "born":
            # PMF -> piecewise-uniform DENSITY: subtract the log bin-widths of the cells each
            # day landed in. Without this, a Born-minus-Gaussian daily difference carries a
            # day-varying Jacobian term that is LARGER on extreme days (tail bins are wider),
            # which would manufacture spurious crisis spikes in the Delta_HO index. End-bin
            # widths use the train-period support (documented approximation). This correction
            # is constant across hidden states, so it does not affect filter/smooth/viterbi.
            out = out - self._log_cell_widths(codes)
        return pd.Series(out, index=ext.index, name="daily_ll")

    def _log_cell_widths(self, codes: np.ndarray) -> np.ndarray:
        """Sum over sites of log(width of the bin the day fell into). (T,) array."""
        n_sites = codes.shape[1]
        W = np.zeros((n_sites, self.d))
        for j in range(n_sites):
            e = np.asarray(self.edges_[j], dtype=float)
            lo = min(self._train_lo[j], e[0])
            hi = max(self._train_hi[j], e[-1])
            bounds = np.concatenate([[lo], e, [hi]])
            W[j] = np.clip(np.diff(bounds), 1e-10, None)
        logw = np.log(W)
        return np.array([logw[np.arange(n_sites), codes[t]].sum() for t in range(codes.shape[0])])

    def heldout_loglik(self, returns: pd.DataFrame) -> float:
        """Per-day held-out log-likelihood (for K selection). Same emission family only."""
        ext, codes = self._prep(returns)
        logB = self._emission_loglik(ext, codes)
        T, K = logB.shape
        la = self.logpi_ + logB[0]
        total = 0.0
        for t in range(1, T):
            c = _logsumexp(la)
            total += c
            la = logB[t] + _logsumexp((la - c)[:, None] + self.logA_, axis=0)
        total += _logsumexp(la)
        return total / T

    # ---- interpretation ------------------------------------------------------
    def regime_cards(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Characterize each regime so it can be NAMED from evidence (crisis / calm /
        momentum / mean-reversion / dispersion ...). Uses viterbi segments of `returns`."""
        from .features import (entanglement_entropy, tail_state_probability,
                               tail_dependence_asymmetry)
        ext, _ = self._prep(returns)
        asset_cols = [c for c in ext.columns if c != "_LAG_EW"]
        path = self.viterbi_path(returns).to_numpy()
        ew = ext[asset_cols].mean(axis=1).to_numpy()
        rows = []
        for k in range(self.K):
            m = path == k
            seg_ar1 = np.nan
            if m.sum() > 10:
                # AR(1) inside contiguous segments only (no cross-regime contamination)
                num = den = 0.0
                start = None
                for t in range(len(m)):
                    if m[t] and start is None:
                        start = t
                    if (not m[t] or t == len(m) - 1) and start is not None:
                        end = t + 1 if m[t] else t
                        seg = ew[start:end]
                        if len(seg) > 5:
                            x, y = seg[:-1] - seg[:-1].mean(), seg[1:] - seg[1:].mean()
                            num += (x * y).sum()
                            den += (x * x).sum()
                        start = None
                seg_ar1 = num / den if den > 0 else np.nan
            sub = ext[asset_cols].to_numpy()[m]
            C = np.corrcoef(sub.T) if m.sum() > 20 else np.full((len(asset_cols),) * 2, np.nan)
            iu = np.triu_indices(len(asset_cols), 1)
            row = dict(regime=k, occupancy=float(m.mean()),
                       ann_vol=float(ew[m].std() * np.sqrt(252)) if m.any() else np.nan,
                       ann_mean=float(ew[m].mean() * 252) if m.any() else np.nan,
                       ew_ar1=float(seg_ar1),
                       avg_corr=float(np.nanmean(C[iu])),
                       expected_days=float(1.0 / max(1e-12, 1 - np.exp(self.logA_[k, k]))))
            if self.emission == "born":
                em = self.emissions_[k]
                row.update(tail_state=float(tail_state_probability(em)),
                           entanglement=float(entanglement_entropy(em)),
                           tail_asym=float(tail_dependence_asymmetry(em)))
            rows.append(row)
        return pd.DataFrame(rows).set_index("regime")


def select_k(returns_train: pd.DataFrame, returns_val: pd.DataFrame, ks=(2, 3, 4),
             **kwargs) -> pd.DataFrame:
    """Fit MS-Born for each K on train, score per-day held-out log-lik on validation."""
    rows = []
    for k in ks:
        m = MarkovSwitchingBorn(n_states=k, **kwargs).fit(returns_train)
        rows.append(dict(K=k, train_loglik=m.train_loglik_,
                         heldout_per_day=m.heldout_loglik(returns_val)))
    return pd.DataFrame(rows).set_index("K")
