"""MPS feature dictionary — the regime / tail-complexity indicators read off each fit.

Each function takes an `MPSFit` (and possibly the recent window / returns block) and returns a scalar
for that rolling window. Stacked over windows these become the candidate feature time-series. The set
covers every "candidate feature" the sponsor lists.

F1  bond dimension at fit quality (effective chi)   F6  tail-state probability
F2  entanglement entropy (one cut)                  F7  parameter / tensor drift
F3  entropy profile (all cuts)                       F8  ordering sensitivity
F4  pairwise mutual information                       F9  reconstruction error (fit-quality test)
F5  NLL of recent returns                             F10 compression ratio

IDENTIFIABILITY (2026-07 external review, accepted — see docs/Response-to-Review.md):
The training objective depends on |psi| only, so any functional of the SIGNED fitted state is NOT
determined by the data distribution P = psi^2/Z — two fits of identical likelihood can report very
different values (verified: same P, S = 0 vs ln 2). Affected — fit diagnostics ONLY, not market
features: F1 (exp-entropy form), F2, F3, F8, F13 correlation_length, F16 schmidt_utilization;
F7 is additionally gauge-noisy for cold starts. Probability-identified alternatives:
`block_mutual_information` (replaces F2 as the cut-dependence gauge), `required_chi` (replaces
F1 — held-out NLL over a chi grid). All P-functionals (F4, F5, F6, F9, F11, F12, F14, F15) are
unaffected.
"""

from __future__ import annotations

import itertools

import numpy as np

from .mps import MPSFit


# --- F1: effective bond dimension (the "chi required" complexity dial) --------------------------
def bond_dimension_at_quality(fit: MPSFit, cut: int | None = None) -> float:
    """F1: EFFECTIVE bond dimension = exp(entanglement entropy) at a cut.

    NOT PROBABILITY-IDENTIFIED (fit diagnostic only): inherits the sign non-identifiability of
    `entanglement_entropy` — see the module header and docs/Response-to-Review.md. For the
    P-identified version of F1 use `required_chi` (held-out NLL over a chi grid)."""
    return float(np.exp(fit.entanglement_entropy(cut)))


def required_chi(returns_block, chi_grid=(1, 2, 4, 8, 16), d=3, sweeps=200, tol=0.01) -> int:
    """F1 (exact): smallest chi whose NLL is within `tol` of the best on the grid — the true
    'bond dimension required for a given fit quality'. Refits per chi (expensive; not for the
    per-window rolling loop — use `bond_dimension_at_quality` there)."""
    from .mps import fit_born_machine
    nlls = {c: fit_born_machine(returns_block, d=d, chi=c, sweeps=sweeps, seed=0).nll_history[-1]
            for c in chi_grid}
    best = min(nlls.values())
    for c in sorted(chi_grid):
        if nlls[c] <= best + tol:
            return int(c)
    return int(max(chi_grid))


# --- F2 / F3: entanglement (wrap the exact MPSFit methods) --------------------------------------
def entanglement_entropy(fit: MPSFit, cut: int | None = None) -> float:
    """F2: von Neumann entropy across a chain cut (Schmidt/SVD). S <= ln(chi).

    NOT PROBABILITY-IDENTIFIED (fit diagnostic only): computed from the SIGNED fitted state; the
    likelihood observes |psi| only, so equal-likelihood fits can report different S (verified:
    same P, S = 0 vs ln 2; rolling series shows saturation + lag-1 autocorr ~0.11 vs 0.96-0.98
    for P-functionals from the same fits). Use `block_mutual_information` as the identified
    cut-dependence gauge. See docs/Response-to-Review.md."""
    return float(fit.entanglement_entropy(cut))


def entropy_profile(fit: MPSFit) -> np.ndarray:
    """F3: entanglement entropy at EVERY cut — the localization signal.

    NOT PROBABILITY-IDENTIFIED (fit diagnostic only): same caveat as F2."""
    return fit.entanglement_profile()


# --- F4: mutual information (nonlinear dependence) ----------------------------------------------
def mutual_information(fit: MPSFit, i: int, j: int) -> float:
    """F4: I(i;j) in nats from the exact 1- and 2-site marginals — captures nonlinear dependence,
    not just linear correlation."""
    a, b = (i, j) if i < j else (j, i)
    p1 = fit.one_site_marginals()
    p2 = fit.two_site_marginal(a, b)
    outer = np.outer(p1[a], p1[b])
    m = (p2 > 1e-15) & (outer > 1e-15)
    return float((p2[m] * np.log(p2[m] / outer[m])).sum())


def mean_mutual_information(fit: MPSFit, adjacent: bool = True) -> float:
    """F4 scalar feature: mean pairwise MI. `adjacent`=True averages over neighbouring chain pairs
    (cheap, N-1 pairs); False averages over ALL pairs (O(N^2))."""
    N = fit.n_sites
    pairs = ([(k, k + 1) for k in range(N - 1)] if adjacent
             else [(i, j) for i in range(N) for j in range(i + 1, N)])
    return float(np.mean([mutual_information(fit, i, j) for i, j in pairs]))


# --- F17: block mutual information (the PROBABILITY-IDENTIFIED cut-dependence gauge) -------------
def block_mutual_information(fit: MPSFit, cut: int | None = None,
                             n_samples: int = 20000, seed: int = 0) -> float:
    """F17: I(left block; right block) across a chain cut, in nats — the probability-identified
    replacement for entanglement entropy (F2), adopted per the 2026-07 review.

    I(L;R) = E_{x~P}[ log P(x) - log P_L(x_L) - log P_R(x_R) ], estimated with the exact perfect
    sampler for the expectation and EXACT block-marginal log-probabilities from the same left/right
    Gram environments the two-site marginals use: P_L(x_L) = v^T Renv[cut] v / Z with
    v = prod_{k<cut} A^(k)[x_k], and symmetrically for P_R. Because every ingredient is a functional
    of P (Gram environments square the cores), the value is invariant to sign reassignments of the
    fitted state. It is also a lower bound: I(L;R) <= 2 S(psi) for any psi with psi^2 = P.
    Cost O(n_samples * N * d * chi^2). Validated against exact enumeration (d^N) on small books.
    """
    N = fit.n_sites
    if cut is None:
        cut = N // 2
    if not (0 < cut < N):
        raise ValueError(f"cut must be interior (1..{N-1}), got {cut}")
    cores = fit._cores3d()
    Lenv, Renv, Z = fit._environments(cores)
    X = fit.sample_bins(int(n_samples), seed=seed)
    logP = fit.log_prob(X)
    logZ = float(np.log(Z))

    def _block_logp(idx_slice, env, side):
        n = X.shape[0]
        if side == "left":
            ks = range(0, cut)
            vec = cores[ks.start][0, X[:, ks.start], :]                    # (n, chi)
            logs = np.zeros(n)
            for k in list(ks)[1:]:
                slc = cores[k][:, X[:, k], :].transpose(1, 0, 2)           # (n, chi_l, chi_r)
                vec = np.einsum("nl,nlr->nr", vec, slc)
                nrm = np.linalg.norm(vec, axis=1, keepdims=True); nrm[nrm < 1e-300] = 1e-300
                logs += np.log(nrm[:, 0]); vec = vec / nrm
        else:
            ks = list(range(cut, N))
            vec = cores[ks[-1]][:, X[:, ks[-1]], 0].T                      # (n, chi)
            logs = np.zeros(n)
            for k in reversed(ks[:-1]):
                slc = cores[k][:, X[:, k], :].transpose(1, 0, 2)
                vec = np.einsum("nlr,nr->nl", slc, vec)
                nrm = np.linalg.norm(vec, axis=1, keepdims=True); nrm[nrm < 1e-300] = 1e-300
                logs += np.log(nrm[:, 0]); vec = vec / nrm
        quad = np.einsum("ni,ij,nj->n", vec, env, vec)
        quad = np.clip(quad, 1e-300, None)
        return np.log(quad) + 2.0 * logs - logZ

    logPL = _block_logp(None, Renv[cut], "left")
    logPR = _block_logp(None, Lenv[cut], "right")
    return float(np.mean(logP - logPL - logPR))


# --- F5: recent-return surprise (change-point) --------------------------------------------------
def recent_nll(fit: MPSFit, recent_samples: np.ndarray) -> float:
    """F5: NLL of the most-recent returns under the fit ('how surprising does now look to a model
    trained on the window?')."""
    return float(-fit.log_prob(recent_samples).mean())


# --- F6: joint-tail probability -----------------------------------------------------------------
def tail_state_probability(fit: MPSFit) -> float:
    """F6: model probability of the joint-extreme corner (every asset in its bottom bin)."""
    return float(fit.tail_state_probability())


# --- F7: tensor-parameter drift (regime-break signal) -------------------------------------------
def parameter_drift(fit_t: MPSFit, fit_tm1: MPSFit) -> float:
    """F7: relative L2 distance between the current and previous (warm-started) tensors.

    WITHDRAWN AS A MARKET FEATURE (2026-07-14 review, BUG-LOG #18): raw-tensor distance is
    gauge/sign-dominated. The committed rolling series was computed COLD-START, where this is pure
    noise: median 1.413 = sqrt(2) (independent random tensors), lag-1 autocorr ~0. Even warm-started
    it mixes gauge drift with distribution drift. Use `distribution_drift` (probability-identified)
    instead. Kept only as a warm-start optimizer diagnostic.
    """
    num = den = 0.0
    for a, b in zip(fit_t.cores, fit_tm1.cores):
        if a.shape != b.shape:
            return float("nan")
        num += float(((a - b) ** 2).sum())
        den += float((b ** 2).sum())
    return float(np.sqrt(num / den)) if den > 0 else 0.0


# --- F7 replacement: probability-identified distribution drift ----------------------------------
def distribution_drift(fit_t: MPSFit, fit_tm1: MPSFit, returns_window_t) -> float:
    """F7 (identified): cross-window predictive deterioration
        drift = mean_t-window[ log P_t(x) - log P_{t-1}(x) ],
    each fit scoring the SAME window-t days through its OWN bin edges (encode) — a functional of
    the two fitted distributions only (sign/gauge invariant), estimating KL-like divergence plus
    genuine data movement. 0 when the learned distribution is unchanged. Replaces raw-tensor
    parameter_drift per the 2026-07 review (BUG-LOG #18)."""
    lp_t = fit_t.log_prob(fit_t.encode(returns_window_t))
    lp_p = fit_tm1.log_prob(fit_tm1.encode(returns_window_t))
    return float((lp_t - lp_p).mean())


# --- F8: ordering sensitivity (itself a novel feature) ------------------------------------------
def ordering_sensitivity(returns_block, d: int = 3, chi: int = 8, sweeps: int = 150,
                         n_orderings: int = 4, seed: int = 0, fitter: str = "adam") -> float:
    """F8: dispersion (std) of the central-cut entanglement across random asset orderings.

    NOT PROBABILITY-IDENTIFIED (fit diagnostic only): a dispersion OF the non-identified F2, so it
    conflates genuine ordering effects with optimizer sign-slack. See docs/Response-to-Review.md. Refits COLD (seed 0, no warm start) once per
    ordering, so the number is EXPENSIVE (n_orderings full fits per call) and NOISY (a std over
    only n_orderings draws); compute on a coarse cadence or behind a flag, not every window.

    fitter="adam" uses mps.fit_born_machine (`sweeps` = Adam steps, ~150+); fitter="dmrg" uses
    dmrg.fit_dmrg (`sweeps` = full DMRG sweeps, ~8-16) — use the same trainer as the pipeline
    whose sensitivity you are measuring."""
    rng = np.random.default_rng(seed)
    cols = list(returns_block.columns)
    if fitter == "dmrg":
        from .dmrg import fit_dmrg as _fit
        fit_one = lambda blk: _fit(blk, d=d, chi=chi, sweeps=sweeps, seed=0)
    else:
        from .mps import fit_born_machine
        fit_one = lambda blk: fit_born_machine(blk, d=d, chi=chi, sweeps=sweeps, seed=0)
    vals = []
    for _ in range(n_orderings):
        order = list(rng.permutation(cols))
        vals.append(fit_one(returns_block[order]).entanglement_entropy())
    return float(np.std(vals))


# --- F9: reconstruction error (fit-quality test) ------------------------------------------------
def reconstruction_error(fit: MPSFit, returns_block, vs_raw: bool = False) -> float:
    """F9: mean absolute off-diagonal error between the MPS-implied correlation matrix and the
    window's BINNED empirical correlation — the correlation of the bin-mean-mapped codes, i.e. the
    'achievable corr' benchmark of scripts/dmrg_ordering_benchmark.py. The MPS only ever sees the
    d-bin codes, so this is the honest fit-quality target: comparing against the RAW-return
    correlation conflates discretization loss (structural, ~constant) with fit failure (the
    signal). Low => the chi cap captures the (binned) dependence; a RISE means the structure got
    richer than the bond dimension can hold — itself a regime signal.

    vs_raw=True restores the pre-fix behavior (benchmark vs raw correlation, discretization loss
    included) — kept only for comparability with runs produced before this fix."""
    C_mps = fit.mps_correlation()
    iu = np.triu_indices(fit.n_sites, 1)
    if vs_raw:
        C_emp = returns_block[fit.columns].corr().to_numpy()
    else:
        codes = fit.encode(returns_block)
        mapped = np.column_stack([fit.bin_values[j][codes[:, j]] for j in range(fit.n_sites)])
        C_emp = np.corrcoef(mapped.T)
    return float(np.abs(C_mps[iu] - C_emp[iu]).mean())


# --- F10: compression ratio ---------------------------------------------------------------------
def compression_ratio(fit: MPSFit) -> float:
    """F10: orders of magnitude the MPS compresses the joint table — log10(d^N / (N*d*chi^2)).
    A structural scalar (≈constant for a given d, N, chi), reported as a headline rather than used
    as a rolling feature."""
    n_params = fit.n_sites * fit.d * fit.chi ** 2
    log_full = fit.n_sites * np.log10(fit.d)
    return float(log_full - np.log10(n_params))


# --- F11: total correlation (multi-information) -------------------------------------------------
def total_correlation(fit: MPSFit, n_samples: int = 20000, seed: int = 0) -> float:
    """F11: total correlation  TC = sum_i H(X_i) - H(X)  — total joint dependence of ALL orders (the
    KL divergence from the fitted joint to the product of its marginals). 0 iff assets are independent;
    larger = more all-order coupling. H(X_i) is exact from the 1-site marginals; H(X) is estimated as
    -E[log P(X)] over MPS samples. NON-REDUNDANT with pairwise correlation (captures 3+-way structure)."""
    p1 = fit.one_site_marginals()
    Hi = 0.0
    for j in range(fit.n_sites):
        p = p1[j][p1[j] > 1e-12]
        Hi += float(-(p * np.log(p)).sum())
    X = fit.sample_bins(int(n_samples), seed=seed)
    Hjoint = float(-fit.log_prob(X).mean())
    return Hi - Hjoint


# --- F12: tail-dependence asymmetry (downside minus upside) -------------------------------------
def tail_dependence_asymmetry(fit: MPSFit, q: float = 0.10) -> float:
    """F12: mean over asset pairs of (lower-tail dependence - upper-tail dependence). > 0 means the
    book co-CRASHES more than it co-RALLIES — a downside-clustering asymmetry that symmetric correlation
    cannot encode (economically: 'fear beats greed'). Read analytically from the 2-site marginals."""
    p1 = fit.one_site_marginals(); d = fit.d
    lower, upper = [], []
    for j in range(fit.n_sites):
        cum = np.cumsum(p1[j]); kl = int(np.searchsorted(cum, q) + 1); lower.append(set(range(min(kl, d))))
        cumr = np.cumsum(p1[j][::-1]); ku = int(np.searchsorted(cumr, q) + 1); upper.append(set(range(d - min(ku, d), d)))
    cores = fit._cores3d(); env = fit._environments(cores); vals = []
    for i in range(fit.n_sites):
        for j in range(i + 1, fit.n_sites):
            p2 = fit.two_site_marginal(i, j, cores, env)
            lamL = sum(p2[a, b] for a in lower[i] for b in lower[j]) / q
            lamU = sum(p2[a, b] for a in upper[i] for b in upper[j]) / q
            vals.append(lamL - lamU)
    return float(np.mean(vals))


# --- F13: correlation length (transfer-matrix decay along the chain) ----------------------------
def correlation_length(fit: MPSFit) -> float:
    """F13: xi = -1 / ln( geomean |lambda_2 / lambda_1| ) of the interior MPS transfer operators.
    Measures how fast dependence decays along the (Fiedler-ordered) chain: large xi = long-range
    coupling (a shock propagates far through the book); small xi = dependence stays local.

    FULLY WITHDRAWN (2026-07-14 review, BUG-LOG #18): per-site transfer-operator eigenvalues are
    NOT gauge-invariant for a finite inhomogeneous MPS (gauge acts two-sidedly with different
    matrices per bond), so this quantity is not even well-defined given the state — and it bounds
    nothing about P's correlation decay (the earlier "upper-bound diagnostic" framing was wrong).
    26% seed spread at <1% objective spread. For an identified alternative, fit a decay length to
    MI(i,j) or binned correlation vs |i-j| from exact two-site marginals."""
    cores = fit._cores3d(); N = fit.n_sites; ratios = []
    for k in range(1, N - 1):
        A = cores[k]; chi = A.shape[0]
        E = np.einsum("asb,csd->acbd", A, A).reshape(chi * chi, chi * chi)
        ev = np.sort(np.abs(np.linalg.eigvals(E)))[::-1]
        if len(ev) > 1 and ev[0] > 1e-12:
            r = ev[1] / ev[0]
            if 0.0 < r < 1.0:
                ratios.append(r)
    if not ratios:
        return 0.0
    return float(-1.0 / np.log(np.exp(np.mean(np.log(ratios)))))


# --- F14: k-way deep-tail corner probability (analytic, subset-averaged) ------------------------
def kway_corner_probability(fit: MPSFit, k: int = 3, q: float = 0.05) -> float:
    """F14: mean over all C(N,k) k-asset subsets of the ANALYTIC joint bottom-corner probability
    P(all k assets in their bottom tail-bin set), contracted exactly from the MPS — the validated
    deep-tail corner math of analysis/deep_tail_copula.py::kway_mps (no sampling noise).

    Tail set per asset = the bottom m bins with m = max(1, round(q*d)). Because `fit.bin_edges`
    are QUANTILE edges (each bin carries ~1/d of the window's empirical mass), the ACTUAL per-asset
    set mass is m/d, NOT the nominal q (e.g. d=12, q=0.05 -> m=1, set mass ~8.3%); the independence
    baseline for the returned value is (m/d)^k. Cost: one O(N chi^2 (chi+d)) chain contraction per
    subset, C(N,k) subsets total — cheap for N <= ~30 at k <= 3, budget accordingly beyond that."""
    m = max(1, int(round(q * fit.d)))
    tail = np.arange(m)
    cores = fit._cores3d()
    Lenv, Renv, Z = fit._environments(cores)
    vals = []
    for S in itertools.combinations(range(fit.n_sites), k):
        i0 = S[0]
        M = cores[i0][:, tail, :]
        La = np.einsum("ac,asb,csd->bd", Lenv[i0], M, M)
        last = i0
        for j in S[1:]:
            for f in range(last + 1, j):
                La = np.einsum("ac,asb,csd->bd", La, cores[f], cores[f])
            Mj = cores[j][:, tail, :]
            La = np.einsum("ac,asb,csd->bd", La, Mj, Mj)
            last = j
        vals.append(float(np.einsum("ac,ac->", La, Renv[last + 1]) / Z))
    return float(np.mean(vals))


# --- F15: independence-normalized pairwise tail dependence --------------------------------------
def tail_dependence_normalized(fit: MPSFit, q: float = 0.10) -> float:
    """F15: independence-normalized lower-tail dependence, mean over ALL asset pairs:
    P(i in tail set AND j in tail set) / [P(i in tail set) * P(j in tail set)], read analytically
    from the exact 2-site marginals. Equals 1 for independent assets BY CONSTRUCTION; > 1 means
    joint crashes cluster beyond independence.

    SUPERSEDES the joint/q reading of mps.py::mps_tail_dependence for feature use: there the
    per-asset tail SET (smallest bins whose model mass first reaches q) generally carries MORE
    than q mass, so dividing the joint by the nominal q both inflates the level and lets it drift
    with the set-mass overshoot. Dividing by the PRODUCT of the two ACTUAL model set masses
    removes that bias. mps.py itself is intentionally left unchanged (other code depends on it)."""
    p1 = fit.one_site_marginals()
    lower, mass = [], []
    for j in range(fit.n_sites):
        cum = np.cumsum(p1[j])
        m = min(int(np.searchsorted(cum, q) + 1), fit.d)
        lower.append(np.arange(m))
        mass.append(float(p1[j][:m].sum()))
    cores = fit._cores3d()
    env = fit._environments(cores)
    vals = []
    for i in range(fit.n_sites):
        for j in range(i + 1, fit.n_sites):
            p2 = fit.two_site_marginal(i, j, cores, env)
            joint = float(p2[np.ix_(lower[i], lower[j])].sum())
            vals.append(joint / max(mass[i] * mass[j], 1e-30))
    return float(np.mean(vals))


# --- F16: Schmidt-capacity utilization ----------------------------------------------------------
def schmidt_utilization(fit: MPSFit, cut: int | None = None) -> float:
    """F16: fraction of bond capacity in use at a cut = exp(entanglement entropy) / chi.

    NOT PROBABILITY-IDENTIFIED (fit diagnostic only): same sign caveat as F2 — near-saturated
    values in the rolling runs are optimizer noise, not data demand (held-out chi* = 2-4).
    See docs/Response-to-Review.md."""
    return float(np.exp(fit.entanglement_entropy(cut)) / fit.chi)
