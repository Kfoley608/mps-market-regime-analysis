"""block_mutual_information (F17): correctness vs exact enumeration, sign-invariance, and the
identifiability property that motivated it (equal-P fits agree on block MI even when their signed
entanglement disagrees)."""
import itertools

import numpy as np
import pandas as pd
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mps_regime.dmrg import fit_dmrg
from mps_regime import features as F


def _toy_returns(n=400, n_assets=5, seed=0):
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((n, 1))
    idio = rng.standard_normal((n, n_assets))
    r = 0.8 * common + 0.6 * idio
    return pd.DataFrame(r, columns=[f"A{k}" for k in range(n_assets)])


def _enumerate_block_mi(fit, cut):
    """Exact I(L;R) by enumerating all d^N configurations."""
    N, d = fit.n_sites, fit.d
    grid = np.array(list(itertools.product(range(d), repeat=N)), dtype=np.int64)
    P = np.exp(fit.log_prob(grid))
    P = P / P.sum()
    nL = d ** cut
    PM = P.reshape(nL, -1)
    PL = PM.sum(1)
    PR = PM.sum(0)
    mask = PM > 1e-300
    outer = np.outer(PL, PR)
    return float((PM[mask] * np.log(PM[mask] / outer[mask])).sum())


def test_block_mi_matches_enumeration():
    rets = _toy_returns()
    fit = fit_dmrg(rets, d=3, chi=4, sweeps=8, seed=0)
    cut = fit.n_sites // 2
    exact = _enumerate_block_mi(fit, cut)
    est = F.block_mutual_information(fit, cut=cut, n_samples=40000, seed=1)
    assert abs(est - exact) < 0.03, f"sampler {est:.4f} vs exact {exact:.4f}"
    assert est >= -0.01  # MI is nonnegative up to MC noise


def test_block_mi_sign_invariant_but_entropy_is_not():
    """The reviewer's counterexample, run through the new feature: two fits with identical P
    (uniform on 2 sites) but S=0 vs S=ln2 must agree on block MI (=0)."""
    from mps_regime.mps import MPSFit

    def make_fit(psi_matrix):
        # MPSFit convention: first core (d, chi), last core (chi, d).
        # With c0 = I, psi(x0, x1) = c_last[x0, x1] = psi_matrix[x0, x1].
        c0 = np.eye(2)
        c1 = np.array(psi_matrix, dtype=np.float64)
        fit = MPSFit.__new__(MPSFit)
        fit.cores = [c0, c1]
        fit.n_sites, fit.d, fit.chi = 2, 2, 2
        fit.columns = ["X", "Y"]
        return fit

    psi_prod = np.array([[0.5, 0.5], [0.5, 0.5]])       # rank 1: S=0
    psi_had  = np.array([[0.5, 0.5], [0.5, -0.5]])      # Hadamard: S=ln2
    f1, f2 = make_fit(psi_prod), make_fit(psi_had)

    # identical P (uniform 1/4)
    grid = np.array(list(itertools.product([0, 1], repeat=2)))
    P1 = np.exp(f1.log_prob(grid)); P2 = np.exp(f2.log_prob(grid))
    assert np.allclose(P1, 0.25, atol=1e-12) and np.allclose(P2, 0.25, atol=1e-12)

    # signed entanglement differs (the non-identified quantity)
    s1 = f1.entanglement_entropy(1); s2 = f2.entanglement_entropy(1)
    assert s1 < 1e-9 and abs(s2 - np.log(2)) < 1e-9

    # block MI agrees (the identified quantity): uniform product -> I = 0 for both
    mi1 = F.block_mutual_information(f1, cut=1, n_samples=20000, seed=0)
    mi2 = F.block_mutual_information(f2, cut=1, n_samples=20000, seed=0)
    assert abs(mi1) < 1e-6 and abs(mi2) < 1e-6, (mi1, mi2)


def test_block_mi_seed_stability_vs_entropy():
    """Across DMRG seeds at ~equal likelihood, block MI must be materially more stable than the
    signed entropy is unstable in the rolling runs (this is a smoke check, not a strict bound)."""
    rets = _toy_returns(seed=3)
    vals_mi = []
    for seed in (0, 1):
        fit = fit_dmrg(rets, d=3, chi=8, sweeps=8, seed=seed)
        vals_mi.append(F.block_mutual_information(fit, n_samples=20000, seed=7))
    assert abs(vals_mi[0] - vals_mi[1]) < 0.15 * max(abs(np.mean(vals_mi)), 0.1)
