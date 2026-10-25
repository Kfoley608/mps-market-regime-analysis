"""Invariance regression suite (review item 9, BUG-LOG #18): amplitude signs, bin grids,
probability normalization. Sign-vs-entanglement identifiability lives in test_block_mi.py."""
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mps_regime import features as F
from mps_regime import mps as M
from mps_regime.dmrg import fit_dmrg


def _toy(n=350, N=5, seed=0):
    rng = np.random.default_rng(seed)
    r = 0.7 * rng.standard_normal((n, 1)) + 0.7 * rng.standard_normal((n, N))
    return pd.DataFrame(r, columns=[f"A{k}" for k in range(N)])


def _fit(seed=0, d=3, chi=4):
    return fit_dmrg(_toy(seed=seed), d=d, chi=chi, sweeps=6, seed=seed)


def test_probability_normalization():
    """P must sum to 1 over the full configuration space (normalization invariant)."""
    fit = _fit()
    grid = np.array(list(itertools.product(range(fit.d), repeat=fit.n_sites)))
    P = np.exp(fit.log_prob(grid))
    assert abs(P.sum() - 1.0) < 1e-8


def test_single_site_sign_gauge_leaves_all_p_functionals_unchanged():
    """Flipping the sign of one site's core is a gauge move on psi (P identical).
    Every P-functional feature must be exactly invariant."""
    fit = _fit()
    flipped = M.MPSFit(
        cores=[(-c if i == 1 else c).copy() for i, c in enumerate(fit.cores)],
        n_sites=fit.n_sites, d=fit.d, chi=fit.chi,
        bin_values=fit.bin_values, bin_std=fit.bin_std, bin_edges=fit.bin_edges,
        columns=fit.columns, nll_history=list(fit.nll_history),
    )
    grid = np.array(list(itertools.product(range(fit.d), repeat=fit.n_sites)))
    assert np.allclose(fit.log_prob(grid), flipped.log_prob(grid), atol=1e-10)

    for fn in (F.tail_state_probability,
               lambda f: F.mean_mutual_information(f, adjacent=False),
               F.tail_dependence_asymmetry,
               lambda f: F.block_mutual_information(f, n_samples=5000, seed=3)):
        a, b = fn(fit), fn(flipped)
        assert abs(a - b) < 1e-9, (fn, a, b)


def test_bin_grid_consistency_encode_vs_discretize():
    """fit.encode must reproduce _discretize's codes on the training block (same digitize
    convention), and encode of OUT-OF-SAMPLE rows must use the TRAIN edges (BUG-LOG #15)."""
    rets = _toy(seed=2)
    tr, te = rets.iloc[:250], rets.iloc[250:]
    fit = fit_dmrg(tr, d=4, chi=3, sweeps=4, seed=0)
    codes_tr, edges_tr, _, _ = M._discretize(tr, 4)
    assert (fit.encode(tr) == codes_tr).all()
    manual_te = np.column_stack([np.digitize(te.to_numpy()[:, k], edges_tr[k])
                                 for k in range(te.shape[1])])
    assert (fit.encode(te) == manual_te).all()
    # and the full-sample edges genuinely differ (the #15 failure mode is detectable)
    codes_full = M._discretize(rets, 4)[0]
    assert (codes_full[250:] != manual_te).any()


def test_binning_rejects_nans():
    """BUG-LOG #18l: NaNs must raise, not silently land in the top bin."""
    from mps_regime.binning import bin_edges
    df = _toy(n=100)
    df.iloc[3, 1] = np.nan
    try:
        bin_edges(df, 3)
        assert False, "expected ValueError on NaN input"
    except ValueError:
        pass
