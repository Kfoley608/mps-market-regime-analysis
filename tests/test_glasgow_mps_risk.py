"""Pure-function tests for the Glasgow MPS extension (protocol v2)."""

import numpy as np
import pandas as pd
import pytest

from week8.glasgow_mps_risk import (
    block_bootstrap_ci,
    christoffersen_pvalues,
    fz0_loss,
    gaussian_copula_samples,
    higher_order_comoments,
    joint_tail_probability,
    kupiec_pvalue,
    marginal_diagnostics,
    nearest_psd,
    portfolio_weights,
    summarize_forecasts,
    summarize_jointtail,
    tail_dependence_pair,
    var_es,
)


def test_higher_order_comoments_are_finite_and_named():
    rng = np.random.default_rng(4)
    market = rng.normal(size=600)
    data = pd.DataFrame(
        {
            "A": market + 0.1 * rng.normal(size=600),
            "B": -market + 0.1 * rng.normal(size=600),
            "C": market**2 + 0.1 * rng.normal(size=600),
            "D": rng.normal(size=600),
        }
    )
    for mode in ("loo", "ew"):
        out = higher_order_comoments(data, market=mode)
        assert list(out.columns) == ["beta", "coskewness", "cokurtosis"]
        assert out.shape == (4, 3)
        assert np.isfinite(out.to_numpy()).all()
    loo = higher_order_comoments(data, market="loo")
    ew = higher_order_comoments(data, market="ew")
    assert not np.allclose(loo["beta"], ew["beta"])


def test_portfolio_weights_are_equal_weighted_and_sorted():
    values = np.arange(10, dtype=float)
    weights, labels = portfolio_weights(values, n_groups=5)
    assert weights.shape == (10, 5)
    assert np.allclose(weights.sum(axis=0), 1.0)
    assert labels[0] == 0
    assert labels[-1] == 4
    assert np.allclose(weights[:, -1] - weights[:, 0],
                       np.array([-0.5, -0.5, 0, 0, 0, 0, 0, 0, 0.5, 0.5]))


def test_tail_dependence_and_var_es():
    x = np.linspace(-1.0, 1.0, 1000)
    lower, upper, asym = tail_dependence_pair(x, x, q=0.05)
    assert lower == pytest.approx(1.0, abs=0.03)
    assert upper == pytest.approx(1.0, abs=0.03)
    assert asym == pytest.approx(0.0, abs=0.05)

    var, es = var_es(x, alpha=0.01)
    assert es <= var
    assert var < 0


def test_joint_tail_probability_and_psd():
    x = np.arange(2000, dtype=float).reshape(500, 4)
    p = joint_tail_probability(x, q=0.1, k=3)
    assert 0.0 <= p <= 1.0
    common = np.quantile(x, 0.1, axis=0)
    p_common = joint_tail_probability(x, q=0.1, k=3, thresholds=common)
    assert p_common == pytest.approx(p)
    cov = np.array([[1.0, 2.0], [2.0, 1.0]])
    fixed = nearest_psd(cov)
    assert np.linalg.eigvalsh(fixed).min() >= 0.0


def test_gaussian_copula_matches_marginals_and_dependence_sign():
    rng = np.random.default_rng(11)
    z = rng.normal(size=800)
    train = pd.DataFrame(
        {
            "A": z + 0.4 * rng.normal(size=800),
            "B": z + 0.4 * rng.normal(size=800),
            "C": rng.standard_t(4, size=800) * 0.01,
        }
    )
    samples = gaussian_copula_samples(train, n_sim=20000, seed=3)
    # marginals match the train distribution: tail rate at train q05 == 5%
    tail_rate, vol_ratio = marginal_diagnostics(samples, train.to_numpy(float), 0.05)
    assert tail_rate == pytest.approx(0.05, abs=0.01)
    assert vol_ratio == pytest.approx(1.0, abs=0.05)
    # dependence sign is preserved
    corr = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
    assert corr > 0.5


def test_fz0_prefers_the_true_quantiles():
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.01, size=20000)
    alpha = 0.05
    true_var = float(np.quantile(r, alpha))
    true_es = float(r[r <= true_var].mean())
    good = np.mean([fz0_loss(x, true_var, true_es, alpha) for x in r])
    shallow = np.mean([fz0_loss(x, true_var * 0.5, true_es * 0.5, alpha) for x in r])
    deep = np.mean([fz0_loss(x, true_var * 2.0, true_es * 2.0, alpha) for x in r])
    assert good < shallow
    assert good < deep
    assert np.isnan(fz0_loss(0.0, -0.01, 0.01, alpha))


def test_kupiec_and_christoffersen_behave():
    rng = np.random.default_rng(5)
    n = 2000
    good_hits = (rng.random(n) < 0.01).astype(int)
    bad_hits = (rng.random(n) < 0.06).astype(int)
    assert kupiec_pvalue(good_hits, 0.01) > 0.05
    assert kupiec_pvalue(bad_hits, 0.01) < 1e-6
    clustered = np.zeros(n, dtype=int)
    clustered[100:140] = 1
    p_ind, p_cc = christoffersen_pvalues(clustered, 0.01)
    assert p_ind < 1e-6
    assert p_cc < 1e-6


def test_block_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(9)
    x = rng.normal(0.5, 1.0, size=500)
    lo, hi = block_bootstrap_ci(x, n_boot=500, seed=1)
    assert lo < 0.5 < hi
    assert hi - lo < 0.5


def _fake_forecasts() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(2)
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    for model, shift in [("gauss_copula", 0.0), ("gaussian", 0.0), ("mps_chi2", -0.001)]:
        for i, date in enumerate(dates):
            actual = -0.02 if i % 17 == 0 else float(rng.normal(0, 0.005))
            var = -0.012 + shift
            rows.append(
                {
                    "date": str(date.date()),
                    "characteristic": "coskewness",
                    "model": model,
                    "actual_bms_return": actual,
                    "var": var,
                    "es": var - 0.004,
                    "var_hit": int(actual < var),
                    "asym_tail_dep": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_summary_has_backtests_and_copula_deltas():
    out = summarize_forecasts(_fake_forecasts(), alpha=0.05)
    assert set(out["model"]) == {"gauss_copula", "gaussian", "mps_chi2"}
    for col in ("kupiec_p", "christoffersen_cc_p", "fz0_mean",
                "fz0_delta_vs_gauss_copula", "var_hit_ci_lo", "var_hit_ci_hi"):
        assert col in out.columns
    non_ref = out[out["model"] != "gauss_copula"]
    assert non_ref["fz0_delta_vs_gauss_copula"].notna().all()
    ref = out[out["model"] == "gauss_copula"]
    assert ref["fz0_delta_vs_gauss_copula"].isna().all()


def test_jointtail_summary_shape():
    rng = np.random.default_rng(3)
    rows = []
    for model in ("gauss_copula", "mps_chi2"):
        for i in range(40):
            base = 0.01 if model == "gauss_copula" else 0.013
            rows.append(
                {
                    "date": f"2020-{1 + i % 12:02d}-01",
                    "model": model,
                    "marginal_tail_rate": 0.05 + rng.normal(0, 0.002),
                    "marginal_vol_ratio": 1.0 + rng.normal(0, 0.01),
                    "joint_tail_prob": base,
                    "joint_tail_prob_rank": base,
                    "joint_tail_lift_vs_gaussian": 1.0,
                    "joint_tail_lift_vs_gauss_copula": base / 0.01,
                    "rank_joint_tail_lift_vs_gauss_copula": base / 0.01,
                }
            )
    out = summarize_jointtail(pd.DataFrame(rows))
    assert set(out["model"]) == {"gauss_copula", "mps_chi2"}
    assert "mean_rank_lift_vs_gauss_copula" in out.columns
    assert "rank_lift_log_ci_lo" in out.columns
