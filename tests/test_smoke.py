"""Smoke + unit tests. Offline only (no network): baselines run on synthetic returns so CI
is deterministic. Grows as modules land (Week 1+)."""

import pathlib

import numpy as np
import pandas as pd
import pytest

import mps_regime


def test_version():
    assert mps_regime.__version__


def test_config_loads():
    from mps_regime.data import load_config, universe_tickers

    cfg_path = pathlib.Path(__file__).resolve().parents[1] / "config" / "universe.yaml"
    cfg = load_config(str(cfg_path))
    tickers = universe_tickers(cfg)
    assert len(tickers) == 12
    assert "TLT" in tickers and "GLD" in tickers
    assert set(cfg["portfolios"]) >= {"equal_weight", "sixty_forty"}


def _synthetic_returns(n_days=600, n_assets=6, rho=0.3, seed=0):
    rng = np.random.default_rng(seed)
    cov = rho * np.ones((n_assets, n_assets)) + (1 - rho) * np.eye(n_assets)
    data = rng.multivariate_normal(np.zeros(n_assets), cov * 1e-4, size=n_days)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_baseline_point_estimates_in_range():
    from mps_regime import baselines as bl

    r = _synthetic_returns()
    assert bl.realized_vol(r) > 0
    assert -1.0 <= bl.average_correlation(r) <= 1.0
    assert 0.0 <= bl.pca_first_explained(r) <= 1.0
    assert 0.0 < bl.absorption_ratio(r) <= 1.0
    assert bl.spectral_entropy(r) >= 0.0


def test_absorption_rises_with_correlation():
    from mps_regime import baselines as bl

    low = bl.absorption_ratio(_synthetic_returns(rho=0.1, seed=1))
    high = bl.absorption_ratio(_synthetic_returns(rho=0.8, seed=1))
    assert high > low  # tighter coupling -> more variance in top eigenvectors


def test_rolling_indicators_shape():
    from mps_regime import baselines as bl

    r = _synthetic_returns(n_days=600)
    panel = bl.rolling_indicators(r, window=250, step=5)
    assert {"realized_vol", "avg_corr", "absorption_ratio", "spectral_entropy",
            "max_drawdown"} <= set(panel.columns)
    assert len(panel) == len(range(250, 601, 5))


def test_drawdown_is_nonpositive():
    from mps_regime import baselines as bl

    r = _synthetic_returns()
    dd = bl.drawdown_series(r)
    assert (dd <= 1e-9).all()              # drawdown is always <= 0
    assert bl.max_drawdown(r) <= 0.0


def test_var_breach_rate_near_alpha():
    from mps_regime import baselines as bl

    r = _synthetic_returns(n_days=2000).mean(axis=1)
    vb = bl.var_breaches(r, window=250, alpha=0.99)
    rate = vb.dropna().mean()
    assert 0.0 < rate < 0.05               # ~1% expected, generous band for synthetic noise


def test_crisis_mask_flags_known_dates():
    from mps_regime import baselines as bl

    idx = pd.bdate_range("2007-01-01", "2026-01-01")
    m = bl.crisis_mask(idx)
    assert m.loc["2020-03-16"]             # COVID crash
    assert m.loc["2008-10-15"]             # GFC
    assert not m.loc["2017-06-15"]         # calm period
    assert m.dtype == bool


def test_classical_panel_has_spec_columns():
    from mps_regime import baselines as bl

    r = _synthetic_returns(n_days=600)
    panel = bl.build_classical_panel(r, window=250, step=5, fetch_overlays=False)
    spec = {"realized_vol", "avg_corr", "pca1_explained", "drawdown",
            "max_drawdown", "var_breach_rate", "crisis"}
    assert spec <= set(panel.columns)
    assert set(panel["crisis"].unique()) <= {0, 1}


def test_crypto_returns_and_quality():
    from mps_regime import crypto

    idx = pd.date_range("2020-01-01", periods=300, freq="D")   # 7-day crypto calendar
    rng = np.random.default_rng(0)
    a = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.05, 300))), index=idx, name="A")
    b = (a * 0.5).rename("B"); b.iloc[:100] = np.nan             # staggered listing
    px = pd.concat([a, b], axis=1)

    rep = crypto.quality_report(px)
    assert rep["n_coins"] == 2
    assert rep["internal_gaps"] == 0                            # pre-listing NaN not counted
    assert rep["negative_prices"] == 0 and rep["zero_prices"] == 0

    core = crypto.to_crypto_returns(px, common_span=True)
    assert core.notna().all().all()                            # common-span balanced (no NaN)
    assert core.index.min() >= idx[100]                        # starts at the youngest listing


def test_crypto_quality_flags_planted_problems():
    """Meta-test: the quality checks must actually CATCH bad data (a cleaning safeguard)."""
    from mps_regime import crypto

    idx = pd.date_range("2018-01-01", periods=200, freq="D")
    rng = np.random.default_rng(1)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 200)))
    px = pd.DataFrame({"A": base, "B": base * 1.1}, index=idx)
    px.iloc[10, 0] = -5.0       # plant a negative price
    px.iloc[20, 1] = 0.0        # plant a zero price
    px.iloc[30, 0] = np.nan     # plant an internal gap (within A's span)

    rep = crypto.quality_report(px)
    assert rep["negative_prices"] >= 1
    assert rep["zero_prices"] >= 1
    assert rep["internal_gaps"] >= 1


def test_crypto_liquid_start_cutoff():
    from mps_regime import crypto

    idx = pd.date_range("2013-01-01", periods=1500, freq="D")   # spans the pre-2015 illiquid era
    rng = np.random.default_rng(2)
    px = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 1500)))}, index=idx)
    full = crypto.to_crypto_returns(px)
    liquid = crypto.to_crypto_returns(px, start=crypto.LIQUID_START)
    assert liquid.index.min() >= pd.Timestamp(crypto.LIQUID_START)   # cutoff applied
    assert len(liquid) < len(full)                                   # earlier data dropped


def test_hyperliquid_returns_and_quality():
    from mps_regime import hyperliquid as hl

    idx = pd.date_range("2023-06-01", periods=200, freq="D")
    rng = np.random.default_rng(3)
    px = pd.DataFrame({"BTC": 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 200))),
                       "ETH": 50 * np.exp(np.cumsum(rng.normal(0, 0.04, 200)))}, index=idx)
    fund = pd.DataFrame({"BTC": rng.normal(0, 5e-4, 200),
                         "ETH": rng.normal(0, 5e-4, 200)}, index=idx)
    rep = hl.quality_report(px, fund)
    assert rep["n_coins"] == 2
    assert rep["negative_prices"] == 0 and rep["zero_prices"] == 0
    assert "funding_span" in rep
    assert hl.to_hl_returns(px).shape[1] == 2


def test_crsp_pivot_returns():
    from mps_regime import crsp           # import must not require the wrds package

    long = pd.DataFrame({
        "ticker": ["A", "A", "B", "B"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
        "ret_adj": [0.01, -0.02, 0.03, 0.00],
    })
    wide = crsp.pivot_returns(long)
    assert list(wide.columns) == ["A", "B"]
    assert wide.shape == (2, 2)
    assert wide.loc[pd.Timestamp("2020-01-01"), "A"] == 0.01


def test_parse_french_factors():
    from mps_regime.data import _parse_french

    text = ("This file was created by ...\n"
            ",Mkt-RF,SMB\n"
            "19630701, -0.67, 0.02\n"
            "19630702,  0.79,-0.26\n"
            "Copyright 2026\n")
    df = _parse_french(text, ["Mkt-RF", "SMB"])
    assert list(df.columns) == ["Mkt-RF", "SMB"]
    assert len(df) == 2                              # header + footer rows skipped
    assert df.iloc[0]["Mkt-RF"] == -0.67


def test_quantile_bins_balanced():
    from mps_regime import binning as bn

    r = _synthetic_returns(n_days=300, n_assets=4)
    binned, edges = bn.quantile_bins(r, n_bins=3)
    assert set(np.unique(binned.to_numpy())) <= {0, 1, 2}
    for c in binned.columns:                       # terciles -> each ~1/3
        frac = binned[c].value_counts(normalize=True)
        assert (frac - 1 / 3).abs().max() < 0.1


def test_discretize_is_permutation():
    from mps_regime import binning as bn

    r = _synthetic_returns(n_days=400, n_assets=6)
    binned, edges, order = bn.discretize(r, n_bins=3)
    assert sorted(order) == sorted(r.columns)
    assert list(binned.columns) == order


def test_mps_born_machine_normalizes_and_samples():
    pytest.importorskip("torch")
    import itertools
    from mps_regime import mps

    rng = np.random.default_rng(0)
    cov = np.array([[1, .6, .2], [.6, 1, .3], [.2, .3, 1.]])
    df = pd.DataFrame(rng.multivariate_normal([0, 0, 0], cov, 1500), columns=list("ABC"))
    fit = mps.fit_born_machine(df, d=3, chi=3, sweeps=150, lr=0.05, seed=0)

    cfg = np.array(list(itertools.product(range(3), repeat=3)))
    P = np.exp(fit.log_prob(cfg))
    assert abs(P.sum() - 1.0) < 1e-3                     # exact normalization
    samp = fit.sample_bins(5000, seed=1)
    assert samp.shape == (5000, 3) and samp.min() >= 0 and samp.max() < 3
    # MPS recovers the sign/ordering of the strongest pair (A,B more dependent than A,C)
    C = fit.mps_correlation()
    assert C[0, 1] > C[0, 2] > 0
    # regime diagnostics are sane
    S = fit.entanglement_entropy()
    assert 0.0 <= S <= np.log(3) + 1e-6              # S <= ln(chi)
    assert 0.0 <= fit.tail_state_probability() <= 1.0
    assert fit.entanglement_profile().shape == (2,)  # N-1 cuts for 3 assets


def test_fiedler_groups_correlated_blocks():
    from mps_regime import binning as bn

    rng = np.random.default_rng(0)
    n = 3000
    fa, fb = rng.normal(size=n), rng.normal(size=n)
    mk = lambda f: f + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"A0": mk(fa), "A1": mk(fa), "A2": mk(fa),
                       "B0": mk(fb), "B1": mk(fb), "B2": mk(fb)})
    _, _, order = bn.discretize(df, n_bins=3)
    posA = sorted(order.index(c) for c in ["A0", "A1", "A2"])
    posB = sorted(order.index(c) for c in ["B0", "B1", "B2"])
    assert posA[-1] - posA[0] == 2                 # block A contiguous on the chain
    assert posB[-1] - posB[0] == 2                 # block B contiguous
