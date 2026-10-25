"""Cryptocurrency data engine — daily USD reference rates from Coin Metrics (Community tier).

Source: Coin Metrics `PriceUSD` reference rate — a manipulation-resistant, true-USD daily rate
(volume-weighted across vetted venues), which handles wash-trading and stablecoin-vs-USD issues
*at the source*. Free Community API, no key required for this endpoint.

Loaders cache to data/ as parquet (gitignored). `load_crypto_prices(snapshot=True)` also writes a
dated, read-only snapshot so backtests stay reproducible even if Coin Metrics revises history.

Crypto-specific cleaning notes (see notebook 04):
  - 24/7 market: the daily series includes weekends. Kept as-is for crypto-only work; when aligning
    with equities, intersect on the equity trading calendar (do that at combine time, not here).
  - true USD: PriceUSD is genuine USD, not a USDT-quoted price (avoids stablecoin peg artifacts).
  - extreme returns: crypto has legitimately huge daily moves (+/-30-50%); flagged, never removed.
  - survivorship: this is a fixed set of major survivors; a broad/delisted-coin universe would need
    Pro + delisted-token retention.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data"
CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"

# Major coins available on the Community tier (SOL/AVAX require Pro).
COINS = ["btc", "eth", "xrp", "ltc", "bch", "ada", "doge", "bnb", "trx", "link", "dot"]

# Recommended modeling start: BTC/LTC pre-2015 is thin/illiquid (microstructure-driven swings, not
# market dynamics), so we model from 2015+ while RETAINING the raw 2010+ data for transparency.
LIQUID_START = "2015-01-01"


def _fetch_one(asset: str, start: str) -> pd.Series | None:
    import requests

    j = requests.get(CM_URL, params={"assets": asset, "metrics": "PriceUSD",
                                     "frequency": "1d", "start_time": start,
                                     "page_size": 10000}, timeout=60).json()
    if "data" not in j or not j["data"]:
        print(f"  skip {asset}: {j.get('error', 'no data')}")
        return None
    rows = j["data"]
    while "next_page_url" in j:
        import requests as _rq
        j = _rq.get(j["next_page_url"], timeout=60).json()
        rows += j.get("data", [])
    times = pd.to_datetime([x["time"] for x in rows], utc=True).tz_localize(None).normalize()
    vals = [float(x["PriceUSD"]) for x in rows]
    return pd.Series(vals, index=times, name=asset.upper()).sort_index()


def load_crypto_prices(coins=COINS, start: str = "2010-01-01", use_cache: bool = True,
                       snapshot: bool = False) -> pd.DataFrame:
    """Daily USD reference-rate price panel (index=date, columns=coin), cached to data/."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / "crypto_cm_prices.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    series = {}
    for a in coins:
        s = _fetch_one(a, start)
        if s is not None:
            series[a.upper()] = s
    prices = pd.DataFrame(series).sort_index()
    prices.index.name = "date"
    try:
        prices.to_parquet(cache)
        if snapshot:
            stamp = prices.index.max().date()
            prices.to_parquet(CACHE_DIR / f"crypto_cm_prices_{stamp}.parquet")
    except Exception:
        pass
    return prices


def to_crypto_returns(prices: pd.DataFrame, kind: str = "log",
                      common_span: bool = False, start: str | None = None) -> pd.DataFrame:
    """Daily returns from the price panel.

    Per-coin returns start at each coin's first listing (NaN before). `common_span=True` trims to
    the window where every coin is present (start = youngest coin's inception) for a balanced panel.
    `start` trims the leading history (pass `LIQUID_START` to drop the thin pre-2015 era for
    modeling); default None keeps full history (for the data-quality audit).
    """
    px = prices.copy()
    if start is not None:
        px = px.loc[start:]
    rets = np.log(px).diff() if kind == "log" else px.pct_change()
    if common_span:
        first = px.dropna(how="any").index.min()
        rets = rets.loc[first:]
        rets = rets.dropna(how="any")
    else:
        rets = rets.dropna(how="all")
    return rets


def load_fear_greed(use_cache: bool = True) -> pd.DataFrame:
    """Crypto Fear & Greed index (Alternative.me, free) — daily 0-100 sentiment + classification.

    Stored as an external **regime benchmark**: a documented, third-party regime label to validate
    the MPS regime signals against (do entropy/NLL spikes line up with Fear/Greed transitions?).
    Columns: fng (int 0-100), classification (Extreme Fear ... Extreme Greed).
    """
    import requests
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / "crypto_fear_greed.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    d = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=30).json()["data"]
    df = pd.DataFrame(
        {"fng": [int(x["value"]) for x in d],
         "classification": [x["value_classification"] for x in d]},
        index=pd.to_datetime([int(x["timestamp"]) for x in d], unit="s").normalize(),
    ).sort_index()
    df.index.name = "date"
    try:
        df.to_parquet(cache)
    except Exception:
        pass
    return df


def quality_report(prices: pd.DataFrame) -> dict:
    """Data-quality summary for the crypto price/return panel."""
    rets = to_crypto_returns(prices)
    # TRUE internal gaps: NaN strictly between a coin's first and last listing date
    # (pre-inception NaNs are expected and excluded).
    internal_gaps = 0
    for c in prices.columns:
        s = prices[c]
        span = s.loc[s.first_valid_index():s.last_valid_index()]
        internal_gaps += int(span.isna().sum())
    return {
        "coins": list(prices.columns),
        "n_coins": prices.shape[1],
        "span": f"{prices.index.min().date()} -> {prices.index.max().date()}",
        "total_days": len(prices),
        "per_coin_start": {c: str(prices[c].first_valid_index().date()) for c in prices.columns},
        "negative_prices": int((prices < 0).sum().sum()),       # impossible for crypto -> expect 0
        "zero_prices": int((prices == 0).sum().sum()),           # would break log-returns -> expect 0
        "internal_gaps": internal_gaps,
        "duplicate_dates": int(prices.index.duplicated().sum()),
        "max_abs_daily_logret": round(float(rets.abs().max().max()), 3),
        "extreme_moves_gt_40pct": int((rets.abs() > 0.40).sum().sum()),
    }
