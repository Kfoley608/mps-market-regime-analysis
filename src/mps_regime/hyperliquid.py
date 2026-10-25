"""Hyperliquid perp / derivatives data engine — from Hyperliquid's official public API.

Source: the Hyperliquid `info` endpoint (free, no key) — the authoritative source we also used in
the hyperscrape perp-pricing work. We pull daily perp candles + hourly funding rates for the top
perps. (The HuggingFace L2-book dataset is tick-level order-book microstructure — overkill for
daily regime/dependence modeling.)

What's distinctive here vs. spot crypto: **funding rates** and the **perp-spot basis** are
derivatives signals spot prices can't provide. Caveat: Hyperliquid launched ~May 2023, so history
is short (~3 yrs) — a complementary signal source, not a primary regime universe.

Cleaning notes (see notebook 05):
  - Drop pre-launch placeholder candles (volume == 0 and 0 trades) — real data starts ~2023-05.
  - Funding is hourly on Hyperliquid; we aggregate to a daily funding rate (sum of the 24 hourly
    rates = total daily funding cost).
  - 24/7 calendar (align to equities only when combining).
"""

from __future__ import annotations

import datetime as dt
import pathlib

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data"
HL_URL = "https://api.hyperliquid.xyz/info"

# Top liquid perps (BTC..LTC overlap our spot set for basis; SOL/AVAX/HYPE add coverage).
PERPS = ["BTC", "ETH", "SOL", "AVAX", "BNB", "DOGE", "LINK", "LTC", "HYPE"]


def _post(payload: dict, retries: int = 5) -> object:
    """POST with retry + backoff (the API drops connections under rapid pagination)."""
    import time

    import requests
    for i in range(retries):
        try:
            return requests.post(HL_URL, json=payload, timeout=40).json()
        except requests.exceptions.RequestException:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def _ms(date: str) -> int:
    return int(dt.datetime.fromisoformat(date).timestamp() * 1000)


def load_hl_prices(coins=PERPS, start: str = "2023-01-01", use_cache: bool = True,
                   snapshot: bool = False) -> pd.DataFrame:
    """Daily perp close-price panel (index=date, columns=coin). Pre-launch zero-volume placeholder
    candles are dropped (real Hyperliquid data starts ~2023-05)."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / "hl_perp_prices.parquet"
    if use_cache and cache.exists():
        prices = pd.read_parquet(cache)
    else:
        end = int(dt.datetime.now().timestamp() * 1000)
        series = {}
        for c in coins:
            candles = _post({"type": "candleSnapshot",
                             "req": {"coin": c, "interval": "1d", "startTime": _ms(start), "endTime": end}})
            if not candles:
                print(f"  skip {c}: no candles")
                continue
            df = pd.DataFrame(candles)
            # CLEAN: drop placeholder candles with no real trading (volume 0 and 0 trades)
            df = df[(df["v"].astype(float) > 0) | (df["n"].astype(int) > 0)]
            idx = pd.to_datetime(df["t"], unit="ms").dt.normalize()
            series[c] = pd.Series(df["c"].astype(float).values, index=idx, name=c)
        prices = pd.DataFrame(series).sort_index()
        prices.index.name = "date"
        try:
            prices.to_parquet(cache)
            if snapshot:
                prices.to_parquet(CACHE_DIR / f"hl_perp_prices_{prices.index.max().date()}.parquet")
        except Exception:
            pass
    # Drop the in-progress UTC day (the latest candle is partial when pulled intraday).
    prices = prices[prices.index < pd.Timestamp(dt.datetime.utcnow().date())]
    return prices


def _funding_paginated(coin: str, start_ms: int) -> list:
    """Page through fundingHistory (the API caps each call at ~500 hourly records)."""
    import time
    out, t = [], start_ms
    now = int(dt.datetime.now().timestamp() * 1000)
    while t < now:
        batch = _post({"type": "fundingHistory", "coin": coin, "startTime": t})
        if not isinstance(batch, list):     # throttle / error response -> wait and retry once
            time.sleep(3.0)
            batch = _post({"type": "fundingHistory", "coin": coin, "startTime": t})
        if not batch:
            break
        out += batch
        last = batch[-1]["time"]
        if last <= t:                       # no forward progress -> done
            break
        t = last + 1
        time.sleep(0.3)                     # gentle pacing to avoid rate-limiting
    return out


def load_hl_funding(coins=PERPS, start: str = "2023-01-01", use_cache: bool = True) -> pd.DataFrame:
    """Daily funding-rate panel (index=date, columns=coin): sum of the 24 hourly funding rates per
    UTC day = the total daily funding cost (a perp/leverage regime signal)."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / "hl_funding_daily.parquet"
    if use_cache and cache.exists():
        funding = pd.read_parquet(cache)
        # drop the in-progress UTC day (its hourly sum is partial)
        return funding[funding.index < pd.Timestamp(dt.datetime.utcnow().date())]

    import time
    series = {}
    for c in coins:
        recs = _funding_paginated(c, _ms(start))
        if not recs:
            print(f"  WARNING: no funding returned for {c}")
            continue
        s = pd.Series([float(r["fundingRate"]) for r in recs],
                      index=pd.to_datetime([r["time"] for r in recs], unit="ms"), name=c)
        # keep only days with a full 24 hourly records -> drops partial first/last UTC days
        g = s.groupby(s.index.normalize())
        daily = g.sum()[g.size() == 24]
        series[c] = daily
        time.sleep(2.0)                                       # pause between coins
    funding = pd.DataFrame(series).sort_index()
    funding.index.name = "date"
    try:
        funding.to_parquet(cache)
    except Exception:
        pass
    return funding


def to_hl_returns(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    rets = np.log(prices).diff() if kind == "log" else prices.pct_change()
    return rets.dropna(how="all")


def quality_report(prices: pd.DataFrame, funding: pd.DataFrame | None = None) -> dict:
    rets = to_hl_returns(prices)
    rep = {
        "coins": list(prices.columns),
        "n_coins": prices.shape[1],
        "price_span": f"{prices.index.min().date()} -> {prices.index.max().date()}",
        "total_days": len(prices),
        "per_coin_start": {c: str(prices[c].first_valid_index().date()) for c in prices.columns},
        "negative_prices": int((prices < 0).sum().sum()),
        "zero_prices": int((prices == 0).sum().sum()),
        "duplicate_dates": int(prices.index.duplicated().sum()),
        "max_abs_daily_logret": round(float(rets.abs().max().max()), 3),
    }
    if funding is not None:
        rep["funding_span"] = f"{funding.index.min().date()} -> {funding.index.max().date()}"
        rep["max_abs_daily_funding"] = round(float(funding.abs().max().max()), 5)
    return rep
