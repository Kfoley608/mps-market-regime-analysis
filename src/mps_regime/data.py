"""Data layer: load the asset universe and build the daily returns panel.

Week-1 task. Pulls adjusted closes for the tickers in config/universe.yaml (yfinance by
default; may switch to a sponsor-provided feed), aligns the calendar, and returns log returns.

Light cleaning only (daily liquid ETFs): keep the common date span where all tickers trade,
forward-fill at most 1 stray missing day, drop the rest. No survivorship / corporate-action
work needed — yfinance auto-adjusted closes already incorporate splits and dividends.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

# Repo root = three levels up from this file (src/mps_regime/data.py -> repo/).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "universe.yaml"
CACHE_DIR = REPO_ROOT / "data"


def load_config(path: str | pathlib.Path = DEFAULT_CONFIG) -> dict:
    """Load the universe / windows / binning / portfolios config."""
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def universe_tickers(config: dict) -> list[str]:
    """Flatten the `universe` section of the config into a flat ticker list (order preserved)."""
    out: list[str] = []
    for _group, val in config["universe"].items():
        if isinstance(val, list):
            out.extend(val)
    return out


def load_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Adjusted-close price panel, columns=tickers (config order), index=trading days.

    Caches to data/prices_<hash>.parquet so re-runs are offline and deterministic.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    key = f"{'-'.join(tickers)}_{start}_{end}".replace(" ", "")
    cache = CACHE_DIR / f"prices_{abs(hash(key)) % (10**10)}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjusted OHLC; 'Close' is the adjusted close
        progress=False,
        group_by="column",
    )
    # yfinance returns a column MultiIndex (field, ticker) for multi-ticker pulls.
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.reindex(columns=tickers)  # enforce config order
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    try:
        close.to_parquet(cache)
    except Exception:  # parquet engine optional; caching is best-effort
        pass
    return close


def to_returns(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    """Daily returns from a price panel ('log' or 'simple').

    Aligns to the common span where every ticker has prices, forward-fills at most one
    stray gap, then computes returns and drops the leading NaN row.
    """
    px = prices.copy()
    # Trim to the first date where all tickers are present (handles staggered ETF inceptions).
    first_valid = px.dropna(how="any").index.min()
    px = px.loc[first_valid:]
    px = px.ffill(limit=1).dropna(how="any")

    if kind == "log":
        rets = np.log(px).diff()
    elif kind == "simple":
        rets = px.pct_change()
    else:
        raise ValueError("kind must be 'log' or 'simple'")
    return rets.dropna(how="any")


def load_index(
    ticker: str,
    start: str,
    end: str | None = None,
) -> pd.Series:
    """Load a single external index LEVEL series (e.g. '^VIX', '^MOVE') as a daily Series.

    Comparison overlays from the sponsor spec ("VIX or MOVE, if available"). Returned unaligned;
    callers reindex onto the returns index. Returns an empty Series if the symbol is
    unavailable (MOVE is intermittently missing on Yahoo) rather than raising.
    """
    import yfinance as yf

    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return pd.Series(dtype=float, name=ticker.lstrip("^"))
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):  # single-ticker pulls can still be 2-D
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index)
        return close.sort_index().rename(ticker.lstrip("^"))
    except Exception:
        return pd.Series(dtype=float, name=ticker.lstrip("^"))


def load_vix(start: str, end: str | None = None) -> pd.Series:
    """CBOE Volatility Index (^VIX) daily close — equity-vol comparison overlay."""
    return load_index("^VIX", start, end)


def load_move(start: str, end: str | None = None) -> pd.Series:
    """ICE BofA MOVE index (^MOVE) — bond-vol overlay; may be empty if Yahoo lacks it."""
    return load_index("^MOVE", start, end)


def _download_french_csv(name: str) -> str:
    """Download + unzip a Ken French Data Library CSV (returns the raw text)."""
    import io
    import zipfile

    import requests

    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        f"{name}_CSV.zip"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    return zf.read(zf.namelist()[0]).decode("latin-1")


def _parse_french(text: str, columns: list[str]) -> pd.DataFrame:
    """Parse the daily rows (YYYYMMDD,...) out of a Ken French CSV into a DataFrame."""
    import re

    rows = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d{8})\s*,(.*)$", line)
        if not m:
            continue
        parts = [p.strip() for p in m.group(2).split(",")]
        if len(parts) < len(columns):
            continue
        try:
            vals = [float(x) for x in parts[: len(columns)]]
        except ValueError:
            continue
        rows.append((m.group(1), *vals))
    df = pd.DataFrame(rows, columns=["date", *columns])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").sort_index()


def load_french_factors(use_cache: bool = True) -> pd.DataFrame:
    """Daily Fama-French 5 factors + Momentum from the Ken French Data Library.

    Returns DECIMAL daily returns with columns:
        Mkt-RF, SMB, HML, RMW, CMA, RF, Mom, Mkt   (Mkt = Mkt-RF + RF)
    Source values are in percent and are divided by 100 here. Cached to data/.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / "french_factors_daily.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    ff5 = _parse_french(
        _download_french_csv("F-F_Research_Data_5_Factors_2x3_daily"),
        ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    )
    mom = _parse_french(
        _download_french_csv("F-F_Momentum_Factor_daily"), ["Mom"]
    )
    df = ff5.join(mom, how="inner") / 100.0     # percent -> decimal
    df["Mkt"] = df["Mkt-RF"] + df["RF"]          # total market return
    try:
        df.to_parquet(cache)
    except Exception:
        pass
    return df


def load_returns(
    config: str | pathlib.Path | dict | None = None,
    kind: str = "log",
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Convenience: config -> tickers -> prices -> returns panel. The Week-1 entry point."""
    cfg = config if isinstance(config, dict) else load_config(config or DEFAULT_CONFIG)
    tickers = universe_tickers(cfg)
    d = cfg.get("data", {})
    prices = load_prices(
        tickers,
        start=d.get("start", "2005-01-01"),
        end=d.get("end"),
        use_cache=use_cache,
    )
    return to_returns(prices, kind=kind)
