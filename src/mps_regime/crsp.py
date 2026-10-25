"""WRDS / CRSP data engine — clean, survivorship-free, point-in-time US equity data.

Requires a one-time `wrds` login that created ~/.pgpass (see the Week-2 worklog). After that,
connections are passwordless, so these loaders run non-interactively. Every pull is cached to
data/ as parquet (gitignored), so re-runs are offline and deterministic.

Loaders:
  load_market_indices()  — CRSP daily value/equal-weighted market + S&P 500 return (crsp.dsi)
  load_sp500_daily()     — point-in-time S&P 500 daily total returns, delisting-adjusted
  pivot_returns()        — long CRSP frame -> wide (date x ticker) returns panel
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data"
WRDS_USERNAME = "kevin_foley"   # override by passing username= to connect()


def connect(username: str = WRDS_USERNAME):
    """Open a passwordless WRDS connection (relies on the ~/.pgpass set up at login)."""
    import wrds

    return wrds.Connection(wrds_username=username)


def load_market_indices(start: str = "1990-01-01", end: str | None = None,
                        use_cache: bool = True) -> pd.DataFrame:
    """CRSP daily market indices: vwretd (value-wtd), ewretd (equal-wtd), sprtrn (S&P 500).

    All include dividends. Indexed by date.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / "crsp_market_indices.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    db = connect()
    end_clause = f" and date <= '{end}'" if end else ""
    df = db.raw_sql(
        f"select date, vwretd, ewretd, sprtrn from crsp.dsi "
        f"where date >= '{start}'{end_clause} order by date",
        date_cols=["date"],
    )
    db.close()
    df = df.set_index("date")
    try:
        df.to_parquet(cache)
    except Exception:
        pass
    return df


def load_sp500_daily(start: str = "2005-01-01", end: str = "2026-12-31",
                     use_cache: bool = True) -> pd.DataFrame:
    """Point-in-time S&P 500 daily total returns (RET incl. dividends, delisting-adjusted).

    Joins crsp.dsf to crsp.msp500list so each stock appears ONLY on the days it was actually
    an index member (no survivorship/look-ahead). Delisting returns from crsp.dsedelist are
    folded into the final-day return. Returns long format:
        [permno, ticker, date, ret, dlret, ret_adj, prc, shrout]
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"crsp_sp500_daily_{start}_{end}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    db = connect()
    panel = db.raw_sql(
        f"""
        select f.permno, f.date, f.ret, f.prc, f.shrout
        from crsp.dsf f
        inner join crsp.msp500list m
          on f.permno = m.permno and f.date >= m.start and f.date <= m.ending
        where f.date between '{start}' and '{end}'
        """,
        date_cols=["date"],
    )
    deli = db.raw_sql(
        "select permno, dlstdt as date, dlret from crsp.dsedelist where dlret is not null",
        date_cols=["date"],
    )
    names = db.raw_sql("select permno, ticker from crsp.dsenames where ticker is not null")
    db.close()

    df = panel.merge(deli, on=["permno", "date"], how="left")
    # Fold the delisting return into that day's return; otherwise keep RET as-is (NaN preserved).
    df["ret_adj"] = df["ret"]
    m = df["dlret"].notna()
    df.loc[m, "ret_adj"] = (1.0 + df.loc[m, "ret"].fillna(0.0)) * (1.0 + df.loc[m, "dlret"]) - 1.0
    # CRSP convention: PRC is stored NEGATIVE when there was no trade (= -(bid+ask)/2). Use |PRC|.
    df["prc"] = df["prc"].abs()
    # Most-recent ticker per permno (for readability).
    tick = names.groupby("permno")["ticker"].last()
    df["ticker"] = df["permno"].map(tick)

    df = df[["permno", "ticker", "date", "ret", "dlret", "ret_adj", "prc", "shrout"]]
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    try:
        df.to_parquet(cache)
    except Exception:
        pass
    return df


def quality_report(df: pd.DataFrame) -> dict:
    """Data-quality summary for a CRSP long panel (run before using it)."""
    rep = {
        "rows": len(df),
        "stocks": int(df["permno"].nunique()),
        "span": f"{df['date'].min().date()} -> {df['date'].max().date()}",
        "nan_ret": int(df["ret"].isna().sum()),
        "negative_prc": int((df["prc"] < 0).sum()),          # should be 0 after the abs() fix
        "dup_permno_date": int(df.duplicated(["permno", "date"]).sum()),
        "extreme_ret_gt_100pct": int((df["ret"].abs() > 1.0).sum()),
        "missing_ticker": int(df["ticker"].isna().sum()),
    }
    return rep


def clean_returns_matrix(long_df: pd.DataFrame, start=None, end=None,
                         min_coverage: float = 0.98) -> pd.DataFrame:
    """Model-ready wide returns panel: pivot to date x PERMNO, restrict to [start,end], keep only
    names present on >= `min_coverage` of those days, then drop any remaining all-NaN rows.

    Pivots on PERMNO (the stable security id), NOT ticker: ~21 tickers are reused across two
    different companies (GM, DELL, DOW, AGN, ...), and pivoting on ticker silently AVERAGES two
    distinct firms into one return. PERMNO keeps identities separate; attach ticker as a label only.
    """
    w = pivot_returns(long_df, value="ret_adj", by="permno")
    if start is not None or end is not None:
        w = w.loc[start:end]
    keep = w.columns[w.notna().mean() >= min_coverage]
    w = w[keep].dropna(how="all")
    return w


def pivot_returns(long_df: pd.DataFrame, value: str = "ret_adj",
                  by: str = "ticker") -> pd.DataFrame:
    """Long CRSP frame -> wide returns panel (index=date, columns=`by`)."""
    return long_df.pivot_table(index="date", columns=by, values=value)
