"""Futures data engine — commodities + FX via DataBento (CME Globex, GLBX.MDP3).

Daily OHLCV for continuous futures. Fills the universe gaps: commodities beyond gold (oil CL,
silver SI, copper HG, nat-gas NG) and FX (EUR 6E, JPY 6J). Gold (GC) is also available alongside
our GLD ETF.

ROLL RULE — open interest (`.n.0`): we use the OPEN-INTEREST-rolled continuous, which always tracks
the contract that is actually being held/traded. The alternative CALENDAR roll (`.c.0`) maps to the
nearest-by-expiration contract, which for thinly-cycled metals (esp. silver) is often an *inactive*
month with no trades that day -> no daily bar -> hundreds of spurious weekday gaps. The OI roll has a
bar essentially every session (verified: SI 2018 had 260 bars under `.n.0` vs 184 under `.c.0`).

CRITICAL cleaning — the roll seam: the continuous is the *raw* stitched series (NOT back-adjusted),
so a close-to-close return spanning a contract roll is spurious (it's the contango/backwardation gap,
not a market move). We detect rolls via `instrument_id` changes and set the roll-day return to NaN.

Requires the DATABENTO_API_KEY env var (read from a local key file; never committed). Data cached
to data/ as parquet (gitignored).
"""

from __future__ import annotations

import datetime as dt
import pathlib

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data"
DATASET = "GLBX.MDP3"
ROLL_RULE = "n"   # continuous roll: "n"=open interest (default, dense), "v"=volume, "c"=calendar

# CME Globex roots -> label. (Dollar index DX is ICE, not on Globex; EUR/JPY proxy the dollar.)
FUTURES = {"CL": "oil", "GC": "gold", "SI": "silver", "HG": "copper",
           "NG": "natgas", "6E": "eurusd", "6J": "jpyusd"}


def load_futures(symbols: list[str] | None = None, start: str = "2010-06-07",
                 end: str | None = None, use_cache: bool = True, snapshot: bool = False):
    """Return (close, roll): wide daily continuous (open-interest-rolled) close prices and a boolean
    roll mask (True where the active contract changed that day). Columns are the futures roots.

    NOTE: DataBento requires an explicit `end` for continuous-contract pulls (open-ended returns
    empty); we default it to today.
    """
    symbols = symbols or list(FUTURES)
    # historical license can't reach the live edge -> default end a few days back
    end = end or (dt.datetime.utcnow().date() - dt.timedelta(days=2)).isoformat()
    CACHE_DIR.mkdir(exist_ok=True)
    c_close = CACHE_DIR / "db_futures_close.parquet"
    c_roll = CACHE_DIR / "db_futures_roll.parquet"
    if use_cache and c_close.exists() and c_roll.exists():
        return pd.read_parquet(c_close), pd.read_parquet(c_roll)

    import databento as db
    import json
    import time
    import threading

    # DataBento throttles/drops reused keep-alive connections when many requests are fired in a loop
    # (RemoteDisconnected / pool-exhausted / silent socket hangs), even though an isolated request is
    # fine. Defenses: (1) a FRESH client (new connection pool) per request so we never reuse a socket
    # the server has dropped; (2) a hard wall-clock timeout via a DAEMON thread so a hung read can't
    # freeze the pull (daemon dies with the process, can't block interpreter exit).
    class _Timeout(Exception):
        pass

    def _get_df(sym, cs, ce, timeout_s=75):   # generous: some chunks are slow-but-working under throttle
        box = {}

        def work():
            try:
                client = db.Historical()                # fresh connection pool every call
                box["df"] = client.timeseries.get_range(
                    dataset=DATASET, symbols=[f"{sym}.{ROLL_RULE}.0"], schema="ohlcv-1d",
                    stype_in="continuous", start=cs, end=ce).to_df()
            except Exception as e:                      # noqa: BLE001
                box["err"] = e

        th = threading.Thread(target=work, daemon=True)
        th.start()
        th.join(timeout_s)
        if th.is_alive():
            raise _Timeout()
        if "err" in box:
            raise box["err"]
        return box["df"]

    def _windows(s0, e0, yrs=4):
        s, e, out, cur = pd.Timestamp(s0), pd.Timestamp(e0), [], pd.Timestamp(s0)
        while cur < e:
            nxt = min(cur + pd.DateOffset(years=yrs), e)
            out.append((cur.date().isoformat(), nxt.date().isoformat())); cur = nxt
        return out

    # --- Resumable pull -------------------------------------------------------------------------
    # Throttling makes a random ~third of chunks fail per session, and the failures don't clear
    # within a retry window. So we persist progress and converge over a few passes: each succeeded
    # chunk's rows are appended to a long-format raw cache and its (symbol|start|end) key is recorded
    # in a manifest. Re-running fetches ONLY the missing chunks. The wide close/roll cache is written
    # only when EVERY requested chunk is done, so the top-level use_cache short-circuit can't return
    # a half-built panel.
    raw_f = CACHE_DIR / "db_futures_raw.parquet"          # long: index=date, cols [symbol, close, iid]
    man_f = CACHE_DIR / "db_futures_manifest.json"
    done = set(json.loads(man_f.read_text())) if man_f.exists() else set()
    raw = pd.read_parquet(raw_f) if raw_f.exists() else None
    rows = [raw] if raw is not None else []

    want = [(s, cs, ce) for s in symbols for cs, ce in _windows(start, end)]
    todo = [(s, cs, ce) for (s, cs, ce) in want if f"{s}|{cs}|{ce}" not in done]
    print(f"chunks: {len(want)} total, {len(want) - len(todo)} cached, {len(todo)} to fetch")

    # GENTLE mode: 2 attempts/chunk with a long pause, and ~6s spacing between chunks so we never
    # hammer the API. Whatever still fails this pass is just left for the next pass (the manifest
    # resumes), so the data converges over a few spaced-out runs instead of one aggressive burst.
    for s, cs, ce in todo:
        for attempt in range(2):
            try:
                d = _get_df(s, cs, ce)
                if len(d):
                    idx = d.index.tz_localize(None).normalize()
                    rows.append(pd.DataFrame({"symbol": s, "close": d["close"].to_numpy(),
                                              "iid": d["instrument_id"].to_numpy()}, index=idx))
                done.add(f"{s}|{cs}|{ce}")
                # persist progress after every success so a crash/throttle never loses ground
                pd.concat(rows).to_parquet(raw_f)
                man_f.write_text(json.dumps(sorted(done)))
                print(f"  ok {s} {cs}->{ce}  ({len(done)}/{len(want)})")
                time.sleep(6)                    # pace: space requests out, don't spam
                break
            except (_Timeout, Exception) as e:   # treat timeouts + connection drops the same
                if attempt == 1:
                    label = "TIMEOUT" if isinstance(e, _Timeout) else f"FAILED ({str(e)[:40]})"
                    print(f"  miss {label} {s} {cs}->{ce}  (will retry next pass)")
                else:
                    time.sleep(15)               # one calm backoff, then a single retry

    missing = [(s, cs, ce) for (s, cs, ce) in want if f"{s}|{cs}|{ce}" not in done]
    raw = pd.concat(rows) if rows else pd.DataFrame(columns=["symbol", "close", "iid"])

    # build the wide close + roll panels from the long raw table
    closes, rolls = {}, {}
    for s in symbols:
        sd = raw[raw["symbol"] == s]
        if sd.empty:
            continue
        sd = sd[~sd.index.duplicated(keep="last")].sort_index()
        closes[s] = pd.Series(sd["close"].to_numpy(), index=sd.index, name=s)
        rmask = pd.Series(sd["iid"].to_numpy(), index=sd.index).diff().ne(0)
        rmask.iloc[0] = False                        # first row isn't a roll
        rolls[s] = pd.Series(rmask.to_numpy(), index=sd.index, name=s)

    close = pd.DataFrame(closes).sort_index(); close.index.name = "date"
    roll = pd.DataFrame(rolls).reindex(close.index).fillna(False).astype(bool)

    if missing:
        print(f"INCOMPLETE: {len(missing)} chunks still missing -> re-run load_futures() to fill the gaps")
    else:
        try:
            close.to_parquet(c_close); roll.to_parquet(c_roll)
            if snapshot:
                close.to_parquet(CACHE_DIR / f"db_futures_close_{close.index.max().date()}.parquet")
            print("COMPLETE: all chunks fetched, wide cache + snapshot written")
        except Exception:
            pass
    return close, roll


def to_futures_returns(close: pd.DataFrame, roll: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    """Daily returns with two contaminated-return classes NaN'd:

    1. **Roll days** — cross-contract close-to-close is spurious (the contango/backwardation gap).
    2. **Non-positive prices** — a log-return is undefined across a price <= 0. This is REAL data, not
       an error: WTI crude (CL) settled NEGATIVE on 2020-04-20 (the storage-crunch event). log() of a
       negative is NaN and would silently corrupt the return into AND out of that day, so we mask both.
       (Even with simple returns a sign-crossing % change is meaningless, so we mask there too.)
    """
    with np.errstate(invalid="ignore"):
        rets = np.log(close).diff() if kind == "log" else close.pct_change(fill_method=None)
    rets = rets.replace([np.inf, -np.inf], np.nan)
    rets = rets.mask(roll.reindex_like(rets).fillna(False))      # 1. roll-day return
    nonpos = close <= 0
    bad = nonpos | nonpos.shift(1, fill_value=False)            # 2. day OF and day AFTER a non-positive price
    rets = rets.mask(bad.reindex_like(rets).fillna(False))
    return rets.dropna(how="all")


def quality_report(close: pd.DataFrame, roll: pd.DataFrame) -> dict:
    rets = to_futures_returns(close, roll)
    # Each contract trades its own calendar, so the UNION index has structural NaNs (not outages).
    # The usable joint sample is the INTERSECTION — days where every contract has a bar.
    common = int(close.notna().all(axis=1).sum())
    neg = close[close < 0].stack()
    return {
        "symbols": list(close.columns),
        "n_symbols": close.shape[1],
        "span": f"{close.index.min().date()} -> {close.index.max().date()}",
        "total_days_union": len(close),
        "common_days_all_present": common,                 # intersection = usable joint sample
        "per_symbol_days": {s: int(close[s].notna().sum()) for s in close.columns},
        "per_symbol_start": {s: str(close[s].first_valid_index().date()) for s in close.columns},
        "negative_prices": int((close < 0).sum().sum()),   # 1 = real: WTI on 2020-04-20
        "negative_price_detail": {str(d.date()): f"{s} {v:.2f}" for (d, s), v in neg.items()},
        "zero_prices": int((close == 0).sum().sum()),
        "duplicate_dates": int(close.index.duplicated().sum()),
        "roll_days_total": int(roll.to_numpy().sum()),
        "max_abs_daily_logret": round(float(rets.abs().max().max()), 3),  # finite after non-positive masking
    }
