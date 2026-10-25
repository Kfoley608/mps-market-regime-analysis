"""Portfolios = evaluation axis. A portfolio is a weight vector over the universe applied on
top of the SAME fitted MPS / realized returns. Special string weights ('equal', 'inverse_vol')
are resolved at eval time from the train window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def resolve_weights(spec, tickers: list[str], train_returns: pd.DataFrame) -> pd.Series:
    """Turn a config weight spec into a normalized weight Series indexed by ticker.

    spec may be:
      - 'equal'       : equal weight across the universe
      - 'inverse_vol' : inverse-volatility (risk parity) from the train window
      - dict          : explicit {ticker: weight}
    """
    raise NotImplementedError("Week 3: resolve equal / inverse_vol / explicit specs.")


def portfolio_returns(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """Realized portfolio return series from weights and an asset returns panel."""
    raise NotImplementedError
