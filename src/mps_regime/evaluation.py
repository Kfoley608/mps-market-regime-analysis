"""Evaluation protocol.

Tier 1 (the honesty layer, run FIRST): regress each MPS feature on the full classical set;
report rolling-window and full-sample (HAC) R^2. Features with R^2 > 0.9 are declared
"repackaged volatility" — that finding ships either way. Later tiers: regime separation,
VaR-breach behavior in stress periods, sensitivity to bond dimension and ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def redundancy_r2(mps_features: pd.DataFrame, classical: pd.DataFrame) -> pd.DataFrame:
    """Tier 1: per-MPS-feature R^2 when regressed on the classical indicator set."""
    raise NotImplementedError("Week 3: OLS per feature, HAC errors via statsmodels/arch.")


def incremental_information(target: pd.Series, classical: pd.DataFrame,
                            mps_features: pd.DataFrame) -> dict:
    """Does adding MPS features improve prediction of `target` (e.g. fwd vol / drawdown)
    over classical-only? Nested-model comparison."""
    raise NotImplementedError


def comparison_table(*series_by_method: pd.Series) -> pd.DataFrame:
    """Assemble the headline method-vs-method comparison table for the report."""
    raise NotImplementedError
