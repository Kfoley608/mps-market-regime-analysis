"""Regime classification / clustering demo.

Cluster the MPS feature time-series into regimes and compare to a classical regime model
(statistical jump model via `jumpmodels`, or a Gaussian HMM). Used both for the deliverable
demo and to show whether MPS features sharpen regime separation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def jump_model_regimes(features: pd.DataFrame, n_states: int = 2):
    """Fit a statistical jump model (jumpmodels) on the feature series -> regime labels."""
    raise NotImplementedError("Week 4-5.")


def hmm_regimes(features: pd.DataFrame, n_states: int = 2):
    """Gaussian HMM baseline regime labels for comparison."""
    raise NotImplementedError("Week 4-5.")
