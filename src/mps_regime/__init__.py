"""mps_regime — Quantum-inspired market regime & tail-complexity feature module.

Fit an MPS (Born machine) to the joint distribution of discretized multi-asset returns on
rolling windows, then extract fit-complexity diagnostics as regime / tail-dependence features.

Pipeline:  data -> binning -> mps (fit) -> features -> evaluation (vs baselines) -> regimes
Portfolios (portfolios.py) are an evaluation axis applied on top of a single fitted MPS.
"""

__version__ = "0.0.1"
