# Selected analysis

This folder keeps the scripts and small result tables behind the main dependence findings.

- `deep_tail_copula.py` compares observed joint downside events with a Gaussian copula using empirical marginals. Run `python analysis/deep_tail_copula.py --synthetic` to check the method on a synthetic example.
- `parallel_rolling_v3.py` builds causal rolling MPS features.
- `tier1_analysis_v3.py` compares those features with classical indicators.
- `production_model.py` contains the regime model used in the later analysis.

The CSV files in `results/` contain the matched-marginal deep-tail comparisons. Other intermediate sweeps and scratch outputs have been left out of this shareable copy. Input price panels are also omitted, so rerunning the market-data scripts requires your own permitted data.
