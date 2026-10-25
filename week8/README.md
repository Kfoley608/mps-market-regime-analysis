# Week 8 — Glasgow higher-order dependence with an MPS Born machine

This workstream extends Cerrato, Crosby, Kim & Zhao (2017), *Relation between
higher order comoments and dependence structure of equity portfolio*,
Journal of Empirical Finance 40, 101–120,
[DOI 10.1016/j.jempfin.2016.11.007](https://doi.org/10.1016/j.jempfin.2016.11.007).

The paper sorts assets by beta, coskewness, and cokurtosis, then studies
time-varying asymmetric tail dependence and one-day-ahead VaR/ES. This project
keeps that research question but replaces the paper's bivariate GHST-GAS copula
with the repository's multivariate MPS Born machine.

## What is new in this project

The MPS is fitted to the full asset-return vector in each causal rolling window.
It then generates joint scenarios from which we calculate:

- lower-tail dependence, upper-tail dependence, and their difference;
- the probability of a k-way joint downside event;
- joint-tail lift relative to a covariance-only Gaussian model;
- one-day-ahead VaR and expected shortfall for high-minus-low portfolios;
- bond-dimension sensitivity (chi=1,2,4 by default).

The interpretation is deliberately aligned with the project description:
MPS is tested as a higher-order dependence and tail-risk representation, not
assumed to be a superior unconditional return predictor.

## Causal protocol

For forecast date t:

1. use only the preceding 250 observations;
2. estimate beta, coskewness, and cokurtosis against the equal-weight market;
3. sort assets into five low-to-high portfolios;
4. fit Gaussian, empirical-bootstrap, and MPS scenario models;
5. forecast the next-day high-minus-low return and evaluate VaR/ES;
6. keep the MPS chain ordering fixed from the first training window.

The default step 21 is practical for a full MPS run. Use step 1 for
the paper's daily rolling cadence. All quantiles, sorting statistics, covariance
estimates, bin edges, and model parameters are training-window only.

## Run

From the repository root:

```text
python week8/glasgow_mps_risk.py --books gfc --smoke
python week8/glasgow_mps_risk.py --books gfc --step 21 --n-sim 20000
python week8/glasgow_mps_risk.py --books gfc primary banks --step 21 --n-sim 20000
```

For a close replication of the paper's daily forecasts:

```text
python week8/glasgow_mps_risk.py --books gfc --step 1 --window 250 --alpha 0.01
```

Set DATA_EXT when parquet books live outside the repository:

```text
$env:DATA_EXT="C:\path\to\data"
python week8/glasgow_mps_risk.py --books gfc --smoke
```

## Outputs

Outputs are written to week8/results/:

- glasgow_comoments_<book>.csv: rolling asset-level beta/coskewness/cokurtosis;
- glasgow_forecasts_<book>.csv.gz: every model/characteristic forecast;
- glasgow_summary_<book>.csv: VaR/ES and dependence summary;
- glasgow_report_<book>.md: presentation-oriented readout scaffold;
- glasgow_metadata_<book>.json: reproducibility metadata.

## Suggested presentation claim

> When individual marginals and pairwise covariance are held fixed, the MPS
> gives a direct estimate of higher-order joint downside structure. Its value
> should be judged by joint-tail lift, asymmetric dependence, and conditional
> VaR/ES backtests—not by an unsupported claim that it predicts ordinary
> daily returns better.

## Important scope note

The original paper uses US S&P 500 and UK FTSE 100 stocks, annual
capitalisation-weighted rebalancing, and a dynamic GHST-GAS copula. The current
repository's books are different and generally lack market-cap weights, so this
implementation uses equal-weight characteristic-sorted portfolios and states
the deviation explicitly. It is therefore a causal MPS extension/replication,
not a literal data-for-data reproduction.
