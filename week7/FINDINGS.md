# What the validation found

The project found higher-order joint downside structure in historical data, and the MPS could represent it in sample. It did not consistently improve out-of-sample tail or regime forecasts. That makes the model most useful here as a way to describe market dependence and test risk-model assumptions.

## Evidence for higher-order dependence

The deep-tail analysis compared observed joint crashes with a Gaussian copula fitted to the same empirical marginals. Across the tested books, three-asset crash probabilities exceeded that matched benchmark. At the 1% tail threshold, the observed-to-benchmark ratio was about 9.9 in the primary equity book, 5.7 in the cross-asset book, and 2.5 in the bank book. These are historical dependence findings, not forecasts. The [comparison tables](../analysis/results/) contain the estimates.

The [synthetic parity test](figures/parity_demo.png) gives a clean model check. Its two distributions have identical single-asset and pairwise behavior, but opposite three-way structure. Correlation and an MPS with bond dimension 1 cannot tell them apart. An MPS with bond dimension 2 can.

## What held up out of sample

The [walk-forward tail test](walkforward_tail_prediction.py) covered six portfolios, seven models, and held-out 126-day blocks. Against the Gaussian copula, the MPS variants recorded one significant win, six significant losses, and eleven ties. Increasing bond dimension often made the held-out result worse. A separate [capacity study](chi_study_books.py) found that concentrated sector books supported more model capacity than heterogeneous cross-asset books.

The [regime calibration test](msborn_calibration.py) found that the model's filtered stress-state probability should not be read as a probability of a future stress event. On the GFC book, days assigned a 90% to 100% stress-state probability were followed by the defined future-stress outcome only 18.3% of the time. A simple trailing-volatility percentile ranked future stress better than the tested HMMs.

Other checks showed where the model remained informative. [Group mutual information](figures/blockmi_heatmap.png) highlighted episodes such as the 2020 market shock and the 2023 regional-bank stress. [Reliability tests](figures/reliability.png) found some measures stable across random seeds, while others were sensitive to bin choice. In a later [Ising comparison](ising_competitor.py), no model family won consistently across books.

## Conclusion

The MPS can expose and describe dependence that a covariance summary misses. In these tests, that extra detail did not produce a dependable forecasting edge. The saved [notebooks](../notebooks/) and [summary results](results/) show the analysis behind both parts of that conclusion.
