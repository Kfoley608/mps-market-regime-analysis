# Market Regimes and Tail Dependence with Matrix Product States

Authored by Kevin Foley

This project studies whether a matrix product state (MPS) model can describe changes in how asset returns move together, especially during market stress. The model estimates a joint distribution of returns. I compare its tail and regime measures with standard indicators and test whether they improve forecasts out of sample.

## How to view the project

You can read the results directly on GitHub. No setup is needed.

1. Start with [the findings](week7/FINDINGS.md). They give the final validation result, including where the MPS worked and where it did not.
2. Open [the MPS walkthrough](notebooks/00-MPS-explained.ipynb) for a visual introduction to the model.
3. Read [the cross-asset analysis](notebooks/16-crossasset-higher-order.ipynb) and [the regime results](notebooks/21-msborn-regime-detection-results.ipynb). GitHub displays the saved charts and tables inside each notebook.
4. Explore the [three risk gauges](week6/Gauges-Explained.md) and their notebooks: [complexity](week6/Gauge-Complexity-Fragility.ipynb), [tail asymmetry](week6/Gauge-Fear-vs-Euphoria.ipynb), and [joint crashes](week6/Gauge-Joint-Crash.ipynb).
5. For the later tests, see [the parity example](week7/figures/parity_demo.png), [the group dependence heatmap](week7/figures/blockmi_heatmap.png), and [the higher-order risk extension](week8/README.md).

The notebooks have saved outputs, so you can follow the analysis without running code or obtaining the input datasets. GitHub may take a moment to render the larger notebooks. If a preview fails, download the notebook and open it in Jupyter or VS Code.

## Main findings

- Historical joint downside moves exceeded a Gaussian copula benchmark with the same estimated marginals. In the deepest three-asset tail test, the measured excess varied by portfolio and reached about 9.9 times the benchmark in the primary equity book. See the [tail comparison tables](analysis/results/).
- The MPS represented higher-order dependence in sample. A synthetic parity example shows a pattern that pairwise correlation cannot distinguish and an MPS with bond dimension 2 can.
- That richer representation did **not** translate into dependable out-of-sample tail or regime forecasts. In the Week 7 tests, simple baselines usually tied or beat the MPS. The strongest use of the model here is describing and diagnosing dependence, not claiming a trading signal.

## What is in this copy

| Folder | Contents |
| --- | --- |
| [`notebooks/`](notebooks/) | Model walkthroughs, cross-asset analysis, rolling comparisons, and regime results |
| [`week6/`](week6/) | Saved gauge notebooks and an explanation of each measure |
| [`week7/`](week7/) | Final validation findings, figures, code, and compact result tables |
| [`week8/`](week8/) | An extension testing characteristic sorts, joint tails, and VaR/ES |
| [`src/mps_regime/`](src/mps_regime/) | Model, data, feature, and evaluation code |
| [`analysis/`](analysis/) | Selected analysis scripts and tail comparison tables |
| [`docs/`](docs/) | Notes on the training method and its early correction |

The original price panels and licensed data are not included. The saved notebook outputs and summary tables are here for reading. [Data notes](data/README.md) explain what would be needed to rerun the analyses.

## Running selected code

Python 3.11 or newer is required. From a terminal in the repository folder:

```bash
python -m venv .venv
python -m pip install -e .
python -m pytest
```

The parity demonstration uses synthetic data and can run without the original market panels:

```bash
python week7/parity_model_demo.py --smoke
```

Market-data analyses need their corresponding price panels. Place your own permitted datasets in `data/` or set `DATA_EXT` to a local data folder. The saved results reflect the datasets and settings used for this project; a new download may produce different numbers.
