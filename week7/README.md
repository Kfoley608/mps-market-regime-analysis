# Validation protocol

The [findings](FINDINGS.md) summarize the final tests. Scripts in this folder compare the MPS with Gaussian copulas, a Gaussian HMM, an Ising model, and simpler baselines on held-out data.

Each analysis estimates bins, thresholds, ordering, and model parameters on its training window, then scores later observations. Price panels are not included in this public copy. Scripts look in `data/`, `data_external/`, or the folder named by `DATA_EXT`.

The small CSVs in `results/` are summary outputs. The figures in `figures/` show the parity example, group dependence, and model reliability.

The [parity demonstration](parity_model_demo.py) uses synthetic data and can run without market panels:

```bash
python week7/parity_model_demo.py --smoke
```
