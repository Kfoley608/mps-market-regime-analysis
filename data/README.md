# Data for reproduction

The input price panels are not distributed with this repository. The notebooks include saved charts and tables for viewing the project without those files.

The analyses used public market sources alongside datasets that require separate access, including CRSP through WRDS and futures data through DataBento. See the loader code in [`src/mps_regime/`](../src/mps_regime/) for the source and cleaning steps. To rerun an analysis, supply the required price panel in `data/` or set `DATA_EXT` to a folder containing it. Check each script for its expected filename and columns.

Do not treat a fresh market-data pull as an exact reproduction of the saved outputs. Vendor revisions, coverage, and access rights can differ.
