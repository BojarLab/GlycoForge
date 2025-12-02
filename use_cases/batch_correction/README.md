# Batch Correction Use Case

This folder bundles the end-to-end example that ships with GlycoForge: simulate cohorts, inject batch effects, run ComBat, and inspect what changed. Think of it as the integration test for the core pipeline.

- `run_correction.ipynb` – interactive walkthrough. Load a config, kick off simulations in both modes, run correction, examine plots.
- `correction.py` – Python entry point (what the notebook calls under the hood). Handles parameter grids, caching per-seed results, and orchestrates metrics.
- `visualization.py` – helper module for plotting batch-effect change, biological preservation, and differential-expression heatmaps.
- `methods.py`, `evaluation.py`, `visualization.py` – glue utilities for batch correction (ComBat setup, metric aggregation).
- `results/` – default output location for notebook runs. Safe to wipe when you want a fresh slate.

## Configs to start from

Copy one of the YAMLs in `sample_config/` at the project root:

- `simplified_mode_config.yaml` – fully synthetic priors.
- `hybrid_mode_config.yaml` – get effect sizes from [data/glycomics_human_leukemia_O_PMID34646384.csv](data/glycomics_human_leukemia_O_PMID34646384.csv) (or your own CSV).

Point the notebook or script at your copy via `config_path`.

## Notebook workflow

1. Open `run_correction.ipynb`.
2. Set `project_root` (already `../..` inside the repo) and update `config_path` if you duplicated the YAML.
3. Execute cells in order:
   - load configuration (handles absolute path resolution for real data);
   - run `run_correction(config)` to simulate, inject, correct, and store metrics;
   - call `plot_parameter_grid_metrics` to auto-scan `results/<mode>/...` and produce figures.
4. Generated tables/plots land in the `results/` subfolder specified by your config.

## Key scripts in motion

- `correction.py::run_correction`
  - reads the config file, builds parameter combinations (seeds × grid), and calls `glycoforge.pipeline.simulate` for each combo;
  - caches intermediate CSV/JSON artifacts per seed;
  - aggregates metrics into `comprehensive_metrics_seed*.json` downstream consumers rely on.
- `visualization.py::plot_parameter_grid_metrics`
  - scans the output directory, computes mean/std across seeds for each metric, draws line charts and 2D heatmaps for parameter grids.



