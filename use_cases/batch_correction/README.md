# Batch Correction Use Case

This folder bundles the end-to-end example that ships with GlycoForge: simulate cohorts, inject batch effects, run ComBat, and inspect what changed. Think of it as the integration test for the core pipeline.

- `run_correction.ipynb` – interactive walkthrough. Load a config, kick off simulations in both modes, run correction, examine plots.
- `correction.py` – Python entry point (where calls `glycoforge.simulate`). Handles parameter grids, caching per-seed results, and orchestrates metrics.

## Configs to start from

Copy and edit one of the YAMLs in `sample_config/` at the project root. Point the notebook or script at your copy via `config_path`.

## Notebook workflow

1. Open `run_correction.ipynb`.
2. Set `project_root` (already `../..` inside the repo) and update `config_path` if you duplicated the YAML.
3. Execute cells in order:
   - load configuration (handles absolute path resolution for real data);
   - run `run_correction(config)` to simulate, inject, correct, and store metrics;
   - call `plot_parameter_grid_metrics` to auto-scan `results/<mode>/...` and produce figures.
4. Generated tables/plots land in the `results/` subfolder specified by your config.




