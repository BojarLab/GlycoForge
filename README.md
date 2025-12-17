# GlycoForge

GlycoForge is a simulation tool for generating glycomic relative-abundance datasets with customizable biological group differences and controllable batch-effect injection

## Environment setup

- Python >= 3.10 
- Core dependency: `glycowork>=1.6.4`, NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn

```bash
git clone https://github.com/BojarLab/GlycoForge.git
cd GlycoForge
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## How the simulator works

We keep everything in the CLR (centered log-ratio) world because glycan abundances live on the simplex.

- First, draw a healthy baseline composition from a Dirichlet prior: `p_H ~ Dirichlet(alpha_H)`.
- Flip to CLR: `z_H = clr(p_H)`.
- For selected glycans, push the signal using real or synthetic effect sizes: `z_U = z_H + m * lambda * d_robust`, where `m` is the differential mask, `lambda` is `bio_strength`, and `d_robust` is the effect vector after `robust_effect_size_processing`.
    - **Simplified mode**: draw synthetic effect sizes (log-fold changes) and pass them through the same robust processing pipeline.
    - **Hybrid mode**: start from the Cohen’s *d* values returned by `glycowork.get_differential_expression`; `define_differential_mask` lets you restrict the injection to significant hits or top-*N* glycans before scaling.
- Invert back to proportions: `p_U = invclr(z_U)` and scale by `k_dir` to get `alpha_U`, note that the healthy and unhealthy Dirichlet strengths use different `k_dir` values, and a separate `variance_ratio` controls their relative magnitude.
- Batch effects ride on top as direction vectors `u_b`, so a clean CLR sample `y_clean` becomes `y_batch = y_clean + kappa_mu * u_b + epsilon`, with `var_b` controlling spread.


## Two modes, one pipeline


There are 2 mode in simulation pipeline `glycoforge/pipeline.py::simulate` entry point; you can swap the configuration:

> Standard configuration files in the `sample_config/` and you can run 2 modes simualtion in the notebook [run_simulation.ipynb](run_simulation.ipynb).

- **Simplified mode (`data_source="simulated"`)** – fully synthetic simulation without real data dependency. You specify the number of glycans (`n_glycans`), and the pipeline:
  1. Initializes a uniform healthy baseline: `alpha_H = ones(n_glycans) * 10`
  2. For each random seed, generates `alpha_U` by randomly scaling a fraction of `alpha_H` values:
     - `up_frac` (default 30%) of glycans are upregulated with scale factors from `up_scale_range=(1.1, 3.0)`
     - `down_frac` (default 30%) are downregulated with scale factors from `down_scale_range=(0.3, 0.9)`
     - Remaining glycans (~40%) stay unchanged
  3. Samples clean cohorts from `Dirichlet(alpha_H)` and `Dirichlet(alpha_U)` with `n_H` healthy and `n_U` unhealthy samples
  4. Defines batch effect direction vectors `u_dict` once per simulation run (fixed seed ensures reproducible batch geometry across parameter sweep)
  5. Applies batch effects controlled by `kappa_mu` (shift strength) and `var_b` (variance scaling)
  6. Grid search over `kappa_mu` and `var_b` produces multiple simulated datasets under the same batch effect structure, enabling fair comparison of batch correction effectiveness
  
  This mode is ideal for controlled experiments where you want to test batch correction methods under known ground truth with varying batch effect strengths.

- **Hybrid mode (`data_source="real"`)** – starts from real glycomics data to preserve biological signal structure. You provide a CSV file or import CSV file from `glycowork.glycan_data`, and the pipeline:
  1. Loads the CSV and extracts healthy/unhealthy sample columns by prefix (configurable via `column_prefix`)
  2. Runs CLR-based differential expression analysis via `glycowork.get_differential_expression` to compute Cohen's *d* effect sizes for each glycan
  3. Reindexes effect sizes to match input glycan order (fills missing glycans with 0.0 if glycowork filters some out)
  4. Applies `differential_mask` to select which glycans receive biological signal injection:
     - `"All"`: inject into all glycans
     - `"significant"`: only glycans marked as significant by `glycowork` (using sample-size-adjusted alpha and corrected p-values)
     - `"Top-N"`: top N glycans by absolute effect size (e.g., `"Top-10"`)
  5. Processes effect sizes through `robust_effect_size_processing`:
     - Centers effect sizes to remove global shift
     - Applies Winsorization to clip extreme outliers (auto-selects percentile 85-99 based on outlier severity, or uses `winsorize_percentile` if specified)
     - Normalizes by baseline (median, Median Absolute Deviation, or 75th percentile via `baseline_method`)
     - Returns normalized effect sizes `d_robust` that caller scales by `bio_strength`
  6. Injects effects in CLR space: `z_U = z_H + mask * bio_strength * d_robust`, where `z_H` is the healthy baseline CLR, `m` is the differential mask
  7. Converts back to proportions: `p_U = invclr(z_U)`
  8. Scales by Dirichlet concentration: `alpha_H = k_dir * p_H` and `alpha_U = (k_dir / variance_ratio) * p_U`
  9. Samples clean cohorts from `Dirichlet(alpha_H)` and `Dirichlet(alpha_U)` with `n_H` healthy and `n_U` unhealthy samples
  10. Defines batch effect direction vectors `u_dict` once per simulation run (**important**: `u_dict` is fixed across all parameter combinations in grid search using a reproducible seed, ensuring fair comparison between different `kappa_mu`/`var_b` settings)
  11. Applies batch effects controlled by `kappa_mu` (shift strength) and `var_b` (variance scaling): `y_batch = y_clean + kappa_mu * sigma * u_b + epsilon`, where `epsilon ~ N(0, sqrt(var_b) * sigma)`
  12. Grid search over `bio_strength`, `k_dir`, `variance_ratio`, `kappa_mu`, `var_b` to systematically test how biological signal strength and batch effects interact under controlled batch effect geometry
  
  This mode keeps the virtual cohort faithful to real biological signal geometry while letting you systematically vary signal strength (`bio_strength`), concentration (`k_dir`), variance (`variance_ratio`), and batch effects for realistic batch correction benchmarking.


## Use Cases

- [use_cases/batch_correction/](use_cases/batch_correction) currently walks through the two-phase simulation + ComBat correction flow, plots, and metrics. The notebook `run_correction.ipynb` walks through both simulation mode step-by-step.

## Limitations and future work

1. **Only two biological groups** – now the generator targets a healthy/unhealthy setup. Supporting >=3 cohorts (e.g., multi-stage disease) will require refactoring the Dirichlet parameter builder and downstream evaluation logic.
2. **Packaging roadmap** – everything is source-first right now. Long term the goal is to publish a trimmed core on PyPI once the API stabilizes.