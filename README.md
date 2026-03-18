<img src="glycoforge_logo.jpg" alt="GlycoForge logo" width="300">

**GlycoForge** is a simulation tool for **generating glycomic relative-abundance datasets** with customizable biological group differences and controllable batch-effect injection.

## Key Features

- **Two simulation modes**: Fully synthetic or templated (extract factor from input reference data + simulate batch effect)
- **Paired multi-glycome simulation**: `simulate_paired()` generates two glycomic datasets (e.g., _N_- and _O_-glycomics) from the same biological samples, with shared batch labels and optional controllable cross-class coupling
- **Controllable effects injection**: Systematic grid search over biological effect or batch effect strength parameters
- **Motif-level effects**: For both bio and batch effects, desired motif differences (e.g., `Neu5Ac: down`) can be introduced. These are propagated in a dynamically constructed biosynthetic network to ensure physiological glycomics data (e.g., corresponding increase in desialylated glycans in the example of `Neu5Ac: down`)
- **MNAR missing data simulation**: Mimics left-censored patterns biased toward low-abundance glycans

## Quick Start

### Installation

* **Python 3.10–3.12 required** (`>=3.10,<3.13`). We recommend creating a dedicated virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

* Core dependency: `glycowork>=1.7.1`

```bash
pip install glycoforge
```

OR

```bash
git clone https://github.com/BojarLab/GlycoForge.git
cd GlycoForge
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Usage

See [run_simulation.ipynb](run_simulation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BojarLab/GlycoForge/blob/main/run_simulation.ipynb) for interactive simulation examples, or [use_cases/batch_correction/](use_cases/batch_correction) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BojarLab/GlycoForge/blob/main/use_cases/batch_correction/run_correction.ipynb) for batch correction workflows, and [benchmarking_batch_effect_removal.ipynb](use_cases/batch_correction/benchmarking_batch_effect_removal.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BojarLab/GlycoForge/blob/main/use_cases/batch_correction/benchmarking_batch_effect_removal.ipynb) for the full six-method benchmark.

## How the simulator works

We keep everything in the CLR (centered log-ratio) space:

- First, draw a healthy baseline composition from a Dirichlet prior: `p_H ~ Dirichlet(alpha_H)`.
- Flip to CLR: `z_H = clr(p_H)`.
- For selected glycans, push the signal using real or synthetic effect sizes: `z_U = z_H + m * lambda * d_robust`, where `m` is the differential mask, `lambda` is `bio_strength`, and `d_robust` is the effect vector after `robust_effect_size_processing`.
    - **Simplified mode**: draw synthetic effect sizes (log-fold changes) and pass them through the same robust processing pipeline.
    - **Hybrid mode**: start from the Cohen's *d* values returned by `glycowork.get_differential_expression`; `define_differential_mask` lets you restrict the injection to significant hits or top-*N* glycans before scaling.
- Invert back to proportions: `p_U = invclr(z_U)` and scale by `k_dir` to get `alpha_U`, note that the healthy and unhealthy Dirichlet strengths use different `k_dir` values, and a separate `variance_ratio` controls their relative magnitude.
- Batch effects ride on top as direction vectors `u_b`, so a clean CLR sample `Y_clean` becomes `Y_with_batch = Y_clean + kappa_mu * sigma * u_b + epsilon`, with `var_b` controlling spread.

## Simulation Modes

`glycoforge.simulate()` generates a single glycomic dataset per entity in two modes controlled by `data_source`. For paired multi-glycome data from the same samples, use `glycoforge.simulate_paired()` instead (see below). Configuration files are in `sample_config/`.

<details>
<summary><b>Synthetic mode (<code>data_source="simulated"</code>)</b> – Fully synthetic simulation (click to show detail introduction)</summary>

<br>

No real data dependency. Ideal for controlled experiments with known ground truth.

**Pipeline steps:**

1. Initializes log-normal healthy baseline: `alpha_H` sampled from log-normal distribution (μ=0, σ=1, fixed seed=42), rescaled to mean of 10
2. For each random seed, generates `alpha_U` by randomly scaling `alpha_H`:
   - `up_frac` (default 30%) upregulated with scale factors from `up_scale_range=(1.1, 3.0)`
   - `down_frac` (default 35%) downregulated with scale factors from `down_scale_range=(0.3, 0.9)`
   - Remaining glycans (~35%) stay unchanged
3. Samples clean cohorts from `Dirichlet(alpha_H)` and `Dirichlet(alpha_U)` with `n_H` healthy and `n_U` unhealthy samples
4. Defines batch effect direction vectors `u_dict` once per simulation run (fixed seed ensures reproducible batch geometry across parameter sweep)
5. Applies batch effects controlled by `kappa_mu` (shift strength) and `var_b` (variance scaling)
6. Optionally applies MNAR (Missing Not At Random) missingness:
   - `missing_fraction`: proportion of missing values (0.0–1.0)
   - `mnar_bias`: intensity-dependent bias (default 2.0, range 0.5–5.0)
   - Left-censored pattern: low-abundance glycans more likely to be missing
7. Grid search over `kappa_mu` and `var_b` produces multiple datasets under identical batch effect structure

**Key parameters:** `n_glycans`, `n_H`, `n_U`, `kappa_mu`, `var_b`, `missing_fraction`, `mnar_bias`

</details>

<details>
<summary><b>Templated mode (<code>data_source="real"</code>)</b> – Extract biological effect from input reference data + simulate batch effect (click to show detail introduction)</summary>

<br>

Starts from real glycomics data to preserve biological signal structure. Accepts CSV file or `glycowork.glycan_data` datasets.

**Pipeline steps:**

1. Loads CSV and extracts healthy/unhealthy sample columns by prefix (configurable via `column_prefix`)
2. Runs CLR-based differential expression via `glycowork.get_differential_expression` to compute Cohen's d effect sizes
3. Reindexes effect sizes to match input glycan order (fills missing glycans with 0.0)
4. Applies `differential_mask` to select which glycans receive biological signal injection:
   - `"All"`: inject into all glycans
   - `"significant"`: only glycans marked significant by glycowork
   - `"Top-N"`: top N glycans by absolute effect size (e.g., `"Top-10"`)
5. Processes effect sizes through `robust_effect_size_processing`:
   - Centers effect sizes to remove global shift
   - Applies Winsorization to clip extreme outliers (auto-selects percentile 85–99, or uses `winsorize_percentile`)
   - Normalizes by baseline (`baseline_method`: median, MAD, or p75)
   - Returns normalized `d_robust` scaled by `bio_strength`
6. Injects effects in CLR space: `z_U = z_H + mask * bio_strength * d_robust`
7. Converts back to proportions: `p_U = invclr(z_U)`
8. Scales by Dirichlet concentration: `alpha_H = k_dir * p_H` and `alpha_U = (k_dir / variance_ratio) * p_U`
9. Samples clean cohorts from `Dirichlet(alpha_H)` and `Dirichlet(alpha_U)` with `n_H` healthy and `n_U` unhealthy samples
10. Defines batch effect direction vectors `u_dict` once per run (fixed seed ensures fair comparison across parameter combinations)
11. Applies batch effects: `y_batch = y_clean + kappa_mu * sigma * u_b + epsilon`, where `epsilon ~ N(0, sqrt(var_b) * sigma)`
12. Optionally applies MNAR missingness (same as Simplified mode)
13. Grid search over `bio_strength`, `k_dir`, `variance_ratio`, `kappa_mu`, `var_b` to systematically test biological signal and batch effect interactions

**Key parameters:** `data_file`, `column_prefix`, `bio_strength`, `k_dir`, `variance_ratio`, `differential_mask`, `winsorize_percentile`, `baseline_method`, `kappa_mu`, `var_b`, `missing_fraction`, `mnar_bias`

</details>

<details>
<summary><b>Paired mode (<code>simulate_paired()</code>)</b> – Two glycomic classes from the same biological samples (click to show detail introduction)</summary>

<br>

Generates two glycomic datasets (e.g., _N_- and _O_-glycomics) that share sample identity: the same `n_H` healthy and `n_U` unhealthy individuals appear in both, so `bio_labels`, `batch_labels`, and column names are identical across glycomes. Each glycome is otherwise independently parameterised (different glycan counts, Dirichlet parameters, biological effect structures, and batch direction vectors).

**Pipeline steps:**

1. Draws independent log-normal healthy baselines `alpha_H_A` and `alpha_H_B` (fixed seeds 42/43 for reproducibility)
2. Generates per-glycome `alpha_U` using independent seeds so biological effects are not correlated by seed reuse
3. Simulates clean compositional data for both glycomes from their respective Dirichlet parameters
4. Optionally injects cross-class coupling in CLR space via shared latent factors `Z ~ N(0, I)`:
   - `Y_A_clr += coupling_strength * (Z @ U_A.T) * sigma_A`
   - `Y_B_clr += coupling_strength * (Z @ U_B.T) * sigma_B`
   - At `coupling_strength=0` the two glycomes are conditionally independent given sample labels; induced HSIC scales as `coupling_strength²`
   - Direction matrices `U_A`, `U_B` can be biased toward motif-matching glycans via `coupling_motif_A/B`
5. Round-trips through `invclr` to restore simplex validity after coupling injection
6. Applies shared batch labels with independent per-glycome direction vectors (same samples in the same batches, but different glycans affected)
7. Applies MNAR missingness independently per glycome (independent seeds prevent artificially correlated missing-value patterns)

**Key parameters:** `n_glycans_A/B`, `bio_strength_A/B`, `k_dir_A/B`, `variance_ratio_A/B`, `coupling_strength`, `n_coupling_components`, `coupling_motif_A/B`, `kappa_mu`, `var_b`, `missing_fraction`, `mnar_bias`

</details>

## Use Cases

The [use_cases/batch_correction/](use_cases/batch_correction) directory demonstrates:
- Call `glycoforge` simulation, and then apply correction workflow
- Six-method batch correction benchmark (ComBat, Percentile, Ratio-ComBat, Harmony, limma-style, Stratified ComBat) across a parameter grid of biological signal strengths and batch effect severities
- Batch correction effectiveness metrics visualization

## Limitation

**Two biological groups only**: Current implementation targets healthy/unhealthy setup. Supporting multi-stage disease (≥3 groups) requires refactoring Dirichlet parameter generation and evaluation metrics.

## Citation

If you use GlycoForge in your research, please cite:

> Hu, S. and Bojar, D. (2026). GlycoForge generates realistic glycomics data under known ground truth for rigorous method benchmarking. *bioRxiv*, doi:10.64898/2026.02.20.707134

**BibTeX:**
```bibtex
@article{hu2026glycoforge,
  title   = {GlycoForge generates realistic glycomics data under known ground truth for rigorous method benchmarking},
  author  = {Hu, Siyu and Bojar, Daniel},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.02.20.707134},
  url     = {https://www.biorxiv.org/content/10.64898/2026.02.20.707134v1}
}
```