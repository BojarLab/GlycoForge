# GlycoForge

GlycoForge is a simulation tool for generating glycomic relative-abundance datasets with customizable biological group differences and controllable batch-effect injection

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

- **Simplified mode (`data_source="simulated"`)** – everything is analytic. You choose Dirichlet strength (`alpha_H`, `k_dir`), decide the fraction of glycans to up/down regulate, and we sample clean cohorts plus their batch-perturbed twins across whichever parameter grid you specify. Great for sensitivity experiments where biology and batch are fully scripted.
- **Hybrid mode (`data_source="real"`)** – point to a CSV (e.g., patient data in `data/`), we run CLR-based differential expression, then pass the effect sizes through a robust processing step (`robust_effect_size_processing`) that clips extreme fold changes before injecting them back into the simulator. That keeps the virtual cohort faithful to the real geometry while still letting you dial `bio_strength`, `k_dir`, `kappa_mu`, `var_b`, etc.


## Examples

Standard configuration files in the `sample_config/` and you can run 2 modes simualtion in the notebook [run_simulation.ipynb](run_simulation.ipynb).


## Use-case

- [use_cases/batch_correction/](use_cases/batch_correction) currently walks through the two-phase simulation + ComBat correction flow, plots, and metrics. The notebook `run_correction.ipynb` walks through both simulation mode step-by-step.

## Limitations and future work

1. **Only two biological groups** – today the generator targets a healthy/unhealthy setup. Supporting >=3 cohorts (e.g., multi-stage disease) will require refactoring the Dirichlet parameter builder and downstream evaluation logic.
2. **Packaging roadmap** – everything is source-first right now. Long term the goal is to publish a trimmed core on PyPI once the API stabilizes.