# Batch Correction Use Case

End-to-end pipeline for batch effect simulation and ComBat correction: simulate cohorts with batch effects, apply ComBat correction, and evaluate effectiveness through comprehensive metrics and visualizations.

## Configuration

Copy and customize YAML configs from `sample_config/` at project root:
- **simplified_mode_config.yaml**: Fully synthetic simulation
- **hybrid_mode_config.yaml**: Extract biological effect from input reference data + batch effect injection


Run interactive examples: 
[run_correction.ipynb](use_cases/batch_correction) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BojarLab/GlycoForge/blob/main/use_cases/batch_correction/run_correction.ipynb).


## ComBat Implementation

The `combat()` function in `methods.py` implements parametric empirical Bayes batch correction:

**Key features:**
- Adapted for CLR/ALR-transformed compositional data
- Parametric shrinkage of batch effect estimates (default `parametric=True`)
- Optional biological covariate preservation via design matrix (`mod` parameter)
- Returns corrected data in same space as input

**Algorithm steps:**
1. Standardize data by removing design matrix effects
2. Estimate batch-specific location (`gamma_hat`) and scale (`delta_hat`) parameters
3. Apply empirical Bayes shrinkage to stabilize estimates
4. Adjust data by removing estimated batch effects

**Usage:**
```python
from methods import combat

# data: glycans x samples (CLR-transformed)
# batch: batch labels for each sample
# mod: optional biological group design matrix
corrected_data = combat(data, batch, mod=None, parametric=True)
```

## Evaluation Metrics

The pipeline generates comprehensive metrics comparing three conditions:
- **Y_clean**: Ground truth without batch effects
- **Y_with_batch**: Data with injected batch effects
- **Y_corrected**: ComBat-corrected data

### Batch Effect Metrics
- **Silhouette Score**: Lower is better (reduced batch clustering)
- **kBET**: Batch entropy test (lower is better)
- **LISI**: Local Inverse Simpson's Index (higher is better for mixing)
- **ARI**: Adjusted Rand Index (lower is better)
- **Compositional Effect Size**: Batch-driven compositional shift (lower is better)
- **PCA Batch Effect**: Variance explained by batch (lower is better)

### Biological Preservation Metrics
- **Biological Variability Preservation**: Ratio of biological variance retained (higher is better)
- **Conserved Differential Proportion**: Fraction of true differential glycans preserved (higher is better)

### Differential Expression Recovery
- **True Positive Rate**: Correctly identified differential glycans
- **False Positive Rate**: Incorrectly identified differential glycans
- **F1 Score, Precision, Recall**: Overall detection quality








