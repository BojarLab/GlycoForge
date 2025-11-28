# /proj/naiss2024-5-630/users/x_siyhu/GlycoForge/use_cases/batch_correction/__init__.py

"""
Batch Correction Use Case
=========================

This module provides a complete workflow for applying and evaluating batch effect correction methods on glycomics data.

It includes:
- Correction methods such as ComBat.
- A suite of evaluation metrics to quantify batch effects and biological signal preservation.
- Visualization tools for inspecting data before and after correction.

Main functions:
- `combat`: Applies the ComBat algorithm for batch correction.
- `check_batch_effect`: Performs a quick statistical check for batch effects.
- `quantify_batch_effect_impact`: Calculates a comprehensive set of metrics to measure batch effect severity.
- `compare_differential_expression`: Compares differential expression results between datasets.
- `plot_pca`: Generates PCA plots colored by batch and biological groups.
- `visualize_batch_correction_results`: Creates summary plots for batch correction effectiveness.

Additional function:
- `run_correction_pipeline`: Runs the full correction pipeline with optional parameter tuning.
"""

from .methods import combat
from .evaluation import (
    check_batch_effect,
    quantify_batch_effect_impact,
    compare_differential_expression,
    evaluate_biological_preservation,
    generate_comprehensive_metrics
)
from .visualization import (
    plot_pca,
    visualize_batch_correction_results,
    visualize_differential_expression_metrics,
    plot_parameter_grid_metrics,
    compare_batch_correction_across_masks,
    compare_differential_expression_across_masks
)
from .correction import run_correction

__all__ = [
    # from methods.py
    'combat',
    
    # from evaluation.py
    'check_batch_effect',
    'quantify_batch_effect_impact',
    'compare_differential_expression',
    'evaluate_biological_preservation',
    'generate_comprehensive_metrics',

    # from visualization.py
    'plot_pca',
    'visualize_batch_correction_results',
    'visualize_differential_expression_metrics',
    'plot_parameter_grid_metrics',
    'compare_batch_correction_across_masks',
    'compare_differential_expression_across_masks',

    # from correction.py
    'run_correction'
]