"""GlycoForge - Simulation Tool for Glycomics Data"""

__version__ = "0.3.0"

# Core simulation interface
from .pipeline import simulate, simulate_paired

# Utility functions
from glycoforge.utils import clr, invclr, parse_simulation_config, plot_pca, check_batch_effect, check_bio_effect
from glycoforge.sim_batch_factor import stratified_batches_from_columns

# Expose core API
__all__ = [
    'simulate',
    'simulate_paired',
    'clr', 
    'invclr',
    'parse_simulation_config',
    'plot_pca',
    'check_batch_effect',
    'check_bio_effect',
    'stratified_batches_from_columns'
]