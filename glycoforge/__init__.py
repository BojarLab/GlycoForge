"""GlycoForge - Simulation Tool for Glycomics Data"""

__version__ = "0.1.2"

# Core simulation interface
from .pipeline import simulate

# Utility functions
from glycoforge.utils import clr, invclr, parse_simulation_config, plot_pca, check_batch_effect, check_bio_effect, load_data_from_glycowork
from glycoforge.sim_batch_factor import stratified_batches_from_columns

# Expose core API
__all__ = [
    'simulate', 
    'clr', 
    'invclr',
    'parse_simulation_config',
    'plot_pca',
    'check_batch_effect',
    'check_bio_effect',
    'load_data_from_glycowork',
    'stratified_batches_from_columns'
]