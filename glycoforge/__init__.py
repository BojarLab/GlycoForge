"""GlycoForge - Simulation Tool for Glycomics Data"""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("glycoforge")
except PackageNotFoundError:  # a source checkout on sys.path, or a frozen app, has no installed metadata
    __version__ = "unknown"

# Core simulation interface
from .pipeline import simulate, simulate_paired, simulate_circadian, glycoforge_power, glycoforge_reliability, motif_template, simulate_known_truth

# Utility functions
from glycoforge.utils import clr, invclr, parse_simulation_config, plot_pca, check_batch_effect, check_bio_effect
from glycoforge.sim_batch_factor import stratified_batches_from_columns

# Expose core API
__all__ = [
    'simulate',
    'simulate_paired',
    'simulate_circadian',
    'glycoforge_power',
    'glycoforge_reliability',
    'motif_template',
    'simulate_known_truth',
    'clr',
    'invclr',
    'parse_simulation_config',
    'plot_pca',
    'check_batch_effect',
    'check_bio_effect',
    'stratified_batches_from_columns'
]