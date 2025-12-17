"""GlycoForge - Simulation and batch correction pipeline for glycomics data"""

__version__ = "0.1.0"

# Core simulation interface
from .pipeline import simulate

# Utility functions
from .utils import clr, invclr, parse_simulation_config, plot_pca

# Expose core API
__all__ = [
    'simulate', 
    'clr', 
    'invclr',
    'parse_simulation_config',
    'plot_pca'
]