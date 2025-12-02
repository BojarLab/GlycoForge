"""GlycoForge - Simulation and batch correction pipeline for glycomics data"""

__version__ = "0.1.0"

# Add current directory to path for imports
import sys
from pathlib import Path
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

# Core simulation interface
from .pipeline import simulate

# Utility functions
from .utils import clr, invclr

# Expose core API
__all__ = ['simulate', 'clr', 'invclr']
