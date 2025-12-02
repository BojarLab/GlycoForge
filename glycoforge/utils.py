import numpy as np
import json
from collections import defaultdict
import glob
import pandas as pd

def clr(x, eps=1e-6):
    x = np.asarray(x)
    
    # Handle zeros by replacing with small epsilon
    x_safe = np.where(x <= 0, eps, x)
    
    # Standard CLR: log(x) - geometric_mean(log(x))
    log_x = np.log(x_safe)
    if x.ndim == 1:
        # Single sample
        geom_mean_log = np.mean(log_x)
        return log_x - geom_mean_log
    else:
        # Multiple samples - compute geometric mean across features (axis=0)
        geom_mean_log = np.mean(log_x, axis=0)
        return log_x - geom_mean_log

def invclr(z, to_percent=True, eps=1e-6):
    z = np.asarray(z, dtype=float)
    z = z - np.mean(z)               # Center to ensure proper simplex
    z = z - np.max(z)                # Numerical stability
    x = np.exp(z)
    x = np.maximum(x, eps)           # Prevent zeros
    x = x / np.sum(x)                # Normalize to 1
    if to_percent:
        x *= 100
    return x

