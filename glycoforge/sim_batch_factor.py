import numpy as np
import pandas as pd
from .utils import invclr  


def define_batch_direction(
        batch_effect_direction=None,  # Manual mode: dict {batch_id: {glycan_index (1-based): direction (+1/-1)}}
        n_glycans=50,                 # Total number of glycans
        n_batches=3,                  # Number of batches (for random mode)
        affected_fraction=(0.05, 1), # Fraction range of affected glycans per batch
        positive_prob=0.6,            # Probability of positive effects
        overlap_prob=0.2,             # Probability of overlap between batches
        u_dict_seed=42,               # Fixed seed for reproducible batch effect structure
        normalize=True,               # CLR normalization
        verbose=False                 # Print generated config
        ):
    
    # Generate random batch_effect_direction if not provided
    if batch_effect_direction is None:
        rng = np.random.default_rng(u_dict_seed)
        batch_effect_direction = {}
        all_affected = set()
        
        for batch_id in range(1, n_batches + 1):
            # Generate number of affected glycans using normal distribution
            frac_mean = (affected_fraction[0] + affected_fraction[1]) / 2
            frac_std = (affected_fraction[1] - affected_fraction[0]) / 4
            frac = np.clip(rng.normal(frac_mean, frac_std), affected_fraction[0], affected_fraction[1])
            n_affected = max(1, int(frac * n_glycans))
            
            # Select glycans with controlled overlap
            if len(all_affected) > 0 and rng.random() < overlap_prob:
                # Allow some overlap
                available = list(range(n_glycans))
            else:
                # Avoid overlap
                available = list(set(range(n_glycans)) - all_affected)
                if len(available) < n_affected:
                    available = list(range(n_glycans))  # Fallback to all if needed
            
            affected_glycans = rng.choice(available, min(n_affected, len(available)), replace=False)
            directions = rng.choice([-1, 1], size=len(affected_glycans), 
                                  p=[1-positive_prob, positive_prob])
            
            batch_effect_direction[batch_id] = {int(g+1): int(d) for g, d in zip(affected_glycans, directions)}
            all_affected.update(affected_glycans)
        
        if verbose:
            print("Generated random batch effect directions:")
            for bid, effects in batch_effect_direction.items():
                pos = sum(1 for d in effects.values() if d > 0)
                neg = sum(1 for d in effects.values() if d < 0)
                print(f"  Batch {bid}: {len(effects)} glycans ({pos}↑, {neg}↓)")
    
    # Convert to direction vectors (original logic)
    u_dict = {}
    for b, effects in batch_effect_direction.items():
        w = np.zeros(n_glycans)
        for j, d in effects.items():
            w[j-1] = d   
        if normalize and np.any(w != 0):
            w_tilde = w - w.mean()
            norm = np.sqrt(np.mean(w_tilde**2))
            w = w_tilde / norm if norm > 0 else w_tilde
        u_dict[b] = w
    
    # Return both normalized u_dict and raw batch_effect_direction
    return u_dict, batch_effect_direction


# according to column name "healthy_" and "unhealthy_"
def stratified_batches_from_columns(columns, n_batches=3, seed=None, verbose=True):

    rng = np.random.default_rng(seed)
    labels = np.array([0 if str(c).startswith("healthy_") else 1 for c in columns])
    batch_labels = np.zeros(len(labels), dtype=int)

    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        rng.shuffle(idx)
        splits = np.array_split(idx, n_batches)
        for b, part in enumerate(splits, start=0):  
            batch_labels[part] = b

    
    batch_groups = {}
    for b in range(n_batches):
        cols_in_batch = [c for c, lab in zip(columns, batch_labels) if lab == b]
        batch_groups[b] = cols_in_batch
        if verbose:
            print(f"Batch {b}: {cols_in_batch}")

    return batch_groups, batch_labels #batch_groups: dict {batch_id: [sample_names]} for plotting，
                                #batch_labels: np.array of 0-based batch assignments for analysis



def estimate_sigma(Y):
    """
    Estimate baseline std (sigma) per glycan (global, across all samples).
    """
    if isinstance(Y, pd.DataFrame):
        sigma = Y.std(axis=1, ddof=1).to_numpy()
    else:  # numpy array
        sigma = np.std(Y, axis=1, ddof=1)
    return sigma



def apply_batch_effect(Y_clean # (samples x glycans) clean CLR matrix
                       , batch_labels # Array of batch labels for each sample (0-based)
                       , u_dict # batch effect direction vectors dict {batch_id: direction_vector}
                       , sigma # Baseline std per glycan (length = n_glycans) in Y_clean_clr data, globally including both healthy and unhealthy
                       , kappa_mu # mean shift strength control per batch (mean_shift = kappa_mu * sigma * u_b)
                       , var_b # variance ratio per batch (var_effect = var_b * baseline_variance)
                       , seed=42 # Random generator for reproducibility
                    ):

    rng = np.random.default_rng(seed)

    Y_with_batch_clr = Y_clean.copy()
    n_samples, n_glycans = Y_clean.shape
    
    for i in range(n_samples):
        b = batch_labels[i]
        if b not in u_dict:
            continue
        u_b = u_dict[b]
        
        # mean shift term
        mean_shift = kappa_mu * sigma * u_b
        
        # variance scaling term (extra noise)
        var_scalor = rng.normal(loc=0.0, scale=np.sqrt(var_b) * sigma, size=n_glycans)
        
        # apply to sample
        Y_with_batch_clr[i, :] += mean_shift + var_scalor

    
    Y_with_batch_compositional = np.zeros_like(Y_with_batch_clr)
    for i in range(n_samples):
        Y_with_batch_compositional[i, :] = invclr(Y_with_batch_clr[i, :])  # Scale to percentage

    return Y_with_batch_clr, Y_with_batch_compositional
