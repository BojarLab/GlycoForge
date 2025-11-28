import numpy as np
import json
from collections import defaultdict
import glob
import pandas as pd

def calculate_batch_correction_changes(before_metrics, after_metrics):
    changes = {}
    change_metrics = ['silhouette', 'kBET', 'LISI', 'ARI', 'compositional_effect_size', 'pca_batch_effect']
    
    for metric in change_metrics:
        if metric in before_metrics and metric in after_metrics:
            before_val = before_metrics[metric]
            after_val = after_metrics[metric]
            if before_val != 0:
                changes[metric] = (after_val - before_val) / before_val * 100
            else:
                changes[metric] = 0  # Handle division by zero
    
    return changes # percentage changes: (after - before) / before * 100

def calculate_biological_preservation(after_metrics):
    preservation = {}
    bio_metrics = ['biological_variability_preservation', 'conserved_differential_proportion']
    
    for metric in bio_metrics:
        if metric in after_metrics:
            # These are already ratios (0-1), convert to percentages
            preservation[metric] = after_metrics[metric] * 100
    
    return preservation # percentage values

def extract_metrics_with_calculations(all_results, random_seeds=None):

    batch_change_metrics = defaultdict(lambda: defaultdict(list))
    batch_preservation_metrics = defaultdict(lambda: defaultdict(list))
    diff_metrics = defaultdict(lambda: defaultdict(list))
    
    # Auto-detect parameter names from first combo
    if all_results:
        first_combo = next(iter(all_results.values()))
        if 'k_dir' in first_combo and 'bio_strength' in first_combo:
            param1_name, param2_name = 'k_dir', 'bio_strength'
        else:
            param1_name, param2_name = 'kappa_mu', 'var_b'
    
    for combo_name, combo_data in all_results.items():
        param1_val = combo_data[param1_name]
        param2_val = combo_data[param2_name]
        output_dir = combo_data['output_dir']
        
        # Find all JSON files in this directory
        json_files = glob.glob(f"{output_dir}/comprehensive_metrics_seed*.json")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    metrics = json.load(f)
                
                # Get before and after batch metrics
                batch_before = metrics.get('batch_effect_metrics', {}).get('before_correction', {})
                batch_after = metrics.get('batch_effect_metrics', {}).get('after_correction', {})
                
                changes = calculate_batch_correction_changes(batch_before, batch_after)
                for metric_name, value in changes.items():
                    batch_change_metrics[metric_name][(param1_val, param2_val)].append(value)
                    
                preservation = calculate_biological_preservation(batch_after)
                for metric_name, value in preservation.items():
                    batch_preservation_metrics[metric_name][(param1_val, param2_val)].append(value)

                diff_overall = metrics.get('differential_expression', {}).get('results', {}).get('overall', {})
                for metric_name, value in diff_overall.items():
                    if isinstance(value, (int, float)):
                        diff_metrics[metric_name][(param1_val, param2_val)].append(value)
                        
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
    
    all_batch_metrics = {**batch_change_metrics, **batch_preservation_metrics}
    
    return all_batch_metrics, diff_metrics

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


def combat(data, batch, mod=None, parametric=True):
    """
    ComBat for CLR/ALR glycomics data.
    data: glycans x samples
    batch: batch labels for each sample, a array-like of shape (n_samples,)
    mod: optional design matrix (e.g., biological group)
    parametric : whether to use parametric empirical Bayes (default True).
    """
    X = np.asarray(data).T  # samples x features
    n, p = X.shape
    batch = pd.Categorical(batch)

    # Design matrix (intercept if None)
    M = np.ones((n, 1)) if mod is None else np.asarray(pd.get_dummies(mod), float)

    # ----- Standardize -----
    B_hat = np.linalg.solve(M.T @ M, M.T @ X)
    grand_mean = M @ B_hat
    s_data = X - grand_mean
    sds = s_data.std(axis=0, ddof=1)
    s_data /= sds

    # ----- Estimate batch effects -----
    n_batch = len(batch.categories)
    gamma_hat = np.zeros((n_batch, p))
    delta_hat = np.zeros((n_batch, p))
    for i, b in enumerate(batch.categories):
        mask = batch == b
        s = s_data[mask]
        gamma_hat[i] = s.mean(0)
        delta_hat[i] = s.var(0, ddof=1)

    # ----- Parametric shrinkage (optional) -----
    if parametric:
        gamma_bar, t2 = gamma_hat.mean(0), gamma_hat.var(0, ddof=1)
        a_prior = (2 * t2 + delta_hat.mean(0)) / delta_hat.var(0)
        b_prior = (delta_hat.mean(0) * a_prior) / 2
        for i, b in enumerate(batch.categories):
            mask = batch == b
            n_b = mask.sum()
            ss = ((s_data[mask] - gamma_hat[i])**2).sum(0)
            delta_hat[i] = (b_prior + 0.5 * ss) / (a_prior + 0.5 * n_b + 1)

    # ----- Adjust -----
    X_adj = s_data.copy()
    for i, b in enumerate(batch.categories):
        mask = batch == b
        X_adj[mask] = ((s_data[mask] - gamma_hat[i]) / np.sqrt(delta_hat[i])) * sds + grand_mean[mask]

    return X_adj.T  # glycans x samples

