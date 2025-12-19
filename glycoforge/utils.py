import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from scipy.stats import f_oneway, shapiro, kruskal



def parse_simulation_config(config):
    """
    Parse complete simulation configuration from YAML.
    Handles all config transformations including batch_effect_direction expansion.

    Args:
        config: Raw dict from yaml.safe_load()

    Returns:
        dict: Parsed config ready for simulate() or correction pipeline
    """
    parsed = {}

    # Direct copy simple parameters
    simple_params = [
        'data_source', 'data_file', 'n_glycans', 'n_H', 'n_U',
        'bio_strength', 'k_dir', 'variance_ratio', 'use_real_effect_sizes',
        'differential_mask', 'column_prefix', 'n_batches',
        'kappa_mu', 'var_b', 'winsorize_percentile', 'baseline_method',
        'u_dict', 'random_seeds', 'output_dir', 'verbose', 'save_csv', 'show_pca_plots',
        'missing_fraction', 'mnar_bias'
    ]

    for key in simple_params:
        if key in config:
            parsed[key] = config[key]

    # Parse batch_effect_direction (complex nested structure)
    if 'batch_effect_direction' in config:
        bed_config = config['batch_effect_direction']

        # Get default parameters
        affected_fraction = config.get('affected_fraction', (0.05, 0.30))
        positive_prob = config.get('positive_prob', 0.6)
        overlap_prob = config.get('overlap_prob', 0.5)

        manual_config, auto_params = _parse_batch_effect_direction(
            bed_config, affected_fraction, positive_prob, overlap_prob
        )

        # Store parsed version - pipeline.py will use this
        parsed['batch_effect_direction'] = manual_config
        parsed['affected_fraction'] = auto_params['affected_fraction']
        parsed['positive_prob'] = auto_params['positive_prob']
        parsed['overlap_prob'] = auto_params['overlap_prob']
    else:
        # Use defaults from config or function defaults
        parsed['affected_fraction'] = config.get('affected_fraction', (0.05, 0.30))
        parsed['positive_prob'] = config.get('positive_prob', 0.6)
        parsed['overlap_prob'] = config.get('overlap_prob', 0.5)

    return parsed


def _parse_batch_effect_direction(bed_config, affected_fraction, positive_prob, overlap_prob):
    """
    Internal helper to parse batch_effect_direction nested structure.
    Expands string keys like "1-50", "3,8,12" to individual indices.

    Returns:
        tuple: (manual_config or None, auto_params dict)
    """
    manual_config = None
    auto_params = {
        'affected_fraction': affected_fraction,
        'positive_prob': positive_prob,
        'overlap_prob': overlap_prob
    }

    if not bed_config or not isinstance(bed_config, dict):
        return manual_config, auto_params

    mode = bed_config.get('mode', 'auto')

    if mode == 'manual':
        raw_manual = bed_config.get('manual', {})
        if raw_manual:
            manual_config = {}

            for batch_id, effects in raw_manual.items():
                batch_id_int = int(batch_id)
                manual_config[batch_id_int] = {}

                for key, direction in effects.items():
                    if isinstance(key, str):
                        if '-' in key and ',' not in key:
                            start, end = map(int, key.split('-'))
                            for idx in range(start, end + 1):
                                manual_config[batch_id_int][idx] = int(direction)
                        elif ',' in key:
                            for idx in map(int, key.split(',')):
                                manual_config[batch_id_int][idx] = int(direction)
                        else:
                            manual_config[batch_id_int][int(key)] = int(direction)
                    else:
                        manual_config[batch_id_int][int(key)] = int(direction)

    # Extract auto parameters if provided
    auto_config = bed_config.get('auto', {})
    if auto_config:
        auto_params['affected_fraction'] = auto_config.get('affected_fraction', affected_fraction)
        auto_params['positive_prob'] = auto_config.get('positive_prob', positive_prob)
        auto_params['overlap_prob'] = auto_config.get('overlap_prob', overlap_prob)

    return manual_config, auto_params


def load_data_from_glycowork(data_file):
    if os.path.exists(data_file):
        return pd.read_csv(data_file)

    # Try loading from glycowork internal datasets
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
            import pkg_resources
            glycowork_path = pkg_resources.resource_filename('glycowork', 'glycan_data')

        # Add .csv extension if not present
        dataset_name = data_file if data_file.endswith('.csv') else f"{data_file}.csv"
        full_path = os.path.join(glycowork_path, dataset_name)

        if os.path.exists(full_path):
            return pd.read_csv(full_path)
    except Exception:
        pass

    # If all fails, try as regular path (will raise clear error message)
    return pd.read_csv(data_file)

def clr(x, eps=1e-6):
    """Centered log-ratio transformation for compositional data.
    
    Parameters:
    -----------
    x : array-like
        Compositional data. Can be 1D (single sample) or 2D (samples x features).
    eps : float
        Small value to replace zeros (default 1e-6).
    Returns:
    --------
    clr_transformed : np.ndarray
        CLR-transformed data with same shape as input.
    """
    x = np.asarray(x, dtype=float)

    # Handle zeros by replacing with small epsilon
    x_safe = np.where(x <= 0, eps, x)

    # Standard CLR: log(x) - geometric_mean(log(x))
    log_x = np.log(x_safe)
    if x.ndim == 1:
        # Single sample: subtract mean across all features
        geom_mean_log = np.mean(log_x)
        return log_x - geom_mean_log
    else:
        # Multiple samples: subtract mean across features for each sample (axis=1)
        geom_mean_log = np.mean(log_x, axis=1, keepdims=True)
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



# Plot PCA for clean and simulated data
def plot_pca(data, #DataFrame (features x samples)
             bio_groups=None, # dict or None, e.g. {'healthy': ['healthy_1', 'healthy_2'], 'unhealthy': ['unhealthy_1']}
             batch_groups=None,
             title="PCA",
             save_path=None):

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data.T)
    sample_names = data.columns.tolist()

    # Helper function to get bio/batch labels for annotation
    def get_bio_label(sample_name):
        if bio_groups:
            for i, (group_name, cols) in enumerate(bio_groups.items()):  # 0-based
                if sample_name in cols:
                    return f"Bio-{i}"
        return ""

    def get_batch_label(sample_name):
        if batch_groups:
            for batch_id, cols in batch_groups.items():
                if sample_name in cols:
                    return f"BE-{batch_id}"
        return ""

    # Setup subplots
    n_plots = sum([bio_groups is not None, batch_groups is not None])
    if n_plots == 0:
        return
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    axes = [axes] if n_plots == 1 else axes
    plot_idx = 0

    # Plot biological groups (with batch annotations)
    if bio_groups is not None:
        bio_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (group_name, cols) in enumerate(bio_groups.items()):
            indices = [sample_names.index(c) for c in cols if c in sample_names]
            axes[plot_idx].scatter(pca_result[indices, 0], pca_result[indices, 1],
                                  c=bio_colors[i % len(bio_colors)], label=group_name, alpha=0.7, s=50)

            # Add batch annotations on bio-colored plot
            for idx in indices:
                batch_label = get_batch_label(sample_names[idx])
                if batch_label:
                    axes[plot_idx].annotate(batch_label, (pca_result[idx, 0], pca_result[idx, 1]),
                                          xytext=(2, 2), textcoords='offset points',
                                          fontsize=8, alpha=0.7)

        axes[plot_idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[plot_idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[plot_idx].set_title(f'{title}\n(colored by bio-groups)')
        axes[plot_idx].legend()
        axes[plot_idx].grid(alpha=0.3)
        plot_idx += 1

    # Plot batch groups (with bio annotations)
    if batch_groups is not None:
        batch_colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#FF6B35']
        for i, (batch_id, cols) in enumerate(sorted(batch_groups.items())):
            indices = [sample_names.index(c) for c in cols if c in sample_names]
            axes[plot_idx].scatter(pca_result[indices, 0], pca_result[indices, 1],
                                  c=batch_colors[i % len(batch_colors)], label=f'Batch {batch_id}', alpha=0.7, s=50)

            # Add bio annotations on batch-colored plot
            for idx in indices:
                bio_label = get_bio_label(sample_names[idx])
                if bio_label:
                    axes[plot_idx].annotate(bio_label, (pca_result[idx, 0], pca_result[idx, 1]),
                                          xytext=(2, 2), textcoords='offset points',
                                          fontsize=8, alpha=0.7)

        axes[plot_idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[plot_idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[plot_idx].set_title(f'{title}\n(colored by batches)')
        axes[plot_idx].legend()
        axes[plot_idx].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()




def _compute_pca_and_stats(data):
    """Compute PCA and return PC, variance explained, and normality test result."""
    X = np.asarray(data).T
    pca = PCA(n_components=min(5, X.shape[0]-1))
    pc = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_[:2].sum()
    total_samples = len(pc[:, 0])

    # Normality test (only for n >= 30)
    norm_p = shapiro(pc[:, 0])[1] if total_samples >= 30 else 0.0

    return pc, var_explained, total_samples, norm_p, X


def _evaluate_bio_effect_details(pc, bio_labels, test_used, ss_total, total_samples, verbose=False):
    """Calculate bio effect with centroid_distance and strength assessment."""
    bio_cat = pd.Categorical(bio_labels)
    pc1_by_bio = [pc[bio_cat==g, 0] for g in bio_cat.categories]

    # Statistical test
    if test_used == "Kruskal-Wallis":
        f_bio, p_bio = kruskal(*pc1_by_bio)
    else:
        f_bio, p_bio = f_oneway(*pc1_by_bio)

    # Effect size (eta²)
    if test_used == "ANOVA":
        ss_bio_between = sum([len(group) * (np.mean(group) - np.mean(pc[:, 0]))**2
                             for group in pc1_by_bio])
        bio_eta = ss_bio_between / ss_total if ss_total > 0 else 0
    else:
        bio_eta = (f_bio - len(bio_cat.categories) + 1) / (total_samples - len(bio_cat.categories) + 1)
        bio_eta = max(0, min(1, bio_eta))

    # Centroid distance
    centroids = [pc[bio_cat==b, :2].mean(axis=0) for b in bio_cat.categories]
    centroid_dist = np.linalg.norm(centroids[0] - centroids[1]) if len(centroids) == 2 else 0

    # Strength assessment
    if p_bio >= 0.05:
        strength = "ABSENT"
        strength_desc = f"No significant signal (p={p_bio:.3e} >= 0.05)"
    elif bio_eta < 0.06:
        strength = "WEAK"
        strength_desc = f"Small effect (eta²={bio_eta:.1%} < 6%)"
    elif bio_eta < 0.14:
        strength = "MODERATE"
        strength_desc = f"Medium effect (6% ≤ eta²={bio_eta:.1%} < 14%)"
    else:
        strength = "STRONG"
        strength_desc = f"Large effect (eta²={bio_eta:.1%} >= 14%)"

    if verbose:
        print(f"Biological effect on PC1: F={f_bio:.2f}, p={p_bio:.3e}")
        print(f"Biological effect size (eta²): {bio_eta:.1%}")
        print(f"Centroid distance (PC1-2): {centroid_dist:.2f}")
        print(f"Signal strength: {strength} - {strength_desc}")

    return {
        'f_statistic': float(f_bio),
        'p_value': float(p_bio),
        'effect_size_eta2': float(bio_eta),
        'centroid_distance': float(centroid_dist),
        'strength': strength,
        'strength_description': strength_desc
    }


def check_batch_effect(data,
                       batch_labels,
                       bio_groups=None,
                       verbose=True
                       ):

    pc, var_explained, total_samples, norm_p, X = _compute_pca_and_stats(data)

    # Batch effect evaluation
    batch_cat = pd.Categorical(batch_labels)
    pc1_by_batch = [pc[batch_cat==b, 0] for b in batch_cat.categories]

    # Choose test
    if total_samples < 30 or norm_p < 0.05:
        f_stat, p_val = kruskal(*pc1_by_batch)
        test_used = "Kruskal-Wallis"
    else:
        f_stat, p_val = f_oneway(*pc1_by_batch)
        test_used = "ANOVA"

    # Batch eta²
    ss_total = np.var(pc[:, 0]) * (len(pc[:, 0]) - 1)
    if test_used == "ANOVA":
        ss_between = sum([len(group) * (np.mean(group) - np.mean(pc[:, 0]))**2
                         for group in pc1_by_batch])
        batch_eta = ss_between / ss_total if ss_total > 0 else 0
    else:
        batch_eta = (f_stat - len(batch_cat.categories) + 1) / (total_samples - len(batch_cat.categories) + 1)
        batch_eta = max(0, min(1, batch_eta))

    if verbose:
        print(f"PC1-2 explain {var_explained:.1%} variance")
        print(f"Batch effect on PC1: F={f_stat:.2f}, p={p_val:.3e} ({test_used})")
        print(f"Batch effect size (eta²): {batch_eta:.1%}")

    results = {
        'pca_variance_explained': float(var_explained),
        'batch_effect': {
            'f_statistic': float(f_stat),
            'p_value': float(p_val),
            'test_used': test_used,
            'effect_size_eta2': float(batch_eta)
        }
    }

    # Bio effect evaluation (if provided)
    if bio_groups is not None:
        bio_effect = _evaluate_bio_effect_details(pc, bio_groups, test_used, ss_total, total_samples, verbose)
        results['bio_effect'] = bio_effect

        # Overall quality assessment
        bio_eta = bio_effect['effect_size_eta2']
        p_bio = bio_effect['p_value']

        if p_val < 0.05 and p_bio < 0.05:
            if batch_eta > bio_eta + 0.1:
                if batch_eta > 0.3:
                    severity = "CRITICAL"
                elif batch_eta > 0.2:
                    severity = "MODERATE"
                else:
                    severity = "MILD"
                severity_description = f"Batch effect ({batch_eta:.1%}) stronger than biological signal ({bio_eta:.1%})"
            else:
                severity = "GOOD"
                severity_description = f"Biological signal ({bio_eta:.1%}) stronger than batch effect ({batch_eta:.1%})"
        elif p_val < 0.05 and p_bio >= 0.05:
            severity = "WARNING"
            severity_description = "Significant batch effect detected, but no significant biological signal"
        elif p_val >= 0.05 and p_bio < 0.05:
            severity = "GOOD"
            severity_description = "Biological signal detected without significant batch effect"
        else:
            severity = "NONE"
            severity_description = "Neither batch nor biological effects are statistically significant"

        if verbose:
            print(f"\nOverall Quality: {severity} - {severity_description}")
            print(f"  (Decision criteria: batch_p={p_val:.3e}, bio_p={p_bio:.3e}, batch_eta²={batch_eta:.1%}, bio_eta²={bio_eta:.1%})")

        # Median variance explained by batch
        batch_dummies = pd.get_dummies(batch_labels).values
        var_batch = np.array([np.corrcoef(X[:, i], batch_dummies.T)[0, 1:].max()**2
                             for i in range(X.shape[1])])
        median_var_batch = float(np.median(var_batch))

        if verbose:
            print(f"Median variance explained by batch across features: {median_var_batch:.1%}")

        results['overall_quality'] = {
            'severity': severity,
            'severity_description': severity_description,
            'median_variance_explained_by_batch': median_var_batch
        }

        return results, pc, var_batch
    else:
        # No bio_groups: return batch-only results
        batch_dummies = pd.get_dummies(batch_labels).values
        var_batch = np.array([np.corrcoef(X[:, i], batch_dummies.T)[0, 1:].max()**2
                             for i in range(X.shape[1])])
        median_var_batch = float(np.median(var_batch))

        if verbose:
            print(f"Median variance explained by batch across features: {median_var_batch:.1%}")

        results['median_variance_explained_by_batch'] = median_var_batch

        return results, pc, var_batch



def check_bio_effect(data_clr, bio_labels, stage_name="", verbose=True):

    pc, var_explained, total_samples, norm_p, _ = _compute_pca_and_stats(data_clr)

    # Choose test
    if total_samples < 30 or norm_p < 0.05:
        test_used = "Kruskal-Wallis"
    else:
        test_used = "ANOVA"

    ss_total = np.var(pc[:, 0]) * (len(pc[:, 0]) - 1)

    # Verbose header
    if verbose and stage_name:
        print(f"\n[{stage_name.upper()}]")
    if verbose:
        print(f"PC1-2 explain {var_explained:.1%} variance")

    # Evaluate bio effect with full details
    bio_effect = _evaluate_bio_effect_details(pc, bio_labels, test_used, ss_total, total_samples, verbose)

    return {
        'pca_variance_explained': float(var_explained),
        'bio_effect': bio_effect
    }, pc


def apply_mnar_missingness(Y_compositional, missing_fraction=0.0, mnar_bias=2.0, seed=42, verbose=True):
    """Apply missing-not-at-random (MNAR) pattern to compositional data.
    Low-intensity glycans have higher probability of being missing.

    Parameters:
    -----------
    Y_compositional : np.ndarray or pd.DataFrame
        Compositional data (samples x glycans), values in percentage or proportions
    missing_fraction : float
        Target fraction of missing values (0.0 to 1.0)
    mnar_bias : float
        Intensity bias parameter. Higher values = stronger bias toward low intensity.
        Typical range: 0.5 (weak) to 5.0 (very strong). Default 2.0.
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print diagnostics

    Returns:
    --------
    Y_missing : pd.DataFrame or np.ndarray
        Data with NaN for missing values
    Y_missing_clr : pd.DataFrame or np.ndarray
        CLR-transformed data (imputed before CLR)
    missing_mask : np.ndarray
        Boolean mask (True = missing)
    diagnostics : dict
        Missingness statistics
    """
    if missing_fraction <= 0:
        if verbose:
            print("Missingness disabled (missing_fraction=0)")
        Y_clr = clr(Y_compositional.values.T if isinstance(Y_compositional, pd.DataFrame) else Y_compositional.T).T
        if isinstance(Y_compositional, pd.DataFrame):
            Y_clr = pd.DataFrame(Y_clr, index=Y_compositional.index, columns=Y_compositional.columns)
        return Y_compositional, Y_clr, np.zeros(Y_compositional.shape, dtype=bool), {}

    rng = np.random.default_rng(seed)
    is_df = isinstance(Y_compositional, pd.DataFrame)
    Y = Y_compositional.values if is_df else Y_compositional
    n_samples, n_glycans = Y.shape
    missing_mask = np.zeros_like(Y, dtype=bool)

    # Phase 1: Per-sample intensity-dependent probability
    for i in range(n_samples):
        sample_values = Y[i, :]
        max_val = np.max(sample_values)
        if max_val > 0:
            normalized_intensity = sample_values / max_val
        else:
            normalized_intensity = np.ones(n_glycans) * 0.5
        prob_missing = (1 - normalized_intensity) ** mnar_bias
        missing_mask[i, :] = rng.random(n_glycans) < prob_missing

    # Phase 2: Global adjustment to target fraction
    target_count = int(missing_fraction * Y.size)
    current_count = np.sum(missing_mask)
    if current_count < target_count:
        available = np.where(~missing_mask)
        n_add = target_count - current_count
        if n_add > 0 and len(available[0]) > 0:
            add_indices = rng.choice(len(available[0]), min(n_add, len(available[0])), replace=False)
            missing_mask[available[0][add_indices], available[1][add_indices]] = True
    elif current_count > target_count:
        missing_indices = np.where(missing_mask)
        n_remove = current_count - target_count
        if n_remove > 0:
            remove_indices = rng.choice(len(missing_indices[0]), n_remove, replace=False)
            missing_mask[missing_indices[0][remove_indices], missing_indices[1][remove_indices]] = False

    # Apply missingness
    Y_missing = Y.copy().astype(float)
    Y_missing[missing_mask] = np.nan

    # Compute CLR with imputation
    Y_for_clr = Y.copy()
    Y_for_clr[missing_mask] = 1e-6
    Y_missing_clr = clr(Y_for_clr.T).T

    # Diagnostics
    intensity_bins = [0, 0.01, 0.1, 1.0, np.inf]
    bin_labels = ['<0.01%', '0.01-0.1%', '0.1-1%', '>1%']
    missing_by_intensity = {}
    for b_idx in range(len(intensity_bins)-1):
        mask = (Y >= intensity_bins[b_idx]) & (Y < intensity_bins[b_idx+1])
        if np.sum(mask) > 0:
            missing_rate = np.sum(missing_mask & mask) / np.sum(mask)
            missing_by_intensity[bin_labels[b_idx]] = float(missing_rate)

    per_sample_missing = np.sum(missing_mask, axis=1)
    diagnostics = {
        'total_missing': int(np.sum(missing_mask)),
        'missing_fraction_actual': float(np.sum(missing_mask) / Y.size),
        'missing_fraction_target': float(missing_fraction),
        'per_sample_missing': per_sample_missing.tolist(),
        'missing_rate_by_intensity': missing_by_intensity,
        'mnar_bias': float(mnar_bias)
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"MNAR MISSINGNESS APPLIED")
        print(f"{'='*60}")
        print(f"Target fraction: {missing_fraction:.1%}")
        print(f"Actual fraction: {diagnostics['missing_fraction_actual']:.1%}")
        print(f"Total missing: {diagnostics['total_missing']}/{Y.size}")
        print(f"MNAR bias: {mnar_bias}")
        print(f"\nMissing rate by intensity:")
        for bin_label, rate in missing_by_intensity.items():
            print(f"  {bin_label:>12}: {rate:.1%}")
        print(f"{'='*60}\n")

    if is_df:
        Y_missing = pd.DataFrame(Y_missing, index=Y_compositional.index, columns=Y_compositional.columns)
        Y_missing_clr = pd.DataFrame(Y_missing_clr, index=Y_compositional.index, columns=Y_compositional.columns)

    return Y_missing, Y_missing_clr, missing_mask, diagnostics
