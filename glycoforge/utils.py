import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from scipy.stats import f_oneway, shapiro, kruskal



def load_data_from_glycowork(data_file):
    if os.path.exists(data_file):
        return pd.read_csv(data_file)
    
    # Try loading from glycowork internal datasets
    try:
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
        strength_desc = f"No significant signal (p={p_bio:.3e} ≥ 0.05)"
    elif bio_eta < 0.06:
        strength = "WEAK"
        strength_desc = f"Small effect (eta²={bio_eta:.1%} < 6%)"
    elif bio_eta < 0.14:
        strength = "MODERATE"
        strength_desc = f"Medium effect (6% ≤ eta²={bio_eta:.1%} < 14%)"
    else:
        strength = "STRONG"
        strength_desc = f"Large effect (eta²={bio_eta:.1%} ≥ 14%)"
    
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
                if verbose:
                    print(f"Warning: {severity_description} - {severity}")
            else:
                severity = "GOOD"
                severity_description = f"Biological signal ({bio_eta:.1%}) stronger than batch effect ({batch_eta:.1%})"
                if verbose:
                    print(f"Good: {severity_description}")
        elif p_val < 0.05 and p_bio >= 0.05:
            severity = "WARNING"
            severity_description = "Significant batch effect detected, but no significant biological signal"
            if verbose:
                print(f"Warning: {severity_description}")
        elif p_val >= 0.05 and p_bio < 0.05:
            severity = "GOOD"
            severity_description = "Biological signal detected without significant batch effect"
            if verbose:
                print(f"Good: {severity_description}")
        else:
            severity = "NONE"
            severity_description = "Neither batch nor biological effects are statistically significant"
            if verbose:
                print(f"Note: {severity_description}")
        
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
