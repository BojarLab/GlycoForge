
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
import os
from collections import defaultdict


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
        if 'bio_strength' in first_combo and 'k_dir' in first_combo:
            param1_name, param2_name = 'bio_strength', 'k_dir'
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


def visualize_batch_correction_results(results, save_path=None, verbose=True):
    """Visualize batch correction effectiveness results using shared calculation functions."""
    
    # Get raw data from results
    raw_data = results.get('raw_data', {})
    config = results.get('config', {})
    n_runs = len(config.get('random_seeds', []))
    
    metric_info = {
        'silhouette': {'name': 'Silhouette Score', 'better': 'Lower', 'type': 'change'},
        'kBET': {'name': 'kBET Score', 'better': 'Lower', 'type': 'change'},
        'LISI': {'name': 'LISI Score', 'better': 'Higher', 'type': 'change'},
        'ARI': {'name': 'Adjusted Rand Index', 'better': 'Lower', 'type': 'change'},
        'compositional_effect_size': {'name': 'Compositional Effect Size', 'better': 'Lower', 'type': 'change'},
        'pca_batch_effect': {'name': 'PCA Batch Effect', 'better': 'Lower', 'type': 'change'},
        'biological_variability_preservation': {'name': 'Bio Variability Preservation', 'better': 'Higher', 'type': 'ratio'},
        'conserved_differential_proportion': {'name': 'Conserved Differential Proportion', 'better': 'Higher', 'type': 'ratio'}
    }
    
    # Calculate metrics using shared functions for each run
    all_change_metrics = []
    all_preservation_metrics = []
    
    # Check if we have the new raw_data format or fallback to old summary_stats
    if raw_data and 'before_correction' in raw_data and 'after_correction' in raw_data:
        # Use raw data to calculate with shared functions
        for run_idx in range(n_runs):
            before_metrics = {k: v[run_idx] if isinstance(v, list) and len(v) > run_idx else v 
                            for k, v in raw_data['before_correction'].items()}
            after_metrics = {k: v[run_idx] if isinstance(v, list) and len(v) > run_idx else v 
                           for k, v in raw_data['after_correction'].items()}
            
            changes = calculate_batch_correction_changes(before_metrics, after_metrics)
            all_change_metrics.append(changes)
            
            preservation = calculate_biological_preservation(after_metrics)
            all_preservation_metrics.append(preservation)
        
        # Create summary stats from calculated values
        summary_stats = {}
        
        # Combine all metrics from all runs
        all_metrics_combined = {}
        for changes in all_change_metrics:
            for k, v in changes.items():
                if k not in all_metrics_combined:
                    all_metrics_combined[k] = []
                all_metrics_combined[k].append(v)
        
        for preservation in all_preservation_metrics:
            for k, v in preservation.items():
                if k not in all_metrics_combined:
                    all_metrics_combined[k] = []
                all_metrics_combined[k].append(v)
        
        # Calculate summary statistics
        for metric, values in all_metrics_combined.items():
            summary_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
    else:
        # Fallback to existing summary_stats if available
        summary_stats = results.get('summary_stats', {})
    
    # Split metrics by type
    change_metrics = [m for m in summary_stats.keys() if m in metric_info and metric_info[m]['type'] == 'change']
    ratio_metrics = [m for m in summary_stats.keys() if m in metric_info and metric_info[m]['type'] == 'ratio']
    
    if verbose:
        print("=" * 60)
        print("VISUALIZE_BATCH_CORRECTION_RESULTS")
        print("=" * 60)
        print(f"Available metrics in summary_stats: {list(summary_stats.keys())}")
        print(f"Change metrics found: {change_metrics}")
        print(f"Ratio metrics found: {ratio_metrics}")
        print(f"COMBAT BATCH CORRECTION EFFECTIVENESS ACROSS {n_runs} RUNS")
        print("-" * 95)
        print(f"{'Metric':<25} {'Direction':<10} {'Mean Value':<12} {'Std Dev':<10} {'Consistency':<10} {'Type':<10}")
        print("-" * 95)
        
        for metric in summary_stats.keys():
            if metric not in metric_info:
                print(f"Warning: Metric '{metric}' not found in metric_info, skipping...")
                continue
                
            stats = summary_stats[metric]
            info = metric_info[metric]
            mean_val = stats['mean']
            std_val = stats['std']
            
            values = stats['values']
            if info['type'] == 'change':
                consistent = all(v < 0 for v in values) if info['better'] == 'Lower' else all(v > 0 for v in values)
            else:  # ratio type
                consistent = all(v > 50 for v in values)
            
            consistency = "Consistent" if consistent else "Mixed"
            
            if info['type'] == 'change':
                print(f"{info['name']:<25} {info['better']:<10} {mean_val:+7.1f}%    {std_val:7.1f}%   {consistency:<10} {'Change':<10}")
            else:
                print(f"{info['name']:<25} {info['better']:<10} {mean_val:7.1f}%     {std_val:7.1f}%   {consistency:<10} {'Preservation':<10}")
        print("=" * 60)
    
    # Create visualization
    if len(change_metrics) > 0 and len(ratio_metrics) > 0:
        # Two subplots with different widths: left plot 3x wider than right plot
        fig = plt.figure(figsize=(16, 8))
        ax1 = plt.subplot(1, 4, (1, 3))  # Takes columns 1-3 (3/4 of width)
        ax2 = plt.subplot(1, 4, 4)       # Takes column 4 (1/4 of width)
    elif len(change_metrics) > 0:
        # Only change metrics
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax2 = None
    elif len(ratio_metrics) > 0:
        # Only ratio metrics
        fig, ax2 = plt.subplots(1, 1, figsize=(6, 8))
        ax1 = None
    else:
        print("No valid metrics found for plotting!")
        return
    
    # Plot 1: Change metrics (percentage change)
    if change_metrics and ax1 is not None:
        change_labels = [f"{metric_info[m]['name']}\n({'↓' if metric_info[m]['better'] == 'Lower' else '↑'} better)" for m in change_metrics]
        change_means = [summary_stats[m]['mean'] for m in change_metrics]
        change_stds = [summary_stats[m]['std'] for m in change_metrics]
        
        change_colors = []
        for i, metric in enumerate(change_metrics):
            if metric_info[metric]['better'] == 'Lower':
                color = 'green' if change_means[i] < 0 else 'red'
            else:
                color = 'green' if change_means[i] > 0 else 'red'
            change_colors.append(color)
        
        bars1 = ax1.bar(range(len(change_metrics)), change_means, yerr=change_stds, 
                       capsize=5, alpha=0.7, color=change_colors, edgecolor='black')
        
        ax1.set_xlabel('Batch Effect Metrics', fontsize=12)
        ax1.set_ylabel('Percentage Change (%)', fontsize=12)
        ax1.set_title(f'Batch Effect Reduction\n (after - before)/before * 100 \n(Mean ± SD across {n_runs} runs)', fontsize=14)
        ax1.set_xticks(range(len(change_metrics)))
        ax1.set_xticklabels(change_labels, rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        for i, (bar, mean_val, std_val) in enumerate(zip(bars1, change_means, change_stds)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (std_val if height >= 0 else -std_val),
                    f'{mean_val:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        # Add legend for change metrics
        legend_elements1 = [Patch(facecolor='green', alpha=0.7, label='Improvement'),
                           Patch(facecolor='red', alpha=0.7, label='Worsening')]
        ax1.legend(handles=legend_elements1, loc='upper right')
    
    # Plot 2: Ratio metrics (preservation percentages)
    if ratio_metrics and ax2 is not None:
        ratio_labels = [f"{metric_info[m]['name']}\n({'↑' if metric_info[m]['better'] == 'Higher' else '↓'} better)" for m in ratio_metrics]
        ratio_means = [summary_stats[m]['mean'] for m in ratio_metrics]
        ratio_stds = [summary_stats[m]['std'] for m in ratio_metrics]
        
        ratio_colors = ['green' if ratio_means[i] > 50 else 'red' for i in range(len(ratio_metrics))]
        
        bars2 = ax2.bar(range(len(ratio_metrics)), ratio_means, yerr=ratio_stds,
                       capsize=5, alpha=0.7, color=ratio_colors, edgecolor='black')
        
        ax2.set_title(f'Biological Signal Preservation\n after/before * 100 \n(Mean ± SD across {n_runs} runs)', fontsize=14)
        ax2.set_xticks(range(len(ratio_metrics)))
        ax2.set_xticklabels(ratio_labels, rotation=45, ha='right')
        ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Perfect Preservation')
        ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Threshold (50%)')
        ax2.grid(True, alpha=0.3)
        
        for i, (bar, mean_val, std_val) in enumerate(zip(bars2, ratio_means, ratio_stds)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std_val + 2,
                    f'{mean_val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Add legend for ratio metrics
        legend_elements2 = [Patch(facecolor='green', alpha=0.7, label='Good(>50%)'),
                           Patch(facecolor='red', alpha=0.7, label='Poor(<=50%)')]
        ax2.legend(handles=legend_elements2, loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    if verbose:
        plt.show()
    else:
        plt.close()


# Create visualization differential expression metrics
def visualize_differential_expression_metrics(output_dir, save_plots=True, verbose=True):
    """
    Visualize key batch correction metrics from JSON files with error bars across runs.
    Uses comprehensive_metrics_seed*.json format.
    """
    
    # Load comprehensive metrics files
    json_files = glob.glob(f"{output_dir}/comprehensive_metrics_seed*.json")
    
    if verbose:
        print("=" * 60)
        print("VISUALIZE_DIFFERENTIAL_EXPRESSION_METRICS")
        print("=" * 60)
        print(f"Found {len(json_files)} comprehensive metrics files.")
        
    if not json_files:
        print(f"Error: No comprehensive metrics files found in {output_dir}")
        return
    
    # Initialize lists to store metrics
    recovery_efficiency = []
    recovery_false_positive_rate = []
    overlap_1v2_rate = []
    overlap_1v3_rate = []
    fp_1v2_rate = []
    fp_1v3_rate = []
    
    # Extract metrics from each file
    for file_path in sorted(json_files):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract differential expression data from comprehensive format
        diff_data = data['differential_expression']
        
        # Extract metrics
        recovery_efficiency.append(diff_data['results']['overall']['recovery_overlap_change_rate'])
        recovery_false_positive_rate.append(diff_data['results']['overall']['recovery_false_positive_change_rate'])
        overlap_1v2_rate.append(diff_data['results']['compare_1v2']['batch_effect_errors']['overlap_rate'])
        overlap_1v3_rate.append(diff_data['results']['compare_1v3']['after_correction_errors']['overlap_rate'])
        
        # Calculate FP rates: gained_signals / dataset1_significant_count * 100
        dataset1_sig_count = diff_data['results']['dataset1']['significant_count']
        gained_1v2_count = diff_data['results']['compare_1v2']['batch_effect_errors']['gained_counts']
        gained_1v3_count = diff_data['results']['compare_1v3']['after_correction_errors']['gained_counts']
        
        fp_1v2_rate.append(gained_1v2_count / dataset1_sig_count * 100)
        fp_1v3_rate.append(gained_1v3_count / dataset1_sig_count * 100)
    
    # Calculate mean and standard error
    metrics_data = {
        'Recovery Overlap\nChange Rate (%)': recovery_efficiency,
        'Recovery FP\nChange Rate (%)': recovery_false_positive_rate,
        '1v2 Overlap\nRate (%)': overlap_1v2_rate,
        '1v3 Overlap\nRate (%)': overlap_1v3_rate,
        '1v2 FP\nRate (%)': fp_1v2_rate,
        '1v3 FP\nRate (%)': fp_1v3_rate
    }

    # Calculate means and stds, handling empty lists
    means = []
    stds = []
    for values in metrics_data.values():
        if len(values) > 0:
            means.append(np.mean(values))
            stds.append(np.std(values, ddof=1) if len(values) > 1 else 0)
        else:
            means.append(0)  # Default value for empty data
            stds.append(0)
    
    if verbose:
        # Check if any metrics have empty data
        for metric_name, values in metrics_data.items():
            if len(values) == 0:
                print(f"Warning: No data for metric '{metric_name}'")
        
        print("Recovery Overlap Change Rate = (overlap_1v3 - overlap_1v2) / dataset1_significant_count * 100")
        print("Recovery FP Change Rate = (gained_1v3 - gained_1v2) / dataset1_significant_count * 100")
        print("=" * 60)
    


    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(metrics_data))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=['#2E8B57', '#2E8B57', '#4169E1', '#4169E1', '#8A2BE2', '#8A2BE2'])
    
    # Customize the plot
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Differential Expression of Batch correction \n \n 1=Clean Data(no BE),\n 2=Data with BE, \n 3=Data after BE correction', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_data.keys(), rotation=15, ha='right')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.5, f'{mean:.1f}±{std:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3)
    
    # Adjust y-axis to accommodate both positive and negative values
    min_val = min(means) - max(stds)
    max_val = max(means) + max(stds)
    y_margin = (max_val - min_val) * 0.1
    ax.set_ylim(min_val - y_margin, max_val + y_margin)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    
    plt.tight_layout()
    if save_plots:
        import os
        os.makedirs(output_dir, exist_ok=True)
        save_path = f"{output_dir}/2_differential_expression_metrics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    if verbose:
        plt.show()
    else:
        plt.close()


def plot_parameter_grid_metrics(all_results=None, results_dir="results", save_path=None, verbose=True):
    """
    Plot batch correction and differential expression metrics across parameter combinations.
    Supports both kappa_mu/var_b and k_dir/bio_strength parameter formats.
    
    Parameters:
    -----------
    all_results : dict, optional
        Pre-loaded results dictionary. If None, will auto-scan results_dir for JSON files.
    results_dir : str, default="results"
        Directory containing parameter combination subdirectories with JSON files.
    save_path : str, optional
        Path prefix for saving plots (will append figure names).
    verbose : bool, default=True
        Whether to print progress information.
    """

    
    if verbose:
        print("=" * 60)
        print("plot_parameter_grid_metrics")
        print("Using shared calculation functions for consistency...")
    
    # Auto-scan if all_results not provided
    if all_results is None or not all_results:
        if verbose:
            print(f"Auto-scanning results directory: {results_dir}")
        
        all_results = {}
        
        # Find all parameter combination directories
        # Support both simple formats (2 params) and hybrid format (4 params)
        combo_dirs = glob.glob(f"{results_dir}/kappa_mu_*_var_b_*") + \
                     glob.glob(f"{results_dir}/*/kappa_mu_*_var_b_*") + \
                     glob.glob(f"{results_dir}/bio_strength_*_k_dir_*") + \
                     glob.glob(f"{results_dir}/*/bio_strength_*_k_dir_*")
        
        if not combo_dirs:
            print(f"Error: No parameter combination directories found in {results_dir}")
            return []
        
        # Auto-detect parameter format from first directory
        first_dir = os.path.basename(combo_dirs[0])
        
        # Check if it's hybrid format (has all 4 parameters)
        if 'bio_strength' in first_dir and 'k_dir' in first_dir and 'kappa_mu' in first_dir and 'var_b' in first_dir:
            # Hybrid format: bio_strength_X_k_dir_Y_kappa_mu_Z_var_b_W
            # Split: ['bio', 'strength', 'X', 'k', 'dir', 'Y', 'kappa', 'mu', 'Z', 'var', 'b', 'W']
            # We care about kappa_mu and var_b (batch params) in hybrid mode
            param1_name, param2_name = 'kappa_mu', 'var_b'
            param1_idx, param2_idx = 8, 11  # Indices for Z and W in the split array
        elif 'bio_strength' in first_dir and 'k_dir' in first_dir:
            # Simple bio format: bio_strength_X_k_dir_Y
            param1_name, param2_name = 'bio_strength', 'k_dir'
            param1_idx, param2_idx = 2, 5  # ['bio', 'strength', 'X', 'k', 'dir', 'Y']
        else:
            # Simple batch format: kappa_mu_X_var_b_Y
            param1_name, param2_name = 'kappa_mu', 'var_b'
            param1_idx, param2_idx = 2, 5  # ['kappa', 'mu', 'X', 'var', 'b', 'Y']
        
        if verbose:
            print(f"Detected parameter format: {param1_name} x {param2_name}")
        
        for combo_dir in combo_dirs:
            combo_name = os.path.basename(combo_dir)
            
            # Extract parameters from directory name
            parts = combo_name.split('_')
            try:
                param1_val = float(parts[param1_idx])
                param2_val = float(parts[param2_idx])
            except (IndexError, ValueError):
                if verbose:
                    print(f"Warning: Could not parse parameters from {combo_name}, skipping...")
                continue
            
            all_results[combo_name] = {
                param1_name: param1_val,
                param2_name: param2_val,
                'results': {},  # Will be populated from JSON files
                'output_dir': combo_dir
            }
        
        if verbose:
            print(f"Found {len(all_results)} parameter combinations:")
            for combo_name in sorted(all_results.keys()):
                data = all_results[combo_name]
                print(f"  {combo_name}: {param1_name}={data[param1_name]}, {param2_name}={data[param2_name]}")
    else:
        # Infer parameter names from existing all_results
        first_combo = next(iter(all_results.values()))
        if 'k_dir' in first_combo:
            param1_name, param2_name = 'k_dir', 'bio_strength'
        else:
            param1_name, param2_name = 'kappa_mu', 'var_b'
    
    # Auto-extract parameter values
    param1_set = set()
    param2_set = set()
    for combo_data in all_results.values():
        param1_set.add(combo_data[param1_name])
        param2_set.add(combo_data[param2_name])
    
    param1_values = sorted(param1_set)
    param2_values = sorted(param2_set)
    
    if verbose:
        print(f"Parameter grid: {param1_name}={param1_values}, {param2_name}={param2_values}")
    
    batch_metrics, diff_metrics = extract_metrics_with_calculations(all_results)
    
    metric_info = {
        'silhouette': {'name': 'Silhouette Score', 'better': 'Lower'},
        'kBET': {'name': 'kBET Score', 'better': 'Lower'},
        'LISI': {'name': 'LISI Score', 'better': 'Higher'},
        'ARI': {'name': 'Adjusted Rand Index', 'better': 'Lower'},
        'compositional_effect_size': {'name': 'Compositional Effect Size', 'better': 'Lower'},
        'pca_batch_effect': {'name': 'PCA Batch Effect', 'better': 'Lower'},
        'biological_variability_preservation': {'name': 'Bio Variability Preservation', 'better': 'Higher'},
        'conserved_differential_proportion': {'name': 'Conserved Differential Proportion', 'better': 'Higher'},
        'recovery_overlap_change_rate': {'name': 'Recovery Overlap Change Rate', 'better': 'Higher'},
        'recovery_false_positive_change_rate': {'name': 'Recovery False Positive Change Rate', 'better': 'Lower'}
    }
    
    # Create parameter combination labels with short display names
    param_short_names = {
        'k_dir': 'k_dir',
        'bio_strength': 'λ',
        'kappa_mu': 'κ_μ',
        'var_b': 'σ²_b'
    }
    
    combo_labels = []
    for param1_val in param1_values:
        for param2_val in param2_values:
            label1 = param_short_names.get(param1_name, param1_name)
            label2 = param_short_names.get(param2_name, param2_name)
            combo_labels.append(f'{label1}={param1_val}, {label2}={param2_val}')
    
    x_pos = np.arange(len(combo_labels))
    plots_created = []
    
    if batch_metrics:
        change_metric_names = ['silhouette', 'kBET', 'LISI', 'ARI', 'compositional_effect_size', 'pca_batch_effect']
        preservation_metric_names = ['biological_variability_preservation', 'conserved_differential_proportion']
        
        change_metrics = {k: v for k, v in batch_metrics.items() if k in change_metric_names}
        preservation_metrics = {k: v for k, v in batch_metrics.items() if k in preservation_metric_names}
        
        if change_metrics:
            fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
            
            academic_colors = {
                'silhouette': "#155D34",     
                'kBET': "#165A92",           
                'LISI': '#2C3E50',         
                'ARI': "#9E20A2",            
                'compositional_effect_size': "#6AA70E",  
                'pca_batch_effect': "#C31D20" 
            }
            
            marker_config = {
                'silhouette': ('o', 'full'),       
                'kBET': ('o', 'none'),              
                'LISI': ('s', 'full'),             
                'ARI': ('^', 'none'),               
                'compositional_effect_size': ('s', 'full'),  
                'pca_batch_effect': ('^', 'none')   
            }
            
            for idx, (metric_name, metric_data) in enumerate(change_metrics.items()):
                means, stds = [], []
                for param1_val in param1_values:
                    for param2_val in param2_values:
                        values = metric_data.get((param1_val, param2_val), [])
                        means.append(np.mean(values) if values else 0)
                        stds.append(np.std(values) if values else 0)
                
                direction = "↑" if metric_info[metric_name]['better'] == 'Higher' else "↓"
                label = f"{metric_info[metric_name]['name']} {direction}"
                
                color = academic_colors.get(metric_name, '#2C3E50')
                marker, fillstyle = marker_config.get(metric_name, ('o', 'full'))
                
                clipped_means = []
                clipped_stds = []
                y_min, y_max = -150, 20
                
                for i, (mean_val, std_val) in enumerate(zip(means, stds)):
                    if mean_val < y_min:
                        clipped_mean = y_min
                        clipped_means.append(clipped_mean)
                        clipped_stds.append(0)
                        ax1.text(x_pos[i], clipped_mean + 5, 
                                f'{mean_val:.0f}±{std_val:.0f}', 
                                ha='center', va='bottom',
                                fontsize=9, fontweight='bold', color=color,
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor=color))
                    elif mean_val > y_max:
                        clipped_mean = y_max
                        clipped_means.append(clipped_mean)
                        clipped_stds.append(0)
                        ax1.text(x_pos[i], clipped_mean - 5, 
                                f'{mean_val:.0f}±{std_val:.0f}', 
                                ha='center', va='top',
                                fontsize=9, fontweight='bold', color=color,
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor=color))
                    else:
                        error_top = mean_val + std_val
                        error_bottom = mean_val - std_val
                        
                        if error_top > y_max or error_bottom < y_min:
                            clipped_std = min(std_val, abs(y_max - mean_val), abs(mean_val - y_min))
                            clipped_stds.append(clipped_std)
                        else:
                            clipped_stds.append(std_val)
                        
                        clipped_means.append(mean_val)
                
                if fillstyle == 'full':
                    face_color = color
                    edge_color = color
                    edge_width = 1
                else:
                    face_color = 'none'  
                    edge_color = color
                    edge_width = 2
                
                linestyle = '-'
                linewidth = 2
                markersize = 10
                
                ax1.errorbar(x_pos, clipped_means, yerr=clipped_stds, 
                            label=label, marker=marker, capsize=3, 
                            color=color, linestyle=linestyle, 
                            markersize=markersize, linewidth=linewidth, 
                            markerfacecolor=face_color,
                            markeredgecolor=edge_color, 
                            markeredgewidth=edge_width)
            
            ax1.set_title('Batch Effect Change Metrics\n(↑ = positive better, ↓ = negative better)\nChange% = (after-before)/before×100', fontweight='bold', fontsize=16)
            ax1.set_ylabel('Percentage Change (%)', fontsize=14)
            ax1.set_xlabel('Parameter Combinations', fontsize=14)
            ax1.set_ylim(-160, 30)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=12)
            ax1.legend(loc='upper right', fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            plt.tight_layout()
            if save_path:
                save_path_1 = f'{save_path}1_batch_effect_change_metrics.png'
                plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"Change metrics plot saved to: {save_path_1}")
            plt.show()
            plots_created.append('change_metrics')
        
        # Figure 2: Preservation metrics
        if preservation_metrics:
            fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
            
            preservation_colors = ['#158945', '#4A4A4A']
            
            for idx, (metric_name, metric_data) in enumerate(preservation_metrics.items()):
                means, stds = [], []
                for param1_val in param1_values:
                    for param2_val in param2_values:
                        values = metric_data.get((param1_val, param2_val), [])
                        means.append(np.mean(values) if values else 0)
                        stds.append(np.std(values) if values else 0)
                
                if metric_name in metric_info:
                    direction = "↑" if metric_info[metric_name]['better'] == 'Higher' else "↓"
                    label = f"{metric_info[metric_name]['name']} {direction}"
                else:
                    direction = "↑"
                    label = f"{metric_name.replace('_', ' ').title()} {direction}"
                
                color = preservation_colors[idx % len(preservation_colors)]
                
                ax2.errorbar(x_pos, means, yerr=stds, 
                            label=label, marker='o', capsize=3, 
                            color=color, linestyle='-', 
                            markersize=10, linewidth=2, 
                            markerfacecolor=color, markeredgecolor=color, markeredgewidth=1)
            
            ax2.set_title('Biological Signal Preservation Metrics\n(↑ = higher better)\nPreservation% = after/before×100', fontweight='bold', fontsize=16)
            ax2.set_ylabel('Preservation Percentage (%)', fontsize=14)
            ax2.set_xlabel('Parameter Combinations', fontsize=14)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=12)
            ax2.legend(loc='upper left', fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Perfect Preservation')
            ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Threshold (50%)')
            
            plt.tight_layout()
            if save_path:
                save_path_2 = f'{save_path}2_preservation_metrics.png'
                plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"Preservation metrics plot saved to: {save_path_2}")
            plt.show()
            plots_created.append('preservation_metrics')
    
    # Figure 3: Differential expression metrics
    if diff_metrics:
        fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
        
        diff_colors = ["#6778B6", "#A46F6F"]
        
        for idx, (metric_name, metric_data) in enumerate(diff_metrics.items()):
            means, stds = [], []
            for param1_val in param1_values:
                for param2_val in param2_values:
                    values = metric_data.get((param1_val, param2_val), [])
                    means.append(np.mean(values) if values else 0)
                    stds.append(np.std(values) if values else 0)
            
            if metric_name in metric_info:
                direction = "↑" if metric_info[metric_name]['better'] == 'Higher' else "↓"
                label = f"{metric_info[metric_name]['name']} {direction}"
            else:
                direction = "↑" if 'recovery' in metric_name and 'false' not in metric_name else "↓"
                label = f"{metric_name.replace('_', ' ').title()} {direction}"
            
            color = diff_colors[idx % len(diff_colors)]
            
            ax3.errorbar(x_pos, means, yerr=stds,
                        label=label, marker='o', capsize=3, 
                        color=color, linestyle='-', 
                        markersize=10, linewidth=2, 
                        markerfacecolor=color, markeredgecolor=color, markeredgewidth=1)
        
        ax3.set_title('Differential Expression Metrics\n(↑ = higher better, ↓ = lower better)', fontweight='bold', fontsize=16)
        ax3.set_xlabel('Parameter Combinations', fontsize=14)
        ax3.set_ylabel('Metric Values', fontsize=14)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=12)
        ax3.legend(loc='upper left', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            save_path_3 = f'{save_path}3_differential_expression_change_metrics.png'
            plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Differential expression metrics plot saved to: {save_path_3}")
        plt.show()
        plots_created.append('differential_expression_change_metrics')

        heatmap_targets = [
            'recovery_overlap_change_rate',
            'recovery_false_positive_change_rate'
        ]
        heatmap_targets = [m for m in heatmap_targets if m in diff_metrics]

        if heatmap_targets:
            # Pre-compute heatmap data so we can share a consistent color scale across plots.
            heatmap_data_map = {}
            vmin, vmax = np.inf, -np.inf

            for metric_name in heatmap_targets:
                heatmap_data = np.full((len(param2_values), len(param1_values)), np.nan)
                for col_idx, param1_val in enumerate(param1_values):
                    for row_idx, param2_val in enumerate(param2_values):
                        values = diff_metrics[metric_name].get((param1_val, param2_val), [])
                        if values:
                            heatmap_data[row_idx, col_idx] = np.mean(values)

                if np.any(~np.isnan(heatmap_data)):
                    current_min = np.nanmin(heatmap_data)
                    current_max = np.nanmax(heatmap_data)
                    vmin = min(vmin, current_min)
                    vmax = max(vmax, current_max)

                heatmap_data_map[metric_name] = heatmap_data

            if not np.isfinite(vmin) or not np.isfinite(vmax):
                # No numeric data available; fall back to a small symmetric range.
                vmin, vmax = -1.0, 1.0
            elif vmin == vmax:
                # Expand a constant range slightly so the colormap remains informative.
                margin = abs(vmin) * 0.1 if vmin != 0 else 1.0
                vmin -= margin
                vmax += margin

            fig_h, axes_h = plt.subplots(1, len(heatmap_targets), figsize=(6 * len(heatmap_targets), 6), constrained_layout=True)
            if not isinstance(axes_h, np.ndarray):
                axes_h = np.array([axes_h])

            scalar_mappables = []
            for ax_h, metric_name in zip(axes_h, heatmap_targets):
                heatmap_data = heatmap_data_map[metric_name]
                im = ax_h.imshow(heatmap_data, origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
                scalar_mappables.append(im)

                ax_h.set_xticks(np.arange(len(param1_values)))
                ax_h.set_xticklabels([f"{val:g}" for val in param1_values], rotation=45, ha='right')
                ax_h.set_yticks(np.arange(len(param2_values)))
                ax_h.set_yticklabels([f"{val:g}" for val in param2_values])

                ax_h.set_xlabel(param_short_names.get(param1_name, param1_name), fontsize=12)
                ax_h.set_ylabel(param_short_names.get(param2_name, param2_name), fontsize=12)

                title = metric_info.get(metric_name, {}).get('name', metric_name.replace('_', ' ').title())
                ax_h.set_title(f"{title}", fontsize=14, fontweight='bold')

                for row_idx in range(len(param2_values)):
                    for col_idx in range(len(param1_values)):
                        if np.isnan(heatmap_data[row_idx, col_idx]):
                            ax_h.text(col_idx, row_idx, 'NA', ha='center', va='center', color='white', fontsize=9, fontweight='bold')

            if scalar_mappables:
                cbar = fig_h.colorbar(scalar_mappables[0], ax=axes_h, fraction=0.046, pad=0.04)
                cbar.ax.set_ylabel('Mean Value', rotation=270, labelpad=15)

            if save_path:
                save_path_h = f'{save_path}3b_differential_expression_heatmaps.png'
                plt.savefig(save_path_h, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"Differential expression heatmaps saved to: {save_path_h}")
            plt.show()
            plots_created.append('differential_expression_heatmaps')
    
    if verbose:
        print(f"Created {len(plots_created)} separate plots: {plots_created}")
    
    return plots_created


def compare_batch_correction_across_masks(mask_dirs, save_path=None, verbose=True):
    """Compare batch correction effectiveness across different mask configurations."""
    
    if verbose:
        print("=" * 60)
        print("COMPARE BATCH CORRECTION ACROSS MASKS")
        print("=" * 60)
    
    # Load data from all mask configurations
    all_mask_data = {}
    
    for mask_name, mask_dir in mask_dirs.items():
        json_files = glob.glob(f"{mask_dir}/comprehensive_metrics_seed*.json")
        
        if not json_files:
            print(f"Warning: No JSON files found in {mask_dir}")
            continue
        
        if verbose:
            print(f"{mask_name}: Found {len(json_files)} seed files")
        
        # Calculate metrics for each seed
        change_metrics_list = []
        preservation_metrics_list = []
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            before_metrics = data['batch_effect_metrics']['before_correction']
            after_metrics = data['batch_effect_metrics']['after_correction']
            
            changes = calculate_batch_correction_changes(before_metrics, after_metrics)
            preservation = calculate_biological_preservation(after_metrics)
            
            change_metrics_list.append(changes)
            preservation_metrics_list.append(preservation)
        
        # Aggregate across seeds
        all_metrics = {}
        for metrics_dict in change_metrics_list:
            for k, v in metrics_dict.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)
        
        for metrics_dict in preservation_metrics_list:
            for k, v in metrics_dict.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)
        
        # Calculate mean and std
        summary = {}
        for metric, values in all_metrics.items():
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        all_mask_data[mask_name] = summary
    
    if not all_mask_data:
        print("Error: No data loaded from any mask configuration!")
        return
    
    # Define metrics
    change_metric_names = ['silhouette', 'kBET', 'LISI', 'ARI', 'compositional_effect_size', 'pca_batch_effect']
    preservation_metric_names = ['biological_variability_preservation', 'conserved_differential_proportion']
    
    metric_display_names = {
        'silhouette': 'Silhouette\nScore\n(↓ better)',
        'kBET': 'kBET\nScore\n(↓ better)',
        'LISI': 'LISI\nScore\n(↑ better)',
        'ARI': 'Adjusted Rand\nIndex\n(↓ better)',
        'compositional_effect_size': 'Compositional\nEffect Size\n(↓ better)',
        'pca_batch_effect': 'PCA Batch\nEffect\n(↓ better)',
        'biological_variability_preservation': 'Bio Variability\nPreservation\n(↑ better)',
        'conserved_differential_proportion': 'Conserved Differential\nProportion\n(↑ better)'
    }
    
    mask_names = list(mask_dirs.keys())
    x_pos = np.arange(len(change_metric_names))
    bar_width = 0.2
    
    # Figure 1: Change metrics (6 metrics)
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    colors = ['#E8E8E8', '#B0C4DE', '#87CEEB', '#4682B4']
    
    for i, mask_name in enumerate(mask_names):
        means = [all_mask_data[mask_name][m]['mean'] for m in change_metric_names]
        stds = [all_mask_data[mask_name][m]['std'] for m in change_metric_names]
        
        offset = (i - 1.5) * bar_width
        bars = ax1.bar(x_pos + offset, means, bar_width, yerr=stds, 
                label=mask_name, capsize=3, alpha=0.85, color=colors[i], edgecolor='black')
        
        # Add value labels on bars (mean only)
        for j, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            label_y = height + (std_val if height >= 0 else -std_val)
            ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{mean_val:.1f}', 
                    ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=9)
    
    ax1.set_ylabel('Percentage Change (%)', fontsize=14)
    ax1.set_title('Batch Effect Reduction Across Mask Configurations (negative is better)\n', fontsize=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([metric_display_names[m] for m in change_metric_names], fontsize=12)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        save_path_1 = f'{save_path}1_compare_batch_effect_change.png'
        plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Change metrics plot saved to: {save_path_1}")
    plt.show()
    
    # Figure 2: Preservation metrics (2 metrics)
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    x_pos_pres = np.arange(len(preservation_metric_names))
    
    for i, mask_name in enumerate(mask_names):
        means = [all_mask_data[mask_name][m]['mean'] for m in preservation_metric_names]
        stds = [all_mask_data[mask_name][m]['std'] for m in preservation_metric_names]
        
        offset = (i - 1.5) * bar_width
        bars = ax2.bar(x_pos_pres + offset, means, bar_width, yerr=stds,
                label=mask_name, capsize=3, alpha=0.85, color=colors[i], edgecolor='black')
        
        # Add value labels on bars (mean only)
        for j, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std_val + 1,
                    f'{mean_val:.1f}', 
                    ha='center', va='bottom', fontsize=10)
    
    ax2.set_ylabel('Preservation Percentage (%)', fontsize=14)
    ax2.set_title('Biological Signal Preservation Across Mask Configurations \n (higher is better)\n', fontsize=15)
    ax2.set_xticks(x_pos_pres)
    ax2.set_xticklabels([metric_display_names[m] for m in preservation_metric_names], fontsize=12)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Perfect (100%)')
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Threshold (50%)')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        save_path_2 = f'{save_path}2_compare_bio_preservation.png'
        plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Preservation metrics plot saved to: {save_path_2}")
    plt.show()
    
    if verbose:
        print("=" * 60)
        print("Comparison completed!")
        print("=" * 60)


def compare_differential_expression_across_masks(mask_dirs, save_path=None, verbose=True):
    """Compare differential expression metrics across different mask configurations."""
    
    if verbose:
        print("=" * 60)
        print("COMPARE DIFFERENTIAL EXPRESSION ACROSS MASKS")
        print("=" * 60)
    
    # Load data from all mask configurations
    all_mask_de_data = {}
    
    for mask_name, mask_dir in mask_dirs.items():
        json_files = glob.glob(f"{mask_dir}/comprehensive_metrics_seed*.json")
        
        if not json_files:
            print(f"Warning: No JSON files found in {mask_dir}")
            continue
        
        if verbose:
            print(f"{mask_name}: Found {len(json_files)} seed files")
        
        # Collect metrics across seeds
        metrics_dict = {
            'recovery_overlap_change_rate': [],
            'recovery_false_positive_change_rate': [],
            'overlap_1v2_rate': [],
            'overlap_1v3_rate': [],
            'fp_1v2_rate': [],
            'fp_1v3_rate': []
        }
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            diff_data = data['differential_expression']
            
            metrics_dict['recovery_overlap_change_rate'].append(
                diff_data['results']['overall']['recovery_overlap_change_rate'])
            metrics_dict['recovery_false_positive_change_rate'].append(
                diff_data['results']['overall']['recovery_false_positive_change_rate'])
            metrics_dict['overlap_1v2_rate'].append(
                diff_data['results']['compare_1v2']['batch_effect_errors']['overlap_rate'])
            metrics_dict['overlap_1v3_rate'].append(
                diff_data['results']['compare_1v3']['after_correction_errors']['overlap_rate'])
            
            dataset1_sig_count = diff_data['results']['dataset1']['significant_count']
            gained_1v2_count = diff_data['results']['compare_1v2']['batch_effect_errors']['gained_counts']
            gained_1v3_count = diff_data['results']['compare_1v3']['after_correction_errors']['gained_counts']
            
            # Avoid division by zero
            if dataset1_sig_count > 0:
                metrics_dict['fp_1v2_rate'].append(gained_1v2_count / dataset1_sig_count * 100)
                metrics_dict['fp_1v3_rate'].append(gained_1v3_count / dataset1_sig_count * 100)
            else:
                metrics_dict['fp_1v2_rate'].append(0.0)
                metrics_dict['fp_1v3_rate'].append(0.0)
        
        # Calculate summary statistics
        summary = {}
        for metric, values in metrics_dict.items():
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        all_mask_de_data[mask_name] = summary
    
    if not all_mask_de_data:
        print("Error: No data loaded from any mask configuration!")
        return
    
    # Define metrics and their display names
    metric_names = [
        'recovery_overlap_change_rate',
        'recovery_false_positive_change_rate',
        'overlap_1v2_rate',
        'overlap_1v3_rate',
        'fp_1v2_rate',
        'fp_1v3_rate'
    ]
    
    metric_display_names = {
        'recovery_overlap_change_rate': 'Recovery Overlap\nChange Rate (%)',
        'recovery_false_positive_change_rate': 'Recovery FP\nChange Rate (%)',
        'overlap_1v2_rate': '1v2 Overlap\nRate (%)',
        'overlap_1v3_rate': '1v3 Overlap\nRate (%)',
        'fp_1v2_rate': '1v2 FP\nRate (%)',
        'fp_1v3_rate': '1v3 FP\nRate (%)'
    }
    
    mask_names = list(mask_dirs.keys())
    x_pos = np.arange(len(metric_names))
    bar_width = 0.2
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#E8E8E8', '#B0C4DE', '#87CEEB', '#4682B4']
    
    for i, mask_name in enumerate(mask_names):
        means = [all_mask_de_data[mask_name][m]['mean'] for m in metric_names]
        stds = [all_mask_de_data[mask_name][m]['std'] for m in metric_names]
        
        offset = (i - 1.5) * bar_width
        bars = ax.bar(x_pos + offset, means, bar_width, yerr=stds,
               label=mask_name, capsize=3, alpha=0.85, color=colors[i], edgecolor='black')
        
        # Add value labels on bars (mean only)
        for j, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            label_y = height + (std_val if height >= 0 else -std_val)
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{mean_val:.1f}', 
                   ha='center', va='bottom' if height >= 0 else 'top', 
                   fontsize=9)
    
    ax.set_title('Differential Expression Recovery Across Mask Configurations \n(1=Clean, 2=With BE, 3=After Correction)\n', fontsize=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([metric_display_names[m] for m in metric_names], fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"DE comparison plot saved to: {save_path}")
    plt.show()
    
    if verbose:
        print("=" * 60)
        print("Comparison completed!")
        print("=" * 60)

