
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import os
from collections import defaultdict
from matplotlib_venn import venn3, venn3_circles

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
    diff_absolute_metrics = defaultdict(lambda: defaultdict(list))  # For absolute rates
    
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
        json_files = glob.glob(f"{output_dir}/correction_metrics_seed*.json")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    metrics = json.load(f)
                
                # Get before and after batch metrics
                batch_before = metrics.get('correction_results', {}).get('batch_correction', {}).get('before', {})
                batch_after = metrics.get('correction_results', {}).get('batch_correction', {}).get('after', {})
                
                changes = calculate_batch_correction_changes(batch_before, batch_after)
                for metric_name, value in changes.items():
                    batch_change_metrics[metric_name][(param1_val, param2_val)].append(value)
                    
                # Get bio preservation metrics (convert to percentage)
                bio_preservation = metrics.get('correction_results', {}).get('bio_preservation', {})
                for metric_name, value in bio_preservation.items():
                    if isinstance(value, (int, float)):
                        batch_preservation_metrics[metric_name][(param1_val, param2_val)].append(value * 100)

                # Get differential expression summary
                diff_summary = metrics.get('correction_results', {}).get('differential_expression', {}).get('summary', {})
                for metric_name, value in diff_summary.items():
                    if isinstance(value, (int, float)):
                        diff_metrics[metric_name][(param1_val, param2_val)].append(value)
                
                # Extract absolute rates directly from the new JSON format
                diff_expr = metrics.get('correction_results', {}).get('differential_expression', {})
                
                # Read pre-calculated rates from Y_with_batch and Y_after_correction
                y_with_batch = diff_expr.get('Y_with_batch', {})
                y_after_correction = diff_expr.get('Y_after_correction', {})
                
                # Extract TP rates (already calculated as rate_pct)
                batch_tp_rate = y_with_batch.get('true_positive', {}).get('rate_pct', 0)
                corrected_tp_rate = y_after_correction.get('true_positive', {}).get('rate_pct', 0)
                
                diff_absolute_metrics['tp_batch_rate'][(param1_val, param2_val)].append(batch_tp_rate)
                diff_absolute_metrics['tp_corrected_rate'][(param1_val, param2_val)].append(corrected_tp_rate)
                
                # Extract FP rates (already calculated as rate_pct)
                batch_fp_rate = y_with_batch.get('false_positive', {}).get('rate_pct', 0)
                corrected_fp_rate = y_after_correction.get('false_positive', {}).get('rate_pct', 0)
                
                diff_absolute_metrics['fp_batch_rate'][(param1_val, param2_val)].append(batch_fp_rate)
                diff_absolute_metrics['fp_corrected_rate'][(param1_val, param2_val)].append(corrected_fp_rate)
                
                # Extract F1, Precision, Recall for Figure 4, 5, 6
                if 'f1_score' in y_with_batch:
                    diff_metrics['batch_f1_score'][(param1_val, param2_val)].append(y_with_batch['f1_score'])
                if 'precision' in y_with_batch:
                    diff_metrics['batch_precision'][(param1_val, param2_val)].append(y_with_batch['precision'])
                if 'recall' in y_with_batch:
                    diff_metrics['batch_recall'][(param1_val, param2_val)].append(y_with_batch['recall'])
                
                if 'f1_score' in y_after_correction:
                    diff_metrics['corrected_f1_score'][(param1_val, param2_val)].append(y_after_correction['f1_score'])
                if 'precision' in y_after_correction:
                    diff_metrics['corrected_precision'][(param1_val, param2_val)].append(y_after_correction['precision'])
                if 'recall' in y_after_correction:
                    diff_metrics['corrected_recall'][(param1_val, param2_val)].append(y_after_correction['recall'])
                        
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
    
    all_batch_metrics = {**batch_change_metrics, **batch_preservation_metrics}
    
    return all_batch_metrics, diff_metrics, diff_absolute_metrics


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

def visualize_differential_expression_metrics(output_dir, save_plots=True, verbose=True):
    """
    Visualize differential expression with Venn diagrams showing overlap of significant glycans
    across Y_clean, Y_with_batch, and Y_corrected datasets.
    """

    
    # Load correction metrics files
    json_files = glob.glob(f"{output_dir}/correction_metrics_seed*.json")
    
    if verbose:
        print("=" * 60)
        print("VISUALIZE_DIFFERENTIAL_EXPRESSION_VENN_DIAGRAMS")
        print("=" * 60)
        print(f"Found {len(json_files)} correction metrics files.")
        
    if not json_files:
        print(f"Error: No correction metrics files found in {output_dir}")
        return
    
    # Collect sets across all runs
    all_y_clean_sets = []
    all_y_batch_sets = []
    all_y_corrected_sets = []
    
    for file_path in sorted(json_files):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        diff_expr = data.get('correction_results', {}).get('differential_expression', {})
        
        y_clean_glycans = set(diff_expr.get('Y_clean', {}).get('significant_glycans', []))
        y_batch_glycans = set(diff_expr.get('Y_with_batch', {}).get('significant_glycans', []))
        y_corrected_glycans = set(diff_expr.get('Y_corrected', {}).get('significant_glycans', []))
        
        all_y_clean_sets.append(y_clean_glycans)
        all_y_batch_sets.append(y_batch_glycans)
        all_y_corrected_sets.append(y_corrected_glycans)
    
    # Create figure with subplots for each run
    n_runs = len(json_files)
    n_cols = min(3, n_runs)  # Max 3 columns
    n_rows = (n_runs + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(7 * n_cols, 6 * n_rows))
    
    for idx, (y_clean, y_batch, y_corrected) in enumerate(zip(all_y_clean_sets, all_y_batch_sets, all_y_corrected_sets)):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        
        # Create Venn diagram
        venn = venn3([y_clean, y_batch, y_corrected],
                     set_labels=('Y_clean', 'Y_with_batch', 'Y_corrected'),
                     set_colors=('#2E8B57', '#FF6347', '#4169E1'),
                     alpha=0.6,
                     ax=ax)
        
        # Add circles
        venn3_circles([y_clean, y_batch, y_corrected], ax=ax, linewidth=1.5)
        
        # Calculate statistics
        total_glycans = max(max(y_clean, default=0), max(y_batch, default=0), max(y_corrected, default=0))
        overlap_clean_corrected = len(y_clean & y_corrected)
        overlap_clean_batch = len(y_clean & y_batch)
        
        seed_num = json_files[idx].split("seed")[1].split(".")[0] if "seed" in json_files[idx] else idx+1
        
        ax.set_title(f'Run {idx + 1} (Seed: {seed_num})\\n' +
                    f'Recovery: Clean∩Corrected={overlap_clean_corrected}/{len(y_clean)} ({overlap_clean_corrected/len(y_clean)*100:.1f}%)\\n' +
                    f'Batch FP: {len(y_batch - y_clean)}, Correction FP: {len(y_corrected - y_clean)}',
                    fontsize=11, fontweight='bold')
    
    plt.suptitle(f'Differential Expression: Venn Diagrams Across {n_runs} Runs\\n' +
                f'Significant Glycan Overlaps (Total glycans: ~{total_glycans})',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        save_path = f"{output_dir}/2_differential_expression_venn.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Venn diagram saved to: {save_path}")
    
    if verbose:
        # Print summary statistics
        print(f"\\nTotal runs: {n_runs}")
        for idx, (y_clean, y_batch, y_corrected) in enumerate(zip(all_y_clean_sets, all_y_batch_sets, all_y_corrected_sets)):
            print(f"\\nRun {idx + 1}:")
            print(f"  Y_clean: {len(y_clean)} significant glycans")
            print(f"  Y_with_batch: {len(y_batch)} significant glycans")
            print(f"  Y_corrected: {len(y_corrected)} significant glycans")
            print(f"  Clean ∩ Batch: {len(y_clean & y_batch)}")
            print(f"  Clean ∩ Corrected: {len(y_clean & y_corrected)}")
            print(f"  Batch ∩ Corrected: {len(y_batch & y_corrected)}")
            print(f"  All 3: {len(y_clean & y_batch & y_corrected)}")
        print("=" * 60)
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
    
    batch_metrics, diff_metrics, diff_absolute_metrics = extract_metrics_with_calculations(all_results)
    
    metric_info = {
        'silhouette': {'name': 'Silhouette Score', 'better': 'Lower'},
        'kBET': {'name': 'kBET Score', 'better': 'Lower'},
        'LISI': {'name': 'LISI Score', 'better': 'Higher'},
        'ARI': {'name': 'Adjusted Rand Index', 'better': 'Lower'},
        'compositional_effect_size': {'name': 'Compositional Effect Size', 'better': 'Lower'},
        'pca_batch_effect': {'name': 'PCA Batch Effect', 'better': 'Lower'},
        'biological_variability_preservation': {'name': 'Bio Variability Preservation', 'better': 'Higher'},
        'conserved_differential_proportion': {'name': 'Conserved Differential Proportion', 'better': 'Higher'}
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
    
 
    if diff_absolute_metrics:
        fig3, (ax3_top, ax3_bottom) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Collect all data for both subplots to determine unified y-axis range
        all_tp_means = []
        all_tp_stds = []
        all_fp_means = []
        all_fp_stds = []
        
        # Store data for filling between curves
        tp_data_dict = {}
        fp_data_dict = {}
        total_glycans_estimate = None
        
        # Subplot 1 (top): True Positive  Rates (2 lines: batch vs corrected)
        tp_metric_names = ['tp_batch_rate', 'tp_corrected_rate']
        tp_labels = ['Y_with_batch (With batch effect)', 'Y_corrected (After correction)']
        tp_colors = ['#FF6347', '#4169E1']  # Red, Blue
        
        for metric_name, label, color in zip(tp_metric_names, tp_labels, tp_colors):
            if metric_name in diff_absolute_metrics:
                metric_data = diff_absolute_metrics[metric_name]
                means, stds = [], []
                for param1_val in param1_values:
                    for param2_val in param2_values:
                        values = metric_data.get((param1_val, param2_val), [])
                        means.append(np.mean(values) if values else 0)
                        stds.append(np.std(values) if values else 0)
                
                tp_data_dict[metric_name] = means
                all_tp_means.extend(means)
                all_tp_stds.extend(stds)
                
                ax3_top.errorbar(x_pos, means, yerr=stds,
                            label=label, marker='o', capsize=3, 
                            color=color, linestyle='-', 
                            markersize=8, linewidth=2, 
                            markerfacecolor=color, markeredgecolor=color, markeredgewidth=1)
        
        # Fill between red and blue in top subplot (blue > red = green, blue < red = red)
        if 'tp_batch_rate' in tp_data_dict and 'tp_corrected_rate' in tp_data_dict:
            red_means = np.array(tp_data_dict['tp_batch_rate'])
            blue_means = np.array(tp_data_dict['tp_corrected_rate'])
            ax3_top.fill_between(x_pos, red_means, blue_means, 
                                where=(blue_means >= red_means), 
                                color='green', alpha=0.2, interpolate=True, label='Improvement')
            ax3_top.fill_between(x_pos, red_means, blue_means, 
                                where=(blue_means < red_means), 
                                color='red', alpha=0.2, interpolate=True, label='Degradation')
        
        # Estimate total glycans from the data
        for combo_data in all_results.values():
            if combo_data[param1_name] == param1_values[0] and combo_data[param2_name] == param2_values[0]:
                json_files = glob.glob(f"{combo_data['output_dir']}/correction_metrics_seed*.json")
                if json_files:
                    with open(json_files[0], 'r') as f:
                        metrics = json.load(f)
                    diff_expr = metrics.get('correction_results', {}).get('differential_expression', {})
                    all_glycans = set()
                    all_glycans.update(diff_expr.get('Y_clean', {}).get('significant_glycans', []))
                    all_glycans.update(diff_expr.get('Y_with_batch', {}).get('significant_glycans', []))
                    all_glycans.update(diff_expr.get('Y_corrected', {}).get('significant_glycans', []))
                    if all_glycans:
                        total_glycans_estimate = max(all_glycans)
                    break
                break
        
        total_glycans_str = f", Total glycans: {total_glycans_estimate}" if total_glycans_estimate else ""
        ax3_top.set_title('Differential Expression: True Positive Rates\n' +
                         f'(Correctly identified / Ground Truth × 100{total_glycans_str})',
                         fontweight='bold', fontsize=15)
        ax3_top.set_ylabel('TP Rate (%)', fontsize=14, fontweight='bold')
        ax3_top.set_xticks(x_pos)
        ax3_top.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=11)
        ax3_top.legend(loc='lower left', fontsize=11)
        ax3_top.grid(True, alpha=0.3, zorder=0)
        ax3_top.axhline(y=100, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Subplot 2 (bottom): False positive rates (2 lines)
        fp_metric_names = ['fp_batch_rate', 'fp_corrected_rate']
        fp_labels = ['Y_with_batch (Batch effect FP)', 'Y_corrected (After correction FP)']
        fp_colors = ['#FF6347', '#4169E1']  # Red, Blue
        
        for metric_name, label, color in zip(fp_metric_names, fp_labels, fp_colors):
            if metric_name in diff_absolute_metrics:
                metric_data = diff_absolute_metrics[metric_name]
                means, stds = [], []
                for param1_val in param1_values:
                    for param2_val in param2_values:
                        values = metric_data.get((param1_val, param2_val), [])
                        means.append(np.mean(values) if values else 0)
                        stds.append(np.std(values) if values else 0)
                
                fp_data_dict[metric_name] = means
                all_fp_means.extend(means)
                all_fp_stds.extend(stds)
                
                ax3_bottom.errorbar(x_pos, means, yerr=stds,
                            label=label, marker='o', capsize=3, 
                            color=color, linestyle='-', 
                            markersize=8, linewidth=2, 
                            markerfacecolor=color, markeredgecolor=color, markeredgewidth=1)
        
        # Fill between red and blue in bottom subplot (blue < red = green, blue > red = red)
        if 'fp_batch_rate' in fp_data_dict and 'fp_corrected_rate' in fp_data_dict:
            red_means = np.array(fp_data_dict['fp_batch_rate'])
            blue_means = np.array(fp_data_dict['fp_corrected_rate'])
            ax3_bottom.fill_between(x_pos, red_means, blue_means, 
                                   where=(blue_means <= red_means), 
                                   color='green', alpha=0.2, interpolate=True, label='Improvement')
            ax3_bottom.fill_between(x_pos, red_means, blue_means, 
                                   where=(blue_means > red_means), 
                                   color='red', alpha=0.2, interpolate=True, label='Degradation')
        
        ax3_bottom.set_title('Differential Expression: False Positive Rates\n' +
                            f'(False positives / Total glycans × 100{total_glycans_str})',
                            fontweight='bold', fontsize=15)
        ax3_bottom.set_ylabel('FP Rate (%)', fontsize=14, fontweight='bold')
        ax3_bottom.set_xlabel('Parameter Combinations', fontsize=14, fontweight='bold')
        ax3_bottom.set_xticks(x_pos)
        ax3_bottom.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=11)
        ax3_bottom.legend(loc='upper left', fontsize=11)
        ax3_bottom.grid(True, alpha=0.3, zorder=0)
        
        # Fixed y-axis range: 0-100%
        ax3_top.set_ylim(0, 100)
        ax3_bottom.set_ylim(0, 100)
        
        plt.tight_layout()
        if save_path:
            save_path_3 = f'{save_path}3_differential_expression_tp_fp_rates.png'
            plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Differential expression TP/FP rates plot saved to: {save_path_3}")
        plt.show()
        plots_created.append('differential_expression_tp_fp_rates')

        # Note: recovery_overlap_change_rate and recovery_false_positive_change_rate
        # have been removed as they are redundant with summary change rates
        heatmap_targets = []
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
    
    # Figure 4: F1 Score comparison
    if diff_metrics:
        f1_metrics = {}
        f1_data_dict = {}
        for metric_name in ['batch_f1_score', 'corrected_f1_score']:
            if metric_name in diff_metrics:
                f1_metrics[metric_name] = diff_metrics[metric_name]
        
        if f1_metrics:
            fig4, ax4 = plt.subplots(1, 1, figsize=(12, 8))
            
            for idx, (metric_name, metric_data) in enumerate(f1_metrics.items()):
                means, stds = [], []
                for param1_val in param1_values:
                    for param2_val in param2_values:
                        values = metric_data.get((param1_val, param2_val), [])
                        means.append(np.mean(values) if values else 0)
                        stds.append(np.std(values) if values else 0)
                
                f1_data_dict[metric_name] = means
                
                label = 'Y_with_batch' if 'batch' in metric_name else 'Y_corrected'
                color = '#FF6347' if 'batch' in metric_name else '#4169E1'
                
                ax4.errorbar(x_pos, means, yerr=stds, label=label, marker='o', capsize=3,
                            color=color, linestyle='-', markersize=8, linewidth=2,
                            markerfacecolor=color, markeredgecolor=color, markeredgewidth=1)
            
            # Fill between red and blue (blue > red = improvement)
            if 'batch_f1_score' in f1_data_dict and 'corrected_f1_score' in f1_data_dict:
                red_means = np.array(f1_data_dict['batch_f1_score'])
                blue_means = np.array(f1_data_dict['corrected_f1_score'])
                ax4.fill_between(x_pos, red_means, blue_means, 
                                where=(blue_means >= red_means), 
                                color='green', alpha=0.2, interpolate=True, label='Improvement')
                ax4.fill_between(x_pos, red_means, blue_means, 
                                where=(blue_means < red_means), 
                                color='red', alpha=0.2, interpolate=True, label='Degradation')
            
            ax4.set_title('F1 Score: Differential Expression Detection Quality\n(Harmonic mean of Precision and Recall)',
                         fontweight='bold', fontsize=15)
            ax4.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Parameter Combinations', fontsize=14, fontweight='bold')
            ax4.set_ylim(0, 1)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=11)
            ax4.legend(loc='lower left', fontsize=11)
            ax4.grid(True, alpha=0.3, zorder=0)
            
            plt.tight_layout()
            if save_path:
                save_path_4 = f'{save_path}4_f1_score.png'
                plt.savefig(save_path_4, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"F1 score plot saved to: {save_path_4}")
            plt.show()
            plots_created.append('f1_score')
    
    # Figure 5: Precision comparison
    if diff_metrics:
        precision_metrics = {}
        precision_data_dict = {}
        for metric_name in ['batch_precision', 'corrected_precision']:
            if metric_name in diff_metrics:
                precision_metrics[metric_name] = diff_metrics[metric_name]
        
        if precision_metrics:
            fig5, ax5 = plt.subplots(1, 1, figsize=(12, 8))
            
            for idx, (metric_name, metric_data) in enumerate(precision_metrics.items()):
                means, stds = [], []
                for param1_val in param1_values:
                    for param2_val in param2_values:
                        values = metric_data.get((param1_val, param2_val), [])
                        means.append(np.mean(values) if values else 0)
                        stds.append(np.std(values) if values else 0)
                
                precision_data_dict[metric_name] = means
                
                label = 'Y_with_batch' if 'batch' in metric_name else 'Y_corrected'
                color = '#FF6347' if 'batch' in metric_name else '#4169E1'
                
                ax5.errorbar(x_pos, means, yerr=stds, label=label, marker='o', capsize=3,
                            color=color, linestyle='-', markersize=8, linewidth=2,
                            markerfacecolor=color, markeredgecolor=color, markeredgewidth=1)
            
            # Fill between red and blue (blue > red = improvement)
            if 'batch_precision' in precision_data_dict and 'corrected_precision' in precision_data_dict:
                red_means = np.array(precision_data_dict['batch_precision'])
                blue_means = np.array(precision_data_dict['corrected_precision'])
                ax5.fill_between(x_pos, red_means, blue_means, 
                                where=(blue_means >= red_means), 
                                color='green', alpha=0.2, interpolate=True, label='Improvement')
                ax5.fill_between(x_pos, red_means, blue_means, 
                                where=(blue_means < red_means), 
                                color='red', alpha=0.2, interpolate=True, label='Degradation')
            
            ax5.set_title('Precision: Correctness of Detected Signals\n(TP / (TP + FP))',
                         fontweight='bold', fontsize=15)
            ax5.set_ylabel('Precision', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Parameter Combinations', fontsize=14, fontweight='bold')
            ax5.set_ylim(0, 1)
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=11)
            ax5.legend(loc='lower left', fontsize=11)
            ax5.grid(True, alpha=0.3, zorder=0)
            
            plt.tight_layout()
            if save_path:
                save_path_5 = f'{save_path}5_precision.png'
                plt.savefig(save_path_5, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"Precision plot saved to: {save_path_5}")
            plt.show()
            plots_created.append('precision')
    
    # Figure 6: Recall comparison
    if diff_metrics:
        recall_metrics = {}
        recall_data_dict = {}
        for metric_name in ['batch_recall', 'corrected_recall']:
            if metric_name in diff_metrics:
                recall_metrics[metric_name] = diff_metrics[metric_name]
        
        if recall_metrics:
            fig6, ax6 = plt.subplots(1, 1, figsize=(12, 8))
            
            for idx, (metric_name, metric_data) in enumerate(recall_metrics.items()):
                means, stds = [], []
                for param1_val in param1_values:
                    for param2_val in param2_values:
                        values = metric_data.get((param1_val, param2_val), [])
                        means.append(np.mean(values) if values else 0)
                        stds.append(np.std(values) if values else 0)
                
                recall_data_dict[metric_name] = means
                
                label = 'Y_with_batch' if 'batch' in metric_name else 'Y_corrected'
                color = '#FF6347' if 'batch' in metric_name else '#4169E1'
                
                ax6.errorbar(x_pos, means, yerr=stds, label=label, marker='o', capsize=3,
                            color=color, linestyle='-', markersize=8, linewidth=2,
                            markerfacecolor=color, markeredgecolor=color, markeredgewidth=1)
            
            # Fill between red and blue (blue > red = improvement)
            if 'batch_recall' in recall_data_dict and 'corrected_recall' in recall_data_dict:
                red_means = np.array(recall_data_dict['batch_recall'])
                blue_means = np.array(recall_data_dict['corrected_recall'])
                ax6.fill_between(x_pos, red_means, blue_means, 
                                where=(blue_means >= red_means), 
                                color='green', alpha=0.2, interpolate=True, label='Improvement')
                ax6.fill_between(x_pos, red_means, blue_means, 
                                where=(blue_means < red_means), 
                                color='red', alpha=0.2, interpolate=True, label='Degradation')
            
            ax6.set_title('Recall (Sensitivity): Coverage of True Signals\n(TP / (TP + FN))',
                         fontweight='bold', fontsize=15)
            ax6.set_ylabel('Recall', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Parameter Combinations', fontsize=14, fontweight='bold')
            ax6.set_ylim(0, 1)
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=11)
            ax6.legend(loc='lower left', fontsize=11)
            ax6.grid(True, alpha=0.3, zorder=0)
            
            plt.tight_layout()
            if save_path:
                save_path_6 = f'{save_path}6_recall.png'
                plt.savefig(save_path_6, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"Recall plot saved to: {save_path_6}")
            plt.show()
            plots_created.append('recall')
    
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
        json_files = glob.glob(f"{mask_dir}/correction_metrics_seed*.json")
        
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
            
            # Extract from new correction_metrics format
            before_metrics = data['correction_results']['batch_correction']['before']
            after_metrics = data['correction_results']['batch_correction']['after']
            
            changes = calculate_batch_correction_changes(before_metrics, after_metrics)
            
            # Extract bio preservation from new format
            bio_preservation = data['correction_results']['bio_preservation']
            preservation = {}
            for metric_name, value in bio_preservation.items():
                if isinstance(value, (int, float)):
                    preservation[metric_name] = value * 100  # Convert to percentage
            
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
        json_files = glob.glob(f"{mask_dir}/correction_metrics_seed*.json")
        
        if not json_files:
            print(f"Warning: No JSON files found in {mask_dir}")
            continue
        
        if verbose:
            print(f"{mask_name}: Found {len(json_files)} seed files")
        
        # Collect metrics across seeds
        metrics_dict = {
            'true_positive_change_pct': [],
            'false_positive_change_pct': [],
            'batch_tp_rate': [],
            'corrected_tp_rate': [],
            'batch_fp_rate': [],
            'corrected_fp_rate': []
        }
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract from new correction_metrics format
            diff_expr = data['correction_results']['differential_expression']
            
            # Get summary metrics
            summary = diff_expr['summary']
            metrics_dict['true_positive_change_pct'].append(summary['true_positive_change_pct'])
            metrics_dict['false_positive_change_pct'].append(summary['false_positive_change_pct'])
            
            # Calculate TP rates (true positive rates)
            y_clean_count = diff_expr['Y_clean']['significant_count']
            batch_tp_count = diff_expr['Y_with_batch']['true_positive']['count']
            corrected_tp_count = diff_expr['Y_after_correction']['true_positive']['count']
            
            if y_clean_count > 0:
                metrics_dict['batch_tp_rate'].append(batch_tp_count / y_clean_count * 100)
                metrics_dict['corrected_tp_rate'].append(corrected_tp_count / y_clean_count * 100)
            else:
                metrics_dict['batch_tp_rate'].append(0.0)
                metrics_dict['corrected_tp_rate'].append(0.0)
            
            # Calculate FP rates relative to total glycans
            all_glycans = set(diff_expr['Y_clean']['significant_glycans']) | \
                         set(diff_expr['Y_with_batch']['significant_glycans']) | \
                         set(diff_expr['Y_corrected']['significant_glycans'])
            total_glycans = max(all_glycans) if all_glycans else 98
            
            batch_fp_count = diff_expr['Y_with_batch']['false_positive']['count']
            corrected_fp_count = diff_expr['Y_after_correction']['false_positive']['count']
            
            metrics_dict['batch_fp_rate'].append(batch_fp_count / total_glycans * 100)
            metrics_dict['corrected_fp_rate'].append(corrected_fp_count / total_glycans * 100)
        
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
        'true_positive_change_pct',
        'false_positive_change_pct',
        'batch_tp_rate',
        'corrected_tp_rate',
        'batch_fp_rate',
        'corrected_fp_rate'
    ]
    
    metric_display_names = {
        'true_positive_change_pct': 'TP Change\n(%)',
        'false_positive_change_pct': 'FP Change\n(%)',
        'batch_tp_rate': 'Batch\nTP Rate (%)',
        'corrected_tp_rate': 'Corrected\nTP Rate (%)',
        'batch_fp_rate': 'Batch\nFP Rate (%)',
        'corrected_fp_rate': 'Corrected\nFP Rate (%)'
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
    
    ax.set_title('Differential Expression Recovery Across Mask Configurations\n(Batch vs Corrected Comparison)\n', fontsize=15)
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

