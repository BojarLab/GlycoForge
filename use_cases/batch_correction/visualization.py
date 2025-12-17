
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
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
    batch_vs_clean_metrics = defaultdict(lambda: defaultdict(list))
    corrected_vs_clean_metrics = defaultdict(lambda: defaultdict(list))
    diff_metrics = defaultdict(lambda: defaultdict(list))
    diff_absolute_metrics = defaultdict(lambda: defaultdict(list))
    
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
        
        json_files = glob.glob(f"{output_dir}/correction_metrics_seed*.json")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    metrics = json.load(f)
                
                batch_before = metrics.get('correction_results', {}).get('batch_correction', {}).get('before', {})
                batch_after = metrics.get('correction_results', {}).get('batch_correction', {}).get('after', {})
                
                changes = calculate_batch_correction_changes(batch_before, batch_after)
                for metric_name, value in changes.items():
                    batch_change_metrics[metric_name][(param1_val, param2_val)].append(value)
                    
                bio_pres = metrics.get('correction_results', {}).get('bio_preservation', {})
                
                # Extract batch_vs_clean metrics (Y_with_batch vs Y_clean)
                batch_vs_clean = bio_pres.get('batch_vs_clean', {})
                for metric_name, value in batch_vs_clean.items():
                    if isinstance(value, (int, float)):
                        batch_vs_clean_metrics[metric_name][(param1_val, param2_val)].append(value * 100)
                
                # Extract corrected_vs_clean metrics (Y_corrected vs Y_clean)
                corrected_vs_clean = bio_pres.get('corrected_vs_clean', {})
                for metric_name, value in corrected_vs_clean.items():
                    if isinstance(value, (int, float)):
                        corrected_vs_clean_metrics[metric_name][(param1_val, param2_val)].append(value * 100)

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
    
    return batch_change_metrics, batch_vs_clean_metrics, corrected_vs_clean_metrics, diff_metrics, diff_absolute_metrics


class SingleRunPlotter:
    """Single parameter combination visualization class"""
    
    def __init__(self, run_dir, verbose=True):
        """
        Initialize the plotter
        
        Parameters:
        - run_dir: Directory containing correction_metrics_seed*.json files
        - verbose: Whether to print detailed information
        """
        self.run_dir = run_dir
        self.verbose = verbose
        self.results = None
        
        # Load data at initialization
        self._load_data()
    
    def plot_single_run_metrics(self, save_path=None):
        """Generate batch correction summary plot"""
        if self.results is None:
            print("Error: No data loaded")
            return
        
        # Visualize batch correction effectiveness results
        raw_data = self.results.get('raw_data', {})
        config = self.results.get('config', {})
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
        
        # Split metrics by type
        change_metrics = [m for m in summary_stats.keys() if m in metric_info and metric_info[m]['type'] == 'change']
        ratio_metrics = [m for m in summary_stats.keys() if m in metric_info and metric_info[m]['type'] == 'ratio']
        
        if self.verbose:
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
            
            ax1.set_ylabel('Percentage Change (%)', fontsize=14, fontweight='bold')
            ax1.set_title(f'Batch Effect Reduction\n (after - before)/before * 100 \n(Mean ± SD across {n_runs} runs)', fontsize=16, fontweight='bold')
            ax1.set_xticks(range(len(change_metrics)))
            ax1.set_xticklabels(change_labels, rotation=45, ha='right', fontsize=12)
            ax1.tick_params(axis='y', labelsize=12)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.grid(True, alpha=0.3)
            
            for i, (bar, mean_val, std_val) in enumerate(zip(bars1, change_means, change_stds)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + (std_val if height >= 0 else -std_val),
                        f'{mean_val:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=11, fontweight='bold')
            
            # Add legend for change metrics
            legend_elements1 = [Patch(facecolor='green', alpha=0.7, label='Improvement'),
                               Patch(facecolor='red', alpha=0.7, label='Worsening')]
            ax1.legend(handles=legend_elements1, loc='upper right', fontsize=11)
        
        # Plot 2: Ratio metrics (preservation percentages)
        if ratio_metrics and ax2 is not None:
            ratio_labels = [f"{metric_info[m]['name']}\n({'↑' if metric_info[m]['better'] == 'Higher' else '↓'} better)" for m in ratio_metrics]
            ratio_means = [summary_stats[m]['mean'] for m in ratio_metrics]
            ratio_stds = [summary_stats[m]['std'] for m in ratio_metrics]
            
            ratio_colors = ['green' if ratio_means[i] > 50 else 'red' for i in range(len(ratio_metrics))]
            
            bars2 = ax2.bar(range(len(ratio_metrics)), ratio_means, yerr=ratio_stds,
                           capsize=5, alpha=0.7, color=ratio_colors, edgecolor='black')
            
            ax2.set_title(f'Biological Signal Preservation\n after/before * 100 \n(Mean ± SD across {n_runs} runs)', fontsize=16, fontweight='bold')
            ax2.set_xticks(range(len(ratio_metrics)))
            ax2.set_xticklabels(ratio_labels, rotation=45, ha='right', fontsize=12)
            ax2.tick_params(axis='y', labelsize=12)
            ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Perfect Preservation')
            ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Threshold (50%)')
            ax2.grid(True, alpha=0.3)
            
            for i, (bar, mean_val, std_val) in enumerate(zip(bars2, ratio_means, ratio_stds)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + std_val + 2,
                        f'{mean_val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # Add legend for ratio metrics
            legend_elements2 = [Patch(facecolor='green', alpha=0.7, label='Good(>50%)'),
                               Patch(facecolor='red', alpha=0.7, label='Poor(<=50%)')]
            ax2.legend(handles=legend_elements2, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        if self.verbose:
            plt.show()
        else:
            plt.close()
    
    def _load_data(self):
        """Load metrics from all seeds in the directory"""
        json_files = glob.glob(f"{self.run_dir}/correction_metrics_seed*.json")
        
        if not json_files:
            if self.verbose:
                print(f"Error: No correction_metrics_seed*.json files found in {self.run_dir}")
                self._suggest_directories()
            return
        
        if self.verbose:
            print(f"Found {len(json_files)} seed files in {self.run_dir}")
        
        # Aggregate metrics across all seeds
        all_before = []
        all_after = []
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            batch_correction = data['correction_results']['batch_correction']
            bio_preservation = data['correction_results']['bio_preservation']
            
            # Combine batch_correction and bio_preservation data
            before = batch_correction['before'].copy()
            after = batch_correction['after'].copy()
            
            # Add bio_preservation metrics to 'after'
            after['biological_variability_preservation'] = bio_preservation['corrected_vs_clean']['biological_variability']
            after['conserved_differential_proportion'] = bio_preservation['corrected_vs_clean']['conserved_differential']
            
            all_before.append(before)
            all_after.append(after)
        
        # Prepare results dictionary
        self.results = {
            'raw_data': {
                'before_correction': {},
                'after_correction': {}
            },
            'config': {
                'random_seeds': list(range(len(json_files)))
            }
        }
        
        # Aggregate metrics
        for metric in all_before[0].keys():
            self.results['raw_data']['before_correction'][metric] = [run[metric] for run in all_before]
        
        for metric in all_after[0].keys():
            self.results['raw_data']['after_correction'][metric] = [run[metric] for run in all_after]
    
    def _suggest_directories(self):
        """Suggest available directories when path is not found"""
        print(f"\nAvailable directories:")
        import os
        parts = self.run_dir.split('/')
        if len(parts) >= 2:
            base_dir = os.path.join(parts[0], parts[1])
            if os.path.exists(base_dir):
                subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
                for d in sorted(subdirs)[:10]:
                    print(f"  - {base_dir}/{d}")




class ParameterGridPlotter:
    """Parameter grid search results visualization class"""
    
    def __init__(self, all_results=None, results_dir="results", verbose=True):
        """
        Initialize the plotter
        
        Parameters:
        - all_results: Pre-loaded results dictionary
        - results_dir: Results directory path (for auto-scanning)
        - verbose: Whether to print detailed information
        """
        self.verbose = verbose
        
        # 1. Initialize data
        if all_results is None or not all_results:
            self.all_results = self._auto_scan_results(results_dir)
        else:
            self.all_results = all_results
        
        # 2. Parse parameters
        self.param1_name, self.param2_name, self.param1_values, self.param2_values = \
            self._parse_parameters()
        
        # 3. Generate common data structures
        self.combo_labels = self._generate_combo_labels()
        self.x_pos = np.arange(len(self.combo_labels))
        
        # 4. Extract all metrics (one-time extraction to avoid repeated IO)
        self.batch_metrics, self.batch_vs_clean, self.corrected_vs_clean, \
        self.diff_metrics, self.diff_absolute_metrics = \
            extract_metrics_with_calculations(self.all_results)
    
    # ==================== Public Interface ====================
    
    def plot_all(self, save_path=None):
        """Plot all charts (one-click generation)"""
        plots = []
        plots.extend(self.plot_batch_change_metrics(save_path))
        plots.extend(self.plot_bio_preservation(save_path))
        plots.extend(self.plot_tp_fp_rates(save_path))
        plots.extend(self.plot_diff_expr_metrics(save_path))
        
        if self.verbose:
            print(f"Created {len(plots)} plots: {plots}")
        
        return plots
    
    def plot_batch_change_metrics(self, save_path=None):
        """Plot batch effect change metrics (Figure 1)"""
        if not self.batch_metrics:
            return []
        
        fig_path = f'{save_path}1_batch_effect_change_metrics.png' if save_path else None
        self._plot_batch_change(fig_path)
        return ['change_metrics']
    
    def plot_bio_preservation(self, save_path=None):
        """Plot biological signal preservation (Figure 2)"""
        if not (self.batch_vs_clean and self.corrected_vs_clean):
            return []
        
        fig_path = f'{save_path}2_bio_signal_preservation.png' if save_path else None
        self._plot_bio_preservation(fig_path)
        return ['bio_signal_preservation']
    
    def plot_tp_fp_rates(self, save_path=None):
        """Plot TP/FP rates (Figure 3)"""
        if not self.diff_absolute_metrics:
            return []
        
        fig_path = f'{save_path}3_differential_expression_tp_fp_rates.png' if save_path else None
        self._plot_tp_fp_rates(fig_path)
        return ['differential_expression_tp_fp_rates']
    
    def plot_diff_expr_metrics(self, save_path=None):
        """Plot differential expression metrics (Figures 4-6: F1/Precision/Recall)"""
        if not self.diff_metrics:
            return []
        
        plots = []
        for metric_type in ['f1_score', 'precision', 'recall']:
            fig_num = {'f1_score': 4, 'precision': 5, 'recall': 6}[metric_type]
            fig_path = f'{save_path}{fig_num}_{metric_type}.png' if save_path else None
            self._plot_diff_expr_metric(metric_type, fig_path)
            plots.append(metric_type)
        
        return plots
    
    # ==================== Internal Methods ====================
    
    def _auto_scan_results(self, results_dir):
        """Auto-scan results directory"""
        if self.verbose:
            print(f"Auto-scanning results directory: {results_dir}")
        
        all_results = {}
        
        # Find all parameter combination directories
        combo_dirs = glob.glob(f"{results_dir}/kappa_mu_*_var_b_*") + \
                     glob.glob(f"{results_dir}/*/kappa_mu_*_var_b_*") + \
                     glob.glob(f"{results_dir}/bio_strength_*_k_dir_*") + \
                     glob.glob(f"{results_dir}/*/bio_strength_*_k_dir_*")
        
        if not combo_dirs:
            print(f"Error: No parameter combination directories found in {results_dir}")
            return {}
        
        # Auto-detect parameter format from first directory
        first_dir = os.path.basename(combo_dirs[0])
        
        if 'bio_strength' in first_dir and 'k_dir' in first_dir and 'kappa_mu' in first_dir and 'var_b' in first_dir:
            param1_name, param2_name = 'kappa_mu', 'var_b'
            param1_idx, param2_idx = 8, 11
        elif 'bio_strength' in first_dir and 'k_dir' in first_dir:
            param1_name, param2_name = 'bio_strength', 'k_dir'
            param1_idx, param2_idx = 2, 5
        else:
            param1_name, param2_name = 'kappa_mu', 'var_b'
            param1_idx, param2_idx = 2, 5
        
        if self.verbose:
            print(f"Detected parameter format: {param1_name} x {param2_name}")
        
        for combo_dir in combo_dirs:
            combo_name = os.path.basename(combo_dir)
            parts = combo_name.split('_')
            try:
                param1_val = float(parts[param1_idx])
                param2_val = float(parts[param2_idx])
            except (IndexError, ValueError):
                if self.verbose:
                    print(f"Warning: Could not parse parameters from {combo_name}, skipping...")
                continue
            
            all_results[combo_name] = {
                param1_name: param1_val,
                param2_name: param2_val,
                'results': {},
                'output_dir': combo_dir
            }
        
        if self.verbose:
            print(f"Found {len(all_results)} parameter combinations:")
            for combo_name in sorted(all_results.keys()):
                data = all_results[combo_name]
                print(f"  {combo_name}: {param1_name}={data[param1_name]}, {param2_name}={data[param2_name]}")
        
        return all_results
    
    def _parse_parameters(self):
        """Parse parameter names and values"""
        if not self.all_results:
            return None, None, [], []
        
        # Infer parameter names from first result
        first_combo = next(iter(self.all_results.values()))
        if 'k_dir' in first_combo:
            param1_name, param2_name = 'k_dir', 'bio_strength'
        elif 'bio_strength' in first_combo:
            param1_name, param2_name = 'bio_strength', 'k_dir'
        else:
            param1_name, param2_name = 'kappa_mu', 'var_b'
        
        # Extract unique parameter values
        param1_set = set()
        param2_set = set()
        for combo_data in self.all_results.values():
            param1_set.add(combo_data[param1_name])
            param2_set.add(combo_data[param2_name])
        
        param1_values = sorted(param1_set)
        param2_values = sorted(param2_set)
        
        if self.verbose:
            print(f"Parameter grid: {param1_name}={param1_values}, {param2_name}={param2_values}")
        
        return param1_name, param2_name, param1_values, param2_values
    
    def _generate_combo_labels(self):
        """Generate parameter combination labels"""
        param_short_names = {
            'k_dir': 'k_dir',
            'bio_strength': 'λ',
            'kappa_mu': 'κ_μ',
            'var_b': 'σ²_b'
        }
        
        combo_labels = []
        for param1_val in self.param1_values:
            for param2_val in self.param2_values:
                label1 = param_short_names.get(self.param1_name, self.param1_name)
                label2 = param_short_names.get(self.param2_name, self.param2_name)
                combo_labels.append(f'{label1}={param1_val}, {label2}={param2_val}')
        
        return combo_labels
    
    def _extract_metric_values(self, metric_data):
        """Extract means and stds from metric data"""
        means, stds = [], []
        for param1_val in self.param1_values:
            for param2_val in self.param2_values:
                values = metric_data.get((param1_val, param2_val), [])
                means.append(np.mean(values) if values else 0)
                stds.append(np.std(values) if values else 0)
        return means, stds
    
    def _plot_batch_change(self, save_path):
        """Internal: Plot batch effect change metrics"""
        change_metric_names = ['silhouette', 'kBET', 'LISI', 'ARI', 'compositional_effect_size', 'pca_batch_effect']
        change_metrics = {k: v for k, v in self.batch_metrics.items() if k in change_metric_names}
        
        if not change_metrics:
            return
        
        metric_info = {
            'silhouette': {'name': 'Silhouette Score', 'better': 'Lower'},
            'kBET': {'name': 'kBET Score', 'better': 'Lower'},
            'LISI': {'name': 'LISI Score', 'better': 'Higher'},
            'ARI': {'name': 'Adjusted Rand Index', 'better': 'Lower'},
            'compositional_effect_size': {'name': 'Compositional Effect Size', 'better': 'Lower'},
            'pca_batch_effect': {'name': 'PCA Batch Effect', 'better': 'Lower'}
        }
        
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
            means, stds = self._extract_metric_values(metric_data)
            
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
                    ax1.text(self.x_pos[i], clipped_mean + 5,
                            f'{mean_val:.0f}±{std_val:.0f}',
                            ha='center', va='bottom',
                            fontsize=9, fontweight='bold', color=color,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor=color))
                elif mean_val > y_max:
                    clipped_mean = y_max
                    clipped_means.append(clipped_mean)
                    clipped_stds.append(0)
                    ax1.text(self.x_pos[i], clipped_mean - 5,
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
            
            ax1.errorbar(self.x_pos, clipped_means, yerr=clipped_stds,
                        label=label, marker=marker, capsize=3,
                        color=color, linestyle=linestyle,
                        markersize=markersize, linewidth=linewidth,
                        markerfacecolor=face_color,
                        markeredgecolor=edge_color,
                        markeredgewidth=edge_width)
        
        ax1.set_title('Batch Effect Change Metrics\n(↑ = positive better, ↓ = negative better)\nChange% = (after-before)/before×100',
                     fontweight='bold', fontsize=16)
        ax1.set_ylabel('Percentage Change (%)', fontsize=14)
        ax1.set_xlabel('Parameter Combinations', fontsize=14)
        ax1.set_ylim(-160, 30)
        ax1.set_xticks(self.x_pos)
        ax1.set_xticklabels(self.combo_labels, rotation=45, ha='right', fontsize=12)
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Change metrics plot saved to: {save_path}")
        plt.show()
    
    def _plot_bio_preservation(self, save_path):
        """Internal: Plot biological signal preservation"""
        BIO_VAR = 'biological_variability'
        CONSERVED = 'conserved_differential'
        
        if not (BIO_VAR in self.batch_vs_clean and BIO_VAR in self.corrected_vs_clean and
                CONSERVED in self.batch_vs_clean and CONSERVED in self.corrected_vs_clean):
            return
        
        fig2, (ax2_top, ax2_bottom) = plt.subplots(2, 1, figsize=(12, 14))
        
        # Extract data for biological variability
        corrected_biovar_means, corrected_biovar_stds = self._extract_metric_values(self.corrected_vs_clean[BIO_VAR])
        batch_biovar_means, batch_biovar_stds = self._extract_metric_values(self.batch_vs_clean[BIO_VAR])
        
        # Extract data for conserved differential
        corrected_conserved_means, corrected_conserved_stds = self._extract_metric_values(self.corrected_vs_clean[CONSERVED])
        batch_conserved_means, batch_conserved_stds = self._extract_metric_values(self.batch_vs_clean[CONSERVED])
        
        # Top subplot: Biological Variability
        improvement_biovar = np.array(corrected_biovar_means) - np.array(batch_biovar_means)
        ax2_top.fill_between(self.x_pos, batch_biovar_means, corrected_biovar_means,
                            where=(improvement_biovar > 0),
                            color='green', alpha=0.2, interpolate=True,
                            label='Improvement (↑ signal recovery)')
        ax2_top.fill_between(self.x_pos, batch_biovar_means, corrected_biovar_means,
                            where=(improvement_biovar <= 0),
                            color='red', alpha=0.2, interpolate=True,
                            label='Degradation (↓ over-correction)')
        
        ax2_top.errorbar(self.x_pos, corrected_biovar_means, yerr=corrected_biovar_stds,
                    label='Y_corrected vs Y_clean',
                    marker='o', capsize=3, color='#2E7D32', linestyle='-',
                    markersize=8, linewidth=2, markerfacecolor='#2E7D32',
                    markeredgecolor='#2E7D32', markeredgewidth=1, zorder=3)
        
        ax2_top.errorbar(self.x_pos, batch_biovar_means, yerr=batch_biovar_stds,
                    label='Y_with_batch vs Y_clean',
                    marker='o', capsize=3, color='#FF6347', linestyle='--',
                    markersize=8, linewidth=2, markerfacecolor='#FF6347',
                    markeredgecolor='#FF6347', markeredgewidth=1, zorder=3)
        
        ax2_top.set_title('Biological Variability Preservation', fontweight='bold', fontsize=16)
        ax2_top.set_ylabel('Preservation Rate (%)', fontsize=14)
        ax2_top.set_xticks(self.x_pos)
        ax2_top.set_xticklabels(self.combo_labels, rotation=45, ha='right', fontsize=12)
        ax2_top.legend(loc='best', fontsize=11)
        ax2_top.grid(True, alpha=0.3)
        
        # Bottom subplot: Conserved Differential
        improvement_conserved = np.array(corrected_conserved_means) - np.array(batch_conserved_means)
        ax2_bottom.fill_between(self.x_pos, batch_conserved_means, corrected_conserved_means,
                               where=(improvement_conserved > 0),
                               color='green', alpha=0.2, interpolate=True,
                               label='Improvement (↑ signal recovery)')
        ax2_bottom.fill_between(self.x_pos, batch_conserved_means, corrected_conserved_means,
                               where=(improvement_conserved <= 0),
                               color='red', alpha=0.2, interpolate=True,
                               label='Degradation (↓ over-correction)')
        
        ax2_bottom.errorbar(self.x_pos, corrected_conserved_means, yerr=corrected_conserved_stds,
                    label='Y_corrected vs Y_clean',
                    marker='s', capsize=3, color='#2E7D32', linestyle='-',
                    markersize=8, linewidth=2, markerfacecolor='#2E7D32',
                    markeredgecolor='#2E7D32', markeredgewidth=1, zorder=3)
        
        ax2_bottom.errorbar(self.x_pos, batch_conserved_means, yerr=batch_conserved_stds,
                    label='Y_with_batch vs Y_clean',
                    marker='s', capsize=3, color='#FF6347', linestyle='--',
                    markersize=8, linewidth=2, markerfacecolor='#FF6347',
                    markeredgecolor='#FF6347', markeredgewidth=1, zorder=3)
        
        ax2_bottom.set_title('Conserved Differential Proportion', fontweight='bold', fontsize=16)
        ax2_bottom.set_ylabel('Preservation Rate (%)', fontsize=14)
        ax2_bottom.set_xlabel('Parameter Combinations', fontsize=14)
        ax2_bottom.set_xticks(self.x_pos)
        ax2_bottom.set_xticklabels(self.combo_labels, rotation=45, ha='right', fontsize=12)
        ax2_bottom.legend(loc='best', fontsize=11)
        ax2_bottom.grid(True, alpha=0.3)
        
        # Set consistent y-axis limits
        all_means = (corrected_biovar_means + batch_biovar_means +
                    corrected_conserved_means + batch_conserved_means)
        y_min = max(0, min(all_means) - 5)
        y_max = min(100, max(all_means) + 5)
        ax2_top.set_ylim(y_min, y_max)
        ax2_bottom.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Biological signal preservation plot saved to: {save_path}")
        plt.show()
    
    def _plot_tp_fp_rates(self, save_path):
        """Internal: Plot TP/FP rates"""
        fig, (ax_tp, ax_fp) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Extract data
        tp_batch = self._extract_metric_values(self.diff_absolute_metrics.get('tp_batch_rate', {}))
        tp_corrected = self._extract_metric_values(self.diff_absolute_metrics.get('tp_corrected_rate', {}))
        fp_batch = self._extract_metric_values(self.diff_absolute_metrics.get('fp_batch_rate', {}))
        fp_corrected = self._extract_metric_values(self.diff_absolute_metrics.get('fp_corrected_rate', {}))
        
        # Get total glycans estimate (shared for both subplots)
        total_glycans_str = self._get_total_glycans_str()
        
        # TP subplot
        tp_config = {
            'title': f'Differential Expression: True Positive Rates\n(Correctly identified / Ground Truth × 100{total_glycans_str})',
            'ylabel': 'TP Rate (%)',
            'ylim': (0, 100),
            'improvement_direction': 'higher',
            'legend_loc': 'lower left',
            'reference_line': 100
        }
        self._plot_comparison_subplot(ax_tp, tp_batch, tp_corrected, tp_config)
        
        # FP subplot
        fp_config = {
            'title': f'Differential Expression: False Positive Rates\n(False positives / Total glycans × 100{total_glycans_str})',
            'ylabel': 'FP Rate (%)',
            'ylim': (0, 100),
            'improvement_direction': 'lower',
            'legend_loc': 'upper left',
            'reference_line': None
        }
        self._plot_comparison_subplot(ax_fp, fp_batch, fp_corrected, fp_config)
        ax_fp.set_xlabel('Parameter Combinations', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Differential expression TP/FP rates plot saved to: {save_path}")
        plt.show()
    
    def _plot_comparison_subplot(self, ax, batch_data, corrected_data, config):
        """Plot a single comparison subplot with batch vs corrected data"""
        batch_means, batch_stds = batch_data
        corrected_means, corrected_stds = corrected_data
        
        # Plot batch data
        ax.errorbar(self.x_pos, batch_means, yerr=batch_stds,
                   label='Y_with_batch', marker='o', capsize=3,
                   color='#FF6347', linestyle='-', markersize=8, linewidth=2,
                   markerfacecolor='#FF6347', markeredgecolor='#FF6347', markeredgewidth=1)
        
        # Plot corrected data
        ax.errorbar(self.x_pos, corrected_means, yerr=corrected_stds,
                   label='Y_corrected', marker='o', capsize=3,
                   color='#4169E1', linestyle='-', markersize=8, linewidth=2,
                   markerfacecolor='#4169E1', markeredgecolor='#4169E1', markeredgewidth=1)
        
        # Fill between for improvement/degradation
        batch_arr = np.array(batch_means)
        corrected_arr = np.array(corrected_means)
        
        if config['improvement_direction'] == 'higher':
            improvement_condition = (corrected_arr >= batch_arr)
        else:  # 'lower'
            improvement_condition = (corrected_arr <= batch_arr)
        
        ax.fill_between(self.x_pos, batch_arr, corrected_arr,
                       where=improvement_condition,
                       color='green', alpha=0.2, interpolate=True, label='Improvement')
        ax.fill_between(self.x_pos, batch_arr, corrected_arr,
                       where=~improvement_condition,
                       color='red', alpha=0.2, interpolate=True, label='Degradation')
        
        # Set labels and styling
        ax.set_title(config['title'], fontweight='bold', fontsize=15)
        ax.set_ylabel(config['ylabel'], fontsize=14, fontweight='bold')
        ax.set_ylim(config['ylim'])
        ax.set_xticks(self.x_pos)
        ax.set_xticklabels(self.combo_labels, rotation=45, ha='right', fontsize=11)
        ax.legend(loc=config['legend_loc'], fontsize=11)
        ax.grid(True, alpha=0.3, zorder=0)
        
        # Add reference line if specified
        if config.get('reference_line') is not None:
            ax.axhline(y=config['reference_line'], color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    def _get_total_glycans_str(self):
        """Extract total glycans estimate for subplot titles"""
        for combo_data in self.all_results.values():
            if (combo_data[self.param1_name] == self.param1_values[0] and
                combo_data[self.param2_name] == self.param2_values[0]):
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
                        return f", Total glycans: {max(all_glycans)}"
                    break
                break
        return ""
    
    def _plot_diff_expr_metric(self, metric_type, save_path):
        """Internal: Plot single differential expression metric (F1/Precision/Recall)"""
        metric_config = {
            'f1_score': {
                'batch_key': 'batch_f1_score',
                'corrected_key': 'corrected_f1_score',
                'title': 'F1 Score: Differential Expression Detection Quality\n(Harmonic mean of Precision and Recall)',
                'ylabel': 'F1 Score',
                'ylim': (0, 1)
            },
            'precision': {
                'batch_key': 'batch_precision',
                'corrected_key': 'corrected_precision',
                'title': 'Precision: Correctness of Detected Signals\n(TP / (TP + FP))',
                'ylabel': 'Precision',
                'ylim': (0, 1)
            },
            'recall': {
                'batch_key': 'batch_recall',
                'corrected_key': 'corrected_recall',
                'title': 'Recall (Sensitivity): Coverage of True Signals\n(TP / (TP + FN))',
                'ylabel': 'Recall',
                'ylim': (0, 1)
            }
        }
        
        config = metric_config[metric_type]
        
        # Check if metrics exist
        if config['batch_key'] not in self.diff_metrics or config['corrected_key'] not in self.diff_metrics:
            return
        
        # Extract data
        batch_means, batch_stds = self._extract_metric_values(self.diff_metrics[config['batch_key']])
        corrected_means, corrected_stds = self._extract_metric_values(self.diff_metrics[config['corrected_key']])
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot batch data
        ax.errorbar(self.x_pos, batch_means, yerr=batch_stds,
                   label='Y_with_batch', marker='o', capsize=3,
                   color='#FF6347', linestyle='-', markersize=8, linewidth=2,
                   markerfacecolor='#FF6347', markeredgecolor='#FF6347', markeredgewidth=1)
        
        # Plot corrected data
        ax.errorbar(self.x_pos, corrected_means, yerr=corrected_stds,
                   label='Y_corrected', marker='o', capsize=3,
                   color='#4169E1', linestyle='-', markersize=8, linewidth=2,
                   markerfacecolor='#4169E1', markeredgecolor='#4169E1', markeredgewidth=1)
        
        # Fill between for improvement/degradation
        batch_arr = np.array(batch_means)
        corrected_arr = np.array(corrected_means)
        ax.fill_between(self.x_pos, batch_arr, corrected_arr,
                       where=(corrected_arr >= batch_arr),
                       color='green', alpha=0.2, interpolate=True, label='Improvement')
        ax.fill_between(self.x_pos, batch_arr, corrected_arr,
                       where=(corrected_arr < batch_arr),
                       color='red', alpha=0.2, interpolate=True, label='Degradation')
        
        # Set labels and styling
        ax.set_title(config['title'], fontweight='bold', fontsize=15)
        ax.set_ylabel(config['ylabel'], fontsize=14, fontweight='bold')
        ax.set_xlabel('Parameter Combinations', fontsize=14, fontweight='bold')
        ax.set_ylim(config['ylim'])
        ax.set_xticks(self.x_pos)
        ax.set_xticklabels(self.combo_labels, rotation=45, ha='right', fontsize=11)
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(True, alpha=0.3, zorder=0)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"{metric_type.capitalize()} plot saved to: {save_path}")
        plt.show()



class MaskComparisonPlotter:
    """Cross-mask comparison visualization class"""
    
    def __init__(self, mask_dirs, verbose=True):
        """
        Initialize the plotter
        
        Parameters:
        - mask_dirs: Dictionary mapping mask names to result directories
        - verbose: Whether to print detailed information
        """
        self.mask_dirs = mask_dirs
        self.verbose = verbose
        self.batch_data = None
        self.de_data = None
        
        # Load all data at initialization
        self._load_batch_data()
        self._load_de_data()
    
    def plot_all(self, save_path=None):
        """Plot all comparisons (batch correction + differential expression)"""
        plots = []
        plots.extend(self.plot_batch_correction(save_path))
        plots.append(self.plot_differential_expression(save_path))
        
        if self.verbose:
            print(f"Created {len(plots)} comparison plots")
        
        return plots
    
    def plot_batch_correction(self, save_path=None):
        """Plot batch correction comparison across masks"""
        if not self.batch_data:
            print("No batch correction data available")
            return []
        
        if self.verbose:
            print("=" * 60)
            print("BATCH CORRECTION COMPARISON")
            print("=" * 60)
        
        # Define metrics
        change_metrics = ['silhouette', 'kBET', 'LISI', 'ARI', 'compositional_effect_size', 'pca_batch_effect']
        preservation_metrics = ['biological_variability_preservation', 'conserved_differential_proportion']
        
        metric_labels = {
            'silhouette': 'Silhouette\nScore\n(↓ better)',
            'kBET': 'kBET\nScore\n(↓ better)',
            'LISI': 'LISI\nScore\n(↑ better)',
            'ARI': 'Adjusted Rand\nIndex\n(↓ better)',
            'compositional_effect_size': 'Compositional\nEffect Size\n(↓ better)',
            'pca_batch_effect': 'PCA Batch\nEffect\n(↓ better)',
            'biological_variability_preservation': 'Bio Variability\nPreservation\n(↑ better)',
            'conserved_differential_proportion': 'Conserved Differential\nProportion\n(↑ better)'
        }
        
        mask_names = list(self.mask_dirs.keys())
        colors = ['#E8E8E8', '#B0C4DE', '#87CEEB', '#4682B4']
        bar_width = 0.2
        
        # Plot 1: Change metrics
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        x_pos = np.arange(len(change_metrics))
        
        for i, mask_name in enumerate(mask_names):
            means = [self.batch_data[mask_name][m]['mean'] for m in change_metrics]
            stds = [self.batch_data[mask_name][m]['std'] for m in change_metrics]
            
            offset = (i - 1.5) * bar_width
            bars = ax1.bar(x_pos + offset, means, bar_width, yerr=stds,
                          label=mask_name, capsize=3, alpha=0.85, 
                          color=colors[i], edgecolor='black')
            
            for bar, mean_val, std_val in zip(bars, means, stds):
                height = bar.get_height()
                label_y = height + (std_val if height >= 0 else -std_val)
                ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                        f'{mean_val:.1f}', ha='center',
                        va='bottom' if height >= 0 else 'top', fontsize=9)
        
        ax1.set_ylabel('Percentage Change (%)', fontsize=14)
        ax1.set_title('Batch Effect Reduction Across Mask Configurations (negative is better)\n', fontsize=15)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([metric_labels[m] for m in change_metrics], fontsize=12)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            path1 = f'{save_path}1_compare_batch_effect_change.png'
            plt.savefig(path1, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Saved: {path1}")
        plt.show()
        
        # Plot 2: Preservation metrics
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        x_pos_pres = np.arange(len(preservation_metrics))
        
        for i, mask_name in enumerate(mask_names):
            means = [self.batch_data[mask_name][m]['mean'] for m in preservation_metrics]
            stds = [self.batch_data[mask_name][m]['std'] for m in preservation_metrics]
            
            offset = (i - 1.5) * bar_width
            bars = ax2.bar(x_pos_pres + offset, means, bar_width, yerr=stds,
                          label=mask_name, capsize=3, alpha=0.85,
                          color=colors[i], edgecolor='black')
            
            for bar, mean_val, std_val in zip(bars, means, stds):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_val + 1,
                        f'{mean_val:.1f}', ha='center', va='bottom', fontsize=10)
        
        ax2.set_ylabel('Preservation Percentage (%)', fontsize=14)
        ax2.set_title('Biological Signal Preservation Across Mask Configurations \n (higher is better)\n', fontsize=15)
        ax2.set_xticks(x_pos_pres)
        ax2.set_xticklabels([metric_labels[m] for m in preservation_metrics], fontsize=12)
        ax2.legend(fontsize=10, loc='lower right')
        ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            path2 = f'{save_path}2_compare_bio_preservation.png'
            plt.savefig(path2, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Saved: {path2}")
        plt.show()
        
        return ['batch_effect_change', 'bio_preservation']
    
    def plot_differential_expression(self, save_path=None):
        """Plot differential expression comparison across masks"""
        if not self.de_data:
            print("No differential expression data available")
            return None
        
        if self.verbose:
            print("=" * 60)
            print("DIFFERENTIAL EXPRESSION COMPARISON")
            print("=" * 60)
        
        metrics = ['true_positive_change_pct', 'false_positive_change_pct',
                   'batch_tp_rate', 'corrected_tp_rate', 'batch_fp_rate', 'corrected_fp_rate']
        
        metric_labels = {
            'true_positive_change_pct': 'TP Change\n(%)',
            'false_positive_change_pct': 'FP Change\n(%)',
            'batch_tp_rate': 'Batch\nTP Rate (%)',
            'corrected_tp_rate': 'Corrected\nTP Rate (%)',
            'batch_fp_rate': 'Batch\nFP Rate (%)',
            'corrected_fp_rate': 'Corrected\nFP Rate (%)'
        }
        
        mask_names = list(self.mask_dirs.keys())
        colors = ['#E8E8E8', '#B0C4DE', '#87CEEB', '#4682B4']
        bar_width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 8))
        x_pos = np.arange(len(metrics))
        
        for i, mask_name in enumerate(mask_names):
            means = [self.de_data[mask_name][m]['mean'] for m in metrics]
            stds = [self.de_data[mask_name][m]['std'] for m in metrics]
            
            offset = (i - 1.5) * bar_width
            bars = ax.bar(x_pos + offset, means, bar_width, yerr=stds,
                         label=mask_name, capsize=3, alpha=0.85,
                         color=colors[i], edgecolor='black')
            
            for bar, mean_val, std_val in zip(bars, means, stds):
                height = bar.get_height()
                label_y = height + (std_val if height >= 0 else -std_val)
                ax.text(bar.get_x() + bar.get_width()/2., label_y,
                       f'{mean_val:.1f}', ha='center',
                       va='bottom' if height >= 0 else 'top', fontsize=9)
        
        ax.set_title('Differential Expression Recovery Across Mask Configurations\n(Batch vs Corrected Comparison)\n', fontsize=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([metric_labels[m] for m in metrics], fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=14)
        ax.legend(fontsize=10, loc='upper left')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Saved: {save_path}")
        plt.show()
        
        return 'differential_expression'
    
    # ==================== Internal Methods ====================
    
    def _load_batch_data(self):
        """Load batch correction data from all masks"""
        all_mask_data = {}
        
        for mask_name, mask_dir in self.mask_dirs.items():
            json_files = glob.glob(f"{mask_dir}/correction_metrics_seed*.json")
            
            if not json_files:
                if self.verbose:
                    print(f"Warning: No JSON files in {mask_dir}")
                continue
            
            if self.verbose:
                print(f"{mask_name}: {len(json_files)} seed files")
            
            all_metrics = {}
            
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                before = data['correction_results']['batch_correction']['before']
                after = data['correction_results']['batch_correction']['after']
                
                changes = calculate_batch_correction_changes(before, after)
                for k, v in changes.items():
                    all_metrics.setdefault(k, []).append(v)
                
                bio_pres = data['correction_results']['bio_preservation']
                for k, v in bio_pres.items():
                    if isinstance(v, (int, float)):
                        all_metrics.setdefault(k, []).append(v * 100)
            
            # Calculate statistics
            summary = {}
            for metric, values in all_metrics.items():
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
            
            all_mask_data[mask_name] = summary
        
        self.batch_data = all_mask_data if all_mask_data else None
    
    def _load_de_data(self):
        """Load differential expression data from all masks"""
        all_mask_de_data = {}
        
        for mask_name, mask_dir in self.mask_dirs.items():
            json_files = glob.glob(f"{mask_dir}/correction_metrics_seed*.json")
            
            if not json_files:
                continue
            
            metrics_dict = {
                'true_positive_change_pct': [], 'false_positive_change_pct': [],
                'batch_tp_rate': [], 'corrected_tp_rate': [],
                'batch_fp_rate': [], 'corrected_fp_rate': []
            }
            
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                diff_expr = data['correction_results']['differential_expression']
                summary = diff_expr['summary']
                
                metrics_dict['true_positive_change_pct'].append(summary['true_positive_change_pct'])
                metrics_dict['false_positive_change_pct'].append(summary['false_positive_change_pct'])
                
                y_clean_count = diff_expr['Y_clean']['significant_count']
                batch_tp = diff_expr['Y_with_batch']['true_positive']['count']
                corrected_tp = diff_expr['Y_after_correction']['true_positive']['count']
                
                if y_clean_count > 0:
                    metrics_dict['batch_tp_rate'].append(batch_tp / y_clean_count * 100)
                    metrics_dict['corrected_tp_rate'].append(corrected_tp / y_clean_count * 100)
                else:
                    metrics_dict['batch_tp_rate'].append(0.0)
                    metrics_dict['corrected_tp_rate'].append(0.0)
                
                all_glycans = set(diff_expr['Y_clean']['significant_glycans']) | \
                             set(diff_expr['Y_with_batch']['significant_glycans']) | \
                             set(diff_expr['Y_corrected']['significant_glycans'])
                total_glycans = max(all_glycans) if all_glycans else 98
                
                batch_fp = diff_expr['Y_with_batch']['false_positive']['count']
                corrected_fp = diff_expr['Y_after_correction']['false_positive']['count']
                
                metrics_dict['batch_fp_rate'].append(batch_fp / total_glycans * 100)
                metrics_dict['corrected_fp_rate'].append(corrected_fp / total_glycans * 100)
            
            # Calculate statistics
            summary = {}
            for metric, values in metrics_dict.items():
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
            
            all_mask_de_data[mask_name] = summary
        
        self.de_data = all_mask_de_data if all_mask_de_data else None

