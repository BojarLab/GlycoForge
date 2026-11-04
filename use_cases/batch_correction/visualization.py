
import glob
import json
import numpy as np
import os
import viscose as vc
from collections import defaultdict

# Aura unless the user chose another palette with viscose.set_palette, before or after importing this
vc.default_palette('aura')


def _finish(fig, save_path, note=None, show=True):
    """Save a figure if given a path, then show it; only an unsaved figure may open the system's image viewer"""
    if save_path:
        fig.save(save_path)
        if note:
            print(note)
    if show:
        fig.show(viewer=not save_path)


def _trace(ax, x, means, stds, color, label, marker='o', dash=None):
    """A line through its means, with error bars"""
    ax.errorbar(x, means, stds, color=color)
    ax.plot(x, means, color=color, lw=2, label=label, marker=marker, dash=dash)


def _shade(ax, x, batch, corrected, higher_better=True, good_label='Improvement', bad_label='Degradation'):
    """Shade the gap between two curves in one hue where correction helped and in the other where it hurt"""
    # Curves can cross between two positions, so the gap is followed on a fine grid, or the shading would leave the curves there
    grid = np.linspace(x[0], x[-1], 50 * len(x))
    batch, corrected = np.interp(grid, x, batch), np.interp(grid, x, corrected)
    above, below = np.maximum(batch, corrected), np.minimum(batch, corrected)
    ax.fill_between(grid, batch, above if higher_better else below, color='b2', alpha=0.35, label=good_label)
    ax.fill_between(grid, batch, below if higher_better else above, color='a2', alpha=0.35, label=bad_label)


def _value_labels(ax, x, means, stds, fmt, size, bold=False):
    """Write each bar's value just past its error bar"""
    for xi, mean, std in zip(x, means, stds):
        ax.text(xi, mean + (std if mean >= 0 else -std), fmt.format(mean), size=size, bold=bold, ha='center', va='bottom' if mean >= 0 else 'top', halo=2.6)


def _verdict_bars(ax, means, stds, good, labels, good_label, bad_label, fmt):
    """Bars in one hue where a metric did well and in the other where it did not"""
    x, means, stds, good = np.arange(len(means)), np.array(means), np.array(stds), np.array(good)
    ax.bar(x[good], means[good], color='b1', label=good_label)
    ax.bar(x[~good], means[~good], color='a1', label=bad_label)
    ax.errorbar(x, means, stds)
    _value_labels(ax, x, means, stds, fmt, 11, bold=True)
    ax.set_xticks(x, labels, rotation=45)
    ax.legend()
    ax.grid()


def _mask_bars(ax, stats, metrics, labels):
    """One group of bars per metric, one bar per mask configuration"""
    x = np.arange(len(metrics))
    width = 0.8 / len(stats)
    for i, (color, (mask_name, data)) in enumerate(zip(ax.pal.colors(len(stats)), stats.items())):
        means, stds = (np.array([data[m][k] for m in metrics]) for k in ('mean', 'std'))
        positions = x + (i - (len(stats) - 1) / 2) * width
        ax.bar(positions, means, width, color=color, label=mask_name)
        ax.errorbar(positions, means, stds)
        _value_labels(ax, positions, means, stds, '{:.1f}', 9)
    ax.set_xticks(x, [labels[m] for m in metrics])
    ax.legend()
    ax.grid()


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
            fig, (ax1, ax2) = vc.subplots(1, 2, figsize=(16, 8), width_ratios=[3, 1])
        elif len(change_metrics) > 0:
            # Only change metrics
            fig, ax1 = vc.subplots(1, 1, figsize=(12, 8))
            ax2 = None
        elif len(ratio_metrics) > 0:
            # Only ratio metrics
            fig, ax2 = vc.subplots(1, 1, figsize=(6, 8))
            ax1 = None
        else:
            print("No valid metrics found for plotting!")
            return
        
        # Plot 1: Change metrics (percentage change)
        if change_metrics and ax1 is not None:
            change_labels = [f"{metric_info[m]['name']}\n({'↓' if metric_info[m]['better'] == 'Lower' else '↑'} better)" for m in change_metrics]
            change_means = [summary_stats[m]['mean'] for m in change_metrics]
            change_stds = [summary_stats[m]['std'] for m in change_metrics]

            change_good = [mean < 0 if metric_info[m]['better'] == 'Lower' else mean > 0 for m, mean in zip(change_metrics, change_means)]
            _verdict_bars(ax1, change_means, change_stds, change_good, change_labels, 'Improvement', 'Worsening', '{:+.1f}%')
            ax1.set_ylabel('Percentage Change (%)')
            ax1.set_title(f'Batch Effect Reduction\n (after - before)/before * 100 \n(Mean ± SD across {n_runs} runs)')
            ax1.axhline(y=0, color='ink', dash=None)

        # Plot 2: Ratio metrics (preservation percentages)
        if ratio_metrics and ax2 is not None:
            ratio_labels = [f"{metric_info[m]['name']}\n({'↑' if metric_info[m]['better'] == 'Higher' else '↓'} better)" for m in ratio_metrics]
            ratio_means = [summary_stats[m]['mean'] for m in ratio_metrics]
            ratio_stds = [summary_stats[m]['std'] for m in ratio_metrics]

            _verdict_bars(ax2, ratio_means, ratio_stds, [mean > 50 for mean in ratio_means], ratio_labels, 'Good (>50%)', 'Poor (<=50%)', '{:.1f}%')
            ax2.set_title(f'Biological Signal Preservation\n after/before * 100 \n(Mean ± SD across {n_runs} runs)')
            ax2.axhline(y=100, color='ink', dash=(4, 3), label='Perfect Preservation')
            ax2.axhline(y=50, color='muted', label='Threshold (50%)')

        _finish(fig, save_path, f"Plot saved to: {save_path}", show=self.verbose)

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

        fig1, ax1 = vc.subplots(1, 1, figsize=(12, 8))
        colors = dict(zip(change_metric_names, fig1.pal.colors(len(change_metric_names))))
        markers = dict(zip(change_metric_names, ['o', 'D', 's', '^', 'P', 'v']))

        for idx, (metric_name, metric_data) in enumerate(change_metrics.items()):
            means, stds = self._extract_metric_values(metric_data)
            
            direction = "↑" if metric_info[metric_name]['better'] == 'Higher' else "↓"
            label = f"{metric_info[metric_name]['name']} {direction}"

            color = colors[metric_name]
            
            clipped_means = []
            clipped_stds = []
            y_min, y_max = -150, 20
            
            for i, (mean_val, std_val) in enumerate(zip(means, stds)):
                if mean_val < y_min:
                    clipped_mean = y_min
                    clipped_means.append(clipped_mean)
                    clipped_stds.append(0)
                    ax1.text(self.x_pos[i], clipped_mean + 5, f'{mean_val:.0f}±{std_val:.0f}', size=9, bold=True, color=color, ha='center', va='bottom', halo=2.6)
                elif mean_val > y_max:
                    clipped_mean = y_max
                    clipped_means.append(clipped_mean)
                    clipped_stds.append(0)
                    ax1.text(self.x_pos[i], clipped_mean - 5, f'{mean_val:.0f}±{std_val:.0f}', size=9, bold=True, color=color, ha='center', va='top', halo=2.6)
                else:
                    error_top = mean_val + std_val
                    error_bottom = mean_val - std_val
                    
                    if error_top > y_max or error_bottom < y_min:
                        clipped_std = min(std_val, abs(y_max - mean_val), abs(mean_val - y_min))
                        clipped_stds.append(clipped_std)
                    else:
                        clipped_stds.append(std_val)
                    
                    clipped_means.append(mean_val)

            _trace(ax1, self.x_pos, clipped_means, clipped_stds, color, label, markers[metric_name])

        ax1.set_title('Batch Effect Change Metrics\n(↑ = positive better, ↓ = negative better)\nChange% = (after-before)/before×100')
        ax1.set_ylabel('Percentage Change (%)')
        ax1.set_xlabel('Parameter Combinations')
        ax1.set_ylim(-160, 30)
        ax1.set_xticks(self.x_pos, self.combo_labels, rotation=45)
        ax1.legend()
        ax1.grid()
        ax1.axhline(y=0, color='ink', dash=None)

        _finish(fig1, save_path, f"Change metrics plot saved to: {save_path}" if self.verbose else None)

    def _plot_bio_preservation(self, save_path):
        """Internal: Plot biological signal preservation"""
        BIO_VAR = 'biological_variability'
        CONSERVED = 'conserved_differential'
        
        if not (BIO_VAR in self.batch_vs_clean and BIO_VAR in self.corrected_vs_clean and
                CONSERVED in self.batch_vs_clean and CONSERVED in self.corrected_vs_clean):
            return

        fig2, axes = vc.subplots(2, 1, figsize=(12, 14))
        all_means = []
        for ax, metric, title, marker in zip(axes, [BIO_VAR, CONSERVED], ['Biological Variability Preservation', 'Conserved Differential Proportion'], ['o', 's']):
            corrected_means, corrected_stds = self._extract_metric_values(self.corrected_vs_clean[metric])
            batch_means, batch_stds = self._extract_metric_values(self.batch_vs_clean[metric])
            _shade(ax, self.x_pos, batch_means, corrected_means, good_label='Improvement (↑ signal recovery)', bad_label='Degradation (↓ over-correction)')
            _trace(ax, self.x_pos, corrected_means, corrected_stds, 'b1', 'Y_corrected vs Y_clean', marker)
            _trace(ax, self.x_pos, batch_means, batch_stds, 'a1', 'Y_with_batch vs Y_clean', marker, dash=(4, 3))
            ax.set_title(title)
            ax.set_ylabel('Preservation Rate (%)')
            ax.set_xticks(self.x_pos, self.combo_labels, rotation=45)
            ax.legend()
            ax.grid()
            all_means += list(corrected_means) + list(batch_means)
        axes[1].set_xlabel('Parameter Combinations')

        # Set consistent y-axis limits
        y_min = max(0, min(all_means) - 5)
        y_max = min(100, max(all_means) + 5)
        for ax in axes:
            ax.set_ylim(y_min, y_max)

        _finish(fig2, save_path, f"Biological signal preservation plot saved to: {save_path}" if self.verbose else None)

    def _plot_tp_fp_rates(self, save_path):
        """Internal: Plot TP/FP rates"""
        fig, (ax_tp, ax_fp) = vc.subplots(2, 1, figsize=(12, 12))
        
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
            'reference_line': 100
        }
        self._plot_comparison_subplot(ax_tp, tp_batch, tp_corrected, tp_config)
        
        # FP subplot
        fp_config = {
            'title': f'Differential Expression: False Positive Rates\n(False positives / Total glycans × 100{total_glycans_str})',
            'ylabel': 'FP Rate (%)',
            'ylim': (0, 100),
            'improvement_direction': 'lower',
            'reference_line': None
        }
        self._plot_comparison_subplot(ax_fp, fp_batch, fp_corrected, fp_config)
        ax_fp.set_xlabel('Parameter Combinations')

        _finish(fig, save_path, f"Differential expression TP/FP rates plot saved to: {save_path}" if self.verbose else None)

    def _plot_comparison_subplot(self, ax, batch_data, corrected_data, config):
        """Plot a single comparison subplot with batch vs corrected data"""
        batch_means, batch_stds = batch_data
        corrected_means, corrected_stds = corrected_data

        _trace(ax, self.x_pos, batch_means, batch_stds, 'a1', 'Y_with_batch')
        _trace(ax, self.x_pos, corrected_means, corrected_stds, 'b1', 'Y_corrected')
        _shade(ax, self.x_pos, batch_means, corrected_means, higher_better=config['improvement_direction'] == 'higher')

        # Set labels and styling
        ax.set_title(config['title'])
        ax.set_ylabel(config['ylabel'])
        ax.set_ylim(*config['ylim'])
        ax.set_xticks(self.x_pos, self.combo_labels, rotation=45)
        ax.legend()
        ax.grid()

        # Add reference line if specified
        if config.get('reference_line') is not None:
            ax.axhline(y=config['reference_line'], dash=(4, 3))

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
        fig, ax = vc.subplots(1, 1, figsize=(12, 8))
        self._plot_comparison_subplot(ax, (batch_means, batch_stds), (corrected_means, corrected_stds), dict(config, improvement_direction='higher'))
        ax.set_xlabel('Parameter Combinations')

        _finish(fig, save_path, f"{metric_type.capitalize()} plot saved to: {save_path}" if self.verbose else None)



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

        stats = {mask_name: self.batch_data[mask_name] for mask_name in self.mask_dirs}

        # Plot 1: Change metrics
        fig1, ax1 = vc.subplots(figsize=(14, 8))
        _mask_bars(ax1, stats, change_metrics, metric_labels)
        ax1.set_ylabel('Percentage Change (%)')
        ax1.set_title('Batch Effect Reduction Across Mask Configurations (negative is better)')
        ax1.axhline(y=0, color='ink', dash=None)
        path1 = f'{save_path}1_compare_batch_effect_change.png' if save_path else None
        _finish(fig1, path1, f"Saved: {path1}" if self.verbose else None)

        # Plot 2: Preservation metrics
        fig2, ax2 = vc.subplots(figsize=(14, 8))
        _mask_bars(ax2, stats, preservation_metrics, metric_labels)
        ax2.set_ylabel('Preservation Percentage (%)')
        ax2.set_title('Biological Signal Preservation Across Mask Configurations\n(higher is better)')
        ax2.axhline(y=100, color='ink', dash=(4, 3))
        ax2.axhline(y=50, color='muted')
        path2 = f'{save_path}2_compare_bio_preservation.png' if save_path else None
        _finish(fig2, path2, f"Saved: {path2}" if self.verbose else None)

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

        fig, ax = vc.subplots(figsize=(14, 8))
        _mask_bars(ax, {mask_name: self.de_data[mask_name] for mask_name in self.mask_dirs}, metrics, metric_labels)
        ax.set_title('Differential Expression Recovery Across Mask Configurations\n(Batch vs Corrected Comparison)')
        ax.set_ylabel('Percentage (%)')
        ax.axhline(y=0, color='ink', dash=None)

        _finish(fig, save_path, f"Saved: {save_path}" if self.verbose else None)

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

