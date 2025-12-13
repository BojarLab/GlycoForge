import os
import sys
import json
import tempfile
import shutil
import pandas as pd
import numpy as np
import itertools
from copy import deepcopy
from glycoforge import simulate
from glycoforge import invclr
from glycoforge.sim_batch_factor import define_batch_direction
from .methods import combat, add_noise_to_zero_variance_features
from .evaluation import (
    quantify_batch_effect_impact,
    evaluate_biological_preservation,
    compare_differential_expression,
    generate_comprehensive_metrics
)
from glycoforge.utils import plot_pca, load_data_from_glycowork


def get_param(config, key, default=None):
    return config.get(key, default)


def parse_parameter_grid(config):
    grid_keys = [k for k, v in config.items() 
                 if isinstance(v, list) and k not in ['random_seeds', 'affected_fraction']]
    
    if not grid_keys:
        return [{}]
    
    keys = list(grid_keys)
    values = [config[k] for k in keys]
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]


def generate_all_data(config, cache_dir):
    verbose = config.get('verbose', True)
    
    param_combinations = parse_parameter_grid(config)
    seeds_to_run = config.get('random_seeds', [42])
    
    # Determine n_glycans: read from real data if in hybrid mode, otherwise use config
    data_source = get_param(config, 'data_source', 'simulated')
    if data_source == 'real':
        data_file = get_param(config, 'data_file')
        if data_file is None:
            raise ValueError("data_file is required when data_source='real'")
        

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        
        df = load_data_from_glycowork(data_file)
        n_glycans = len(df)  # Number of glycans (rows in CSV)
        if verbose:
            print(f"[Hybrid Mode] Detected {n_glycans} glycans from real data: {data_file}")
    else:
        n_glycans = get_param(config, 'n_glycans', 50)
    
    # Generate global u_dict once for all seeds to ensure consistent batch structure
    # Note: This logic is kept for future extensibility (e.g., if affected_fraction becomes 
    # part of the parameter grid). Currently, simulate() uses a fixed u_dict_seed=42 by default,
    # but generating u_dict here avoids redundant computation across multiple runs and provides
    # explicit control over batch effect structure consistency.
    u_dict_global = define_batch_direction(
        n_glycans=n_glycans,
        n_batches=get_param(config, 'n_batches', 3),
        affected_fraction=get_param(config, 'affected_fraction', (0.05, 0.30)),
        positive_prob=get_param(config, 'positive_prob', 0.6),
        overlap_prob=get_param(config, 'overlap_prob', 0.5),
        verbose=verbose
    )
    
    global_metadata = {
        'u_dict': {k: v.tolist() for k, v in u_dict_global.items()},
        'config': config
    }
    with open(f"{cache_dir}/_metadata.json", 'w') as f:
        json.dump(global_metadata, f, indent=2)
    
    if verbose:
        print("=" * 60)
        print("PHASE 1: DATA GENERATION")
        print("=" * 60)
        print(f"Parameter combinations: {len(param_combinations)}")
        print(f"Seeds per combination: {len(seeds_to_run)}")
        print(f"Total tasks: {len(param_combinations) * len(seeds_to_run)}")
        print("=" * 60)
    
    total_tasks = len(param_combinations) * len(seeds_to_run)
    task_idx = 0
    
    for combo_idx, params in enumerate(param_combinations):
        combo_name = "_".join([f"{k}_{v}" for k, v in params.items()])
        combo_cache_dir = f"{cache_dir}/{combo_name}"
        os.makedirs(combo_cache_dir, exist_ok=True)
        
        if verbose and params:
            print(f"\nProcessing combination {combo_idx + 1}/{len(param_combinations)}: {params}")
        
        for seed in seeds_to_run:
            task_idx += 1
            seed_cache_dir = f"{combo_cache_dir}/seed{seed}"
            os.makedirs(seed_cache_dir, exist_ok=True)
            
            if verbose:
                print(f"  [{task_idx}/{total_tasks}] Generating data: seed={seed}")
            
            simulate_kwargs = {
                'data_source': get_param(config, 'data_source', 'simulated'),
                'data_file': get_param(config, 'data_file'),
                'n_glycans': get_param(config, 'n_glycans', 50),
                'n_H': get_param(config, 'n_H', 15),
                'n_U': get_param(config, 'n_U', 15),
                'bio_strength': params.get('bio_strength', get_param(config, 'bio_strength', 1.5)),
                'k_dir': params.get('k_dir', get_param(config, 'k_dir', 100)),
                'variance_ratio': get_param(config, 'variance_ratio', 1.5),
                'use_real_effect_sizes': get_param(config, 'use_real_effect_sizes', False),
                'differential_mask': get_param(config, 'differential_mask', 'All'),
                'column_prefix': get_param(config, 'column_prefix'),
                'winsorize_percentile': get_param(config, 'winsorize_percentile', None),
                'baseline_method': get_param(config, 'baseline_method', 'median'),
                'n_batches': get_param(config, 'n_batches', 3),
                'affected_fraction': get_param(config, 'affected_fraction', (0.05, 0.30)),
                'positive_prob': get_param(config, 'positive_prob', 0.6),
                'overlap_prob': get_param(config, 'overlap_prob', 0.5),
                'kappa_mu': params.get('kappa_mu', get_param(config, 'kappa_mu', 1.0)),
                'var_b': params.get('var_b', get_param(config, 'var_b', 0.5)),
                'random_seeds': [seed],
                'u_dict': u_dict_global,  # Use shared u_dict for all seeds
                'output_dir': seed_cache_dir,
                'verbose': False,
                'save_csv': True,
                'show_pca_plots': False
            }
            
            simulate(**simulate_kwargs)
    
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 1 COMPLETED")
        print("=" * 60)


def process_corrections(config, cache_dir):
    verbose = config.get('verbose', True)
    save_csv = config.get('save_csv', False)
    show_pca_plots = config.get('show_pca_plots', False)
    base_output_dir = config.get('output_dir', 'results/')
    
    with open(f"{cache_dir}/_metadata.json", 'r') as f:
        global_metadata = json.load(f)
    
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 2: BATCH CORRECTION & EVALUATION")
        print("=" * 60)
    
    all_results = []
    
    combo_dirs = sorted([d for d in os.listdir(cache_dir) 
                        if os.path.isdir(f"{cache_dir}/{d}") and not d.startswith('_')])
    
    for combo_idx, combo_name in enumerate(combo_dirs):
        combo_cache_dir = f"{cache_dir}/{combo_name}"
        combo_output_dir = f"{base_output_dir}/{combo_name}"
        os.makedirs(combo_output_dir, exist_ok=True)
        
        if verbose:
            print(f"\nProcessing combination {combo_idx + 1}/{len(combo_dirs)}: {combo_name}")
        
        seed_dirs = sorted([d for d in os.listdir(combo_cache_dir) 
                           if d.startswith('seed')])
        
        combo_results = []
        
        for seed_idx, seed_dir in enumerate(seed_dirs):
            seed = int(seed_dir.replace('seed', ''))
            seed_cache_path = f"{combo_cache_dir}/{seed_dir}"
            
            if verbose:
                print(f"  [{seed_idx + 1}/{len(seed_dirs)}] Processing seed={seed}")
            
            Y_clean = pd.read_csv(f"{seed_cache_path}/1_Y_clean_seed{seed}.csv", index_col=0)
            Y_clean_clr = pd.read_csv(f"{seed_cache_path}/1_Y_clean_clr_seed{seed}.csv", index_col=0)
            Y_with_batch = pd.read_csv(f"{seed_cache_path}/2_Y_with_batch_seed{seed}.csv", index_col=0)
            Y_with_batch_clr = pd.read_csv(f"{seed_cache_path}/2_Y_with_batch_clr_seed{seed}.csv", index_col=0)
            
            with open(f"{seed_cache_path}/metadata_seed{seed}.json", 'r') as f:
                metadata = json.load(f)
            
            batch_labels = np.array(metadata['sample_info']['batch_labels'])
            bio_labels = np.array(metadata['sample_info']['bio_labels'])
            bio_groups = metadata['sample_info']['bio_groups']
            batch_groups = metadata['sample_info']['batch_groups']
            batch_check_results = metadata.get('quality_checks', {}).get('Y_with_batch', {})
            
            batch_metrics_before = quantify_batch_effect_impact(
                Y_with_batch_clr, 
                batch_labels, 
                bio_groups, 
                verbose=False
            )
            
            Y_with_batch_clr_fixed = add_noise_to_zero_variance_features(
                Y_with_batch_clr, 
                noise_level=1e-10, 
                random_seed=seed
            )
            
            Y_corrected_clr_values = combat(Y_with_batch_clr_fixed.values, batch_labels)
            Y_corrected_clr = pd.DataFrame(
                Y_corrected_clr_values, 
                index=Y_with_batch_clr_fixed.index, 
                columns=Y_with_batch_clr_fixed.columns
            )
            
            Y_corrected = pd.DataFrame(index=Y_corrected_clr.index, columns=Y_corrected_clr.columns)
            for sample in Y_corrected_clr.columns:
                Y_corrected[sample] = invclr(Y_corrected_clr[sample].values)
            
            batch_metrics_after = quantify_batch_effect_impact(
                Y_corrected_clr, 
                batch_labels, 
                bio_groups, 
                verbose=False
            )
            
            bio_preservation_metrics = evaluate_biological_preservation(
                clean_data=Y_clean_clr,
                corrected_data=Y_corrected_clr,
                bio_labels=bio_labels
            )
            batch_metrics_after.update(bio_preservation_metrics)
            
            diff_expr_results = compare_differential_expression(
                dataset1=Y_clean,
                dataset2=Y_with_batch,
                dataset3=Y_corrected,
                verbose=False
            )
            
            if save_csv:
                Y_clean.to_csv(f"{combo_output_dir}/1_Y_clean_seed{seed}.csv", float_format="%.16f")
                Y_clean_clr.to_csv(f"{combo_output_dir}/1_Y_clean_clr_seed{seed}.csv", float_format="%.16f")
                Y_with_batch.to_csv(f"{combo_output_dir}/2_Y_with_batch_seed{seed}.csv", float_format="%.16f")
                Y_with_batch_clr.to_csv(f"{combo_output_dir}/2_Y_with_batch_clr_seed{seed}.csv", float_format="%.16f")
                Y_corrected.to_csv(f"{combo_output_dir}/3_Y_after_combat_seed{seed}.csv", float_format="%.16f")
                Y_corrected_clr.to_csv(f"{combo_output_dir}/3_Y_after_combat_clr_seed{seed}.csv", float_format="%.16f")
            
            # Save metadata from Phase 1 to final output directory
            metadata_output_path = f"{combo_output_dir}/metadata_seed{seed}.json"
            with open(metadata_output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Build run_config from new metadata structure
            run_config = {
                'seed': seed,
                'data_source': metadata.get('data_source', 'simulated')
            }
            
            # Add data_file for hybrid mode
            if 'data_file' in metadata:
                run_config['data_file'] = metadata['data_file']
            if 'use_real_effect_sizes' in metadata:
                run_config['use_real_effect_sizes'] = metadata['use_real_effect_sizes']
            
            # Merge bio_parameters and batch_parameters
            if 'bio_parameters' in metadata:
                run_config.update(metadata['bio_parameters'])
            if 'batch_parameters' in metadata:
                run_config.update(metadata['batch_parameters'])
            
            generate_comprehensive_metrics(
                seed=seed,
                output_dir=combo_output_dir,
                batch_metrics_before=batch_metrics_before,
                batch_metrics_after=batch_metrics_after,
                diff_expr_results=diff_expr_results,
                run_config={'key_parameters': run_config},
                batch_check_results=batch_check_results
            )
            
            if show_pca_plots:
                plot_pca(Y_clean_clr, bio_groups=bio_groups, 
                        title=f"Seed {seed}: Clean Data")
                plot_pca(Y_with_batch_clr, bio_groups=bio_groups, batch_groups=batch_groups,
                        title=f"Seed {seed}: With Batch Effects")
                plot_pca(Y_corrected_clr, bio_groups=bio_groups, batch_groups=batch_groups,
                        title=f"Seed {seed}: After ComBat")
            
            combo_results.append({
                'seed': seed,
                'metrics_before': batch_metrics_before,
                'metrics_after': batch_metrics_after,
                'diff_expr': diff_expr_results
            })
        
        all_results.append({
            'combo': combo_name,
            'results': combo_results
        })
    
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 2 COMPLETED")
        print("=" * 60)
    
    return all_results


def cleanup_cache(cache_dir):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)


def run_correction(config):
    verbose = config.get('verbose', True)
    
    DEBUG_MODE = os.environ.get('GLYCOFORGE_DEBUG', '0') == '1'
    KEEP_CACHE = os.environ.get('GLYCOFORGE_KEEP_CACHE', '0') == '1'
    
    if DEBUG_MODE:
        cache_dir = 'debug_cache/'
        os.makedirs(cache_dir, exist_ok=True)
        if verbose:
            print(f"[DEBUG MODE] Using cache: {cache_dir}")
    else:
        cache_dir = tempfile.mkdtemp(prefix='glycoforge_cache_')
        if verbose:
            print(f"[SYSTEM] Using temp cache: {cache_dir}")
    
    try:
        generate_all_data(config, cache_dir)
        
        results = process_corrections(config, cache_dir)
        
        if verbose:
            print("\n" + "=" * 60)
            print("BATCH CORRECTION PIPELINE COMPLETED")
            print(f"Results saved to: {config.get('output_dir', 'results/')}")
            print("=" * 60)
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"\n[ERROR] Pipeline failed")
            if DEBUG_MODE or KEEP_CACHE:
                print(f"[DEBUG] Cache preserved at: {cache_dir}")
        raise e
        
    finally:
        if not (DEBUG_MODE and KEEP_CACHE):
            cleanup_cache(cache_dir)
            if verbose:
                print("[SYSTEM] Cache cleaned")
