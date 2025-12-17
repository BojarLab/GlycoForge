import os
import json
import pandas as pd
import numpy as np
from glycoforge import simulate, invclr,plot_pca
from .methods import combat, add_noise_to_zero_variance_features
from .evaluation import (
    quantify_batch_effect_impact,
    evaluate_biological_preservation,
    compare_differential_expression,
    generate_comprehensive_metrics
)


def generate_all_data(config):
    """
    Phase 1: Data generation using glycoforge.simulate()
    
    Delegates all data generation to glycoforge.simulate(), which handles:
    - Grid search for list parameters (kappa_mu, var_b, bio_strength, etc.)
    - Multiple random seeds
    - Directory structure creation
    - Saving CSV files and metadata.json to disk
    
    Returns:
        Result object from simulate() (data is already saved to disk)
    """
    verbose = config.get('verbose', True)
    output_dir = config.get('output_dir', 'results/')
    
    if verbose:
        print("=" * 60)
        print("PHASE 1: DATA GENERATION")
        print("Calling glycoforge.simulate() for grid search and data generation")
        print(f"Output directory: {output_dir}")
        print("=" * 60)
    
    # Single call handles all grid combinations and seeds
    result = simulate(**config)
    
    if verbose:
        if isinstance(result, dict) and 'metadata' not in result:
            print(f"\nGrid search completed: {len(result)} parameter combinations")
        else:
            print("\nSingle run completed")
        print(f"All data saved to: {output_dir}")
        print("=" * 60)
        print("PHASE 1 COMPLETED")
        print("=" * 60)
    
    return result


def process_corrections(config):
    """
    Phase 2: Batch correction and evaluation
    
    Reads data generated in Phase 1 from disk, applies ComBat correction,
    and evaluates correction effectiveness.
    """
    verbose = config.get('verbose', True)
    save_csv = config.get('save_csv', False)
    show_pca_plots = config.get('show_pca_plots', False)
    output_dir = config.get('output_dir', 'results/')
    
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 2: BATCH CORRECTION & EVALUATION")
        print("Reading data from Phase 1 and applying ComBat correction")
        print("=" * 60)
    
    all_results = []
    
    # Discover all parameter combination directories created by simulate()
    combo_dirs = sorted([d for d in os.listdir(output_dir) 
                        if os.path.isdir(f"{output_dir}/{d}")])
    
    # Handle single run (no grid search subdirectories)
    if not combo_dirs:
        combo_dirs = ['.']
    
    for combo_idx, combo_name in enumerate(combo_dirs):
        combo_dir = f"{output_dir}/{combo_name}" if combo_name != '.' else output_dir
        
        if verbose and combo_name != '.':
            print(f"\nProcessing combination {combo_idx + 1}/{len(combo_dirs)}: {combo_name}")
        
        # Discover all metadata files (one per seed)
        metadata_files = sorted([
            f for f in os.listdir(combo_dir) 
            if f.startswith('metadata_seed') and f.endswith('.json')
        ])
        
        combo_results = []
        
        for seed_idx, metadata_file in enumerate(metadata_files):
            # Extract seed number from filename
            seed = int(metadata_file.replace('metadata_seed', '').replace('.json', ''))
            
            if verbose:
                print(f"  [{seed_idx + 1}/{len(metadata_files)}] Processing seed={seed}")
            
            # Read data files generated in Phase 1
            Y_clean = pd.read_csv(f"{combo_dir}/1_Y_clean_seed{seed}.csv", index_col=0)
            Y_clean_clr = pd.read_csv(f"{combo_dir}/1_Y_clean_clr_seed{seed}.csv", index_col=0)
            Y_with_batch = pd.read_csv(f"{combo_dir}/2_Y_with_batch_seed{seed}.csv", index_col=0)
            Y_with_batch_clr = pd.read_csv(f"{combo_dir}/2_Y_with_batch_clr_seed{seed}.csv", index_col=0)
            
            with open(f"{combo_dir}/metadata_seed{seed}.json", 'r') as f:
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
                with_batch_data=Y_with_batch_clr,
                corrected_data=Y_corrected_clr,
                bio_labels=bio_labels
            )
            batch_metrics_after['bio_preservation'] = bio_preservation_metrics
            
            diff_expr_results = compare_differential_expression(
                dataset1=Y_clean,
                dataset2=Y_with_batch,
                dataset3=Y_corrected,
                verbose=False
            )
            
            # Save corrected data (Phase 1 data already saved by simulate())
            if save_csv:
                Y_corrected.to_csv(f"{combo_dir}/3_Y_after_combat_seed{seed}.csv", float_format="%.16f")
                Y_corrected_clr.to_csv(f"{combo_dir}/3_Y_after_combat_clr_seed{seed}.csv", float_format="%.16f")
            
            # Generate comprehensive metrics using metadata from Phase 1
            generate_comprehensive_metrics(
                seed=seed,
                output_dir=combo_dir,
                batch_metrics_before=batch_metrics_before,
                batch_metrics_after=batch_metrics_after,
                diff_expr_results=diff_expr_results,
                metadata=metadata,
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


def run_correction(config):
    """
    Run batch correction pipeline in two phases:
    
    Phase 1: Data generation (delegated to glycoforge.simulate)
    Phase 2: Batch correction and evaluation
    
    All data is saved directly to the output directory specified in config.
    """
    verbose = config.get('verbose', True)
    output_dir = config.get('output_dir', 'results/')
    
    if verbose:
        print(f"[SYSTEM] Output directory: {output_dir}")
    
    try:
        # Phase 1: Generate data
        generate_all_data(config)
        
        # Phase 2: Apply corrections and evaluate
        results = process_corrections(config)
        
        if verbose:
            print("\n" + "=" * 60)
            print("BATCH CORRECTION PIPELINE COMPLETED")
            print(f"Results saved to: {output_dir}")
            print("=" * 60)
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"\n[ERROR] Pipeline failed: {e}")
        raise e
