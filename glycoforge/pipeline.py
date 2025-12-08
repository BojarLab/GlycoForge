from glycowork.motif.analysis import get_differential_expression
import numpy as np
import pandas as pd
import os
import json

from plot import plot_pca
from sim_bio_factor import create_bio_groups, simulate_clean_data, generate_alpha_U, define_dirichlet_params_from_real_data, define_differential_mask
from sim_batch_factor import define_batch_direction, stratified_batches_from_columns, apply_batch_effect, estimate_sigma
from utils import clr
from metrics import check_batch_effect


def simulate(
    data_source="simulated",
    data_file=None,
    n_glycans=50,
    n_H=15,
    n_U=15,
    bio_strength=1.5,
    k_dir=100,
    variance_ratio=1.5,
    use_real_effect_sizes=False,
    differential_mask=None,
    column_prefix=None,
    n_batches=3,
    affected_fraction=(0.05, 0.30),
    positive_prob=0.6,
    overlap_prob=0.5,
    kappa_mu=1.0,
    var_b=0.5,
    max_fold_change=3.0,
    scaling_strategy="clip",
    u_dict=None,
    random_seeds=[42],
    output_dir="results/",
    verbose=False,
    save_csv=True,
    show_pca_plots=None
):
    # Capture original config for metadata
    differential_mask_config = differential_mask

    # Define parameters supported for grid search
    grid_search_params = {
        'kappa_mu': kappa_mu,
        'var_b': var_b,
        'bio_strength': bio_strength,
        'k_dir': k_dir,
        'variance_ratio': variance_ratio,
        'max_fold_change': max_fold_change,
        'scaling_strategy': scaling_strategy
    }
    
    # Identify which parameters are lists/tuples (requiring grid search)
    list_params = {k: v for k, v in grid_search_params.items() if isinstance(v, (list, tuple))}
    
    if list_params:
        import itertools
        
        # Extract keys and values for product
        keys = list(list_params.keys())
        values = list(list_params.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*values))
        
        all_grid_results = {}
        
        if verbose:
            print("=" * 60)
            print(f"STARTING GRID SEARCH: {len(combinations)} combinations")
            print(f"Varying parameters: {', '.join(keys)}")
            print("=" * 60)
            
        for combo in combinations:
            # Create a dictionary for this specific combination
            current_params = dict(zip(keys, combo))
            
            # Construct subdirectory name dynamically
            # e.g., "bio_1.5_kdir_100_kappa_2.0"
            dir_name_parts = [f"{k}_{v}" for k, v in current_params.items()]
            sub_dir = os.path.join(output_dir, "_".join(dir_name_parts))
            
            if verbose:
                param_str = ", ".join([f"{k}={v}" for k, v in current_params.items()])
                print(f"\n>>> Grid Run: {param_str}")
            
            # Prepare arguments for recursive call
            # Start with all original arguments
            kwargs = {
                'data_source': data_source,
                'data_file': data_file,
                'n_glycans': n_glycans,
                'n_H': n_H,
                'n_U': n_U,
                'bio_strength': bio_strength,
                'k_dir': k_dir,
                'variance_ratio': variance_ratio,
                'use_real_effect_sizes': use_real_effect_sizes,
                'differential_mask': differential_mask,
                'column_prefix': column_prefix,
                'n_batches': n_batches,
                'affected_fraction': affected_fraction,
                'positive_prob': positive_prob,
                'overlap_prob': overlap_prob,
                'kappa_mu': kappa_mu,
                'var_b': var_b,
                'max_fold_change': max_fold_change,
                'scaling_strategy': scaling_strategy,
                'u_dict': u_dict,
                'random_seeds': random_seeds,
                'output_dir': sub_dir,
                'verbose': verbose,
                'save_csv': save_csv,
                'show_pca_plots': show_pca_plots
            }
            
            # Update with current grid values
            kwargs.update(current_params)
            
            # Recursive call
            result = simulate(**kwargs)
            
            # Store result with a key representing the combination
            key_name = "_".join(dir_name_parts)
            all_grid_results[key_name] = result
            
        return all_grid_results

    
    if data_source not in ["simulated", "real"]:
        raise ValueError(f"data_source must be 'simulated' or 'real', got '{data_source}'")
    
    if data_source == "real" and data_file is None:
        raise ValueError("data_file is required when data_source='real'")
    
    if show_pca_plots is None:
        show_pca_plots = verbose
    
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("UNIFIED BATCH CORRECTION PIPELINE")
        print("=" * 60)
        print(f"Mode: {data_source.upper()}")
        if data_source == "real":
            print(f"Data file: {data_file}")
            print(f"Use real effect sizes: {use_real_effect_sizes}")
        print(f"Processing {len(random_seeds)} random seeds")
        print(f"Parameters: n_glycans={n_glycans}, n_H={n_H}, n_U={n_U}")
        print(f"Bio signal: bio_strength={bio_strength}, k_dir={k_dir}, variance_ratio={variance_ratio}")
        print(f"  → Healthy k_dir={k_dir:.1f}, Unhealthy k_dir={k_dir/variance_ratio:.1f}")
        print(f"Batch: n_batches={n_batches}, kappa_mu={kappa_mu}, var_b={var_b}")
        print(f"Output: {output_dir}")
        print("=" * 60)
    
    # Step 1: Prepare alpha_H based on data source
    if data_source == "simulated":
        alpha_H = np.ones(n_glycans) * 10
        real_effect_sizes = None
        alpha_U_base = None  # Will generate synthetically in loop
        
    elif data_source == "real":
        df = pd.read_csv(data_file)
        
        # Get column prefixes (with defaults)
        if column_prefix is None:
            column_prefix = {}
        healthy_prefix = column_prefix.get('healthy', 'R7')
        unhealthy_prefix = column_prefix.get('unhealthy', 'BM')
        
        # Find columns by prefix
        r7_cols = [c for c in df.columns if c.startswith(healthy_prefix)]
        bm_cols = [c for c in df.columns if c.startswith(unhealthy_prefix)]
        
        if not r7_cols or not bm_cols:
            raise ValueError(
                f"No columns found with prefixes: healthy='{healthy_prefix}', unhealthy='{unhealthy_prefix}'. "
                f"Available columns: {df.columns.tolist()[:10]}... "
                f"Please check 'column_prefix' in config."
            )
        
        # Get actual number of glycans from real data
        n_glycans_real = df.shape[0]
        
        # Preprocess data to avoid zero-variance issues in glycowork
        # Add tiny random noise to zero values to prevent constant imputation
        rng = np.random.default_rng(42)
        numeric_cols = r7_cols + bm_cols
        
        # Create a copy to avoid modifying original dataframe if needed elsewhere
        df_processed = df.copy()
        
        # Apply jitter only to zero values
        for col in numeric_cols:
            zero_mask = df_processed[col] == 0
            if zero_mask.any():
                # Generate noise between 1e-6 and 1.1e-6
                noise = rng.uniform(1e-6, 1.1e-6, size=zero_mask.sum())
                df_processed.loc[zero_mask, col] = noise
                
        if verbose:
            print(f"[Real Data] Applied jitter to zero values to prevent zero-variance issues")
        
        results = get_differential_expression(
            df_processed, 
            group1=r7_cols,
            group2=bm_cols,
            transform="CLR",
            impute=True
        )
        
        # Reindex effect sizes to match input glycan order
        # get_differential_expression may filter out some glycans, causing length mismatch
        if 'glycan' in df.columns:
            glycan_order = df['glycan'].astype(str).tolist()
        else:
            glycan_order = df.index.astype(str).tolist()
        
        # Convert results to indexed Series for alignment
        if 'glycan' in results.columns:
            effect_series = results.set_index('glycan')['Effect size']
            if 'significant' in results.columns:
                significant_series = results.set_index('glycan')['significant']
            else:
                significant_series = None
        else:
            effect_series = results['Effect size']
            if 'significant' in results.columns:
                significant_series = results['significant']
            else:
                significant_series = None
        
        # Reindex to full glycan list, fill missing with 0.0 (no effect injected)
        effect_series_full = effect_series.reindex(glycan_order).fillna(0.0)
        real_effect_sizes = effect_series_full.values
        
        if significant_series is not None:
            significant_full = significant_series.reindex(glycan_order).fillna(False)
            significant_mask_aligned = significant_full.values
        else:
            significant_mask_aligned = None
        
        if verbose:
            n_original = len(effect_series)
            n_full = len(real_effect_sizes)
            n_missing = n_full - n_original
            if n_missing > 0:
                print(f"[Real Data] Reindexed effect sizes: {n_original} → {n_full} glycans ({n_missing} missing filled with 0.0)")
        
        if use_real_effect_sizes:
            # Extract healthy baseline from mean of all healthy samples
            healthy_ref = df[r7_cols].mean(axis=1).values
            
            # Handle zeros in healthy reference
            healthy_ref = np.array(healthy_ref, dtype=float)
            if np.any(healthy_ref == 0):
                if verbose:
                    print(f"[Real Data] Found {(healthy_ref == 0).sum()} zeros in healthy mean")
            
            # Normalize to proportions
            p_h = healthy_ref / np.sum(healthy_ref)
            
            # Resolve differential_mask using helper
            differential_mask = define_differential_mask(
                differential_mask, 
                n_glycans=len(p_h),
                effect_sizes=real_effect_sizes,
                significant_mask=significant_mask_aligned,
                verbose=verbose
            )
            
            if verbose:
                n_differential = int(differential_mask.sum())
                print(f"[Real Data] {n_differential}/{len(differential_mask)} glycans will have effects injected")
            
            # Call new function with real effect sizes via CLR-space injection
            alpha_H, alpha_U_base = define_dirichlet_params_from_real_data(
                p_h=p_h,
                effect_sizes=real_effect_sizes,
                differential_mask=differential_mask,
                bio_strength=bio_strength,  # Use user-specified value directly
                k_dir=k_dir,
                variance_ratio=variance_ratio,
                max_fold_change=max_fold_change, # Use user-specified value
                scaling_strategy=scaling_strategy, # Use user-specified value
                min_alpha=0.5,
                max_alpha=None,
                verbose=verbose
            )
            
            # Override n_glycans with actual data size
            n_glycans = n_glycans_real
            
            if verbose:
                print(f"[Real Data] Used CLR-space injection for effect sizes")
                print(f"[Real Data] Actual n_glycans from data: {n_glycans}")
                print(f"[Real Data] alpha_H: [{alpha_H.min():.3f}, {alpha_H.max():.3f}]")
                print(f"[Real Data] alpha_U: [{alpha_U_base.min():.3f}, {alpha_U_base.max():.3f}]")
        else:
            # Still need to match real data size even if not using real effect sizes
            n_glycans = n_glycans_real
            alpha_H = np.ones(n_glycans) * 10
            alpha_U_base = None  # Will generate synthetically in loop
        
        if verbose:
            print(f"Loaded real data: {len(r7_cols)} healthy, {len(bm_cols)} unhealthy")
            print(f"Number of glycans: {n_glycans}")
            print(f"Effect sizes range: [{min(real_effect_sizes):.3f}, {max(real_effect_sizes):.3f}]")
    
    # Step 2: Define batch direction vectors
    if u_dict is None:
        u_dict = define_batch_direction(
            batch_effects=None,
            n_glycans=n_glycans,
            n_batches=n_batches,
            affected_fraction=affected_fraction,
            positive_prob=positive_prob,
            overlap_prob=overlap_prob,
            # seed parameter removed - uses default fixed seed (42) for reproducibility
            verbose=verbose
        )
    
    if verbose:
        print(f"Batch direction vectors: {[len(v) for v in u_dict.values()]}")
    
    # Step 3-9: Multi-run loop
    all_runs_results = []
    
    for run_idx, seed in enumerate(random_seeds):
        if verbose:
            print(f"\n--- Run {run_idx + 1}/{len(random_seeds)} (seed={seed}) ---")
        
        # Generate alpha_U per-run
        if use_real_effect_sizes:
            # Use alpha_U from define_dirichlet_params (based on real effect sizes)
            alpha_U = alpha_U_base
            if verbose:
                print(f"[Real Data] Using alpha_U from real effect sizes")
        else:
            # Generate alpha_U synthetically
            alpha_U, delta = generate_alpha_U(alpha_H, up_frac=0.3, down_frac=0.35, seed=seed)
        
        if verbose:
            print(f"alpha_U range: [{alpha_U.min():.2f}, {alpha_U.max():.2f}]")
        
        # Step 3: Generate clean data
        P, labels = simulate_clean_data(alpha_H, alpha_U, n_H, n_U, seed=seed, verbose=verbose)
        glycan_index = np.arange(1, P.shape[1] + 1)
        Y_clean = pd.DataFrame(
            P.T, 
            index=glycan_index,
            columns=[f"healthy_{i+1}" for i in range(np.sum(labels==0))] +
                    [f"unhealthy_{i+1}" for i in range(np.sum(labels==1))]
        )
        Y_clean.index.name = "glycan_index"
        
        Y_clean_clr = clr(Y_clean.values.T).T
        Y_clean_clr = pd.DataFrame(Y_clean_clr, index=Y_clean.index, columns=Y_clean.columns)
        
        if save_csv:
            Y_clean.to_csv(f"{output_dir}/1_Y_clean_seed{seed}.csv", float_format="%.32f")
            Y_clean_clr.to_csv(f"{output_dir}/1_Y_clean_clr_seed{seed}.csv", float_format="%.32f")
        
        # Step 4: Apply batch effects
        batch_groups, batch_labels = stratified_batches_from_columns(
            Y_clean_clr.columns, 
            n_batches=n_batches, 
            seed=seed, 
            verbose=verbose
        )
        
        Y_clean_T = Y_clean_clr.T.values
        sigma = estimate_sigma(Y_clean_clr)
        
        Y_with_batch_clr_T, Y_with_batch_T = apply_batch_effect(
            Y_clean=Y_clean_T,
            batch_labels=batch_labels,
            u_dict=u_dict,
            sigma=sigma,
            kappa_mu=kappa_mu,
            var_b=var_b,
            seed=seed
        )

        Y_with_batch_clr = pd.DataFrame(Y_with_batch_clr_T.T, index=Y_clean_clr.index, columns=Y_clean_clr.columns)
        Y_with_batch = pd.DataFrame(Y_with_batch_T.T, index=Y_clean_clr.index, columns=Y_clean_clr.columns)

        if save_csv:
            Y_with_batch.to_csv(f"{output_dir}/2_Y_with_batch_seed{seed}.csv", float_format="%.32f")
            Y_with_batch_clr.to_csv(f"{output_dir}/2_Y_with_batch_clr_seed{seed}.csv", float_format="%.32f")
        
        # Step 5: Quick batch effect check
        bio_groups, bio_labels = create_bio_groups(
            Y_clean_clr, 
            {'Healthy': ['healthy'], 'Unhealthy': ['unhealthy']}
        )
        if verbose:
            print("\n" + "=" * 60)
            print("QUICK BATCH EFFECT CHECK")
            print("=" * 60)
        check_batch_effect_results, _, _ = check_batch_effect(Y_with_batch_clr, batch_labels, bio_labels, verbose=verbose)
        if verbose:
            print("=" * 60 + "\n")
        
        # Step 6: PCA plots
        if show_pca_plots:
            plot_pca(Y_clean_clr, bio_groups=bio_groups, 
                    title=f"Run {run_idx + 1}: Clean Data")
            plot_pca(Y_with_batch_clr, bio_groups=bio_groups, batch_groups=batch_groups,
                    title=f"Run {run_idx + 1}: With Batch Effects")
        
        # Step 7: Save metadata JSON
        batch_groups_serializable = {k: list(v) for k, v in batch_groups.items()}
        bio_groups_serializable = {k: list(v) for k, v in bio_groups.items()}
        
        # Construct key_parameters
        key_parameters = {
            'n_H': n_H,
            'n_U': n_U,
            'bio_strength': bio_strength,
            'k_dir': k_dir,
            'variance_ratio': variance_ratio,
            'k_dir_H': k_dir,
            'k_dir_U': k_dir / variance_ratio,
            'kappa_mu': kappa_mu,
            'var_b': var_b,
            'affected_fraction': list(affected_fraction) if isinstance(affected_fraction, tuple) else affected_fraction,
            'positive_prob': positive_prob,
            'overlap_prob': overlap_prob,
            'differential_mask_config': differential_mask_config if isinstance(differential_mask_config, (str, type(None))) else "Custom Array"
        }
        
        if data_source == "real":
            key_parameters['differential_mask_sum'] = int(differential_mask.sum()) if differential_mask is not None else 0

        # Construct metadata with ordered keys
        metadata = {
            'seed': seed,
            'data_source': data_source,
        }
        
        if data_source == "real":
            metadata['data_file'] = data_file
            metadata['use_real_effect_sizes'] = use_real_effect_sizes
            
        metadata.update({
            'n_glycans': n_glycans,
            'n_samples': n_H + n_U,
            'n_batches': n_batches,
            'key_parameters': key_parameters,
            'quickly_check_batch_effect': check_batch_effect_results,
            'data_info': {
                'bio_labels': bio_labels.tolist(),
                'batch_labels': batch_labels.tolist(),
                'bio_groups': bio_groups_serializable,
                'alpha_H': alpha_H.tolist(),
                'alpha_U': alpha_U.tolist(),
                'differential_mask_values': differential_mask.tolist() if differential_mask is not None and hasattr(differential_mask, 'tolist') else None
            },
            'detailed_batch_info': {
                'u_dict_keys': list(u_dict.keys()),
                'u_dict': {k: v.tolist() for k, v in u_dict.items()},  # Save full u_dict
                'affected_glycans_per_batch': {k: len(v) for k, v in u_dict.items()},
                'batch_groups': batch_groups_serializable,
                'sigma_mean': float(np.mean(sigma)),
                'sigma_std': float(np.std(sigma))
            }
        })
        
        metadata_path = f"{output_dir}/metadata_seed{seed}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if verbose:
            print(f"Metadata saved: {metadata_path}")
        
        all_runs_results.append(metadata)
    
    if verbose:
        print("=" * 60)
        print("PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Processed {len(random_seeds)} seeds successfully")
        print(f"Results in: {output_dir}")
        print("=" * 60)
    
    return {
        'metadata': all_runs_results,
        'config': {
            'data_source': data_source,
            'n_glycans': n_glycans,
            'n_H': n_H,
            'n_U': n_U,
            'n_batches': n_batches,
            'kappa_mu': kappa_mu,
            'var_b': var_b,
            'random_seeds': random_seeds,
            'affected_fraction': affected_fraction,
            'positive_prob': positive_prob,
            'overlap_prob': overlap_prob,
            'output_dir': output_dir
        }
    }
