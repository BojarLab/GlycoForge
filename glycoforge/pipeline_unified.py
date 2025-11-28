from glycowork.motif.analysis import get_differential_expression
import numpy as np
import pandas as pd
import os

from plot import plot_pca
from sim_bio_factor import create_bio_groups, simulate_clean_data, generate_alpha_U, add_noise_to_zero_variance_features,  define_dirichlet_params_from_real_data
from sim_batch_factor import define_batch_direction, stratified_batches_from_columns, apply_batch_effect, estimate_sigma
from utils import clr, invclr, combat
from metrics import quantify_batch_effect_impact, evaluate_biological_preservation, compare_differential_expression, generate_comprehensive_metrics, check_batch_effect


def unified_batch_correction_pipeline(
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
    n_batches=3,
    affected_fraction=(0.05, 0.30),
    positive_prob=0.6,
    overlap_prob=0.5,
    kappa_mu=1.0,
    var_b=0.5,
    u_dict=None,
    random_seeds=[42],
    output_dir="results/",
    verbose=False,
    save_csv=True,
    show_pca_plots=None
):

    
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
        bm_cols = [c for c in df.columns if c.startswith("BM")]
        r7_cols = [c for c in df.columns if c.startswith("R7")]
        
        # Get actual number of glycans from real data
        n_glycans_real = df.shape[0]
        
        results = get_differential_expression(
            df, 
            group1=r7_cols,
            group2=bm_cols,
            transform="CLR",
            impute=True
        )
        
        real_effect_sizes = results['Effect size'].tolist()
        
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
            
            # Handle differential_mask: default to all zeros (no effects injected)
            if differential_mask is None:
                differential_mask = np.zeros(len(p_h))
                if verbose:
                    print(f"[Real Data] No differential_mask provided → using zeros (no effects injected)")
            else:
                differential_mask = np.asarray(differential_mask, dtype=float)
                if len(differential_mask) != len(p_h):
                    raise ValueError(f"differential_mask length ({len(differential_mask)}) must match number of glycans ({len(p_h)})")
            
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
                max_effect=3.0,
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
            Y_clean.to_csv(f"{output_dir}/1_Y_clean_seed{seed}.csv", float_format="%.6f")
            Y_clean_clr.to_csv(f"{output_dir}/1_Y_clean_clr_seed{seed}.csv", float_format="%.6f")
        
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
            Y_with_batch.to_csv(f"{output_dir}/2_Y_with_batch_seed{seed}.csv", float_format="%.6f")
            Y_with_batch_clr.to_csv(f"{output_dir}/2_Y_with_batch_clr_seed{seed}.csv", float_format="%.6f")
        
        # Step 5: Quantify batch effects before correction
        bio_groups, bio_labels = create_bio_groups(
            Y_clean_clr, 
            {'Healthy': ['healthy'], 'Unhealthy': ['unhealthy']}
        )
        batch_metrics_before = quantify_batch_effect_impact(
            Y_with_batch_clr, 
            batch_labels, 
            bio_groups, 
            verbose=verbose
        )
        
        # Step 5.5: Quick batch effect check before correction
        if verbose:
            print("\n" + "=" * 60)
            print("QUICK BATCH EFFECT CHECK (Before Correction)")
            print("=" * 60)
        batch_check_results, _, _ = check_batch_effect(Y_with_batch_clr, batch_labels, bio_labels, verbose=verbose)
        if verbose:
            print("=" * 60 + "\n")
        
        # Step 6: ComBat correction
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
        
        if save_csv:
            Y_corrected_clr.to_csv(f"{output_dir}/3_Y_after_combat_clr_seed{seed}.csv", float_format="%.6f")
            Y_corrected.to_csv(f"{output_dir}/3_Y_after_combat_seed{seed}.csv", float_format="%.6f")
        
        if verbose:
            print(f"Corrected data saved: seed{seed}")
        
        # Step 7: Quantify batch effects after correction
        batch_metrics_after = quantify_batch_effect_impact(
            Y_corrected_clr, 
            batch_labels, 
            bio_groups, 
            verbose=verbose
        )
        
        bio_preservation_metrics = evaluate_biological_preservation(
            clean_data=Y_clean_clr,
            corrected_data=Y_corrected_clr,
            bio_labels=bio_labels
        )
        batch_metrics_after.update(bio_preservation_metrics)
        
        # Step 7.5: PCA plots
        if show_pca_plots:
            plot_pca(Y_clean_clr, bio_groups=bio_groups, 
                    title=f"Run {run_idx + 1}: Clean Data")
            plot_pca(Y_with_batch_clr, bio_groups=bio_groups, batch_groups=batch_groups,
                    title=f"Run {run_idx + 1}: With Batch Effects")
            plot_pca(Y_corrected_clr, bio_groups=bio_groups, batch_groups=batch_groups,
                    title=f"Run {run_idx + 1}: After ComBat")
        
        # Step 8: Differential expression analysis
        diff_expr_results = compare_differential_expression(
            dataset1=Y_clean if not save_csv else None,
            dataset2=Y_with_batch if not save_csv else None,
            dataset3=Y_corrected if not save_csv else None,
            dataset1_path=f"{output_dir}/1_Y_clean_seed{seed}.csv" if save_csv else None,
            dataset2_path=f"{output_dir}/2_Y_with_batch_seed{seed}.csv" if save_csv else None,
            dataset3_path=f"{output_dir}/3_Y_after_combat_seed{seed}.csv" if save_csv else None,
            verbose=verbose
        )
        
        # Step 9: Save comprehensive metrics
        run_config = {
            'seed': seed,
            'data_source': data_source,
            'n_glycans': n_glycans,
            'n_H': n_H,
            'n_U': n_U,
            'n_batches': n_batches,
            'k_dir': k_dir,
            'variance_ratio': variance_ratio,
            'k_dir_H': k_dir,
            'k_dir_U': k_dir / variance_ratio,
            'bio_strength': bio_strength,
            'kappa_mu': kappa_mu,
            'var_b': var_b,
            'affected_fraction': affected_fraction,
            'positive_prob': positive_prob,
            'overlap_prob': overlap_prob
        }
        
        if data_source == "real":
            run_config['data_file'] = data_file
            run_config['use_real_effect_sizes'] = use_real_effect_sizes
            run_config['differential_mask_sum'] = int(differential_mask.sum()) if differential_mask is not None else 0
        
        generate_comprehensive_metrics(
            seed=seed,
            output_dir=output_dir,
            batch_metrics_before=batch_metrics_before,
            batch_metrics_after=batch_metrics_after,
            diff_expr_results=diff_expr_results,
            run_config=run_config,
            batch_check_results=batch_check_results
        )
        
        all_runs_results.append({
            'run': run_idx + 1,
            'seed': seed,
            'metrics_before': batch_metrics_before.copy(),
            'metrics_after': batch_metrics_after.copy(),
            'differential_expression': diff_expr_results
        })
    
    # Step 10: Compute summary statistics
    all_metrics = set()
    for result in all_runs_results:
        all_metrics.update(result['metrics_before'].keys())
        all_metrics.update(result['metrics_after'].keys())
    
    metrics_names = list(all_metrics)
    ratio_metrics = ['biological_variability_preservation', 'conserved_differential_proportion']
    
    improvement_data = {metric: [] for metric in metrics_names}
    
    for result in all_runs_results:
        for metric in metrics_names:
            before = result['metrics_before'].get(metric, None)
            after = result['metrics_after'].get(metric, None)
            
            if after is None:
                continue
            
            if metric in ratio_metrics:
                pct_change = after * 100
            else:
                if before is not None and before != 0:
                    pct_change = ((after - before) / abs(before)) * 100
                else:
                    pct_change = 0
            
            improvement_data[metric].append(pct_change)
    
    summary_stats = {}
    for metric in metrics_names:
        improvements = improvement_data[metric]
        if len(improvements) > 0:
            summary_stats[metric] = {
                'mean': np.mean(improvements),
                'std': np.std(improvements, ddof=1) if len(improvements) > 1 else 0,
                'values': improvements
            }
    
    if verbose:
        print("=" * 60)
        print("PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Processed {len(random_seeds)} seeds successfully")
        print(f"Results in: {output_dir}")
        print("=" * 60)
    
    return {
        'all_runs_results': all_runs_results,
        'summary_stats': summary_stats,
        'improvement_data': improvement_data,
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
