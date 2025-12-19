from glycowork.motif.analysis import get_differential_expression
import numpy as np
import pandas as pd
import os
import json
import warnings
import contextlib
import io
from .sim_bio_factor import create_bio_groups, simulate_clean_data, generate_alpha_U, define_dirichlet_params_from_real_data, define_differential_mask
from .sim_batch_factor import define_batch_direction, stratified_batches_from_columns, apply_batch_effect, estimate_sigma
from .utils import clr, plot_pca, check_batch_effect, check_bio_effect, load_data_from_glycowork, apply_mnar_missingness


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
    batch_effect_direction=None,
    affected_fraction=(0.05, 1),
    positive_prob=0.6,
    overlap_prob=0.5,
    kappa_mu=1.0,
    var_b=0.5,
    winsorize_percentile=None,
    baseline_method="median",
    u_dict=None,
    missing_fraction=0.0,
    mnar_bias=2.0,
    random_seeds=[42],
    output_dir="results/",
    verbose=False,
    save_csv=True,
    show_pca_plots=None
):
    # Parse config if batch_effect_direction contains nested structure
    if batch_effect_direction is not None and isinstance(batch_effect_direction, dict) and 'mode' in batch_effect_direction:
        from .utils import parse_simulation_config
        temp_config = {
            'batch_effect_direction': batch_effect_direction,
            'affected_fraction': affected_fraction,
            'positive_prob': positive_prob,
            'overlap_prob': overlap_prob
        }
        parsed = parse_simulation_config(temp_config)
        batch_effect_direction = parsed.get('batch_effect_direction')
        affected_fraction = parsed.get('affected_fraction', affected_fraction)
        positive_prob = parsed.get('positive_prob', positive_prob)
        overlap_prob = parsed.get('overlap_prob', overlap_prob)

    # Capture original config for metadata
    differential_mask_config = differential_mask

    # Define parameters supported for grid search
    grid_search_params = {
        'kappa_mu': kappa_mu,
        'var_b': var_b,
        'bio_strength': bio_strength,
        'k_dir': k_dir,
        'variance_ratio': variance_ratio,
        'winsorize_percentile': winsorize_percentile,
        'baseline_method': baseline_method,
        'missing_fraction': missing_fraction,
        'mnar_bias': mnar_bias
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
                'batch_effect_direction': batch_effect_direction,
                'affected_fraction': affected_fraction,
                'positive_prob': positive_prob,
                'overlap_prob': overlap_prob,
                'kappa_mu': kappa_mu,
                'var_b': var_b,
                'winsorize_percentile': winsorize_percentile,
                'baseline_method': baseline_method,
                'u_dict': u_dict,
                'missing_fraction': missing_fraction,
                'mnar_bias': mnar_bias,
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
        print(f"Missingness: fraction={missing_fraction:.1%}, bias={mnar_bias}")
        print(f"Output: {output_dir}")
        print("=" * 60)

    # Step 1: Prepare alpha_H based on data source
    bio_debug_info = None  # Initialize debug info storage

    if data_source == "simulated":
        alpha_H = np.ones(n_glycans) * 10
        real_effect_sizes = None
        alpha_U_base = None  # Will generate synthetically in loop

    elif data_source == "real":
        df = load_data_from_glycowork(data_file)

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
            print(f"[Real Data] Calling get_differential_expression with:")
            print(f"  - group1 (disease/unhealthy): {len(bm_cols)} samples (expected: {unhealthy_prefix})")
            print(f"    → {bm_cols[:min(3, len(bm_cols))]}...")
            print(f"  - group2 (control/healthy): {len(r7_cols)} samples (expected: {healthy_prefix})")
            print(f"    → {r7_cols[:min(3, len(r7_cols))]}...")
            print(f"  - transform='CLR', impute=True")
            print(f"    [WARNING] Effect size convention: positive = upregulated in disease")

        # Convention: group1 = disease/unhealthy (BM), group2 = control/healthy (R7)
        # This ensures positive effect sizes indicate upregulation in disease
        # Suppress glycowork output when verbose=False and capture messages
        glycowork_messages = []
        if not verbose:
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture), \
                 warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                results = get_differential_expression(
                    df_processed,
                    group1=bm_cols,
                    group2=r7_cols,
                    transform="CLR",
                    impute=True
                )
            captured_stdout = stdout_capture.getvalue().strip()
            if captured_stdout:
                glycowork_messages.append(f"get_differential_expression: {captured_stdout}")
            if w:
                for warning in w:
                    glycowork_messages.append(f"{warning.category.__name__}: {warning.message}")
        else:
            results = get_differential_expression(
                df_processed,
                group1=bm_cols,
                group2=r7_cols,
                transform="CLR",
                impute=True
            )

        # Handle cases where glycowork filters out some glycans or returns NaN
        # Create aligned effect size array matching original glycan order
        if len(results) != n_glycans_real:
            if verbose:
                print(f"[Real Data] Warning: get_differential_expression returned {len(results)} rows, expected {n_glycans_real}")
                print(f"[Real Data] Aligning effect sizes using index mapping, filling missing with 0.0")

            # Initialize with zeros for all glycans
            aligned_effect_sizes = np.zeros(n_glycans_real)
            aligned_significant = np.zeros(n_glycans_real, dtype=bool)  # Also align significant mask

            # Map results back to original positions using DataFrame index
            for idx, (effect_size, significant) in enumerate(zip(results['Effect size'], results.get('significant', [False] * len(results)))):
                original_idx = results.index[idx]
                if original_idx < n_glycans_real:
                    aligned_effect_sizes[original_idx] = effect_size if not pd.isna(effect_size) else 0.0
                    aligned_significant[original_idx] = significant if not pd.isna(significant) else False

            real_effect_sizes = aligned_effect_sizes.tolist()
            significant_mask = aligned_significant
        else:
            # Normal case: lengths match, just handle NaN
            real_effect_sizes = results['Effect size'].fillna(0.0).tolist()
            significant_mask = results['significant'].values if 'significant' in results.columns else None

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
            # Use aligned significant_mask that matches n_glycans_real
            differential_mask = define_differential_mask(
                differential_mask,
                n_glycans=len(p_h),
                effect_sizes=real_effect_sizes,
                significant_mask=significant_mask,  # Already aligned above
                verbose=verbose
            )

            if verbose:
                n_differential = int(differential_mask.sum())
                print(f"[Real Data] {n_differential}/{len(differential_mask)} glycans will have effects injected")

            # Call new function with real effect sizes via CLR-space injection
            alpha_H, alpha_U_base, bio_debug_info = define_dirichlet_params_from_real_data(
                p_h=p_h,
                effect_sizes=real_effect_sizes,
                differential_mask=differential_mask,
                bio_strength=bio_strength,
                k_dir=k_dir,
                variance_ratio=variance_ratio,
                winsorize_percentile=winsorize_percentile,
                baseline_method=baseline_method,
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

        # Quick check: Original real data bio effect (only in hybrid mode)
        original_data_bio_check = None
        if use_real_effect_sizes:
            bio_labels_real = [0] * len(r7_cols) + [1] * len(bm_cols)
            real_data_values = df[r7_cols + bm_cols].values.T
            real_data_clr = clr(real_data_values).T
            real_data_clr_df = pd.DataFrame(real_data_clr, columns=r7_cols + bm_cols)
            original_data_bio_check, _ = check_bio_effect(
                real_data_clr_df, bio_labels_real,
                stage_name="Original Real Data", verbose=verbose
            )

        if verbose:
            print(f"Loaded real data: {len(r7_cols)} healthy, {len(bm_cols)} unhealthy")
            print(f"Number of glycans: {n_glycans}")
            print(f"Effect sizes range: [{min(real_effect_sizes):.3f}, {max(real_effect_sizes):.3f}]")

    # Step 2: Define batch direction vectors
    # Default: use the provided batch_effect_direction as raw value
    batch_effect_direction_raw = batch_effect_direction

    if u_dict is None:
        # Generate u_dict and get the actual raw direction (be generated in auto mode)
        u_dict, batch_effect_direction_raw = define_batch_direction(
            batch_effect_direction=batch_effect_direction,
            n_glycans=n_glycans,
            n_batches=n_batches,
            affected_fraction=affected_fraction,
            positive_prob=positive_prob,
            overlap_prob=overlap_prob,
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

        # Quick check: Simulated clean data bio effect (all modes)
        bio_labels_sim = [0] * n_H + [1] * n_U
        Y_clean_bio_check, _ = check_bio_effect(
            Y_clean_clr, bio_labels_sim,
            stage_name="Simulated Clean Data (Y_clean)", verbose=verbose
        )

        # Show injection success summary (only in hybrid mode with verbose)
        if use_real_effect_sizes and verbose:
            print("\n" + "=" * 60)
            print("  BIO INJECTION SUCCESS CHECK")
            print("=" * 60)
            if bio_debug_info is not None:
                injection = np.array(bio_debug_info['injection'])
                n_injected = np.sum(injection != 0)
                print(f"  Injected {n_injected}/{n_glycans} glycans")
                print(f"  Injection range: [{injection.min():.2f}, {injection.max():.2f}] CLR units")
                if original_data_bio_check is not None:
                    orig_eta = original_data_bio_check['bio_effect']['effect_size_eta2']
                    sim_eta = Y_clean_bio_check['bio_effect']['effect_size_eta2']
                    print(f"  Original data eta²: {orig_eta:.1%} → Simulated data eta²: {sim_eta:.1%}")
                    print(f"  Enhancement: {sim_eta/orig_eta:.2f}× stronger" if orig_eta > 0 else "")
            print("=" * 60 + "\n")

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

        # Step 4.5: Apply MNAR missingness
        Y_missing, Y_missing_clr, missing_mask, missing_diagnostics = apply_mnar_missingness(
            Y_with_batch,
            missing_fraction=missing_fraction,
            mnar_bias=mnar_bias,
            seed=seed,
            verbose=verbose
        )
        if missing_fraction > 0 and save_csv:
            Y_missing.to_csv(f"{output_dir}/3_Y_with_batch_and_missing_seed{seed}.csv", float_format="%.32f")
            Y_missing_clr.to_csv(f"{output_dir}/3_Y_with_batch_and_missing_clr_seed{seed}.csv", float_format="%.32f")
        
        # Use Y_missing_clr for subsequent analysis if missingness applied
        Y_for_analysis = Y_missing_clr if missing_fraction > 0 else Y_with_batch_clr

        # Step 5: Quick batch effect check
        bio_groups, bio_labels = create_bio_groups(
            Y_clean_clr,
            {'Healthy': ['healthy'], 'Unhealthy': ['unhealthy']}
        )
        if verbose:
            print("\n" + "=" * 60)
            print("QUICK BATCH EFFECT CHECK")
            print("=" * 60)
        check_batch_effect_results, _, _ = check_batch_effect(Y_for_analysis, batch_labels, bio_labels, verbose=verbose)
        if verbose:
            print("=" * 60 + "\n")

        # Step 6: PCA plots
        if show_pca_plots:
            plot_pca(Y_clean_clr, bio_groups=bio_groups,
                    title=f"Run {run_idx + 1}: Clean Data")
            plot_pca(Y_with_batch_clr, bio_groups=bio_groups, batch_groups=batch_groups,
                    title=f"Run {run_idx + 1}: With Batch Effects")
            if missing_fraction > 0:
                plot_pca(Y_missing_clr, bio_groups=bio_groups, batch_groups=batch_groups,
                        title=f"Run {run_idx + 1}: With Batch + Missingness")

        # Step 7: Save metadata JSON
        batch_groups_serializable = {k: list(v) for k, v in batch_groups.items()}
        bio_groups_serializable = {k: list(v) for k, v in bio_groups.items()}

        # Construct bio_parameters
        bio_parameters = {
            'n_H': n_H,
            'n_U': n_U,
            'bio_strength': bio_strength,
            'k_dir': k_dir,
            'k_dir_H': k_dir,
            'k_dir_U': k_dir / variance_ratio,
            'variance_ratio': variance_ratio,
            'differential_mask_config': differential_mask_config if isinstance(differential_mask_config, (str, type(None))) else "Custom Array"
        }

        if data_source == "real":
            bio_parameters['differential_mask_sum'] = int(differential_mask.sum()) if differential_mask is not None else 0

        # Construct batch_parameters
        batch_parameters = {
            'n_batches': n_batches,
            'kappa_mu': kappa_mu,
            'var_b': var_b,
            'affected_fraction': list(affected_fraction) if isinstance(affected_fraction, tuple) else affected_fraction,
            'positive_prob': positive_prob,
            'overlap_prob': overlap_prob,
            'missing_fraction': missing_fraction,
            'mnar_bias': mnar_bias,
            'sigma_mean': float(np.mean(sigma)),
            'sigma_std': float(np.std(sigma))
        }

        # Construct quality_checks (in data processing order)
        quality_checks = {}

        # Step 1: Original data check (only for hybrid mode)
        if use_real_effect_sizes and original_data_bio_check is not None:
            quality_checks['original_data'] = original_data_bio_check

        # Step 2: Y_clean check (all modes)
        if Y_clean_bio_check is not None:
            quality_checks['Y_clean'] = Y_clean_bio_check

        # Step 3: Y_with_batch check (all modes)
        quality_checks['Y_with_batch'] = check_batch_effect_results

        # Step 4: Missingness diagnostics (if applied)
        if missing_fraction > 0:
            quality_checks['missingness'] = missing_diagnostics

        # Construct dirichlet_parameters
        dirichlet_parameters = {
            'alpha_H': alpha_H.tolist(),
            'alpha_U': alpha_U.tolist(),
            'differential_mask': differential_mask.tolist() if differential_mask is not None and hasattr(differential_mask, 'tolist') else None
        }

        # Construct sample_info
        sample_info = {
            'bio_labels': bio_labels.tolist(),
            'batch_labels': batch_labels.tolist(),
            'bio_groups': bio_groups_serializable,
            'batch_groups': batch_groups_serializable
        }

        # Construct metadata with new structure
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
            'bio_parameters': bio_parameters,
            'batch_parameters': batch_parameters,
            'quality_checks': quality_checks,
            'dirichlet_parameters': dirichlet_parameters,
            'sample_info': sample_info
        })

        # Add data processing information for transparency and debugging
        if data_source == "real":
            # Get prefix info (handle both dict and None cases)
            prefix_config = column_prefix if column_prefix is not None else {}
            healthy_prefix_used = prefix_config.get('healthy', 'R7')
            unhealthy_prefix_used = prefix_config.get('unhealthy', 'BM')

            # Add captured glycowork messages first (if any)
            if glycowork_messages:
                metadata['glycowork_messages'] = glycowork_messages

            metadata['differential_expression_config'] = {
                'jitter_applied': True,
                'jitter_range': [1e-6, 1.1e-6],
                'differential_expression_config': {
                    'group1_type': 'disease',
                    'group1_prefix': unhealthy_prefix_used,
                    'group2_type': 'control',
                    'group2_prefix': healthy_prefix_used,
                    'transform': 'CLR',
                    'impute': True,
                    'convention': 'positive effect size = upregulated in disease (group1 > group2)'
                } if use_real_effect_sizes else None
            }

        # Add debug info (optional, only in hybrid mode)
        if bio_debug_info is not None:
            metadata['bio_injection_debug'] = bio_debug_info

        # Record batch_effect_direction configuration
        batch_direction_config = {
            'mode': None,
            'manual': None,
            'auto': None
        }

        if batch_effect_direction is not None:
            # Manual mode: batch_effect_direction was provided
            batch_direction_config['mode'] = 'manual'
            batch_direction_config['manual'] = {
                batch_id: {idx: int(direction) for idx, direction in effects.items()}
                for batch_id, effects in batch_effect_direction.items()
            }
        else:
            # Auto mode: using random generation
            batch_direction_config['mode'] = 'auto'
            batch_direction_config['auto'] = {
                'affected_fraction': affected_fraction,
                'positive_prob': positive_prob,
                'overlap_prob': overlap_prob
            }

        metadata['batch_parameters']['batch_effect_direction'] = batch_direction_config

        # Prepare batch_effect_direction_raw for serialization
        batch_effect_direction_raw_serializable = None
        if batch_effect_direction_raw is not None:
            batch_effect_direction_raw_serializable = {
                str(batch_id): {int(glycan_idx): int(direction)
                                for glycan_idx, direction in effects.items()}
                for batch_id, effects in batch_effect_direction_raw.items()
            }

        metadata['batch_injection_debug'] = {
            'batch_effect_direction_raw': batch_effect_direction_raw_serializable,
            'u_dict': {k: v.tolist() for k, v in u_dict.items()},
            'affected_glycans_per_batch': {k: len(v) for k, v in u_dict.items()}
        }

        metadata_path = f"{output_dir}/metadata_seed{seed}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, separators=(',', ': '))

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
            'missing_fraction': missing_fraction,
            'mnar_bias': mnar_bias,
            'random_seeds': random_seeds,
            'affected_fraction': affected_fraction,
            'positive_prob': positive_prob,
            'overlap_prob': overlap_prob,
            'output_dir': output_dir
        }
    }
