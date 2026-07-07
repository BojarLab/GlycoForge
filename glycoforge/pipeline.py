from glycowork.glycan_data.loader import glycomics_data_loader as _gcl
from glycowork.motif.analysis import get_differential_expression
from glycowork.motif.graph import subgraph_isomorphism
import numpy as np
import pandas as pd
import os
import json
import warnings
import contextlib
import io
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from glycoforge.sim_bio_factor import create_bio_groups, simulate_clean_data, generate_alpha_U, define_bio_injection_from_real_data, define_differential_mask, calibrate_pair_corr
from glycoforge.sim_batch_factor import define_batch_direction, stratified_batches_from_columns, apply_batch_effect, estimate_sigma
from glycoforge.utils import clr, invclr, plot_pca, check_batch_effect, check_bio_effect, load_data_from_glycowork, apply_mnar_missingness, find_compositional_pairs


def _build_copula_ref(n_glycans, glycan_class=None, return_candidates=False):
  """Build pooled CLR reference matrix and LW covariance from glycowork datasets.
  Mirrors the synthetic-mode pooling logic in simulate() but as a reusable helper
  for simulate_paired, which needs separate refs for glycome A and B."""
  _class_tag = {'N': '_n_', 'O': '_o_', 'GSL': '_gsl_'}.get(glycan_class)
  _all_ds_names = [n for n in dir(_gcl) if not n.startswith('_') and isinstance(getattr(_gcl, n), pd.DataFrame)
                   and (_class_tag is None or _class_tag in n.lower())]
  _pooled_clr, _pooled_R, _candidates = [], [], []
  for _n in _all_ds_names:
    _df = getattr(_gcl, _n).copy()
    if _n.startswith('time_series'):
      _seqs = [str(c) for c in _df.columns[1:]]
      _mat = _df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').T.reset_index(drop=True)
    else:
      _seqs = [str(g) for g in _df['glycan'].tolist()] if 'glycan' in _df.columns else []
      _mat = _df.drop(columns=['glycan'], errors='ignore').select_dtypes(include=[np.number]).reset_index(drop=True)
    if not _seqs or _mat.empty or len(_seqs) < n_glycans:
      continue
    _seen, _keep = set(), []
    for _i, _g in enumerate(_seqs):
      if _g not in _seen:
        _seen.add(_g)
        _keep.append(_i)
    _sub = _mat.iloc[_keep].reset_index(drop=True)
    _top = _sub.mean(axis=1).nlargest(n_glycans).index.tolist()
    _vals = _sub.iloc[_top].values.T
    _vals = _vals[~np.isnan(_vals).any(axis=1)]
    if _vals.shape[0] < 2:
      continue
    _clr_i = clr(_vals)
    if _clr_i.shape[1] < n_glycans:
      continue
    _lw_i = LedoitWolf().fit(_clr_i).covariance_
    _std_i = np.sqrt(np.diag(_lw_i))
    _std_i = np.where(_std_i < 1e-10, 1.0, _std_i)
    _R_i = _lw_i / np.outer(_std_i, _std_i)
    np.fill_diagonal(_R_i, 1.0)
    _pooled_clr.append(_clr_i)
    _pooled_R.append(_R_i)
    _candidates.append((_n, _seqs, _sub))
  if not _pooled_R:
      raise ValueError(
          f"No glycowork datasets found with >= {n_glycans} glycans for class '{glycan_class}'. Try a lower n_glycans or glycan_class=None.")
  _R_consensus = np.mean(_pooled_R, axis=0)
  np.fill_diagonal(_R_consensus, 1.0)
  _clr_all = np.vstack(_pooled_clr)
  _pooled_std = np.std(_clr_all, axis=0)
  _pooled_std = np.where(_pooled_std < 1e-10, 1.0, _pooled_std)
  _Sigma = _R_consensus * np.outer(_pooled_std, _pooled_std)
  if return_candidates:
      return _clr_all, _Sigma, _candidates, len(_pooled_R), _class_tag
  return _clr_all, _Sigma


def simulate(
    data_source="simulated",
    data_file=None,
    glycan_class="N", # only used when data_source = "simulated"
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
    mnar_bias=1.0,
    glycan_sequences=None,
    motif_rules = None,
    motif_bias = 0.8,
    pair_corr_target = "auto", # None disables coupling injection; "auto" or a float sets the substrate-product target r
    batch_motif_rules = None,  # {batch_id: {motif: direction}}
    batch_motif_bias=0.8,
    batch_mode="additive",
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
        'mnar_bias': mnar_bias,
        'batch_mode': batch_mode,
        'motif_bias': motif_bias,
        'batch_motif_bias': batch_motif_bias
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
                'glycan_class': glycan_class,
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
                'glycan_sequences': glycan_sequences,
                'motif_rules': motif_rules,
                'motif_bias': motif_bias,
                'pair_corr_target': pair_corr_target,
                'batch_motif_rules': batch_motif_rules,
                'batch_motif_bias': batch_motif_bias,
                'batch_mode': batch_mode,
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
        _clr_all_syn, _Sigma_syn, _candidates, _n_datasets, _class_tag = _build_copula_ref(n_glycans, glycan_class=glycan_class, return_candidates=True)
        _K_bio = min(3, n_glycans - 1)
        _, _evecs_syn = np.linalg.eigh(_Sigma_syn)
        _top_evecs_syn = _evecs_syn[:, -_K_bio:]
        # Build alpha_H from pooled mean abundance profile
        _p_h_syn = np.maximum(np.mean(np.vstack([
            _ds_mat.iloc[_ds_mat.mean(axis = 1).nlargest(n_glycans).index].mean(axis = 1).values
            for _, _, _ds_mat in _candidates if _ds_mat.mean(axis = 1).nlargest(n_glycans).shape[0] == n_glycans
        ]), axis = 0), 1e-6)
        _p_h_syn /= _p_h_syn.sum()
        alpha_H = _p_h_syn * 10 * n_glycans
        if glycan_sequences is None:
            _rep_name, _rep_seqs, _rep_mat = max(_candidates, key = lambda t: len(t[1]))
            _top_pos = _rep_mat.mean(axis = 1).nlargest(n_glycans).index.tolist()
            glycan_sequences = [_rep_seqs[i] for i in _top_pos]
        real_effect_sizes = None
        if motif_rules is not None and glycan_sequences is not None:
            _alpha_U_motif, _ = generate_alpha_U(
                alpha_H, up_frac = 0.3, down_frac = 0.35,
                glycan_sequences = glycan_sequences[:n_glycans],
                motif_rules = motif_rules, motif_bias = motif_bias,
                seed = 42, verbose = verbose
            )
            alpha_U_base = _alpha_U_motif
        else:
            alpha_U_base = None
        if verbose:
            print(f"[Synthetic] Pooled covariance from {_n_datasets} datasets ({n_glycans} glycans each, {_clr_all_syn.shape[0]} total samples)")
            if _class_tag:
                print(f"[Synthetic] Glycan class filter: {_class_tag}")
    elif data_source == "real":
        df = load_data_from_glycowork(data_file)
        if glycan_sequences is None and 'glycan' in df.columns:
            glycan_sequences = df['glycan'].tolist()
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
                np.random.seed(42)
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
            alpha_H, alpha_U_base, bio_debug_info = define_bio_injection_from_real_data(
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
            # Compute MVN sampler parameters once (reused across all seeds).
            # We use Ledoit-Wolf shrinkage because n_samples << n_glycans is typical in glycomics,
            # making the raw sample covariance rank-deficient and numerically unusable.
            _clr_H_real = clr(df_processed[r7_cols].values.T)  # (n_H, n_glycans)
            _clr_U_real = clr(df_processed[bm_cols].values.T)  # (n_U, n_glycans)
            _clr_all_real = np.vstack([_clr_H_real, _clr_U_real])
            Sigma_mvn = LedoitWolf().fit(_clr_all_real).covariance_  # shrunk pooled covariance
            if verbose:
                print(f"[Copula] Ledoit-Wolf covariance estimated from {_clr_all_real.shape[0]} samples × {n_glycans} features")
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
            verbose=verbose,
            glycan_sequences=glycan_sequences,
            batch_motif_rules=batch_motif_rules,
            motif_bias=batch_motif_bias
        )
    if verbose:
        print(f"Batch direction vectors: {[len(v) for v in u_dict.values()]}")
        # Build calibrated substrate-product correlation targets once, reused across all seeds.
        # Injects biosynthetic coupling the shrinkage-regularized copula covariance cannot supply.
        # Templated (real effect sizes): "auto" measures within-group r from the real CLR; a float overrides.
        # Synthetic: pooled-by-rank reference has no glycan-identity coupling to measure, so "auto" falls back to the real-data median (~0.5)
    pair_corr_cal = None
    if pair_corr_target is not None and motif_rules is not None and glycan_sequences is not None:
        _pairs_pc = find_compositional_pairs(list(glycan_sequences[:n_glycans]), motif_rules, verbose = verbose,
                                             prefix = "PairCorr ")
        _sp = list(zip(_pairs_pc['substrates'], _pairs_pc['products']))
        if _sp and use_real_effect_sizes:
            _pc_raw = []
            for _si, _pi in _sp:
                _rv = [v for v in [
                    np.corrcoef(_clr_H_real[:, _si], _clr_H_real[:, _pi])[0, 1] if _clr_H_real.shape[0] > 2 else np.nan,
                    np.corrcoef(_clr_U_real[:, _si], _clr_U_real[:, _pi])[0, 1] if _clr_U_real.shape[0] > 2 else np.nan]
                       if not np.isnan(v)]
                if pair_corr_target == "auto" and _rv:
                    _pc_raw.append((_si, _pi, float(np.mean(_rv))))
                elif pair_corr_target != "auto":
                    _pc_raw.append((_si, _pi, float(pair_corr_target)))
            if _pc_raw:
                pair_corr_cal = calibrate_pair_corr(_pc_raw, Sigma_mvn, _clr_all_real)
        elif _sp and data_source == "simulated":
            _t = 0.5 if pair_corr_target == "auto" else float(pair_corr_target)
            pair_corr_cal = calibrate_pair_corr([(_si, _pi, _t) for _si, _pi in _sp], _Sigma_syn, _clr_all_syn)
        if verbose:
            print(
                f"[PairCorr] Injecting {0 if pair_corr_cal is None else len(pair_corr_cal)} substrate-product correlations")
    # Step 3-9: Multi-run loop
    all_runs_results = []
    for run_idx, seed in enumerate(random_seeds):
        if verbose:
            print(f"\n--- Run {run_idx + 1}/{len(random_seeds)} (seed={seed}) ---")
        # Generate alpha_U per-run
        if use_real_effect_sizes:
            # Use alpha_U from define_bio_injection_from_real_data (real effect sizes)
            alpha_U = alpha_U_base
            if verbose:
                print(f"[Real Data] Using alpha_U from real effect sizes")
        else:
            if alpha_U_base is not None:
                alpha_U = alpha_U_base
                _p_H_syn = alpha_H / alpha_H.sum()
                _p_U_syn = alpha_U / alpha_U.sum()
                _injection_dir_syn = clr(_p_U_syn) - clr(_p_H_syn)
            else:
                _rng_bio = np.random.default_rng(seed + 999)
                _signs_syn = np.sign(_rng_bio.standard_normal(_K_bio))
                _injection_dir_syn = (_top_evecs_syn * _signs_syn).T
                alpha_U = alpha_H.copy()
        # Step 3: Generate clean data
        if use_real_effect_sizes:
            P, labels = simulate_clean_data(alpha_H, alpha_U, n_H, n_U, seed = seed, verbose = verbose,
                                            real_clr_ref = _clr_all_real, Sigma_lw = Sigma_mvn,
                                            injection = np.array(bio_debug_info['injection']),
                                            pair_corr = pair_corr_cal)
        else:
            _use_scale = (alpha_U_base is None)
            P, labels = simulate_clean_data(alpha_H, alpha_U, n_H, n_U, seed = seed, verbose = verbose,
                                            real_clr_ref = _clr_all_syn, Sigma_lw = _Sigma_syn,
                                            injection = _injection_dir_syn, bio_strength = bio_strength,
                                            scale_injection = _use_scale, pair_corr = pair_corr_cal)
        if glycan_sequences is not None:
            glycan_index = glycan_sequences[:n_glycans]
            index_name = "glycan"
        else:
            glycan_index = np.arange(1, P.shape[1] + 1)
            index_name = "glycan_index"
        Y_clean = pd.DataFrame(
            P.T,
            index=glycan_index,
            columns=[f"healthy_{i + 1}" for i in range(np.sum(labels == 0))] +
                    [f"unhealthy_{i + 1}" for i in range(np.sum(labels == 1))]
        )
        Y_clean.index.name = index_name
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
            seed=seed,
            batch_motif_rules=batch_motif_rules,
            glycan_sequences=glycan_sequences,
            batch_mode = batch_mode
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
        # Construct bio_signal_params
        bio_signal_params  = {
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
            'bio_signal_params': bio_signal_params,
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


def simulate_paired(
  # ── Shared sample structure ────────────────────────────────────────────────
  # These are imposed identically on both glycomes; they encode the fact that
  # every sample appears in both datasets (e.g., same patient, same tube).
  n_H=15,
  n_U=15,
  n_batches=3,
  # ── Glycome A ──────────────────────────────────────────────────────────────
  n_glycans_A=50,
  glycan_class_A="N",
  bio_strength_A=1.5,
  k_dir_A=100,
  variance_ratio_A=1.5,
  up_frac_A=0.3,
  down_frac_A=0.35,
  glycan_sequences_A=None,
  motif_rules_A=None,
  motif_bias_A=0.8,
  batch_motif_rules_A=None,
  # ── Glycome B ──────────────────────────────────────────────────────────────
  n_glycans_B=50,
  glycan_class_B="O",
  bio_strength_B=1.5,
  k_dir_B=100,
  variance_ratio_B=1.5,
  up_frac_B=0.3,
  down_frac_B=0.35,
  glycan_sequences_B=None,
  motif_rules_B=None,
  motif_bias_B=0.8,
  batch_motif_rules_B=None,
  batch_mode="additive",
  # ── Cross-class coupling (optional) ────────────────────────────────────────
  # At coupling_strength=0 the two glycomes are statistically independent
  # conditioned on the shared sample structure. Increasing it injects shared
  # latent variation proportional to coupling_strength², making cross-glycome
  # HSIC detectable. See inject_coupling in sim_coupled.py for the full model.
  coupling_strength=0.5,
  n_coupling_components=1,
  coupling_motif_A=None,   # {motif: any} — direction values ignored, presence biases U_A
  coupling_motif_B=None,
  coupling_motif_bias=0.8,
  # ── Shared batch parameters ────────────────────────────────────────────────
  # Both glycomes share batch labels (same sample → same batch), but batch
  # direction vectors are generated independently so the specific glycans
  # affected and their directions differ between glycomes.
  kappa_mu=1.0,
  var_b=0.5,
  affected_fraction=(0.05, 1),
  positive_prob=0.6,
  overlap_prob=0.5,
  # ── Missingness ────────────────────────────────────────────────────────────
  # Applied independently per glycome (different random seeds) so that the
  # missing-value patterns are not artificially correlated.
  missing_fraction=0.0,
  mnar_bias=1.0,
  # ── Meta ───────────────────────────────────────────────────────────────────
  random_seeds=None,
  output_dir="results/paired/",
  verbose=False,
  save_csv=True,
):
  """Simulate two paired glycomic datasets measured on the same biological samples.

  The fundamental guarantee of this function is sample identity: the same n_H healthy
  and n_U unhealthy individuals appear in both glycomes, so bio_labels, batch_labels,
  and column names are shared. This is the correct data-generating process for studies
  that measure, for example, N- and O-glycomics from the same patient serum aliquots.

  Each glycome is otherwise independently parameterised — different glycan counts,
  different Dirichlet concentration parameters, different biological effect structures,
  and different batch direction vectors. The only shared quantities are the sample-level
  group memberships and batch assignments.

  Cross-class coupling is controlled by *coupling_strength* (default 0). At zero the
  two CLR matrices are conditionally independent given the sample labels. Increasing it
  injects shared latent factors in CLR space (see inject_coupling in sim_coupled.py),
  making HSIC between the two glycomes detectable. The coupling is added after clean
  data generation and before batch effects, modelling biochemical co-regulation (e.g.
  shared sugar nucleotide pools) that would be present in vivo but attenuated by
  downstream sample handling variation.

  Parameters
  ----------
  n_H, n_U : int
      Number of healthy and unhealthy samples. Identical for both glycomes.
  n_glycans_A/B : int
      Number of glycan features in each glycome.
  glycan_class_A/B : str or None
      Glycan class filter for pooled reference construction ('N', 'O', 'GSL', or None
      for no filtering). Defaults 'N'/'O' reflecting the canonical paired use case.
      Set to None to pool across all classes when class-matched datasets are scarce.
  bio_strength_A/B : float
      Scaling of the biological effect in CLR space for each glycome. Higher values
      produce more separated healthy/unhealthy distributions.
  k_dir_A/B : float
      Dirichlet concentration parameter controlling within-group variance. Higher
      values produce tighter, less variable samples.
  variance_ratio_A/B : float
      Ratio of unhealthy to healthy variance (unhealthy k_dir = k_dir / variance_ratio).
  up_frac_A/B, down_frac_A/B : float
      Fraction of glycans up/down-regulated in the unhealthy group.
  glycan_sequences_A/B : list of str or None
      IUPAC glycan sequences for motif-based effects. If None, glycans are indexed
      numerically and motif_rules are ignored.
  motif_rules_A/B : dict or None
      {motif: direction} for biological effects. Passed to generate_alpha_U.
  motif_bias_A/B : float
      Strength of motif preference in biological effect generation.
  batch_motif_rules_A/B : dict or None
      {batch_id: {motif: direction}} for batch effect generation per glycome.
  coupling_strength : float
      Magnitude of cross-class coupling injection. 0 = fully independent glycomes
      with shared sample structure only; 1.0 adds one sigma of shared variation.
      The induced HSIC scales approximately as coupling_strength². Typical useful
      range is 0–3 for benchmarking HSIC detection sensitivity.
  n_coupling_components : int
      Number of independent shared latent dimensions. 1 gives a rank-1 shared
      structure; higher values spread the coupling across multiple axes.
  coupling_motif_A/B : dict or None
      {motif: any} — values ignored; keys used to bias the coupling direction
      matrix toward glycans matching those motifs. Useful for modelling known
      biochemical links, e.g. shared fucosylation affecting both N- and O-glycans.
  coupling_motif_bias : float
      Weight multiplier for motif-matching glycans in the coupling direction matrix.
  kappa_mu : float
      Mean-shift magnitude for batch effects (shared parameter, applied independently
      to each glycome using its own direction vectors and sigma estimates).
  var_b : float
      Variance inflation magnitude for batch effects.
  affected_fraction : tuple of float
      (min, max) fraction of glycans affected per batch, drawn per run.
  positive_prob, overlap_prob : float
      Passed to define_batch_direction for both glycomes.
  missing_fraction : float
      Target fraction of missing values. Applied independently to each glycome
      so that missing-value patterns are not artificially correlated.
  mnar_bias : float
      MNAR intensity bias; higher values make low-abundance glycans more likely
      to be missing (models MS detection limits).
  random_seeds : list of int or None
      One run is produced per seed. Seeds are offset internally (+1000, +2000, etc.)
      to ensure glycome A and B have independent biological and coupling draws while
      remaining fully reproducible.
  output_dir : str
  verbose : bool
  save_csv : bool

  Returns
  -------
  list of dict, one entry per seed. Each dict contains:
    Y_A_clean, Y_B_clean : pd.DataFrame, shape (n_glycans × n_samples)
        Compositional data (% scale) after coupling injection, before batch effects.
    Y_A_clean_clr, Y_B_clean_clr : pd.DataFrame
        CLR-transformed equivalent of the above.
    Y_A_final, Y_B_final : pd.DataFrame
        CLR data after batch effects and missingness — the analysis-ready output.
    bio_labels : np.ndarray, shape (n_samples,)
        0 = healthy, 1 = unhealthy. Identical for A and B.
    batch_labels : np.ndarray, shape (n_samples,)
        0-based batch assignments. Identical for A and B.
    batch_groups : dict {batch_id: [sample_names]}
    All scalar config parameters and per-run diagnostics (missingness stats).
  """
  from glycoforge.sim_coupled import inject_coupling
  if random_seeds is None:
    random_seeds = [42]
  n_samples = n_H + n_U
  os.makedirs(output_dir, exist_ok=True)
  # Fixed-seed baseline Dirichlet parameters. Seed 42 for A, 43 for B so the two
  # glycomes have structurally independent abundance distributions from the start.
  rng_alpha = np.random.default_rng(42)
  raw_A = rng_alpha.lognormal(0, 1.0, n_glycans_A)
  alpha_H_A = raw_A / np.mean(raw_A) * 10
  rng_alpha_B = np.random.default_rng(43)
  raw_B = rng_alpha_B.lognormal(0, 1.0, n_glycans_B)
  alpha_H_B = raw_B / np.mean(raw_B) * 10
  # Batch direction vectors are generated once (u_dict_seed fixed) so the batch
  # structure is identical across runs; only the biological and coupling draws vary.
  # Seeds 42 and 43 again keep A and B independent.
  _clr_ref_A, _Sigma_A = _build_copula_ref(n_glycans_A, glycan_class = glycan_class_A)
  _clr_ref_B, _Sigma_B = _build_copula_ref(n_glycans_B, glycan_class = glycan_class_B)
  _K_bio_A = min(3, n_glycans_A - 1)
  _, _evecs_A = np.linalg.eigh(_Sigma_A)
  _top_evecs_A = _evecs_A[:, -_K_bio_A:]
  _K_bio_B = min(3, n_glycans_B - 1)
  _, _evecs_B = np.linalg.eigh(_Sigma_B)
  _top_evecs_B = _evecs_B[:, -_K_bio_B:]
  u_dict_A, _ = define_batch_direction(
      n_glycans = n_glycans_A, n_batches = n_batches,
      affected_fraction = affected_fraction, positive_prob = positive_prob,
      overlap_prob = overlap_prob, u_dict_seed = 42,
      glycan_sequences = glycan_sequences_A, batch_motif_rules = batch_motif_rules_A,
      verbose = verbose
  )
  u_dict_B, _ = define_batch_direction(
      n_glycans = n_glycans_B, n_batches = n_batches,
      affected_fraction = affected_fraction, positive_prob = positive_prob,
      overlap_prob = overlap_prob, u_dict_seed = 43,
      glycan_sequences = glycan_sequences_B, batch_motif_rules = batch_motif_rules_B,
      verbose = verbose
  )
  sample_cols = (
    [f"healthy_{i+1}" for i in range(n_H)] +
    [f"unhealthy_{i+1}" for i in range(n_U)]
  )
  idx_A = (list(glycan_sequences_A[:n_glycans_A]) if glycan_sequences_A
           else list(np.arange(1, n_glycans_A + 1)))
  idx_B = (list(glycan_sequences_B[:n_glycans_B]) if glycan_sequences_B
           else list(np.arange(1, n_glycans_B + 1)))
  all_runs = []
  for run_idx, seed in enumerate(random_seeds):
    if verbose:
      print(f"\n{'='*60}")
      print(f"SIMULATE_PAIRED  run {run_idx+1}/{len(random_seeds)}  (seed={seed})")
      print(f"{'='*60}")
    # Generate alpha_U independently per glycome. A uses seed directly; B uses
    # seed+1000 so their biological effect draws are never correlated by seed reuse.
    alpha_U_A, _ = generate_alpha_U(
      alpha_H_A, up_frac=up_frac_A, down_frac=down_frac_A,
      glycan_sequences=glycan_sequences_A, motif_rules=motif_rules_A,
      motif_bias=motif_bias_A, seed=seed, verbose=verbose
    )
    alpha_U_B, _ = generate_alpha_U(
      alpha_H_B, up_frac=up_frac_B, down_frac=down_frac_B,
      glycan_sequences=glycan_sequences_B, motif_rules=motif_rules_B,
      motif_bias=motif_bias_B, seed=seed + 1000, verbose=verbose
    )
    # Simulate clean compositional data: (n_samples × n_glycans)
    _has_motif_A = (motif_rules_A is not None and glycan_sequences_A is not None)
    _has_motif_B = (motif_rules_B is not None and glycan_sequences_B is not None)
    if _has_motif_A:
        _p_H_A = alpha_H_A / alpha_H_A.sum()
        _p_U_A = alpha_U_A / alpha_U_A.sum()
        _inj_dir_A = clr(_p_U_A) - clr(_p_H_A)
    else:
        _rng_bio_A = np.random.default_rng(seed + 999)
        _signs_A = np.sign(_rng_bio_A.standard_normal(_K_bio_A))
        _inj_dir_A = (_top_evecs_A * _signs_A).T
    if _has_motif_B:
        _p_H_B = alpha_H_B / alpha_H_B.sum()
        _p_U_B = alpha_U_B / alpha_U_B.sum()
        _inj_dir_B = clr(_p_U_B) - clr(_p_H_B)
    else:
        _rng_bio_B = np.random.default_rng(seed + 1999)
        _signs_B = np.sign(_rng_bio_B.standard_normal(_K_bio_B))
        _inj_dir_B = (_top_evecs_B * _signs_B).T
    P_A, bio_labels = simulate_clean_data(
        alpha_H_A, alpha_U_A, n_H, n_U, seed = seed, verbose = verbose,
        real_clr_ref = _clr_ref_A, Sigma_lw = _Sigma_A,
        injection = _inj_dir_A, bio_strength = bio_strength_A, scale_injection = not _has_motif_A
    )
    P_B, _ = simulate_clean_data(
        alpha_H_B, alpha_U_B, n_H, n_U, seed = seed + 1, verbose = verbose,
        real_clr_ref = _clr_ref_B, Sigma_lw = _Sigma_B,
        injection = _inj_dir_B, bio_strength = bio_strength_B, scale_injection = not _has_motif_B
    )
    Y_A_clr = clr(P_A)   # (n_samples × n_glycans_A)
    Y_B_clr = clr(P_B)
    # ── Optional coupling injection ────────────────────────────────────────
    # seed+2000 for the latent Z draw so it is independent of all bio and
    # alpha_U draws. The round-trip through invclr after injection restores
    # the simplex constraint.
    coupling_meta = {}
    if coupling_strength > 0:
      Y_A_clr, Y_B_clr, Z = inject_coupling(
        Y_A_clr, Y_B_clr,
        coupling_strength=coupling_strength,
        n_coupling_components=n_coupling_components,
        coupling_motif_A=coupling_motif_A,
        coupling_motif_B=coupling_motif_B,
        coupling_motif_bias=coupling_motif_bias,
        glycan_sequences_A=glycan_sequences_A,
        glycan_sequences_B=glycan_sequences_B,
        seed=seed + 2000,
        verbose=verbose
      )
      coupling_meta['Z_shape'] = list(Z.shape)
      coupling_meta['Z_std'] = float(Z.std())
    # Round-trip to simplex to restore compositional validity after injection
    P_A_post = np.array([invclr(Y_A_clr[i], to_percent=True) for i in range(n_samples)])
    P_B_post = np.array([invclr(Y_B_clr[i], to_percent=True) for i in range(n_samples)])
    # Recompute CLR from round-tripped compositions for strict internal consistency
    Y_A_clr_clean = clr(P_A_post)
    Y_B_clr_clean = clr(P_B_post)
    # Convert to (glycans × samples) DataFrames matching pipeline convention
    Y_A_clean = pd.DataFrame(P_A_post.T, index=idx_A, columns=sample_cols)
    Y_B_clean = pd.DataFrame(P_B_post.T, index=idx_B, columns=sample_cols)
    Y_A_clean_clr = pd.DataFrame(Y_A_clr_clean.T, index=idx_A, columns=sample_cols)
    Y_B_clean_clr = pd.DataFrame(Y_B_clr_clean.T, index=idx_B, columns=sample_cols)
    if save_csv:
      Y_A_clean.to_csv(f"{output_dir}/A_1_clean_seed{seed}.csv", float_format="%.32f")
      Y_B_clean.to_csv(f"{output_dir}/B_1_clean_seed{seed}.csv", float_format="%.32f")
      Y_A_clean_clr.to_csv(f"{output_dir}/A_1_clean_clr_seed{seed}.csv", float_format="%.32f")
      Y_B_clean_clr.to_csv(f"{output_dir}/B_1_clean_clr_seed{seed}.csv", float_format="%.32f")
    # ── Shared batch labels, independent effect directions ─────────────────
    # stratified_batches_from_columns uses sample names to stratify by bio group,
    # so both glycomes get the same assignments from the same sample_cols.
    batch_groups, batch_labels = stratified_batches_from_columns(
      sample_cols, n_batches=n_batches, seed=seed, verbose=verbose
    )
    sigma_A = estimate_sigma(Y_A_clean_clr)
    sigma_B = estimate_sigma(Y_B_clean_clr)
    # apply_batch_effect expects (n_samples × n_glycans); .T.values transposes
    Y_A_batch_clr_T, Y_A_batch_T = apply_batch_effect(
      Y_clean=Y_A_clean_clr.T.values, batch_labels=batch_labels,
      u_dict=u_dict_A, sigma=sigma_A, kappa_mu=kappa_mu, var_b=var_b, seed=seed,
      batch_motif_rules=batch_motif_rules_A, glycan_sequences=glycan_sequences_A, batch_mode=batch_mode
    )
    Y_B_batch_clr_T, Y_B_batch_T = apply_batch_effect(
      Y_clean=Y_B_clean_clr.T.values, batch_labels=batch_labels,
      u_dict=u_dict_B, sigma=sigma_B, kappa_mu=kappa_mu, var_b=var_b, seed=seed,
      batch_motif_rules=batch_motif_rules_B, glycan_sequences=glycan_sequences_B, batch_mode=batch_mode
    )
    # Back to (glycans × samples) DataFrames
    Y_A_batch_clr = pd.DataFrame(Y_A_batch_clr_T.T, index=idx_A, columns=sample_cols)
    Y_B_batch_clr = pd.DataFrame(Y_B_batch_clr_T.T, index=idx_B, columns=sample_cols)
    Y_A_batch = pd.DataFrame(Y_A_batch_T.T, index=idx_A, columns=sample_cols)
    Y_B_batch = pd.DataFrame(Y_B_batch_T.T, index=idx_B, columns=sample_cols)
    if save_csv:
      Y_A_batch.to_csv(f"{output_dir}/A_2_batch_seed{seed}.csv", float_format="%.32f")
      Y_B_batch.to_csv(f"{output_dir}/B_2_batch_seed{seed}.csv", float_format="%.32f")
      Y_A_batch_clr.to_csv(f"{output_dir}/A_2_batch_clr_seed{seed}.csv", float_format="%.32f")
      Y_B_batch_clr.to_csv(f"{output_dir}/B_2_batch_clr_seed{seed}.csv", float_format="%.32f")
    # ── Missingness applied independently per glycome ──────────────────────
    # seed+1 for B so that each glycome's missing-value pattern is drawn from
    # an independent stream; using the same seed would make missingness trivially
    # correlated across classes (the same glycans absent in both datasets).
    Y_A_missing, Y_A_missing_clr, _, diag_A = apply_mnar_missingness(
      Y_A_batch, missing_fraction=missing_fraction,
      mnar_bias=mnar_bias, seed=seed, verbose=verbose
    )
    Y_B_missing, Y_B_missing_clr, _, diag_B = apply_mnar_missingness(
      Y_B_batch, missing_fraction=missing_fraction,
      mnar_bias=mnar_bias, seed=seed + 1, verbose=verbose
    )
    if missing_fraction > 0 and save_csv:
      Y_A_missing.to_csv(f"{output_dir}/A_3_missing_seed{seed}.csv", float_format="%.32f")
      Y_B_missing.to_csv(f"{output_dir}/B_3_missing_seed{seed}.csv", float_format="%.32f")
    Y_A_final = Y_A_missing_clr if missing_fraction > 0 else Y_A_batch_clr
    Y_B_final = Y_B_missing_clr if missing_fraction > 0 else Y_B_batch_clr
    run_meta = {
      'seed': seed,
      'n_H': n_H,
      'n_U': n_U,
      'n_glycans_A': n_glycans_A,
      'n_glycans_B': n_glycans_B,
      'coupling_strength': coupling_strength,
      'n_coupling_components': n_coupling_components,
      'coupling_meta': coupling_meta,
      'n_batches': n_batches,
      'kappa_mu': kappa_mu,
      'var_b': var_b,
      'missing_fraction': missing_fraction,
      'mnar_bias': mnar_bias,
      'bio_labels': bio_labels.tolist(),
      'batch_labels': batch_labels.tolist(),
      'batch_groups': {k: list(v) for k, v in batch_groups.items()},
      'missingness_A': diag_A,
      'missingness_B': diag_B,
    }
    with open(f"{output_dir}/metadata_seed{seed}.json", 'w') as f:
      json.dump(run_meta, f, indent=2)
    if verbose:
      print(f"\nRun {run_idx+1} complete — outputs in {output_dir}")
      print("=" * 60)
    all_runs.append({
      **run_meta,
      'Y_A_clean': Y_A_clean,
      'Y_A_clean_clr': Y_A_clean_clr,
      'Y_B_clean': Y_B_clean,
      'Y_B_clean_clr': Y_B_clean_clr,
      'Y_A_final': Y_A_final,
      'Y_B_final': Y_B_final,
    })
  return all_runs


def simulate_circadian(data_file=None, zt_seq=[12, 18, 0, 6, 12, 18, 0, 6, 12], reps=None, cum_seq=None, q_thresh=0.05,
                       amp_scale=2.0, sim_zt_seq=None, sim_reps=None, sim_cum_seq=None, n_batches=3, kappa_mu=1.0,
                       var_b=0.5, affected_fraction=(0.05, 1), positive_prob=0.6, overlap_prob=0.5,
                       batch_motif_rules=None, batch_motif_bias=0.8, batch_mode="additive", missing_fraction=0.0,
                       mnar_bias=1.0, random_seeds=[42], output_dir="results/circadian/", verbose=False, save_csv=True):
    """Simulate circadian glycomics data grounded on a real time-course dataset.

    Fits per-glycan cosinor rhythms and the arrhythmic residual backbone from data_file
    (define_circadian_injection_from_real_data), then generates synthetic samples by an
    empirical-marginal Gaussian copula on the residuals (the copula_emp model, the winner of
    the model bake-off on detection-yield calibration and amplitude fidelity) with the fitted
    cosinor mean curve injected in CLR space and scaled by amp_scale. Clean compositions are
    then pushed through the standard GlycoForge subsystem: batch direction vectors, batch
    effects, and MNAR missingness, with batches stratified across ZT phases so the batch
    factor is orthogonal to the rhythm. Outputs match simulate/simulate_paired conventions.

    Parameters
    ----------
    data_file : str
        Path to a glycan x sample CSV (sample columns named T{n}_ZT{zt}_R{rep}) or a glycowork
        dataset name. Required.
    zt_seq, reps, cum_seq : list or None
        The real data's design used for fitting: ZT phase, replicate count, and cumulative
        hours per timepoint. reps/cum_seq default to 5 replicates and 6 h spacing.
    q_thresh : float
        fdr_tsbh q cutoff defining the ground-truth rhythmic glycan set.
    amp_scale : float
        Multiplier on the injected cosinor amplitude (default 2.0, calibrated so the simulated
        detection yield matches the real study).
    sim_zt_seq, sim_reps, sim_cum_seq : list or None
        Design of the simulated study; default to the fitted design. Vary these (e.g. more
        replicates or denser sampling) for power analysis.
    n_batches, kappa_mu, var_b, affected_fraction, positive_prob, overlap_prob, batch_motif_rules, batch_motif_bias, batch_mode :
        Batch-effect parameters, passed to define_batch_direction and apply_batch_effect.
    missing_fraction, mnar_bias : float
        MNAR missingness parameters, passed to apply_mnar_missingness.
    random_seeds : list of int
        One run per seed.
    output_dir : str
    verbose, save_csv : bool

    Returns
    -------
    list of dict, one per seed. Each holds the run metadata (design, batch assignment,
    ground-truth rhythmic mask, amplitude, acrophase_ZT, missingness diagnostics) plus
    Y_clean, Y_clean_clr (compositional and CLR clean data, glycans x samples), Y_final
    (CLR after batch effects and missingness), and the cum/zt vectors.
    """
    from scipy.stats import norm
    from glycoforge.sim_circadian import define_circadian_injection_from_real_data
    if data_file is None:
        raise ValueError("data_file is required: a path to a glycan x sample CSV, or a glycowork dataset name.")
    # ── Fit cosinor + residual backbone from the real data ─────────────────
    params = define_circadian_injection_from_real_data(data_file, zt_seq=zt_seq, reps=reps, cum_seq=cum_seq, q_thresh=q_thresh)
    glycans = list(params["glycans"])
    n_glycans = len(glycans)
    period = params["period"]
    omega = 2 * np.pi / period
    M, cos_c, sin_c = params["mesor"], params["cos_coef"], params["sin_coef"]
    R, cov = params["residuals"], params["cov"]
    sim_zt_seq = sim_zt_seq if sim_zt_seq is not None else zt_seq
    sim_reps = sim_reps if sim_reps is not None else (reps if reps is not None else [5] * len(sim_zt_seq))
    sim_cum_seq = sim_cum_seq if sim_cum_seq is not None else (cum_seq if cum_seq is not None else [i * 6.0 for i in range(len(sim_zt_seq))])
    cum = np.concatenate([np.full(r, sim_cum_seq[i]) for i, r in enumerate(sim_reps)])
    zt = np.concatenate([np.full(r, sim_zt_seq[i]) for i, r in enumerate(sim_reps)])
    N = len(cum)
    cols = []
    for i, r in enumerate(sim_reps):
        for j in range(r):
            cols.append(f"T{len(cols) + 1}_ZT{int(sim_zt_seq[i])}_R{j + 1}")
    # ── Copula correlation factor ──────────────
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    C = cov / np.outer(std, std)
    jit = 1e-8
    while True:
        try:
            L = np.linalg.cholesky(C + jit * np.eye(n_glycans))
            break
        except np.linalg.LinAlgError:
            jit *= 10
    os.makedirs(output_dir, exist_ok=True)
    all_runs = []
    for seed in random_seeds:
        rng = np.random.default_rng(seed)
        # ── Clean data: empirical-marginal Gaussian copula + cosinor injection ──
        U = np.clip(norm.cdf(rng.standard_normal((N, n_glycans)) @ L.T), 1e-6, 1 - 1e-6)
        res = np.empty((N, n_glycans))
        for g in range(n_glycans):
            res[:, g] = np.quantile(R[:, g], U[:, g])
        clr_clean = amp_scale * (cos_c[None, :] * np.cos(omega * cum)[:, None] + sin_c[None, :] * np.sin(omega * cum)[:, None]) + M[None, :] + res
        comp = np.exp(clr_clean)
        comp = comp / comp.sum(axis=1, keepdims=True) * 100
        Y_clean = pd.DataFrame(comp.T, index=glycans, columns=cols)
        Y_clean.index.name = "glycan"
        Y_clean_clr = pd.DataFrame(clr(Y_clean.values.T).T, index=glycans, columns=cols)
        # ── Phase-stratified batches so batch is orthogonal to ZT ──────────
        batch_labels = np.empty(N, dtype=int)
        for phase in np.unique(cum):
            members = np.where(cum == phase)[0]
            rng.shuffle(members)
            batch_labels[members] = np.arange(len(members)) % n_batches
        batch_groups = {b: [cols[k] for k in range(N) if batch_labels[k] == b] for b in range(n_batches)}
        # ── Batch effects + missingness via existing GlycoForge subsystem ──
        u_dict, _ = define_batch_direction(
            batch_effect_direction=None, n_glycans=n_glycans, n_batches=n_batches, affected_fraction=affected_fraction,
            positive_prob=positive_prob, overlap_prob=overlap_prob, verbose=verbose, glycan_sequences=glycans,
            batch_motif_rules=batch_motif_rules, motif_bias=batch_motif_bias)
        sigma = estimate_sigma(Y_clean_clr)
        Y_batch_clr_T, Y_batch_T = apply_batch_effect(
            Y_clean=Y_clean_clr.T.values, batch_labels=batch_labels, u_dict=u_dict, sigma=sigma, kappa_mu=kappa_mu,
            var_b=var_b, seed=seed, batch_motif_rules=batch_motif_rules, glycan_sequences=glycans, batch_mode=batch_mode)
        Y_batch_clr = pd.DataFrame(Y_batch_clr_T.T, index=glycans, columns=cols)
        Y_batch = pd.DataFrame(Y_batch_T.T, index=glycans, columns=cols)
        Y_missing, Y_missing_clr, _, missing_diag = apply_mnar_missingness(
            Y_batch, missing_fraction=missing_fraction, mnar_bias=mnar_bias, seed=seed, verbose=verbose)
        Y_final = Y_missing_clr if missing_fraction > 0 else Y_batch_clr
        if save_csv:
            Y_clean.to_csv(f"{output_dir}/1_clean_seed{seed}.csv", float_format="%.32f")
            Y_clean_clr.to_csv(f"{output_dir}/1_clean_clr_seed{seed}.csv", float_format="%.32f")
            Y_batch.to_csv(f"{output_dir}/2_batch_seed{seed}.csv", float_format="%.32f")
            Y_batch_clr.to_csv(f"{output_dir}/2_batch_clr_seed{seed}.csv", float_format="%.32f")
            if missing_fraction > 0:
                Y_missing.to_csv(f"{output_dir}/3_missing_seed{seed}.csv", float_format="%.32f")
        run_meta = {
            "seed": seed, "n_glycans": n_glycans, "n_samples": N, "period": period, "amp_scale": amp_scale,
            "sim_zt_seq": list(sim_zt_seq), "sim_reps": list(sim_reps), "sim_cum_seq": list(sim_cum_seq),
            "cum": cum.tolist(), "zt": zt.tolist(), "batch_labels": batch_labels.tolist(),
            "batch_groups": {int(k): v for k, v in batch_groups.items()}, "rhythmic": params["rhythmic"].tolist(),
            "amplitude": params["amplitude"].tolist(), "acrophase_ZT": params["acrophase_ZT"].tolist(),
            "n_batches": n_batches, "kappa_mu": kappa_mu, "var_b": var_b, "batch_mode": batch_mode,
            "missing_fraction": missing_fraction, "mnar_bias": mnar_bias, "missingness": missing_diag,
            "affected_fraction": list(affected_fraction) if isinstance(affected_fraction, tuple) else affected_fraction}
        with open(f"{output_dir}/metadata_seed{seed}.json", "w") as f:
            json.dump(run_meta, f, indent=2)
        if verbose:
            print(f"[circadian] seed {seed}: {N} samples, {int(params['rhythmic'].sum())} ground-truth rhythmic glycans -> {output_dir}")
        all_runs.append({**run_meta, "Y_clean": Y_clean, "Y_clean_clr": Y_clean_clr, "Y_final": Y_final, "cum": cum, "zt": zt})
    return all_runs


def glycoforge_power(
    reference_dataset="human_serum_bacteremia_N_PMID33535571",
    reference_df=None,
    n_glycans=30,
    frac_differential=0.2,
    glycan_effects=(0.3, 0.5, 0.8, 1.2),
    target_motif="Neu5Ac(a2-3)Gal",
    motif_shifts=(0.3, 0.5, 0.8, 1.1),
    sample_sizes=(3, 6, 10, 16, 25, 40),
    n_seeds=40,
    motif_n_seeds=20,
    power_target=0.8,
    ground_truth_seed=0,
    include_glycan=True,
    include_motif=True,
    progress=None
):
    """Monte-Carlo power analysis for a glycomics study design. Returns a matplotlib
    Figure and a tidy results DataFrame. Panel A injects a defined Cohen's d into a
    random subset of glycans (glycan-level detection); panel B coherently regulates the
    carriers of target_motif and tests at the motif level via glycowork's motif
    aggregation. Bands are 95% Monte-Carlo intervals across seeds. Pass progress(done,
    total) to receive progress callbacks during the run."""
    ds = reference_df.copy() if reference_df is not None else getattr(_gcl, reference_dataset).copy()
    if "glycan" not in ds.columns:
        raise ValueError("reference_df must follow the glycowork convention: a 'glycan' column of IUPAC sequences plus numeric sample-abundance columns.")
    ref_label = "user dataset" if reference_df is not None else reference_dataset
    seqs_all = [str(x) for x in ds["glycan"].tolist()]
    mat = ds.drop(columns=["glycan"]).select_dtypes("number")
    seen, keep = set(), []
    for i, s in enumerate(seqs_all):
        if s not in seen:
            seen.add(s)
            keep.append(i)
    seqs_all = [seqs_all[i] for i in keep]
    mat = mat.iloc[keep].reset_index(drop=True)
    top = mat.mean(axis = 1).nlargest(n_glycans).index.tolist()
    n_glycans = len(top)
    seqs = [seqs_all[i] for i in top]
    vals = mat.iloc[top].values.T
    vals = vals[~np.isnan(vals).any(axis=1)]
    clr_ref = clr(vals)
    Sigma = LedoitWolf().fit(clr_ref).covariance_
    sd = clr_ref.std(axis=0)
    panels = (["glycan"] if include_glycan else []) + (["motif"] if include_motif else [])
    if not panels:
        raise ValueError("Enable at least one of include_glycan / include_motif.")
    carrier = None
    if include_motif:
        carrier = np.array([bool(subgraph_isomorphism(s, target_motif)) for s in seqs])
        if carrier.sum() < 3:
            raise ValueError(f"target_motif '{target_motif}' carried by only {int(carrier.sum())} glycans; pick a more common motif.")
    total = 0
    if include_glycan:
        total += len(glycan_effects) * len(sample_sizes) * n_seeds
    if include_motif:
        total += len(motif_shifts) * len(sample_sizes) * motif_n_seeds
    done = 0

    def tick():
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, total)

    def cols(n):
        return [f"H{i}" for i in range(n)] + [f"U{i}" for i in range(n)]

    def sim_df(n, inj, seed):
        P, _ = simulate_clean_data(np.ones(n_glycans), np.ones(n_glycans), n, n, seed=seed, real_clr_ref=clr_ref, Sigma_lw=Sigma, injection=inj)
        d = pd.DataFrame(P.T, columns=cols(n))
        d.insert(0, "glycan", seqs)
        return d

    rng = np.random.default_rng(ground_truth_seed)
    n_diff = max(2, int(frac_differential * n_glycans))
    diff_idx = rng.choice(n_glycans, n_diff, replace=False)
    gmask = np.zeros(n_glycans, bool)
    gmask[diff_idx] = True
    gsign = np.zeros(n_glycans)
    gsign[diff_idx] = np.resize([1.0, -1.0], n_diff)

    def glycan_curve(d):
        inj = np.where(gmask, gsign * d * sd, 0.0)
        means, sems = [], []
        for n in sample_sizes:
            vs = []
            for seed in range(n_seeds):
                df = sim_df(n, inj, seed)
                np.random.seed(seed)
                with contextlib.redirect_stdout(io.StringIO()):
                    res = get_differential_expression(df, group1=cols(n)[n:], group2=cols(n)[:n], transform="CLR", impute=True)
                flag = dict(zip(res["Glycan"], res["significant"]))
                sig = np.array([bool(flag.get(s, False)) for s in seqs])
                vs.append(sig[gmask].mean())
                tick()
            a = np.array(vs)
            means.append(a.mean())
            sems.append(a.std() / np.sqrt(len(a)))
        return np.array(means), np.array(sems)

    def motif_curve(shift):
        inj = shift * sd * carrier
        inj = inj - inj.mean()
        means, sems, eff = [], [], []
        for n in sample_sizes:
            det, es = [], []
            for seed in range(motif_n_seeds):
                df = sim_df(n, inj, seed)
                np.random.seed(seed)
                with contextlib.redirect_stdout(io.StringIO()):
                    res = get_differential_expression(df, group1=cols(n)[n:], group2=cols(n)[:n], motifs=True, transform="CLR", impute=True)
                row = res[res["Glycan"] == target_motif]
                if len(row):
                    det.append(float(row["significant"].iloc[0]))
                    es.append(abs(float(row["Effect size"].iloc[0])))
                tick()
            a = np.array(det)
            means.append(a.mean())
            sems.append(a.std() / np.sqrt(len(a)))
            eff.append(np.mean(es) if es else np.nan)
        return np.array(means), np.array(sems), eff[-1]

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.2), sharey=len(panels) > 1, squeeze=False)
    axes = axes[0]
    x = np.array(sample_sizes)
    rows = []
    for ax, panel in zip(axes, panels):
        if panel == "glycan":
            for d in glycan_effects:
                m, se = glycan_curve(d)
                lo, hi = np.clip(m - 1.96 * se, 0, 1), np.clip(m + 1.96 * se, 0, 1)
                ln, = ax.plot(x, m, marker="o", label=f"d = {d}")
                ax.fill_between(x, lo, hi, alpha=0.18, color=ln.get_color())
                for n, p, l, h in zip(sample_sizes, m, lo, hi):
                    rows.append({"level": "glycan", "effect": f"d={d}", "n_per_group": n, "power": p, "ci_low": l, "ci_high": h})
            ax.set_title(f"(A) Glycan-level detection\n{n_diff}/{n_glycans} glycans truly differential")
        else:
            for shift in motif_shifts:
                m, se, realized = motif_curve(shift)
                lo, hi = np.clip(m - 1.96 * se, 0, 1), np.clip(m + 1.96 * se, 0, 1)
                ln, = ax.plot(x, m, marker="s", label=f"d \u2248 {realized:.2f}")
                ax.fill_between(x, lo, hi, alpha=0.18, color=ln.get_color())
                for n, p, l, h in zip(sample_sizes, m, lo, hi):
                    rows.append({"level": "motif", "effect": f"d~{realized:.2f}", "n_per_group": n, "power": p, "ci_low": l, "ci_high": h})
            ax.set_title(f"(B) Motif-level detection\ntarget: {target_motif} ({int(carrier.sum())}/{n_glycans} carriers)")
        ax.axhline(power_target, ls="--", c="grey", lw=1)
        ax.set_xlabel("Samples per group (n_H = n_U)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(title="Effect size (Cohen's d)")
    axes[0].set_ylabel("Power")
    fig.suptitle(f"GlycoForge power analysis. Reference: {ref_label}", y=1.02)
    fig.tight_layout()
    return fig, pd.DataFrame(rows)