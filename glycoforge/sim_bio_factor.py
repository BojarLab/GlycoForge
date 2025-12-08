import numpy as np
from utils import clr, invclr
from scipy.stats import dirichlet


def create_bio_groups(data, prefix_mapping):
    bio_groups = {}
    for group_name, prefixes in prefix_mapping.items():
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        
        group_cols = []
        for col in data.columns:
            col_lower = str(col).lower()
            if any(col_lower.startswith(prefix.lower()) for prefix in prefixes):
                group_cols.append(col)
        
        if group_cols:
            bio_groups[group_name] = group_cols
    

    bio_labels = np.zeros(len(data.columns), dtype=int)
    for group_id, (group_name, cols) in enumerate(bio_groups.items()):
        for col in cols:
            if col in data.columns:
                col_idx = data.columns.get_loc(col)
                bio_labels[col_idx] = group_id
    
    return bio_groups, bio_labels  # bio_labels: 0-based array 


# Simplified simulation of alhpa_H and alpha_U for clean data
def generate_alpha_U(alpha_H,
                     up_frac=0.3,
                      down_frac=0.3,
                     up_scale_range=(1.1, 3.0),
                     down_scale_range=(0.3, 0.9),
                     seed=None):
    """Generate alpha_U with heterogeneous up/down scaling."""
    rng = np.random.default_rng(seed)
    n = len(alpha_H)
    alpha_U = alpha_H.copy()
    delta = np.ones(n)
    
    n_up = int(up_frac * n)
    n_down = int(down_frac * n)
    
    up_idx = rng.choice(n, n_up, replace=False)
    down_candidates = np.setdiff1d(np.arange(n), up_idx)
    down_idx = rng.choice(down_candidates, n_down, replace=False)
    
    up_scales = rng.uniform(*up_scale_range, size=n_up)
    down_scales = rng.uniform(*down_scale_range, size=n_down)
 
    alpha_U[up_idx] *= up_scales
    alpha_U[down_idx] *= down_scales
    delta[up_idx] = up_scales
    delta[down_idx] = down_scales
    
    alpha_U = np.clip(alpha_U, 1e-3, None)
    return alpha_U, delta

def robust_effect_size_processing(effect_sizes, bio_strength=1.0, max_fold_change=10.0, scaling_strategy="clip", verbose=False):
    """
    Robust effect size processing pipeline with fold-change safety limits.
    
    Parameters:
    -----------
    effect_sizes : array-like
        Raw Cohen's d effect sizes.
    bio_strength : float
        Scaling factor for biological signal strength.
    max_fold_change : float
        Maximum allowed fold change (ratio) for any feature.
        Converted to CLR limit via ln(max_fold_change).
    scaling_strategy : str, "clip" or "scale"
        - "clip": Cap individual features that exceed the limit (preserves others).
        - "scale": Scale down ALL features if any exceeds limit (preserves ratios).
    """
    effect_sizes = np.array(effect_sizes)
    
    if verbose:
        print(f"  [Debug] Input effect sizes: range=[{np.min(effect_sizes):.3f}, {np.max(effect_sizes):.3f}], mean={np.mean(effect_sizes):.3f}, std={np.std(effect_sizes):.3f}")

    # Step 1: Center (Required for CLR additivity)
    # We removed standardization and tanh to keep the logic simple and intuitive.
    # Centering ensures that the geometric mean of the changes is 1 (sum in CLR is 0).
    processed = effect_sizes - np.mean(effect_sizes)
    
    # Step 2: Calculate Raw Injection
    raw_injection = processed * bio_strength
    
    # Step 3: Apply Safety Limits based on Max Fold Change
    # Limit in CLR space = ln(max_fold_change)
    limit = np.log(max_fold_change)
    max_val = np.max(np.abs(raw_injection))
    
    final_injection = raw_injection.copy()
    
    if max_val > limit:
        if verbose:
            print(f"  [Warning] Max injection ({max_val:.2f}) exceeds limit ({limit:.2f} = {max_fold_change}x fold change).")
            
        if scaling_strategy == "scale":
            # Strategy 1: Global Scaling
            scale_factor = limit / max_val
            final_injection = raw_injection * scale_factor
            if verbose:
                print(f"  -> Strategy 'scale': All effects scaled down by factor {scale_factor:.3f} to preserve relative ratios.")
                
        elif scaling_strategy == "clip":
            # Strategy 2: Local Clipping (Default)
            final_injection = np.clip(raw_injection, -limit, limit)
            n_clipped = np.sum(np.abs(raw_injection) > limit)
            if verbose:
                print(f"  -> Strategy 'clip': {n_clipped} features clipped to limit. Others unchanged.")
                print(f"  -> Tip: Use scaling_strategy='scale' if you want to preserve relative effect sizes.")
    else:
        if verbose:
            print(f"  [Debug] Injection within limits (max={max_val:.2f} < limit={limit:.2f}). No scaling needed.")
            
    # Return the FINAL INJECTION vector, not just the processed effect sizes
    # Note: The calling function expects "d_robust", which it then multiplies by bio_strength.
    # To maintain compatibility, we return final_injection / bio_strength
    # so that: (final_injection / bio_strength) * bio_strength = final_injection
    
    if abs(bio_strength) > 1e-9:
        return final_injection / bio_strength
    else:
        return np.zeros_like(final_injection)

#  handle NaN in CLR data
def add_noise_to_zero_variance_features(data, #glycans x samples
                                        noise_level=1e-6,  #Relative noise magnitude based on the row mean.
                                        random_seed=42):
    rng = np.random.default_rng(random_seed)
    data_with_noise = data.copy()
    
    for glycan_idx in range(data.shape[0]):
        row_values = data.iloc[glycan_idx, :].values
        
        # Detect zero-variance features (within floating-point precision)
        if np.std(row_values) < 1e-12:
            # Determine noise amplitude relative to the row mean
            row_mean = np.mean(row_values)
            if row_mean > 0:
                noise_amplitude = row_mean * noise_level
            else:
                noise_amplitude = noise_level  # Absolute noise if mean is zero
            
            # Add independent random noise to each sample
            noise = rng.normal(0, noise_amplitude, len(row_values))
            data_with_noise.iloc[glycan_idx, :] += noise
            
            # Ensure all values remain positive
            data_with_noise.iloc[glycan_idx, :] = np.maximum(
                data_with_noise.iloc[glycan_idx, :], 
                noise_level  # Minimum safeguard value
            )
    
    return data_with_noise



def define_dirichlet_params_from_real_data(p_h, # array-like Baseline healthy distribution (proportions, should sum to ~1).
                                           effect_sizes, # array-like Cohen's d effect sizes from differential expression analysis.
                                           differential_mask, # array-like Binary mask (0 or 1) indicating which glycans are significantly differential.
                                           bio_strength=1.0, # float Biological effect scaling parameter 'lambda'.
                                           k_dir=100, # float Dirichlet concentration parameter for Healthy group.
                                           variance_ratio=1.0, # float Unhealthy variance / Healthy variance.
                                           max_fold_change=10.0, # float Maximum allowed fold change (ratio).
                                           scaling_strategy="clip", # str "clip" or "scale".
                                           min_alpha=0.5, # float Minimum alpha value to avoid zero parameters.
                                           max_alpha=None, # float or None Maximum alpha value to prevent extreme concentrations.
                                           verbose=True):
    """
    Generate Dirichlet parameters using real effect sizes via CLR-space injection.
    -----------
    variance_ratio : float, default=1.0
        Ratio of Unhealthy variance to Healthy variance.
        - variance_ratio = 1.0: Equal variance (conservative)
        - variance_ratio = 1.5: Unhealthy 50% more variable (default)
        - variance_ratio = 2.0: Unhealthy 2× more variable (severe disease)
        - variance_ratio > 3.0: Extreme (use with caution)
    """
    p_h = np.asarray(p_h, dtype=float)
    effect_sizes = np.asarray(effect_sizes, dtype=float)
    differential_mask = np.asarray(differential_mask, dtype=float)
    
    # Validate array lengths for alignment
    if len(effect_sizes) != len(p_h):
        raise ValueError(
            f"Effect sizes length ({len(effect_sizes)}) does not match p_h length ({len(p_h)}). "
            f"Effect sizes must be reindexed to align with input glycan order in pipeline.py. "
            f"This usually happens when get_differential_expression filters out some glycans."
        )
    
    if len(differential_mask) != len(p_h):
        raise ValueError(
            f"Differential mask length ({len(differential_mask)}) does not match p_h length ({len(p_h)}). "
            f"Mask must be generated with n_glycans={len(p_h)}."
        )
    
    # Handle zero values with glycan-wise imputation
    zero_mask = (p_h == 0)
    if np.any(zero_mask):
        for i in np.where(zero_mask)[0]:
            non_zero_values = p_h[p_h > 0]
            if len(non_zero_values) > 0:
                # Use 10% of minimum non-zero value as imputation
                p_h[i] = np.min(non_zero_values) * 0.1
            else:
                # Fallback: use detection limit
                p_h[i] = 0.001
        
        if verbose:
            print(f"[define_dirichlet_params_from_real_data] Imputed {zero_mask.sum()} zero values in p_h")
    
    # Step 1: Normalize p_h to sum = 1
    p_h = p_h / np.sum(p_h)
    
    # Step 2: Transform to CLR space
    z_h = clr(p_h)
    
    # Step 3: Robust effect size processing
    # We use standardize=False to preserve the relative magnitude of real effect sizes,
    # only centering and clipping them to avoid numerical instability.
    d_robust = robust_effect_size_processing(
        effect_sizes, 
        bio_strength=bio_strength,
        max_fold_change=max_fold_change,
        scaling_strategy=scaling_strategy,
        verbose=verbose
    )
    
    if verbose:
        print(f"[define_dirichlet_params_from_real_data] Raw effect sizes: mean={np.mean(effect_sizes):.3f}, std={np.std(effect_sizes):.3f}, range=[{np.min(effect_sizes):.3f}, {np.max(effect_sizes):.3f}]")
        print(f"[define_dirichlet_params_from_real_data] Robust effect sizes (normalized): mean={np.mean(d_robust):.3f}, std={np.std(d_robust):.3f}, range=[{np.min(d_robust):.3f}, {np.max(d_robust):.3f}]")
        print(f"[define_dirichlet_params_from_real_data] Differential mask: {int(differential_mask.sum())}/{len(differential_mask)} glycans are significant")
    
    # Step 4: Inject effects in CLR space (only for significant glycans)
    # z(U) = z(H) + m × bio_strength × d_robust
    # Note: We do NOT multiply by feature sigma here because in high-sparsity data,
    # sigma can be artificially large (due to imputation of zeros), leading to exploded values.
    # Treating d_robust as direct CLR increments is a safer approximation.
    z_u = z_h + differential_mask * bio_strength * d_robust
    
    if verbose:
        # Calculate injection for debug
        injection = differential_mask * bio_strength * d_robust
        effect_magnitude = np.abs(injection)
        print(f"[define_dirichlet_params_from_real_data] Injected effect magnitude (CLR units): mean={np.mean(effect_magnitude[differential_mask > 0]):.3f}, max={np.max(effect_magnitude):.3f}")
    
    # Step 5: Transform back to simplex
    p_u = invclr(z_u, to_percent=False)
    
    if verbose:
        # Find top 3 features with largest absolute injection
        injection_mag = np.abs(injection)
        top_indices = np.argsort(injection_mag)[-3:]
        
        print(f"\n  [Debug] Top 3 features with largest absolute injection (CLR space):")
        for idx in top_indices:
            # Only show if there is actual injection
            if injection_mag[idx] > 0:
                print(f"    Glycan {idx+1}:")
                print(f"      p_h (base prop): {p_h[idx]*100:.2f}%")
                print(f"      z_h (base CLR):  {z_h[idx]:.3f}")
                print(f"      d_robust:        {d_robust[idx]:.3f}")
                print(f"      injection:       {injection[idx]:.3f} ( = {d_robust[idx]:.2f} * {bio_strength} )")
                print(f"      z_u (new CLR):   {z_u[idx]:.3f}")
                print(f"      p_u (new prop):  {p_u[idx]*100:.2f}%")
                print(f"      Change:          {p_h[idx]*100:.2f}% -> {p_u[idx]*100:.2f}%")

    # Step 6: Scale by Dirichlet concentration with variance control
    k_dir_H = k_dir
    k_dir_U = k_dir / variance_ratio
    
    alpha_H = k_dir_H * p_h
    alpha_U = k_dir_U * p_u
    
    # Step 7: Clip alpha values to ensure valid range
    alpha_H = np.clip(alpha_H, min_alpha, max_alpha)
    alpha_U = np.clip(alpha_U, min_alpha, max_alpha)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[define_dirichlet_params_from_real_data] Dirichlet Concentration Parameters")
        print(f"{'='*60}")
        print(f"  Healthy k_dir:    {k_dir_H:.1f}")
        print(f"  Unhealthy k_dir:  {k_dir_U:.1f}")
        print(f"  Variance ratio:   {variance_ratio:.2f} (Unhealthy/Healthy)")
        print(f"    → Unhealthy variance is {variance_ratio:.2f}× Healthy")
        print(f"  alpha_H range:    [{alpha_H.min():.3f}, {alpha_H.max():.3f}]")
        print(f"  alpha_U range:    [{alpha_U.min():.3f}, {alpha_U.max():.3f}]")
        print(f"  Alpha ratio (max/min): H={alpha_H.max()/alpha_H.min():.1f}×, U={alpha_U.max()/alpha_U.min():.1f}×")
        if max_alpha is not None:
            clipped_h = (k_dir_H * p_h > max_alpha).sum()
            clipped_u = (k_dir_U * p_u > max_alpha).sum()
            if clipped_h > 0 or clipped_u > 0:
                print(f"  Clipped: {clipped_h} alpha_H, {clipped_u} alpha_U values to max_alpha={max_alpha}")
        print(f"{'='*60}\n")
    
    return alpha_H, alpha_U


def simulate_clean_data(alpha_H, alpha_U, n_H, n_U, seed=None, verbose=True):
    """
    Simulate clean glycomics data from Dirichlet distributions.
    """
    rng = np.random.default_rng(seed)
    healthy_samples = dirichlet.rvs(alpha_H, size=n_H, random_state=rng) * 100
    unhealthy_samples = dirichlet.rvs(alpha_U, size=n_U, random_state=rng) * 100

    P = np.vstack([healthy_samples, unhealthy_samples])   # shape (n_H+n_U, n_glycans)
    
    if verbose:
        print("=" * 60)
        print("SIMULATE_CLEAN_DATA")
        print("=" * 60)
        print(f"Simulated data Y_clean shape: {P.shape}")
        print(f"Min value: {P.min():.2e}")
        print(f"Max value: {P.max():.2f}")
        print(f"Zero values: {(P == 0).sum()}")
        print("=" * 60)
    
    labels = np.array([0]*n_H + [1]*n_U)                  # 0=healthy, 1=unhealthy

    return P, labels


def define_differential_mask(mask_input, n_glycans, effect_sizes=None, significant_mask=None, verbose=False):
    """Generate differential mask from string config or array."""
    if mask_input is None or (isinstance(mask_input, str) and mask_input.lower() in ["null", "none"]):
        if verbose: print("[Mask] Mode: Null (all zeros)")
        return np.zeros(n_glycans)
    
    if isinstance(mask_input, str):
        mode = mask_input.strip().lower()
        if mode in ["all", "full"]:
            if verbose: print("[Mask] Mode: All (all ones)")
            return np.ones(n_glycans)
        
        if mode in ["significant", "de-based"]:
            if significant_mask is None: raise ValueError("'DE' mode requires significant_mask from analysis")
            if verbose: print(f"[Mask] Mode: DE-based ({int(np.sum(significant_mask))} features)")
            return np.asarray(significant_mask, dtype=float)
            
        if mode.startswith("top-"):
            try:
                n = int(mode.split("-")[1])
                if effect_sizes is None: raise ValueError("'Top-N' mode requires effect_sizes")
                top_idx = np.argsort(np.abs(effect_sizes))[-n:]
                mask = np.zeros(n_glycans)
                mask[top_idx] = 1.0
                if verbose: print(f"[Mask] Mode: Top-{n} by effect size")
                return mask
            except (IndexError, ValueError):
                raise ValueError(f"Invalid Top-N format: {mask_input}")

    # Array input
    mask = np.asarray(mask_input, dtype=float)
    if len(mask) != n_glycans:
        raise ValueError(f"Mask length {len(mask)} != n_glycans {n_glycans}")
    return mask

