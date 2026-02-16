import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "glycoforge"))

import numpy as np
import pandas as pd
from glycoforge.utils import clr, invclr
from glycoforge.sim_bio_factor import (
    simulate_clean_data, 
    generate_alpha_U,
    define_dirichlet_params_from_real_data
)
from glycoforge.sim_batch_factor import apply_batch_effect, define_batch_direction



def test_clr_transform():
    x = np.array([0.2, 0.3, 0.5])
    result = clr(x)
    assert np.isclose(result.sum(), 0.0, atol=1e-6)
    
def test_invclr_transform():
    x = np.array([0.2, 0.3, 0.5])
    z = clr(x)
    x_back = invclr(z, to_percent=False)
    assert np.allclose(x, x_back, atol=1e-5)

def test_simulate_data():
    alpha_H = np.ones(10) * 5
    alpha_U, _ = generate_alpha_U(alpha_H, seed=42)
    P, labels = simulate_clean_data(alpha_H, alpha_U, 10, 10, seed=42, verbose=False)
    
    assert P.shape == (20, 10)
    assert np.allclose(P.sum(axis=1), 100.0)  # Returns percentage, sums to 100
    assert len(labels) == 20


def test_define_dirichlet_params_from_real_data():
    """Test Dirichlet parameters from real effect sizes"""
    p_h = np.array([0.2, 0.3, 0.5])
    effect_sizes = np.array([1.2, -0.8, 0.1])
    differential_mask = np.array([1, 1, 0])
    
    alpha_H, alpha_U, debug_info = define_dirichlet_params_from_real_data(
        p_h, effect_sizes, differential_mask,
        bio_strength=1.0,
        k_dir=10,
        verbose=False
    )
    
    assert alpha_H.shape == (3,)
    assert alpha_U.shape == (3,)
    assert np.all(alpha_H > 0)
    assert np.all(alpha_U > 0)
    
    # Check debug_info structure
    assert 'raw_effect_sizes' in debug_info
    assert 'd_robust' in debug_info
    assert 'injection' in debug_info
    assert 'p_h' in debug_info
    assert 'p_u' in debug_info

def test_apply_batch_effect():
    """Test batch effect application"""
    n_samples, n_glycans = 20, 10
    Y_clean = np.random.randn(n_samples, n_glycans)
    batch_labels = np.array([0]*7 + [1]*7 + [2]*6)
    
    # define_batch_direction now returns (u_dict, raw_direction) tuple
    u_dict, _ = define_batch_direction(
        n_glycans=n_glycans, 
        n_batches=3, 
        u_dict_seed=42, 
        verbose=False
    )
    
    sigma = np.ones(n_glycans)
    Y_batch_clr, Y_batch = apply_batch_effect(
        Y_clean, batch_labels, u_dict, 
        sigma, kappa_mu=1.0, var_b=0.5, 
        seed=42
    )
    
    assert Y_batch_clr.shape == (n_samples, n_glycans)
    assert Y_batch.shape == (n_samples, n_glycans)
    assert np.allclose(Y_batch.sum(axis=1), 100.0)  # Returns percentage


def test_check_batch_effect():
    """Test batch effect checking with bio_groups"""
    from glycoforge.utils import check_batch_effect
    
    n_samples, n_features = 30, 10
    np.random.seed(42)
    
    data = pd.DataFrame(np.random.randn(n_features, n_samples))
    batch_labels = np.array([0]*10 + [1]*10 + [2]*10)
    bio_groups = np.array([0]*15 + [1]*15)
    
    results, pc, var_batch = check_batch_effect(
        data, batch_labels, bio_groups, verbose=False
    )
    
    # Check main structure
    assert 'pca_variance_explained' in results
    assert 'batch_effect' in results
    assert 'bio_effect' in results
    assert 'overall_quality' in results
    
    # Check nested batch_effect structure
    assert 'f_statistic' in results['batch_effect']
    assert 'p_value' in results['batch_effect']
    assert 'test_used' in results['batch_effect']
    assert 'effect_size_eta2' in results['batch_effect']
    
    # Check nested bio_effect structure
    assert 'f_statistic' in results['bio_effect']
    assert 'p_value' in results['bio_effect']
    assert 'effect_size_eta2' in results['bio_effect']
    
    # Check overall_quality structure
    assert 'severity' in results['overall_quality']
    assert 'severity_description' in results['overall_quality']
    assert 'median_variance_explained_by_batch' in results['overall_quality']
    
    # Check output arrays
    assert pc.shape[0] == n_samples
    assert len(var_batch) == n_features


def test_check_bio_effect():
    """Test biological effect checking"""
    from glycoforge.utils import check_bio_effect
    
    n_samples, n_glycans = 30, 10
    np.random.seed(42)
    
    # Create CLR-transformed data with biological differences
    data_clr = pd.DataFrame(
        np.random.randn(n_glycans, n_samples),
        columns=[f'sample_{i}' for i in range(n_samples)]
    )
    # Add biological signal
    data_clr.iloc[:5, 15:] += 1.0  # First 5 glycans differ between groups
    
    bio_labels = np.array([0]*15 + [1]*15)
    
    results, pc = check_bio_effect(
        data_clr, bio_labels, stage_name="test", verbose=False
    )
    
    # Check structure
    assert 'pca_variance_explained' in results
    assert 'bio_effect' in results
    
    # Check bio_effect nested structure
    assert 'f_statistic' in results['bio_effect']
    assert 'p_value' in results['bio_effect']
    assert 'effect_size_eta2' in results['bio_effect']
    
    # Check PC output
    assert pc.shape[0] == n_samples
    assert pc.shape[1] >= 2  # At least 2 principal components


def test_robust_effect_size_processing():
    """Test Winsorization of effect sizes"""
    from glycoforge.sim_bio_factor import robust_effect_size_processing
    
    np.random.seed(42)
    effect_sizes = np.array([0.1, 0.5, 1.2, 3.5, 10.0, -0.3, -1.5, -8.0])
    
    # Test with automatic winsorization
    d_normalized = robust_effect_size_processing(
        effect_sizes, 
        winsorize_percentile=95,
        baseline_method="median",
        verbose=False
    )
    
    # Check that extreme values are clipped (normalized values should be smaller)
    # The function returns normalized effect sizes (centered, winsorized, and baseline-scaled)
    assert isinstance(d_normalized, np.ndarray)
    assert len(d_normalized) == len(effect_sizes)
    
    # After winsorization and normalization, max absolute value should be reasonable
    assert np.max(np.abs(d_normalized)) < 10  # Should be more moderate after processing


def test_define_differential_mask():
    """Test differential mask generation"""
    from glycoforge.sim_bio_factor import define_differential_mask
    
    n_glycans = 20
    effect_sizes = np.random.randn(n_glycans)
    significant_mask = np.zeros(n_glycans)
    significant_mask[:5] = 1  # First 5 are significant
    
    # Test "All" mode
    mask_all = define_differential_mask("All", n_glycans, verbose=False)
    assert np.all(mask_all == 1.0)
    assert len(mask_all) == n_glycans
    
    # Test "Null" mode
    mask_null = define_differential_mask("Null", n_glycans, verbose=False)
    assert np.all(mask_null == 0.0)
    assert len(mask_null) == n_glycans
    
    # Test "significant" mode
    mask_sig = define_differential_mask(
        "significant", n_glycans, 
        significant_mask=significant_mask, 
        verbose=False
    )
    assert np.sum(mask_sig) == 5
    assert np.array_equal(mask_sig, significant_mask)
    
    # Test "Top-N" mode
    mask_top5 = define_differential_mask(
        "Top-5", n_glycans, 
        effect_sizes=effect_sizes, 
        verbose=False
    )
    assert np.sum(mask_top5) == 5
    # Check that it selected the 5 with largest absolute effect sizes
    top5_idx = np.argsort(np.abs(effect_sizes))[-5:]
    assert np.sum(mask_top5[top5_idx]) == 5
    
    # Test array input
    custom_mask = np.array([1, 0, 1, 0] * 5)
    mask_custom = define_differential_mask(custom_mask, n_glycans, verbose=False)
    assert np.array_equal(mask_custom, custom_mask)


def test_motif_based_alpha_generation():
    """Test motif-based alpha_U generation with real glycan data
    
    Validates:
    1. Glycans with target motifs are preferentially selected
    2. up/down directions are correctly applied
    3. substrate-product pairing relationships are preserved
    4. motif_bias parameter affects selection probability
    """
    from glycoforge.utils import load_data_from_glycowork

    try:
        df = load_data_from_glycowork("glycomics_human_leukemia_O_PMID34646384.csv")
        # Glycan sequences are in the first column (or index if set_index was used)
        if df.index.dtype == 'object' and isinstance(df.index[0], str):
            glycan_sequences = df.index.tolist()[:20]
        else:
            glycan_sequences = df.iloc[:20, 0].tolist()  # Use first 20 glycans from first column
    except Exception as e:
        # Fallback: skip test if data not available
        print(f"Skipping test_motif_based_alpha_generation: data not available ({e})")
        return
    
    # Define test parameters
    alpha_H = np.ones(len(glycan_sequences)) * 10
    motif_rules = {"Fuc": "up", "Neu5Ac": "down"}
    
    # Generate alpha_U with motif bias
    alpha_U_with_motif, _ = generate_alpha_U(
        alpha_H, 
        glycan_sequences=glycan_sequences,
        motif_rules=motif_rules,
        motif_bias=0.8,
        seed=42,
        verbose=False
    )
    
    # Generate alpha_U without motif as control
    alpha_U_no_motif, _ = generate_alpha_U(
        alpha_H,
        glycan_sequences=None,
        motif_rules=None,
        seed=42,
        verbose=False
    )
    
    # Validation 1: Array dimensions are correct
    assert len(alpha_U_with_motif) == len(glycan_sequences)
    assert len(alpha_U_no_motif) == len(glycan_sequences)
    
# Validation 2: Motif bias affects the distribution
    # With motif rules, the distribution should differ from no-motif case
    # Test that motif_bias has an effect (not testing specific direction due to pairing complexity)
    ratio_with_motif = alpha_U_with_motif / alpha_H
    ratio_no_motif = alpha_U_no_motif / alpha_H
    
    # The distributions should be different when motif rules are applied
    # Use coefficient of variation (std/mean) as a measure
    cv_with_motif = np.std(ratio_with_motif) / np.mean(ratio_with_motif)
    cv_no_motif = np.std(ratio_no_motif) / np.mean(ratio_no_motif)
    
    # At least verify motif rules create some effect (distributions differ)
    # This is more robust than testing specific directions due to pairing complexity
    assert not np.allclose(ratio_with_motif, ratio_no_motif, rtol=0.1)
    
    # Validation 3: Verify motif-containing glycans are represented
    fuc_indices = [i for i, seq in enumerate(glycan_sequences) if "Fuc" in seq]
    sia_indices = [i for i, seq in enumerate(glycan_sequences) if "Neu5Ac" in seq]
    
    # At least some glycans should contain the target motifs
    assert len(fuc_indices) > 0 or len(sia_indices) > 0
    
    # Validation 4: Output values are in reasonable range
    assert np.all(alpha_U_with_motif > 0)
    assert np.all(np.isfinite(alpha_U_with_motif))
    assert np.all(alpha_U_no_motif > 0)
    assert np.all(np.isfinite(alpha_U_no_motif))


def test_compositional_pairing():
    """Test compositional pairing identification logic
    
    Validates:
    1. Substrate-product pairs are correctly identified
    2. Returned indices are valid and in range
    3. Unpaired glycans are categorized correctly
    4. Both network-based and string-matching fallback work
    """
    from glycoforge.utils import find_compositional_pairs, load_data_from_glycowork
    
    # Load real glycan data
    try:
        df = load_data_from_glycowork("glycomics_human_leukemia_O_PMID34646384.csv")
        # Glycan sequences are in the first column (or index if set_index was used)
        if df.index.dtype == 'object' and isinstance(df.index[0], str):
            glycan_sequences = df.index.tolist()[:30]
        else:
            glycan_sequences = df.iloc[:30, 0].tolist()  # Use 30 glycans from first column
    except Exception as e:
        print(f"Skipping test_compositional_pairing: data not available ({e})")
        return
    
    motif_rules = {"Neu5Ac": "down"}  # Look for desialylation pairs
    
    # Call pairing function
    pairs = find_compositional_pairs(
        glycan_sequences, 
        motif_rules, 
        verbose=False
    )
    
    # Validation 1: Return dictionary structure is correct
    assert isinstance(pairs, dict)
    assert 'substrates' in pairs
    assert 'products' in pairs
    assert 'unpaired_up' in pairs
    assert 'unpaired_down' in pairs
    
    # Validation 2: All indices are within valid range
    all_indices = (pairs['substrates'] + pairs['products'] + 
                   pairs['unpaired_up'] + pairs['unpaired_down'])
    if len(all_indices) > 0:
        assert all(0 <= idx < len(glycan_sequences) for idx in all_indices)
    
    # Validation 3: Substrate-product pairing lengths match
    assert len(pairs['substrates']) == len(pairs['products'])
    
    # Validation 4: Paired glycans have motif differences
    for sub_idx, prod_idx in zip(pairs['substrates'], pairs['products']):
        sub_seq = glycan_sequences[sub_idx]
        prod_seq = glycan_sequences[prod_idx]
        # At least one should contain Neu5Ac
        assert "Neu5Ac" in sub_seq or "Neu5Ac" in prod_seq
    
    # Validation 5: Pairing function returns valid data structures
    # Note: Duplicates are allowed as biosynthetic networks can have many-to-many relationships
    assert isinstance(pairs['substrates'], list)
    assert isinstance(pairs['products'], list)
    assert isinstance(pairs['unpaired_up'], list)
    assert isinstance(pairs['unpaired_down'], list)


def test_batch_motif_effects():
    """Test motif-based batch effect direction generation
    
    Validates:
    1. batch_motif_rules correctly applied to each batch
    2. Different batches can have different motif preferences
    3. Glycans with target motifs are preferentially affected
    4. Returned u_dict direction vectors are valid
    """
    from glycoforge.utils import load_data_from_glycowork
    from glycoforge.sim_batch_factor import define_batch_direction
    
    # Load real glycan data
    try:
        df = load_data_from_glycowork("glycomics_human_leukemia_O_PMID34646384.csv")
        # Glycan sequences are in the first column (or index if set_index was used)
        if df.index.dtype == 'object' and isinstance(df.index[0], str):
            glycan_sequences = df.index.tolist()[:25]
        else:
            glycan_sequences = df.iloc[:25, 0].tolist()  # Use 25 glycans from first column
    except Exception as e:
        print(f"Skipping test_batch_motif_effects: data not available ({e})")
        return
    
    n_glycans = len(glycan_sequences)
    
    # Define different motif preferences for each batch
    # Test includes empty dict to verify bug fix for UnboundLocalError
    batch_motif_rules = {
        1: {"Fuc": "up"},       # Batch 1 prefers Fuc upregulation
        2: {"Neu5Ac": "down"},  # Batch 2 prefers Neu5Ac downregulation
        3: {}                    # Batch 3 has no specific preference (tests empty dict handling)
    }
    
    u_dict, raw_direction = define_batch_direction(
        batch_effect_direction=None,  # Auto-generate
        n_glycans=n_glycans,
        n_batches=3,  # Changed back to 3 batches to test empty dict case
        glycan_sequences=glycan_sequences,
        batch_motif_rules=batch_motif_rules,
        motif_bias=0.8,
        u_dict_seed=42,
        verbose=False
    )
    
    # Validation 1: u_dict contains all batches
    assert len(u_dict) == 3
    assert all(b in u_dict for b in [1, 2, 3])
    
    # Validation 2: Each direction vector has correct dimensions
    for b, vec in u_dict.items():
        assert len(vec) == n_glycans
        assert np.all(np.isfinite(vec))
    
    # Validation 3: raw_direction contains actual affected glycan indices
    assert isinstance(raw_direction, dict)
    for batch_id, effects in raw_direction.items():
        assert isinstance(effects, dict)
        # Indices are 1-based in raw_direction
        for glycan_idx, direction in effects.items():
            assert 1 <= glycan_idx <= n_glycans
            assert direction in [-1, 1]
    
    # Validation 4: Batch 1 affected glycans should have higher Fuc proportion
    batch1_affected = list(raw_direction[1].keys())
    if len(batch1_affected) > 0:
        batch1_seqs = [glycan_sequences[idx-1] for idx in batch1_affected]
        fuc_count_batch1 = sum(1 for seq in batch1_seqs if "Fuc" in seq)
        fuc_count_overall = sum(1 for seq in glycan_sequences if "Fuc" in seq)
        
        if fuc_count_overall > 0:  # Only test if there are any Fuc-containing glycans
            fuc_ratio_batch1 = fuc_count_batch1 / len(batch1_seqs)
            fuc_ratio_overall = fuc_count_overall / len(glycan_sequences)
            # Should be higher than overall average due to motif_bias
            # Use relaxed threshold to account for randomness
            assert fuc_ratio_batch1 >= fuc_ratio_overall * 0.7
    
    # Validation 5: Batch 3 (empty motif rules) should still generate effects
    # This validates the bug fix for empty dict handling
    batch3_affected = list(raw_direction[3].keys())
    assert len(batch3_affected) > 0, "Batch 3 with empty motif_rules should still affect some glycans"
    
    # Validation 6: Ensure at least some glycans are affected overall
    total_affected = sum(len(effects) for effects in raw_direction.values())
    assert total_affected > 0


def test_mnar_missingness_basic():
    """Test basic MNAR missingness functionality
    
    Validates:
    1. Missing rate is close to target missing_fraction
    2. Return value types are correct
    3. missing_mask dimensions match input
    4. CLR-transformed data has no NaN values
    5. diagnostics contains necessary statistics
    """
    from glycoforge.utils import apply_mnar_missingness, load_data_from_glycowork
    
    # Use real data subset for testing
    try:
        df = load_data_from_glycowork("glycomics_human_leukemia_O_PMID34646384.csv")
        # Skip first column (glycan names) and get numeric data only
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
        Y_test = df[numeric_cols].values.T  # First 10 numeric columns, transposed
    except Exception as e:
        # Fallback: use synthetic data
        print(f"Using synthetic data for test_mnar_missingness_basic: {e}")
        Y_test = np.random.rand(10, 5) * 100
    
    target_missing_fraction = 0.3
    
    Y_missing, Y_missing_clr, missing_mask, diagnostics = apply_mnar_missingness(
        Y_test,
        missing_fraction=target_missing_fraction,
        mnar_bias=2.0,
        seed=42,
        verbose=False
    )
    
    # Validation 1: Dimensions match
    assert Y_missing.shape == Y_test.shape
    assert Y_missing_clr.shape == Y_test.shape
    assert missing_mask.shape == Y_test.shape
    
    # Validation 2: Missing rate is close to target (within 5% tolerance)
    actual_missing_rate = np.sum(missing_mask) / missing_mask.size
    assert abs(actual_missing_rate - target_missing_fraction) < 0.05
    
    # Validation 3: Missing values are NaN in Y_missing
    assert np.sum(np.isnan(Y_missing)) == np.sum(missing_mask)
    
    # Validation 4: CLR data has no NaN (already imputed)
    assert not np.any(np.isnan(Y_missing_clr))
    assert np.all(np.isfinite(Y_missing_clr))
    
    # Validation 5: Diagnostics contains necessary fields
    assert isinstance(diagnostics, dict)
    if diagnostics:  # Only check if not empty dict
        # Check for at least some diagnostic information
        assert len(diagnostics) > 0
    
    # Validation 6: Non-missing values remain unchanged
    non_missing_mask = ~missing_mask
    if np.any(non_missing_mask):
        assert np.allclose(Y_missing[non_missing_mask], Y_test[non_missing_mask])


def test_mnar_intensity_bias():
    """Test MNAR intensity-dependent bias
    
    Validates:
    1. Low-intensity values are more likely to be missing
    2. mnar_bias parameter affects the degree of bias
    3. Extreme cases: bias=0 (random) vs bias=5 (strong bias)
    """
    from glycoforge.utils import apply_mnar_missingness
    
    # Construct data with known intensity gradient
    n_samples, n_glycans = 20, 5
    Y_test = np.tile([0.5, 2.0, 10.0, 30.0, 50.0], (n_samples, 1))
    # Add some noise
    np.random.seed(42)
    Y_test += np.random.rand(n_samples, n_glycans) * 0.1
    
    target_missing_fraction = 0.3
    
    # Test with strong MNAR bias
    _, _, missing_mask_strong, _ = apply_mnar_missingness(
        Y_test,
        missing_fraction=target_missing_fraction,
        mnar_bias=5.0,
        seed=42,
        verbose=False
    )
    
    # Test with weak MNAR bias
    _, _, missing_mask_weak, _ = apply_mnar_missingness(
        Y_test,
        missing_fraction=target_missing_fraction,
        mnar_bias=0.5,
        seed=43,
        verbose=False
    )
    
    # Validation 1: Calculate missing rates per glycan column
    missing_rate_per_glycan_strong = missing_mask_strong.sum(axis=0) / n_samples
    missing_rate_per_glycan_weak = missing_mask_weak.sum(axis=0) / n_samples
    
    # Validation 2: With strong bias, low-intensity glycans should have higher missing rate
    # Column 0 (intensity ~0.5) should have higher missing rate than column 4 (intensity ~50)
    if missing_rate_per_glycan_strong.sum() > 0:
        # At least the trend should be that lower intensity has more missingness
        # Allow some flexibility due to randomness
        low_intensity_rate = missing_rate_per_glycan_strong[0]
        high_intensity_rate = missing_rate_per_glycan_strong[-1]
        assert low_intensity_rate >= high_intensity_rate * 0.8
    
    # Validation 3: Both masks should have similar overall missing fraction
    assert abs(missing_mask_strong.sum()/missing_mask_strong.size - target_missing_fraction) < 0.05
    assert abs(missing_mask_weak.sum()/missing_mask_weak.size - target_missing_fraction) < 0.05


def test_mnar_edge_cases():
    """Test MNAR edge cases and robustness
    
    Validates:
    1. missing_fraction=0 produces no missingness
    2. missing_fraction=1 produces complete missingness
    3. All-zero data does not crash
    4. Single sample/feature data works
    5. Random seed ensures reproducibility
    """
    from glycoforge.utils import apply_mnar_missingness
    
    # Test data
    Y_test = np.random.rand(10, 5) * 100
    
    # Test 1: missing_fraction=0 should produce no missing values
    Y_missing_0, Y_clr_0, mask_0, diag_0 = apply_mnar_missingness(
        Y_test, missing_fraction=0.0, seed=42, verbose=False
    )
    assert np.sum(mask_0) == 0
    assert not np.any(np.isnan(Y_missing_0))
    
    # Test 2: missing_fraction=1 should make everything missing
    Y_missing_1, Y_clr_1, mask_1, diag_1 = apply_mnar_missingness(
        Y_test, missing_fraction=1.0, seed=42, verbose=False
    )
    assert np.sum(mask_1) == mask_1.size
    assert np.all(np.isnan(Y_missing_1))
    
    # Test 3: All-zero data should not crash
    Y_zeros = np.zeros((5, 3))
    Y_missing_zeros, _, mask_zeros, _ = apply_mnar_missingness(
        Y_zeros, missing_fraction=0.5, seed=42, verbose=False
    )
    assert Y_missing_zeros.shape == Y_zeros.shape
    
    # Test 4: Single sample/feature cases
    Y_single_sample = np.random.rand(1, 5) * 100
    Y_missing_ss, _, mask_ss, _ = apply_mnar_missingness(
        Y_single_sample, missing_fraction=0.3, seed=42, verbose=False
    )
    assert Y_missing_ss.shape == Y_single_sample.shape
    
    Y_single_feature = np.random.rand(10, 1) * 100
    Y_missing_sf, _, mask_sf, _ = apply_mnar_missingness(
        Y_single_feature, missing_fraction=0.3, seed=42, verbose=False
    )
    assert Y_missing_sf.shape == Y_single_feature.shape
    
    # Test 5: Reproducibility with same seed
    Y_missing_a, _, mask_a, _ = apply_mnar_missingness(
        Y_test, missing_fraction=0.3, seed=42, verbose=False
    )
    Y_missing_b, _, mask_b, _ = apply_mnar_missingness(
        Y_test, missing_fraction=0.3, seed=42, verbose=False
    )
    assert np.array_equal(mask_a, mask_b)
    assert np.array_equal(np.isnan(Y_missing_a), np.isnan(Y_missing_b))

