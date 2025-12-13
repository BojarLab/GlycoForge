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
    
    u_dict = define_batch_direction(
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

