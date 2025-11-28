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
    
    alpha_H, alpha_U = define_dirichlet_params_from_real_data(
        p_h, effect_sizes, differential_mask,
        bio_strength=1.0,
        k_dir=10,
        verbose=False
    )
    
    assert alpha_H.shape == (3,)
    assert alpha_U.shape == (3,)
    assert np.all(alpha_H > 0)
    assert np.all(alpha_U > 0)

def test_apply_batch_effect():
    """Test batch effect application"""
    n_samples, n_glycans = 20, 10
    Y_clean = np.random.randn(n_samples, n_glycans)
    batch_labels = np.array([0]*7 + [1]*7 + [2]*6)
    
    u_dict = define_batch_direction(
        n_glycans=n_glycans, 
        n_batches=3, 
        seed=42, 
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

# def test_quantify_batch_effect_impact():
#     """Test batch effect quantification metrics"""
#     n_glycans, n_samples = 10, 30
#     np.random.seed(42)
    
#     # Create data with batch effects
#     data = np.random.randn(n_glycans, n_samples)
#     batch_labels = np.array([0]*10 + [1]*10 + [2]*10)
    
#     # Add batch effect
#     data[:5, :10] += 2  # Batch 0
#     data[:5, 20:30] -= 2  # Batch 2
    
#     df = pd.DataFrame(data)
#     bio_groups = {
#         'Healthy': list(range(15)),
#         'Unhealthy': list(range(15, 30))
#     }
    
#     metrics = quantify_batch_effect_impact(
#         df, batch_labels, bio_groups, verbose=False
#     )
    
#     assert 'silhouette' in metrics
#     assert 'kBET' in metrics
#     assert 'LISI' in metrics
#     assert 'ARI' in metrics
#     assert 'compositional_effect_size' in metrics
#     assert 'pca_batch_effect' in metrics
#     assert -1 <= metrics['silhouette'] <= 1

# def test_evaluate_biological_preservation():
#     """Test biological preservation metrics"""
#     n_glycans, n_samples = 10, 20
#     np.random.seed(42)
    
#     clean_data = pd.DataFrame(np.random.randn(n_glycans, n_samples))
#     corrected_data = clean_data + np.random.randn(n_glycans, n_samples) * 0.1
#     bio_labels = np.array([0]*10 + [1]*10)
    
#     preservation = evaluate_biological_preservation(
#         clean_data, corrected_data, bio_labels
#     )
    
#     assert 'biological_variability_preservation' in preservation
#     assert 'conserved_differential_proportion' in preservation
#     assert 0 <= preservation['biological_variability_preservation'] <= 1
#     assert 0 <= preservation['conserved_differential_proportion'] <= 1

# def test_check_batch_effect():
#     """Test batch effect checking"""
#     n_samples, n_features = 30, 10
#     np.random.seed(42)
    
#     data = pd.DataFrame(np.random.randn(n_features, n_samples))
#     batch_labels = np.array([0]*10 + [1]*10 + [2]*10)
#     bio_labels = np.array([0]*15 + [1]*15)
    
#     results, pc, var_batch = check_batch_effect(
#         data, batch_labels, bio_labels, verbose=False
#     )
    
#     assert 'pca_variance_explained' in results
#     assert 'batch_effect' in results
#     assert 'biological_effect' in results
#     assert 'severity' in results
#     assert pc.shape[0] == n_samples
#     assert len(var_batch) == n_features
