import json
import numpy as np
import os
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
from glycowork.motif.analysis import get_differential_expression
from glycoforge.sim_bio_factor import create_bio_groups
import contextlib
import io
import warnings


def pca_batch_effect(data, batch):
    """
    Calculate batch effect strength using PCA.
    Measures how well batches are separated in the first principal component.
    
    Returns:
    - Batch separation score: higher values indicate stronger batch effects
    - Ideal value after correction: close to 0
    """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    
    # Calculate batch centers in PC1 space
    batch_centers = []
    for batch_id in np.unique(batch):
        mask = batch == batch_id
        if np.sum(mask) > 0:  # Ensure batch has samples
            center = np.mean(pca_result[mask, 0])  # PC1 center for this batch
            batch_centers.append(center)
    
    # Batch separation = standard deviation of batch centers
    # Higher values = stronger batch effects
    batch_separation = np.std(batch_centers) if len(batch_centers) > 1 else 0.0
    
    return batch_separation


def silhouette(data, batch):
    return silhouette_score(data, batch)


def kBET(data, batch, k=25):
    """
    k-nearest neighbor Batch Effect Test (kBET).
    
    Parameters:
    - data: Input data matrix
    - batch: Batch labels
    - k: Number of nearest neighbors to consider
    
    Returns:
    - Mean chi-square statistic (higher values indicate stronger batch effects)
    """
    k = min(k, len(data) - 1)
    
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(data)
    distances, indices = nn.kneighbors(data)
    
    n_batches = len(np.unique(batch))
    batch_frequencies = np.zeros((len(batch), n_batches))
    
    for i, idx in enumerate(indices):
        batch_frequencies[i] = np.bincount(batch[idx], minlength=n_batches)

    expected_freq = np.mean(batch_frequencies, axis=0)
    expected_freq = np.where(expected_freq == 0, 1e-6, expected_freq)  # Avoid division by zero
    
    chi_square_stats = np.sum((batch_frequencies - expected_freq)**2 / expected_freq, axis=1)
    
    return np.mean(chi_square_stats)


def lisi(data, batch, perplexity=30):
    """
    Local Inverse Simpson's Index (LISI).
    
    Parameters:
    - data: Input data matrix
    - batch: Batch labels
    - perplexity: Perplexity parameter for distance calculations
    
    Returns:
    - Mean LISI score (higher values indicate better batch mixing)
    """

    perplexity = min(perplexity, len(data) - 1)
    
    distances = squareform(pdist(data))
    
    # Avoid division by zero
    distances = np.where(distances == 0, 1e-10, distances)
    
    P = np.exp(-distances / perplexity)

    # Normalize probabilities (exclude self)
    np.fill_diagonal(P, 0)
    P = P / np.sum(P, axis=1, keepdims=True)
    
    # One-hot encode batch labels
    batch_onehot = pd.get_dummies(batch).values
    lisi_scores = 1 / np.sum(np.dot(P, batch_onehot)**2, axis=1)
    
    return np.mean(lisi_scores)


def adjusted_rand_index(data, batch):
    """
    Adjusted Rand Index for measuring similarity between k-means clusters and batch labels
    K = # of batches
    
    Parameters:
    - data: Input data matrix
    - batch: Batch labels
    
    Returns:
    - ARI score between k-means clusters and batch labels
    """
    n_clusters = len(np.unique(batch))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    return adjusted_rand_score(batch, cluster_labels)


def compositional_effect_size(data, batch, is_clr=False):
    """
    Calculate compositional effect size measuring batch differences.
    
    Args:
        data: Input data matrix
        batch: Batch labels
        is_clr: Whether the input data is already CLR-transformed
    
    Returns:
        Mean effect size across all features
    """
    if not is_clr:
        # For compositional data, apply CLR transformation with numerical stability
        clr_data = np.log(data + 1e-6) - np.mean(np.log(data + 1e-6), axis=1, keepdims=True)
    else:
        # Already CLR-transformed data, use directly
        clr_data = data
    
    # Calculate batch means
    batch_means = [np.mean(clr_data[batch == b], axis=0) for b in np.unique(batch)]
    batch_means = np.array(batch_means)
    
    # Calculate effect sizes (max - min across batches for each feature)
    effect_sizes = np.max(batch_means, axis=0) - np.min(batch_means, axis=0)
    
    # Return mean effect size, handling potential NaN values
    result = np.mean(effect_sizes)
    return result if not np.isnan(result) else 0.0


def preservation_of_biological_variability(data_before, data_after, biological_groups):
    """
    Calculate preservation of biological variability after batch correction.
    
    Parameters:
    - data_before: Data before correction
    - data_after: Data after correction
    - biological_groups: Biological group labels
    
    Returns:
    - Correlation between F-statistics before and after correction
    """
    def f_statistic(data, groups):
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if n_groups < 2:
            return np.zeros(data.shape[1])
        
        group_means = [np.mean(data[groups == g], axis=0) for g in unique_groups]
        overall_mean = np.mean(data, axis=0)
        
        # Between-group variance
        between_group_var = np.sum([len(data[groups == g]) * (gm - overall_mean)**2 
                                   for g, gm in zip(unique_groups, group_means)], axis=0) / (n_groups - 1)
        
        # Within-group variance
        within_group_vars = [np.var(data[groups == g], axis=0) for g in unique_groups]
        within_group_var = np.mean(within_group_vars, axis=0)
        
        # Avoid division by zero
        within_group_var = np.where(within_group_var == 0, 1e-10, within_group_var)
        
        return between_group_var / within_group_var
    
    f_before = f_statistic(data_before, biological_groups)
    f_after = f_statistic(data_after, biological_groups)
    
    # Handle edge cases
    if len(f_before) < 2 or np.all(f_before == f_before[0]) or np.all(f_after == f_after[0]):
        return 0.0
    
    try:
        correlation = np.corrcoef(f_before, f_after)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    except:
        return 0.0


def proportion_conserved_differential(data_before, data_after, threshold=0.05):
    """
    Calculate proportion of conserved differential relationships.
    
    Parameters:
    - data_before: Data before batch effect correction
    - data_after: Data after correction
    - threshold: Correlation threshold for significant relationships
    
    Returns:
    - Proportion of conserved relationships
    """
    def differential_relationships(data):
        corr = np.corrcoef(data.T)
        upper_triangle = np.triu(np.ones_like(corr, dtype=bool), k=1)
        return corr[upper_triangle] > threshold
    
    diff_before = differential_relationships(data_before)
    diff_after = differential_relationships(data_after)
    
    if len(diff_before) == 0 or np.sum(diff_before) == 0:
        return 0.0
    
    return np.sum(diff_before & diff_after) / np.sum(diff_before)




def compare_differential_expression(dataset1=None, #simulated clean data
                                    dataset2=None, #batch-affected data
                                    dataset3=None, #combat corrected data
                                   dataset1_path=None, 
                                   dataset2_path=None, 
                                   dataset3_path=None,
                                   bio_group_prefixes={'Healthy': ['healthy'], 'Unhealthy': ['unhealthy']}, #dict, biological group prefix mapping
                                   verbose=True):
    
    # Extract dataset names 
    if dataset1_path:
        dataset1_name = os.path.splitext(os.path.basename(dataset1_path))[0]
    else:
        dataset1_name = "dataset1_memory"
        
    if dataset2_path:
        dataset2_name = os.path.splitext(os.path.basename(dataset2_path))[0]
    else:
        dataset2_name = "dataset2_memory"
        
    if dataset3_path:
        dataset3_name = os.path.splitext(os.path.basename(dataset3_path))[0]
    elif dataset3 is not None:
        dataset3_name = "dataset3_memory"
    else:
        dataset3_name = None
    
    # Helper function to load and format dataset
    def load_dataset(path):
        try:
            # Try loading with glycan_index as index first
            df = pd.read_csv(path, index_col=0)
            # Check if this looks like a data matrix (no glycan column)
            if 'glycan' not in df.columns:
                # Add glycan column for consistency with get_differential_expression
                df_orig = pd.read_csv("glycowork/glycan_data/glycomics_human_leukemia_O_PMID34646384.csv")
                df_formatted = pd.DataFrame({'glycan': df_orig['glycan'].tolist()[:len(df)]})
                for col in df.columns:
                    df_formatted[col] = df[col].values
                return df_formatted
            else:
                return df
        except:
            # Fallback to regular CSV loading
            return pd.read_csv(path)
    
    # Load datasets - use provided DataFrames or load from paths
    if dataset1 is not None:
        df1 = dataset1.copy()
        # Ensure glycan column exists for get_differential_expression compatibility
        if 'glycan' not in df1.columns:
            df1 = df1.reset_index()
            if df1.columns[0] != 'glycan':
                df1.rename(columns={df1.columns[0]: 'glycan'}, inplace=True)
    else:
        df1 = load_dataset(dataset1_path)
        
    if dataset2 is not None:
        df2 = dataset2.copy()
        # Ensure glycan column exists for get_differential_expression compatibility
        if 'glycan' not in df2.columns:
            df2 = df2.reset_index()
            if df2.columns[0] != 'glycan':
                df2.rename(columns={df2.columns[0]: 'glycan'}, inplace=True)
    else:
        df2 = load_dataset(dataset2_path)
        
    if dataset3 is not None:
        df3 = dataset3.copy()
        # Ensure glycan column exists for get_differential_expression compatibility
        if 'glycan' not in df3.columns:
            df3 = df3.reset_index()
            if df3.columns[0] != 'glycan':
                df3.rename(columns={df3.columns[0]: 'glycan'}, inplace=True)
    elif dataset3_path:
        df3 = load_dataset(dataset3_path)
    else:
        df3 = None
    
    # Validate shapes and structure
    if df1.shape[0] != df2.shape[0]:
        raise ValueError(f"Dataset shape mismatch: {df1.shape[0]} vs {df2.shape[0]} glycans")
    if df3 is not None and df1.shape[0] != df3.shape[0]:
        raise ValueError(f"Dataset shape mismatch: {df1.shape[0]} vs {df3.shape[0]} glycans")
    
    # Create groups for differential expression analysis 
    bio_groups_dict, _ = create_bio_groups(df1, bio_group_prefixes) 
    groups1 = bio_groups_dict  
    groups2 = bio_groups_dict  
    groups3 = bio_groups_dict if df3 is not None else None 
    
    # Validate sample names are consistent 
    if not df1.columns.equals(df2.columns):
        raise ValueError("Sample column names must be identical between dataset1 and dataset2")
    if df3 is not None and not df1.columns.equals(df3.columns):
        raise ValueError("Sample column names must be identical between dataset1 and dataset3")    
    if verbose:
        print("=" * 60)
        print("COMPARE_DIFFERENTIAL_EXPRESSION")
        print("=" * 60)
        print(f"Dataset 1 data range: [{df1.iloc[:, 1:].min().min():.3f}, {df1.iloc[:, 1:].max().max():.3f}]")
        print(f"Dataset 2 data range: [{df2.iloc[:, 1:].min().min():.3f}, {df2.iloc[:, 1:].max().max():.3f}]")
        if df3 is not None:
            print(f"Dataset 3 data range: [{df3.iloc[:, 1:].min().min():.3f}, {df3.iloc[:, 1:].max().max():.3f}]")
            print(f"Dataset 3 zero/negative values: {(df3.iloc[:, 1:] <= 0).sum().sum()}")
    
    # Perform differential expression analysis
    # Capture glycowork output messages and warnings
    captured_messages = []
    captured_warnings = []
    
    f1 = io.StringIO()
    with contextlib.redirect_stdout(f1), warnings.catch_warnings(record=True) as w1:
        warnings.simplefilter("always")
        res1 = get_differential_expression(df=df1, group1=groups1['Healthy'], group2=groups1['Unhealthy'], transform="CLR", motifs=False)
    msg1 = f1.getvalue().strip()
    if msg1:
        captured_messages.append(f"Dataset1 ({dataset1_name}): {msg1}")
    if w1:
        for warning in w1:
            captured_warnings.append(f"Dataset1 ({dataset1_name}): {warning.category.__name__}: {warning.message}")
    
    f2 = io.StringIO()
    with contextlib.redirect_stdout(f2), warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        res2 = get_differential_expression(df=df2, group1=groups2['Healthy'], group2=groups2['Unhealthy'], transform="CLR", motifs=False)
    msg2 = f2.getvalue().strip()
    if msg2:
        captured_messages.append(f"Dataset2 ({dataset2_name}): {msg2}")
    if w2:
        for warning in w2:
            captured_warnings.append(f"Dataset2 ({dataset2_name}): {warning.category.__name__}: {warning.message}")
    
    if df3 is not None:
        f3 = io.StringIO()
        with contextlib.redirect_stdout(f3), warnings.catch_warnings(record=True) as w3:
            warnings.simplefilter("always")
            res3 = get_differential_expression(df=df3, group1=groups3['Healthy'], group2=groups3['Unhealthy'], transform="CLR", motifs=False)
        msg3 = f3.getvalue().strip()
        if msg3:
            captured_messages.append(f"Dataset3 ({dataset3_name}): {msg3}")
        if w3:
            for warning in w3:
                captured_warnings.append(f"Dataset3 ({dataset3_name}): {warning.category.__name__}: {warning.message}")
    
    if verbose:
        if captured_messages:
            print("\nDifferential expression analysis messages:")
            for msg in captured_messages:
                print(f"  {msg}")
        if captured_warnings:
            print("\nWarnings during analysis:")
            for warn in captured_warnings:
                print(f"  {warn}")
    
    # Extract significant glycan indices
    sig1 = sorted([idx + 1 for idx in res1[res1['significant'] == True].index])
    sig2 = sorted([idx + 1 for idx in res2[res2['significant'] == True].index])
    if df3 is not None:
        sig3 = sorted([idx + 1 for idx in res3[res3['significant'] == True].index])
    
    # Calculate overlaps and differences for dataset1 vs dataset2
    overlap_1v2 = sorted(set(sig1) & set(sig2))
    lost_signals_1v2 = sorted(set(sig1) - set(sig2))
    gained_signals_1v2 = sorted(set(sig2) - set(sig1))
    
    # Compile results
    results = {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "dataset1_name": dataset1_name,
            "dataset2_name": dataset2_name,
            "dataset1_path": dataset1_path if dataset1_path else "memory",
            "dataset2_path": dataset2_path if dataset2_path else "memory",
            "total_glycans": len(res1),
            "bio_groups": {k: len(v) for k, v in groups1.items()},
            "analysis_messages": captured_messages if captured_messages else [],
            "warnings": captured_warnings if captured_warnings else []
        },
        "results": {
            "dataset1": {
                "significant_count": len(sig1),
                "significant_indices": sig1
            },
            "dataset2": {
                "significant_count": len(sig2),
                "significant_indices": sig2
            },
            "compare_1v2": {
                "batch_effect_errors": {
                    "tp_count": len(overlap_1v2),
                    "tp_indices": overlap_1v2,
                    "fp_count": len(gained_signals_1v2),
                    "fp_indices": gained_signals_1v2,
                    "fn_count": len(lost_signals_1v2),
                    "fn_indices": lost_signals_1v2
                }
            }
        }
    }
    
    # Add third dataset comparison if provided
    if df3 is not None:
        overlap_1v3 = sorted(set(sig1) & set(sig3))
        lost_signals_1v3 = sorted(set(sig1) - set(sig3))
        gained_signals_1v3 = sorted(set(sig3) - set(sig1))
        
        
        results["metadata"]["dataset3_name"] = dataset3_name
        results["metadata"]["dataset3_path"] = dataset3_path if dataset3_path else "memory"
        results["results"]["dataset3"] = {
            "significant_count": len(sig3),
            "significant_indices": sig3
        }
        results["results"]["compare_1v3"] = {
            "after_correction_errors": {
                "tp_count": len(overlap_1v3),
                "tp_indices": overlap_1v3,
                "fp_count": len(gained_signals_1v3),
                "fp_indices": gained_signals_1v3,
                "fn_count": len(lost_signals_1v3),
                "fn_indices": lost_signals_1v3
            }
        }
    
    # Print summary
    if verbose:
        print(f"Dataset 1 ({dataset1_name}): {len(sig1)}/{len(res1)} significant glycans")
        print(f"Dataset 2 ({dataset2_name}): {len(sig2)}/{len(res2)} significant glycans")
        print(f"Compare 1vs2 - Batch Effect Errors:")
        print(f"  TP: {len(overlap_1v2)} glycans, FP: {len(gained_signals_1v2)} glycans, FN: {len(lost_signals_1v2)} glycans")
        
        if df3 is not None:
            print(f"Dataset 3 ({dataset3_name}): {len(sig3)}/{len(res3)} significant glycans")
            print(f"Compare 1vs3 - After Correction Errors:")
            print(f"  TP: {len(overlap_1v3)} glycans, FP: {len(gained_signals_1v3)} glycans, FN: {len(lost_signals_1v3)} glycans")
        print("=" * 60)
    
    return results



def quantify_batch_effect_impact(Y_with_batch_clr, #DataFrame (glycans x samples) in CLR space with batch effects
                                  batch_labels,  # Array of batch labels for each sample (0-based)
                                  bio_groups, # Dict of biological groups {'Healthy': [...], 'Unhealthy': [...]}
                                  verbose=True
                                  ): 
    # Transpose to (samples x features) format required by metrics
    data_T = Y_with_batch_clr.T.values
    
    # Store results
    metrics = {}
    
    if verbose:
        print("=" * 60)
        print("QUANTIFY_BATCH_EFFECT_IMPACT")
        print("=" * 60)
        print(f"Data shape: {data_T.shape} (samples x features)")
        print(f"Batch labels: {np.unique(batch_labels)}")
        print(f"Batch distribution: {np.bincount(batch_labels)}")
        if bio_groups:
            print(f"Biological groups: {[f'{k}({len(v)})' for k, v in bio_groups.items()]}")
        print("-" * 60)
    
    # 1. Silhouette Score (batch separation)
    silhouette_score = silhouette(data_T, batch_labels)
    metrics['silhouette'] = silhouette_score
    if verbose:
        print("1. Silhouette Score (higher = close to +1 = stronger batch effects)")
        print(f"   Silhouette Score: {silhouette_score:.4f}")
    
    # 2. kBET (k-nearest neighbor Batch Effect Test)
    k_neighbors = min(25, len(data_T) - 1)
    kbet_score = kBET(data_T, batch_labels, k=k_neighbors)
    metrics['kBET'] = kbet_score
    if verbose:
        print("2. kBET (higher = close to +1 = stronger batch effects)")
        print(f"   kBET Score: {kbet_score:.4f}")
    
    # 3. LISI (Local Inverse Simpson's Index)
    perplexity_val = min(30, len(data_T) - 1)
    lisi_score = lisi(data_T, batch_labels, perplexity=perplexity_val)
    metrics['LISI'] = lisi_score
    if verbose:
        print("3. LISI (higher = Close to # of batch = better local batch mixing)")
        print(f"   LISI Score: {lisi_score:.4f}")
    
    # 4. Adjusted Rand Index (clustering agreement with batches)
    ari_score = adjusted_rand_index(data_T, batch_labels)
    metrics['ARI'] = ari_score
    if verbose:
        print("4. Adjusted Rand Index (higher = close to +1 = stronger batch effects)")
        print(f"   ARI Score: {ari_score:.4f}")
    
    # 5. Compositional Effect Size
    comp_effect = compositional_effect_size(data_T, batch_labels, is_clr=True)
    metrics['compositional_effect_size'] = comp_effect
    if verbose:
        print("5. Compositional Effect Size (higher > 1.5 stronger batch effects)")
        print(f"   Compositional Effect Size: {comp_effect:.4f}")
    
    # 6. PCA Batch Effect
    pca_batch_score = pca_batch_effect(data_T, batch_labels)
    metrics['pca_batch_effect'] = pca_batch_score
    if verbose:
        print("6. PCA Batch Effect (higher = stronger batch effects in PCA space)")
        print(f"   PCA Batch Effect: {pca_batch_score:.4f}")
        print("=" * 60)
    
    return metrics




def evaluate_biological_preservation(clean_data, with_batch_data, corrected_data, bio_labels):
    
    # Calculate Y_with_batch vs Y_clean (baseline corruption)
    bio_var_batch_vs_clean = preservation_of_biological_variability(
        data_before=clean_data.T.values,
        data_after=with_batch_data.T.values,
        biological_groups=bio_labels
    )
    
    conserved_batch_vs_clean = proportion_conserved_differential(
        data_before=clean_data.T.values,
        data_after=with_batch_data.T.values,
        threshold=0.05
    )
    
    # Calculate Y_corrected vs Y_clean (correction effectiveness)
    bio_var_corrected_vs_clean = preservation_of_biological_variability(
        data_before=clean_data.T.values,
        data_after=corrected_data.T.values,
        biological_groups=bio_labels
    )
    
    conserved_corrected_vs_clean = proportion_conserved_differential(
        data_before=clean_data.T.values,
        data_after=corrected_data.T.values,
        threshold=0.05
    )
    
    return {
        'batch_vs_clean': {
            'description': 'Y_with_batch vs Y_clean (baseline corruption level)',
            'biological_variability': bio_var_batch_vs_clean,
            'conserved_differential': conserved_batch_vs_clean
        },
        'corrected_vs_clean': {
            'description': 'Y_corrected vs Y_clean (after batch correction)',
            'biological_variability': bio_var_corrected_vs_clean,
            'conserved_differential': conserved_corrected_vs_clean
        }
    }


def generate_comprehensive_metrics(seed, output_dir, 
                                 batch_metrics_before,
                                 batch_metrics_after,
                                 diff_expr_results,
                                 metadata,
                                 batch_check_results=None):
    
    # Build run_info
    run_info = {
        "seed": seed,
        "analysis_timestamp": datetime.now().isoformat(),
        "output_dir": output_dir
    }
    
    # Build simulation_config directly from metadata
    simulation_config = {
        "data_source": metadata.get('data_source', 'simulated'),
        "bio_parameters": metadata.get('bio_parameters', {}),
        "batch_parameters": metadata.get('batch_parameters', {})
    }
    
    # Add hybrid mode specific parameters
    if metadata.get('data_source') == 'real':
        simulation_config['data_file'] = metadata.get('data_file')
        simulation_config['use_real_effect_sizes'] = metadata.get('use_real_effect_sizes', False)
        if metadata.get('column_prefix'):
            simulation_config['column_prefix'] = metadata.get('column_prefix')
    
    bio_preservation = batch_metrics_after.pop('bio_preservation', {})
    
    # Calculate improvement metrics for all 6 batch correction metrics
    improvement = {}
    
    # Metrics where lower is better (reduction is improvement)
    lower_better_metrics = ['silhouette', 'kBET', 'ARI', 'compositional_effect_size', 'pca_batch_effect']
    for metric in lower_better_metrics:
        if metric in batch_metrics_before and metric in batch_metrics_after:
            before_val = batch_metrics_before[metric]
            after_val = batch_metrics_after[metric]
            if before_val != 0:
                reduction = ((before_val - after_val) / before_val) * 100
                improvement[f'{metric}_reduction_pct'] = round(reduction, 2)
    
    # Metrics where higher is better (increase is improvement)
    if 'LISI' in batch_metrics_before and 'LISI' in batch_metrics_after:
        before_val = batch_metrics_before['LISI']
        after_val = batch_metrics_after['LISI']
        if before_val != 0:
            increase = ((after_val - before_val) / before_val) * 100
            improvement['LISI_increase_pct'] = round(increase, 2)
    
    # Build correction_results
    correction_results = {
        "batch_correction": {
            "before": batch_metrics_before.copy(),
            "after": batch_metrics_after.copy(),
            "improvement": improvement
        }
    }
    
    if bio_preservation:
        correction_results['bio_preservation'] = bio_preservation
    
    # Process differential expression results
    if diff_expr_results:
        de_results = diff_expr_results.get('results', {})
        de_processed = {
            "Y_clean": {
                "significant_count": de_results.get('dataset1', {}).get('significant_count', 0),
                "significant_glycans": de_results.get('dataset1', {}).get('significant_indices', [])
            },
            "Y_with_batch": {
                "significant_count": de_results.get('dataset2', {}).get('significant_count', 0),
                "significant_glycans": de_results.get('dataset2', {}).get('significant_indices', [])
            },
            "Y_corrected": {
                "significant_count": de_results.get('dataset3', {}).get('significant_count', 0),
                "significant_glycans": de_results.get('dataset3', {}).get('significant_indices', [])
            }
        }
        
        # Process batch impact
        batch_errors = de_results.get('compare_1v2', {}).get('batch_effect_errors', {})
        y_clean_count = de_results.get('dataset1', {}).get('significant_count', 0)
        y_batch_count = de_results.get('dataset2', {}).get('significant_count', 0)
        total_glycans = diff_expr_results.get('metadata', {}).get('total_glycans', 98)
        
        # Calculate confusion matrix for batch impact
        # TP = overlap (Y_clean ∩ Y_batch), FP = gained (Y_batch - Y_clean)
        # FN = lost (Y_clean - Y_batch), TN = total - TP - FP - FN
        batch_tp_count = batch_errors.get('tp_count', 0)
        batch_tp_indices = batch_errors.get('tp_indices', [])
        batch_fp_count = batch_errors.get('fp_count', 0)
        batch_fp_indices = batch_errors.get('fp_indices', [])
        batch_fn_count = batch_errors.get('fn_count', 0)
        batch_fn_indices = batch_errors.get('fn_indices', [])
        batch_tn_count = total_glycans - batch_tp_count - batch_fp_count - batch_fn_count
        
        # Calculate TN indices: all glycans except TP, FP, FN
        all_glycans = set(range(1, total_glycans + 1))
        batch_tn_indices = sorted(list(all_glycans - set(batch_tp_indices) - set(batch_fp_indices) - set(batch_fn_indices)))
        
        # Calculate performance metrics for Y_with_batch
        batch_precision = batch_tp_count / max(batch_tp_count + batch_fp_count, 1)
        batch_recall = batch_tp_count / max(batch_tp_count + batch_fn_count, 1)
        batch_f1 = 2 * batch_precision * batch_recall / max(batch_precision + batch_recall, 1e-10)
        
        # Add confusion matrix to Y_with_batch (preserve significant_count and significant_glycans)
        de_processed['Y_with_batch'].update({
            "true_positive": {
                "count": batch_tp_count,
                "rate_pct": round(batch_tp_count / max(y_clean_count, 1) * 100, 2),
                "indices": batch_tp_indices
            },
            "false_positive": {
                "count": batch_fp_count,
                "rate_pct": round(batch_fp_count / max(total_glycans, 1) * 100, 2),
                "indices": batch_fp_indices
            },
            "true_negative": {
                "count": batch_tn_count,
                "rate_pct": round(batch_tn_count / max(total_glycans, 1) * 100, 2),
                "indices": batch_tn_indices
            },
            "false_negative": {
                "count": batch_fn_count,
                "rate_pct": round(batch_fn_count / max(y_clean_count, 1) * 100, 2),
                "indices": batch_fn_indices
            },
            "f1_score": round(batch_f1, 4),
            "precision": round(batch_precision, 4),
            "recall": round(batch_recall, 4)
        })
        
        # Process correction recovery
        correction_errors = de_results.get('compare_1v3', {}).get('after_correction_errors', {})
        y_corrected_count = de_results.get('dataset3', {}).get('significant_count', 0)
        
        # Calculate confusion matrix for correction recovery
        # TP = overlap (Y_clean ∩ Y_corrected), FP = gained (Y_corrected - Y_clean)
        # FN = lost (Y_clean - Y_corrected), TN = total - TP - FP - FN
        corrected_tp_count = correction_errors.get('tp_count', 0)
        corrected_tp_indices = correction_errors.get('tp_indices', [])
        corrected_fp_count = correction_errors.get('fp_count', 0)
        corrected_fp_indices = correction_errors.get('fp_indices', [])
        corrected_fn_count = correction_errors.get('fn_count', 0)
        corrected_fn_indices = correction_errors.get('fn_indices', [])
        corrected_tn_count = total_glycans - corrected_tp_count - corrected_fp_count - corrected_fn_count
        
        # Calculate TN indices: all glycans except TP, FP, FN
        corrected_tn_indices = sorted(list(all_glycans - set(corrected_tp_indices) - set(corrected_fp_indices) - set(corrected_fn_indices)))
        
        # Calculate performance metrics for Y_after_correction
        corrected_precision = corrected_tp_count / max(corrected_tp_count + corrected_fp_count, 1)
        corrected_recall = corrected_tp_count / max(corrected_tp_count + corrected_fn_count, 1)
        corrected_f1 = 2 * corrected_precision * corrected_recall / max(corrected_precision + corrected_recall, 1e-10)
        
        # Add confusion matrix to Y_after_correction (preserve significant_count and significant_glycans)
        de_processed['Y_after_correction'] = de_processed.get('Y_corrected', {}).copy()
        de_processed['Y_after_correction'].update({
            "true_positive": {
                "count": corrected_tp_count,
                "rate_pct": round(corrected_tp_count / max(y_clean_count, 1) * 100, 2),
                "indices": corrected_tp_indices
            },
            "false_positive": {
                "count": corrected_fp_count,
                "rate_pct": round(corrected_fp_count / max(total_glycans, 1) * 100, 2),
                "indices": corrected_fp_indices
            },
            "true_negative": {
                "count": corrected_tn_count,
                "rate_pct": round(corrected_tn_count / max(total_glycans, 1) * 100, 2),
                "indices": corrected_tn_indices
            },
            "false_negative": {
                "count": corrected_fn_count,
                "rate_pct": round(corrected_fn_count / max(y_clean_count, 1) * 100, 2),
                "indices": corrected_fn_indices
            },
            "f1_score": round(corrected_f1, 4),
            "precision": round(corrected_precision, 4),
            "recall": round(corrected_recall, 4)
        })
        
        # Process summary with all change rates
        overall = de_results.get('overall', {})
        
        # Calculate change rates for all confusion matrix elements
        tp_change = (corrected_tp_count - batch_tp_count) / max(batch_tp_count, 1) * 100
        fp_change = (corrected_fp_count - batch_fp_count) / max(batch_fp_count, 1) * 100
        tn_change = (corrected_tn_count - batch_tn_count) / max(batch_tn_count, 1) * 100
        fn_change = (corrected_fn_count - batch_fn_count) / max(batch_fn_count, 1) * 100
        
        # Calculate change rates for derived metrics
        f1_change = (corrected_f1 - batch_f1) / max(batch_f1, 1e-10) * 100
        precision_change = (corrected_precision - batch_precision) / max(batch_precision, 1e-10) * 100
        recall_change = (corrected_recall - batch_recall) / max(batch_recall, 1e-10) * 100
        
        de_processed['summary'] = {
            "true_positive_change_pct": round(tp_change, 2),
            "false_positive_change_pct": round(fp_change, 2),
            "true_negative_change_pct": round(tn_change, 2),
            "false_negative_change_pct": round(fn_change, 2),
            "f1_score_change_pct": round(f1_change, 2),
            "precision_change_pct": round(precision_change, 2),
            "recall_change_pct": round(recall_change, 2)
        }
        
        correction_results['differential_expression'] = de_processed
    
    # Add quality_checks if available
    if batch_check_results:
        correction_results["quality_checks"] = {
            "Y_with_batch": batch_check_results
        }
    
    # Assemble final structure
    correction_data = {
        "run_info": run_info,
        "simulation_config": simulation_config,
        "correction_results": correction_results
    }
    
    output_file = f"{output_dir}/correction_metrics_seed{seed}.json"
    with open(output_file, 'w') as f:
        json.dump(correction_data, f, indent=2, ensure_ascii=False)
    
    print(f"Correction metrics saved to: {output_file}")
    
    return correction_data
