
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def combat(data, batch, mod=None, parametric=True):
    """
    ComBat for CLR/ALR glycomics data.
    data: glycans x samples
    batch: batch labels for each sample, a array-like of shape (n_samples,)
    mod: optional design matrix (e.g., biological group)
    parametric : whether to use parametric empirical Bayes (default True).
    """
    X = np.asarray(data).T  # samples x features
    n, p = X.shape
    batch = pd.Categorical(batch)

    # Design matrix (intercept if None)
    if mod is None:
      M = np.ones((n, 1))
    else:
      bio_dummies = pd.get_dummies(mod, drop_first=True).values
      M = np.hstack([np.ones((n, 1)), bio_dummies])

    # ----- Standardize -----
    B_hat = np.linalg.solve(M.T @ M, M.T @ X)
    grand_mean = M @ B_hat
    s_data = X - grand_mean
    sds = s_data.std(axis=0, ddof=1)
    s_data /= sds

    # ----- Estimate batch effects -----
    n_batch = len(batch.categories)
    gamma_hat = np.zeros((n_batch, p))
    delta_hat = np.zeros((n_batch, p))
    for i, b in enumerate(batch.categories):
        mask = batch == b
        s = s_data[mask]
        gamma_hat[i] = s.mean(0)
        delta_hat[i] = s.var(0, ddof=1)

    # ----- Parametric shrinkage (optional) -----
    if parametric:
        gamma_bar, t2 = gamma_hat.mean(0), gamma_hat.var(0, ddof=1)
        a_prior = (2 * t2 + delta_hat.mean(0)) / delta_hat.var(0)
        b_prior = (delta_hat.mean(0) * a_prior) / 2
        for i, b in enumerate(batch.categories):
            mask = batch == b
            n_b = mask.sum()
            ss = ((s_data[mask] - gamma_hat[i])**2).sum(0)
            delta_hat[i] = (b_prior + 0.5 * ss) / (a_prior + 0.5 * n_b + 1)

    # ----- Adjust -----
    X_adj = s_data.copy()
    for i, b in enumerate(batch.categories):
        mask = batch == b
        X_adj[mask] = ((s_data[mask] - gamma_hat[i]) / np.sqrt(delta_hat[i])) * sds + grand_mean[mask]

    return X_adj.T  # glycans x samples


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


def percentile_normalization(data, batch, reference_batch=None):
  X = np.asarray(data).T
  batch = pd.Categorical(batch)
  if reference_batch is None:
    reference_batch = batch.categories[0]
  ref_mask = batch == reference_batch
  ref_data = X[ref_mask]
  ref_percentiles = np.percentile(ref_data, np.arange(0, 101), axis=0)
  X_corrected = np.zeros_like(X)
  for i, b in enumerate(batch.categories):
    mask = batch == b
    batch_data = X[mask]
    for sample_idx in np.where(mask)[0]:
      sample = X[sample_idx]
      from scipy.stats import rankdata
      ranks = rankdata(sample, method='average')
      percentile_positions = (ranks - 1) / (len(sample) - 1) * 100
      corrected_sample = np.zeros_like(sample)
      for feature_idx in range(len(sample)):
        percentile_pos = percentile_positions[feature_idx]
        lower_idx = int(np.floor(percentile_pos))
        upper_idx = int(np.ceil(percentile_pos))
        if lower_idx == upper_idx:
          corrected_sample[feature_idx] = ref_percentiles[lower_idx, feature_idx]
        else:
          weight = percentile_pos - lower_idx
          corrected_sample[feature_idx] = (1 - weight) * ref_percentiles[lower_idx, feature_idx] + weight * ref_percentiles[upper_idx, feature_idx]
      X_corrected[sample_idx] = corrected_sample
  X_corrected = np.maximum(X_corrected, 1e-10)
  X_corrected = X_corrected / X_corrected.sum(axis=1, keepdims=True)
  return X_corrected.T


def ratio_preserving_combat(data, batch, mod=None, parametric=True):
  X = np.asarray(data).T
  n, p = X.shape
  batch = pd.Categorical(batch)
  X_safe = np.maximum(X, 1e-10)
  X_safe = X_safe / X_safe.sum(axis=1, keepdims=True)
  log_ratios = np.log(X_safe)
  log_ratios = log_ratios - log_ratios.mean(axis=1, keepdims=True)
  if mod is None:
    M = np.ones((n, 1))
  else:
    bio_dummies = pd.get_dummies(mod, drop_first=True).values
    M = np.hstack([np.ones((n, 1)), bio_dummies])
  B_hat = np.linalg.solve(M.T @ M, M.T @ log_ratios)
  grand_mean = M @ B_hat
  s_data = log_ratios - grand_mean
  sds = s_data.std(axis=0, ddof=1)
  sds = np.where(sds < 1e-10, 1.0, sds)
  s_data = s_data / sds
  n_batch = len(batch.categories)
  gamma_hat = np.zeros((n_batch, p))
  delta_hat = np.zeros((n_batch, p))
  for i, b in enumerate(batch.categories):
    mask = batch == b
    s = s_data[mask]
    gamma_hat[i] = s.mean(0)
    delta_hat[i] = s.var(0, ddof=1)
  if parametric:
    gamma_bar, t2 = gamma_hat.mean(0), gamma_hat.var(0, ddof=1)
    a_prior = (2 * t2 + delta_hat.mean(0)) / delta_hat.var(0)
    b_prior = (delta_hat.mean(0) * a_prior) / 2
    for i, b in enumerate(batch.categories):
      mask = batch == b
      n_b = mask.sum()
      ss = ((s_data[mask] - gamma_hat[i])**2).sum(0)
      delta_hat[i] = (b_prior + 0.5 * ss) / (a_prior + 0.5 * n_b + 1)
  X_adj = s_data.copy()
  for i, b in enumerate(batch.categories):
    mask = batch == b
    X_adj[mask] = ((s_data[mask] - gamma_hat[i]) / np.sqrt(delta_hat[i])) * sds + grand_mean[mask]
  X_corrected = np.exp(X_adj)
  X_corrected = np.maximum(X_corrected, 1e-10)
  X_corrected = X_corrected / X_corrected.sum(axis=1, keepdims=True)
  return X_corrected.T


def harmony_correction(data, batch, max_iter=10, sigma=0.1, n_clusters=None):
  """Harmony-style iterative correction in PCA space."""
  X = np.asarray(data).T
  batch = pd.Categorical(batch)
  n_batches = len(batch.categories)
  if n_clusters is None:
    n_clusters = min(50, X.shape[0]//2)
  pca = PCA(n_components=min(20, X.shape[0]-1, X.shape[1]))
  Z = pca.fit_transform(X)
  sigma = np.std(Z)
  for iteration in range(max_iter):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(Z)
    Z_corrected = Z.copy()
    for cluster_id in np.unique(clusters):
      cluster_mask = clusters == cluster_id
      if cluster_mask.sum() < 2:
        continue
      cluster_centers = []
      batch_counts = []
      for b in batch.categories:
        batch_cluster_mask = cluster_mask & (batch == b)
        if batch_cluster_mask.sum() > 0:
          cluster_centers.append(Z[batch_cluster_mask].mean(axis=0))
          batch_counts.append(batch_cluster_mask.sum())
      if len(cluster_centers) < 2:
        continue
      weights = np.array(batch_counts) / sum(batch_counts)
      global_center = np.average(cluster_centers, axis=0, weights=weights)
      for b in batch.categories:
        batch_cluster_mask = cluster_mask & (batch == b)
        if batch_cluster_mask.sum() > 0:
          batch_center = Z[batch_cluster_mask].mean(axis=0)
          correction = global_center - batch_center
          adaptive_sigma = sigma / (iteration + 1)
          Z_corrected[batch_cluster_mask] += adaptive_sigma * correction
    Z = Z_corrected
  X_corrected = pca.inverse_transform(Z)
  X_corrected = np.maximum(X_corrected, 0)
  row_sums = X_corrected.sum(axis=1, keepdims=True)
  row_sums = np.where(row_sums == 0, 1, row_sums)
  X_corrected = X_corrected / row_sums
  return X_corrected.T


def limma_style_correction(data, batch, mod=None):
  """Simple linear model batch correction like limma::removeBatchEffect."""
  X = np.asarray(data).T  # samples x features
  batch = pd.Categorical(batch)
  batch_dummies = pd.get_dummies(batch).values[:, 1:]  # Drop first batch as reference
  if mod is not None:
    bio_dummies = pd.get_dummies(mod, drop_first=True).values
    design = np.hstack([np.ones((X.shape[0], 1)), bio_dummies, batch_dummies])
    n_bio = 1 + bio_dummies.shape[1]
  else:
    design = np.hstack([np.ones((X.shape[0], 1)), batch_dummies])
    n_bio = 1
  coeffs = np.linalg.solve(design.T @ design, design.T @ X)
  batch_effects = design[:, n_bio:] @ coeffs[n_bio:]
  X_corrected = X - batch_effects
  X_corrected = np.maximum(X_corrected, 1e-10)
  X_corrected = X_corrected / X_corrected.sum(axis=1, keepdims=True)
  return X_corrected.T


def stratified_combat(data, batch, bio_group):
  """Apply ComBat separately within each biological group."""
  X = np.asarray(data).T
  batch = pd.Categorical(batch)
  bio_group = pd.Categorical(bio_group)
  X_corrected = np.zeros_like(X)
  for bio in bio_group.categories:
    bio_mask = bio_group == bio
    if bio_mask.sum() < 4:
      X_corrected[bio_mask] = X[bio_mask]
      continue
    group_batch = batch[bio_mask]
    n_unique_batches = len(np.unique(group_batch))
    if n_unique_batches < 2:
      X_corrected[bio_mask] = X[bio_mask]
      continue
    group_data = X[bio_mask]
    corrected = combat(group_data.T, group_batch).T
    X_corrected[bio_mask] = corrected
  X_corrected = np.maximum(X_corrected, 1e-10)
  X_corrected = X_corrected / X_corrected.sum(axis=1, keepdims=True)
  return X_corrected.T
