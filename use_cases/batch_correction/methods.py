
import numpy as np
import pandas as pd

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
    M = np.ones((n, 1)) if mod is None else np.asarray(pd.get_dummies(mod), float)

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

