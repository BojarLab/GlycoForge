import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import f_oneway, shapiro, kruskal

def check_batch_effect(data, 
                       batch_labels, 
                       bio_groups=None,
                       verbose=True
                       ):
    
    X = np.asarray(data).T
    pca = PCA(n_components=min(5, X.shape[0]-1))
    pc = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_[:2].sum()
    
    batch_cat = pd.Categorical(batch_labels)
    pc1_by_batch = [pc[batch_cat==b, 0] for b in batch_cat.categories]
    
    total_samples = len(pc[:, 0])
    if total_samples < 30:
        f_stat, p_val = kruskal(*pc1_by_batch)
        test_used = "Kruskal-Wallis"
    else:
        _, norm_p = shapiro(pc[:, 0])
        if norm_p < 0.05:
            f_stat, p_val = kruskal(*pc1_by_batch)
            test_used = "Kruskal-Wallis"
        else:
            f_stat, p_val = f_oneway(*pc1_by_batch)
            test_used = "ANOVA"
    
    if test_used == "ANOVA":
        ss_total = np.var(pc[:, 0]) * (len(pc[:, 0]) - 1)
        ss_between = sum([len(group) * (np.mean(group) - np.mean(pc[:, 0]))**2 
                         for group in pc1_by_batch])
        batch_eta = ss_between / ss_total if ss_total > 0 else 0
    else:
        batch_eta = (f_stat - len(batch_cat.categories) + 1) / (total_samples - len(batch_cat.categories) + 1)
        batch_eta = max(0, min(1, batch_eta))
    
    if verbose:
        print(f"PC1-2 explain {var_explained:.1%} variance")
        print(f"Batch effect on PC1: F={f_stat:.2f}, p={p_val:.3e} ({test_used})")
        print(f"Batch effect size (eta²): {batch_eta:.1%}")
    
    results = {
        'pca_variance_explained': float(var_explained),
        'batch_effect': {
            'f_statistic': float(f_stat),
            'p_value': float(p_val),
            'test_used': test_used,
            'effect_size_eta2': float(batch_eta)
        }
    }
    
    severity = None
    severity_description = None
    
    if bio_groups is not None:
        bio_cat = pd.Categorical(bio_groups)
        pc1_by_bio = [pc[bio_cat==g, 0] for g in bio_cat.categories]
        
        if total_samples < 30 or (total_samples >= 30 and norm_p < 0.05):
            f_bio, p_bio = kruskal(*pc1_by_bio)
        else:
            f_bio, p_bio = f_oneway(*pc1_by_bio)
        
        if test_used == "ANOVA":
            ss_bio_between = sum([len(group) * (np.mean(group) - np.mean(pc[:, 0]))**2 
                                 for group in pc1_by_bio])
            bio_eta = ss_bio_between / ss_total if ss_total > 0 else 0
        else:
            bio_eta = (f_bio - len(bio_cat.categories) + 1) / (total_samples - len(bio_cat.categories) + 1)
            bio_eta = max(0, min(1, bio_eta))
        
        if verbose:
            print(f"Biological effect on PC1: F={f_bio:.2f}, p={p_bio:.3e}")
            print(f"Biological effect size (eta²): {bio_eta:.1%}")
        
        results['biological_effect'] = {
            'f_statistic': float(f_bio),
            'p_value': float(p_bio),
            'effect_size_eta2': float(bio_eta)
        }
        
        if p_val < 0.05 and p_bio < 0.05:
            if batch_eta > bio_eta + 0.1:
                if batch_eta > 0.3:
                    severity = "CRITICAL"
                elif batch_eta > 0.2:
                    severity = "MODERATE"
                else:
                    severity = "MILD"
                severity_description = f"Batch effect ({batch_eta:.1%}) stronger than biological signal ({bio_eta:.1%})"
                if verbose:
                    print(f"Warning: {severity_description} - {severity}")
            else:
                severity = "GOOD"
                severity_description = f"Biological signal ({bio_eta:.1%}) stronger than batch effect ({batch_eta:.1%})"
                if verbose:
                    print(f"Good: {severity_description}")
        elif p_val < 0.05 and p_bio >= 0.05:
            severity = "WARNING"
            severity_description = "Significant batch effect detected, but no significant biological signal"
            if verbose:
                print(f"Warning: {severity_description}")
        elif p_val >= 0.05 and p_bio < 0.05:
            severity = "GOOD"
            severity_description = "Biological signal detected without significant batch effect"
            if verbose:
                print(f"Good: {severity_description}")
        else:
            severity = "NONE"
            severity_description = "Neither batch nor biological effects are statistically significant"
            if verbose:
                print(f"Note: {severity_description}")
        
        results['severity'] = severity
        results['severity_description'] = severity_description
    
    batch_dummies = pd.get_dummies(batch_labels).values
    var_batch = np.array([np.corrcoef(X[:, i], batch_dummies.T)[0, 1:].max()**2 
                         for i in range(X.shape[1])])
    median_var_batch = float(np.median(var_batch))
    
    if verbose:
        print(f"Median variance explained by batch across features: {median_var_batch:.1%}")
    
    results['median_variance_explained_by_batch'] = median_var_batch
    
    return results, pc, var_batch

