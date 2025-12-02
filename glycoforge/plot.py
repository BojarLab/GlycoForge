import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# Plot PCA for clean and simulated data
def plot_pca(data, #DataFrame (features x samples)
             bio_groups=None, # dict or None, e.g. {'healthy': ['healthy_1', 'healthy_2'], 'unhealthy': ['unhealthy_1']}
             batch_groups=None, 
             title="PCA", 
             save_path=None):

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data.T)
    sample_names = data.columns.tolist()
    
    # Helper function to get bio/batch labels for annotation
    def get_bio_label(sample_name):
        if bio_groups:
            for i, (group_name, cols) in enumerate(bio_groups.items()):  # 0-based
                if sample_name in cols:
                    return f"Bio-{i}"
        return ""
    
    def get_batch_label(sample_name):
        if batch_groups:
            for batch_id, cols in batch_groups.items():
                if sample_name in cols:
                    return f"BE-{batch_id}"
        return ""
    
    # Setup subplots
    n_plots = sum([bio_groups is not None, batch_groups is not None])
    if n_plots == 0:
        return
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    axes = [axes] if n_plots == 1 else axes
    plot_idx = 0
    
    # Plot biological groups (with batch annotations)
    if bio_groups is not None:
        bio_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (group_name, cols) in enumerate(bio_groups.items()):
            indices = [sample_names.index(c) for c in cols if c in sample_names]
            axes[plot_idx].scatter(pca_result[indices, 0], pca_result[indices, 1],
                                  c=bio_colors[i % len(bio_colors)], label=group_name, alpha=0.7, s=50)
            
            # Add batch annotations on bio-colored plot
            for idx in indices:
                batch_label = get_batch_label(sample_names[idx])
                if batch_label:
                    axes[plot_idx].annotate(batch_label, (pca_result[idx, 0], pca_result[idx, 1]),
                                          xytext=(2, 2), textcoords='offset points', 
                                          fontsize=8, alpha=0.7)
        
        axes[plot_idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[plot_idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[plot_idx].set_title(f'{title}\n(colored by bio-groups)')
        axes[plot_idx].legend()
        axes[plot_idx].grid(alpha=0.3)
        plot_idx += 1
    
    # Plot batch groups (with bio annotations)
    if batch_groups is not None:
        batch_colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#FF6B35']
        for i, (batch_id, cols) in enumerate(sorted(batch_groups.items())):
            indices = [sample_names.index(c) for c in cols if c in sample_names]
            axes[plot_idx].scatter(pca_result[indices, 0], pca_result[indices, 1],
                                  c=batch_colors[i % len(batch_colors)], label=f'Batch {batch_id}', alpha=0.7, s=50)
            
            # Add bio annotations on batch-colored plot
            for idx in indices:
                bio_label = get_bio_label(sample_names[idx])
                if bio_label:
                    axes[plot_idx].annotate(bio_label, (pca_result[idx, 0], pca_result[idx, 1]),
                                          xytext=(2, 2), textcoords='offset points', 
                                          fontsize=8, alpha=0.7)
        
        axes[plot_idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[plot_idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[plot_idx].set_title(f'{title}\n(colored by batches)')
        axes[plot_idx].legend()
        axes[plot_idx].grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
