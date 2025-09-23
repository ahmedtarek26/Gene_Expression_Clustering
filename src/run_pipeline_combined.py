"""
Run Gaussian mixture model clustering with comprehensive reporting and visualisations.

This script loads a features matrix (and optional labels), performs standard preprocessing
(variable gene selection, scaling, optional PCA), fits a Gaussian mixture model across a
range of component numbers, selects the best model by BIC, and generates several
visual aids: a BIC curve, a bar chart of cluster sizes, a contingency heatmap (if
labels are provided) and a 2D scatter plot of the clustered samples. The script
prints a concise table of BIC results, reports the Adjusted Rand Index (ARI) and
cluster purity summary when labels are available, and computes the mean silhouette
score to help non‑experts gauge clustering quality.

Example usage (synthetic data):

    python run_pipeline_combined.py \
        --features synthetic_features.csv \
        --labels synthetic_labels.csv \
        --n-variable-genes 10 \
        --n-pca-components 5 \
        --k-min 2 --k-max 6 \
        --covariance-type full \
        --n-init 3 \
        --random-state 0

For real data, adjust the --n-variable-genes and --n-pca-components to suit your dataset.
All generated figures will be saved in the specified output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from preprocess import load_and_preprocess
from model_select import select_gmm_bic, print_bic_results
from metrics import compute_ari, compute_silhouette


def run_pipeline_combined(
    features_path: str,
    labels_path: Optional[str],
    n_variable_genes: Optional[int],
    n_pca_components: Optional[int],
    k_min: int,
    k_max: int,
    covariance_type: str,
    n_init: int,
    random_state: Optional[int],
    output_dir: str,
) -> None:
    """Run clustering pipeline and produce a comprehensive report and plots.

    Parameters
    ----------
    features_path : str
        Path to the features CSV file (samples × genes).
    labels_path : str, optional
        Path to the labels CSV file (one label per sample) or None.
    n_variable_genes : int, optional
        Number of most variable genes to retain for clustering. If None, keep all.
    n_pca_components : int, optional
        Number of principal components to retain. If None, no PCA is applied.
    k_min : int
        Minimum number of mixture components to evaluate.
    k_max : int
        Maximum number of mixture components to evaluate.
    covariance_type : str
        Covariance type for the GMM ('full' or 'diag').
    n_init : int
        Number of random initialisations for EM.
    random_state : int, optional
        Random seed for reproducibility.
    output_dir : str
        Directory to save the output figures.
    """

    # Ensure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    X, y = load_and_preprocess(
        features_path,
        labels_path,
        n_variable_genes,
        n_pca_components,
        random_state,
    )

    # Range of cluster counts to evaluate
    k_range = list(range(k_min, k_max + 1))

    # Fit models and select the best by BIC
    best_model, results = select_gmm_bic(
        X,
        k_range=k_range,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=random_state,
    )

    # Print BIC results and best K
    print("Clustering results (lower BIC is better):")
    print_bic_results(results)
    print(f"\nBest number of clusters (K) = {best_model.n_components}")

    # Predict cluster assignments
    labels_pred = best_model.predict(X)

    # Evaluate if labels available
    if y is not None:
        ari = compute_ari(y, labels_pred)
        print(f"Adjusted Rand Index (against provided labels): {ari:.3f}")

        # Cluster purity summary
        df = pd.DataFrame({"Cluster": labels_pred, "Label": y})
        cluster_purity = df.groupby("Cluster")["Label"].apply(lambda col: col.value_counts().iloc[0] / col.size)
        print("\nCluster summary (size and top label purity):")
        for cluster, purity in cluster_purity.items():
            size = (labels_pred == cluster).sum()
            top_label = df[df["Cluster"] == cluster]["Label"].value_counts().idxmax()
            print(f"Cluster {cluster}: {size} samples — top label {top_label} ({purity:.1%} purity)")

        # Contingency table
        print("\nContingency table (clusters vs labels):")
        print(pd.crosstab(df["Cluster"], df["Label"]))

    # Silhouette score for any case
    sil_score = compute_silhouette(X, labels_pred)
    if not np.isnan(sil_score):
        print(f"Mean Silhouette Score: {sil_score:.3f}")

    # Plot BIC curve
    bic_values = [res.bic for res in results]
    plt.figure(figsize=(6, 4))
    plt.plot(k_range, bic_values, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Bayesian Information Criterion (BIC)')
    plt.title('Model selection: BIC vs K')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    bic_path = out_path / 'bic_curve.png'
    plt.savefig(bic_path, bbox_inches='tight')
    plt.close()
    print(f"Saved BIC curve to {bic_path}")

    # Plot cluster sizes bar chart
    cluster_sizes = np.bincount(labels_pred)
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(cluster_sizes) + 1), cluster_sizes)
    plt.xlabel('Cluster index')
    plt.ylabel('Number of samples')
    plt.title('Cluster sizes')
    bar_path = out_path / 'cluster_sizes_bar.png'
    plt.savefig(bar_path, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster sizes bar chart to {bar_path}")

    # Contingency heatmap
    if y is not None:
        contingency = pd.crosstab(df["Cluster"], df["Label"])
        plt.figure(figsize=(8, 4))
        im = plt.imshow(contingency.values, aspect='auto', cmap='viridis')
        plt.colorbar(im)
        plt.xticks(range(len(contingency.columns)), contingency.columns, rotation=45, ha='right')
        plt.yticks(range(len(contingency.index)), contingency.index)
        plt.title('Contingency heatmap (clusters vs labels)')
        for i in range(contingency.shape[0]):
            for j in range(contingency.shape[1]):
                val = contingency.iloc[i, j]
                color = 'white' if val > contingency.values.max() / 2 else 'black'
                plt.text(j, i, val, ha='center', va='center', color=color)
        plt.xlabel('True label')
        plt.ylabel('Cluster')
        plt.tight_layout()
        heatmap_path = out_path / 'contingency_heatmap.png'
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()
        print(f"Saved contingency heatmap to {heatmap_path}")

    # Create 2D representation for scatter
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=random_state)
        X_2d = pca.fit_transform(X)
    else:
        # Already 1 or 2 dimensions
        if X.shape[1] == 1:
            X_2d = np.hstack([X, np.zeros((X.shape[0], 1))])
        else:
            X_2d = X[:, :2]

    # Plot PCA scatter by cluster
    plt.figure(figsize=(6, 5))
    for k in range(best_model.n_components):
        mask = labels_pred == k
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=20, alpha=0.7, label=f'Cluster {k + 1}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('2D representation of clusters')
    plt.legend(loc='best', frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    pca_path = out_path / 'pca_clusters.png'
    plt.savefig(pca_path, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA scatter to {pca_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run GMM clustering with BIC model selection and generate a comprehensive summary, "
            "including multiple plots and metrics."
        )
    )
    parser.add_argument('--features', type=str, required=True, help='Path to features CSV (samples × genes).')
    parser.add_argument('--labels', type=str, default=None, help='Optional path to labels CSV.')
    parser.add_argument('--n-variable-genes', type=int, default=None, help='Number of variable genes to select (None = all).')
    parser.add_argument('--n-pca-components', type=int, default=None, help='Number of PCA components to retain (None = no PCA).')
    parser.add_argument('--k-min', type=int, default=2, help='Minimum number of clusters to evaluate.')
    parser.add_argument('--k-max', type=int, default=10, help='Maximum number of clusters to evaluate.')
    parser.add_argument('--covariance-type', type=str, choices=['full', 'diag'], default='full', help='Covariance type for the GMM.')
    parser.add_argument('--n-init', type=int, default=3, help='Number of random initialisations for EM.')
    parser.add_argument('--random-state', type=int, default=None, help='Random seed for reproducibility.')
    parser.add_argument('--output-dir', type=str, default='figs', help='Directory to save output figures.')
    args = parser.parse_args()

    run_pipeline_combined(
        features_path=args.features,
        labels_path=args.labels,
        n_variable_genes=args.n_variable_genes,
        n_pca_components=args.n_pca_components,
        k_min=args.k_min,
        k_max=args.k_max,
        covariance_type=args.covariance_type,
        n_init=args.n_init,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
