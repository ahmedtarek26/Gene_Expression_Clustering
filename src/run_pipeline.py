"""
Run Gaussian mixture model clustering with BIC model selection and produce user‑friendly outputs.

This script loads a features matrix (and optional labels), performs standard preprocessing
(variable gene selection, scaling, optional PCA), fits a Gaussian mixture model across a
range of component numbers, selects the best model by BIC, and generates a couple of
visual aids: a BIC curve and a 2D scatter plot of the clustered samples. The script also
prints a concise table of results and, if ground‑truth labels are provided, reports the
Adjusted Rand Index (ARI) to help non‑experts gauge how well the clustering aligns with
known classes.

Example usage (synthetic data):

    python run_pipeline.py \
        --features synthetic_features.csv \
        --labels synthetic_labels.csv \
        --n-variable-genes 10 \
        --n-pca-components 5 \
        --k-min 2 --k-max 6 \
        --covariance-type full \
        --n-init 3 \
        --random-state 0

For real data, adjust the --n-variable-genes and --n-pca-components to suit your dataset.
The generated figures (BIC curve and PCA scatter) will be saved in the current working
directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

from preprocess import load_and_preprocess
from model_select import select_gmm_bic, print_bic_results
from metrics import compute_ari


def run_pipeline(
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
    """Run clustering pipeline and produce plots.

    Parameters
    ----------
    features_path : str
        Path to the features CSV file (samples x genes).
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

    # Create output directory if it doesn't exist
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

    # Define range of K values to evaluate
    k_range = list(range(k_min, k_max + 1))

    # Perform model selection using BIC
    best_model, results = select_gmm_bic(
        X,
        k_range=k_range,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=random_state,
    )

    # Print concise results table for users without technical background
    print("Clustering results (lower BIC is better):")
    print_bic_results(results)
    print(f"\nBest number of clusters (K) = {best_model.n_components}")

    # If labels provided, compute ARI for user comprehension
    if y is not None:
        labels_pred = best_model.predict(X)
        ari = compute_ari(y, labels_pred)
        print(f"Adjusted Rand Index (against provided labels): {ari:.3f}")

    # Plot BIC curve
    bic_values = [res.bic for res in results]
    plt.figure(figsize=(6, 4))
    plt.plot(k_range, bic_values, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Bayesian Information Criterion (BIC)')
    plt.title('Model selection: BIC vs K')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    bic_plot_path = out_path / 'bic_curve.png'
    plt.savefig(bic_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved BIC curve to {bic_plot_path}")

    # Prepare 2D representation for scatter plot
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=random_state)
        X_2d = pca.fit_transform(X)
    else:
        # If already 2D or 1D, take first two columns/pad zeros
        if X.shape[1] == 1:
            X_2d = np.hstack([X, np.zeros((X.shape[0], 1))])
        else:
            X_2d = X[:, :2]

    # Plot PCA scatter coloured by cluster assignment
    labels_pred = best_model.predict(X)
    plt.figure(figsize=(6, 5))
    for k in range(best_model.n_components):
        mask = labels_pred == k
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1], s=20, alpha=0.7,
            label=f'Cluster {k + 1}'
        )
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('2D representation of clusters')
    plt.legend(loc='best', frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    pca_plot_path = out_path / 'pca_clusters.png'
    plt.savefig(pca_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA scatter to {pca_plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run GMM clustering with BIC model selection and generate easy‑to‑read outputs, "
            "including a BIC curve and a 2D scatter plot."
        )
    )
    parser.add_argument('--features', type=str, required=True, help='Path to features CSV (samples x genes).')
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

    run_pipeline(
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
