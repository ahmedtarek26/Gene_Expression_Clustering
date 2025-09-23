#!/usr/bin/env python3
"""
run_pipeline_summary.py
User-friendly script to cluster gene expression data using Gaussian mixture models and produce a summary report.

Usage:
python run_pipeline_summary.py --features <features_csv> --labels <labels_csv> --n-variable-genes 2000 --n-pca-components 50 --k-min 2 --k-max 10 --covariance-type full --n-init 5 --random-state 0 --output-dir figs

This script loads and preprocesses gene expression data, performs Gaussian mixture model clustering across a range of cluster numbers, selects the model with lowest BIC, prints a summary table with cluster sizes and top labels, computes the adjusted Rand Index if labels are provided, and saves BIC and PCA scatter plots for visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from preprocess import load_and_preprocess
from model_select import select_gmm_bic, print_bic_results
from metrics import compute_ari


def run_pipeline_summary(
    features_path,
    labels_path=None,
    n_variable_genes=2000,
    n_pca_components=50,
    k_min=2,
    k_max=10,
    covariance_type="full",
    n_init=5,
    random_state=None,
    output_dir="figs",
):
    # Load and preprocess data
    X, y = load_and_preprocess(
        features_path,
        labels_path,
        n_variable_genes=n_variable_genes,
        n_pca_components=n_pca_components,
        random_state=random_state,
    )

    # Determine range of K values
    ks = list(range(k_min, k_max + 1))
    print(f"Fitting GMMs for K from {k_min} to {k_max}...")

    # Model selection
    best_model, results = select_gmm_bic(
        X, k_range=ks, covariance_type=covariance_type, n_init=n_init, random_state=random_state
    )
    print("\nBIC results:")
    print_bic_results(results)

        results = [{'n_components': res.k, 'bic': res.bic} for res in results]

    # Print best K and ARI if labels provided
    best_k = best_model.n_components
    print(f"\nBest number of clusters (lowest BIC): {best_k}")

    labels_pred = best_model.predict(X)

    if y is not None:
        ari = compute_ari(y, labels_pred)
        print(f"Adjusted Rand Index (ARI): {ari:.4f}")

    # Cluster summary and contingency table if labels available
    if y is not None:
        df = pd.DataFrame({"Cluster": labels_pred, "Label": y})
        cluster_summary = df.groupby("Cluster")["Label"].apply(
            lambda col: col.value_counts().iloc[0] / col.size
        )
        print("\nCluster summary (size and top label purity):")
        for cluster, purity in cluster_summary.iteritems():
            size = (labels_pred == cluster).sum()
            top_label = df[df["Cluster"] == cluster]["Label"].value_counts().idxmax()
            print(f"Cluster {cluster}: {size} samples — top label {top_label} ({purity:.1%} purity)")
        print("\nContingency table (clusters vs labels):")
        print(pd.crosstab(df["Cluster"], df["Label"]))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot BIC curve
    k_vals = [r["n_components"] for r i    k_vals = [res.k for res in results]
    bic_vals = [r["bic"] for r in results]
    plt.figure()
    plt.plot(k_vals, bic_vals, marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("BIC")
    plt.title("BIC vs Number of clusters")
    plt.savefig(os.path.join(output_dir, "bic_curve.png"))
    plt.close()

    # Plot PCA scatter with clusters
    if X.shape[1] > 2:
        # Use PCA for 2D visualization only if dimension > 2
        pca = PCA(n_components=2, random_state=random_state)
        X_vis = pca.fit_transform(X)
    else:
        X_vis = X
    plt.figure()
    scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels_pred, cmap="tab10", s=30)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"PCA scatter plot (K={best_k})")
    # Add legend to indicate clusters
    legend_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=scatter.cmap(i), markersize=6) for i in range(best_k)]
    legend_labels = [f"Cluster {i}" for i in range(best_k)]
    plt.legend(legend_handles, legend_labels, title="Clusters", loc="best", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_clusters.png"))
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GMM clustering pipeline with summary reporting.")
    parser.add_argument("--features", required=True, help="Path to features CSV (samples × genes).")
    parser.add_argument("--labels", default=None, help="Path to labels CSV (one label per sample).")
    parser.add_argument("--n-variable-genes", type=int, default=2000, help="Number of highly variable genes to keep.")
    parser.add_argument("--n-pca-components", type=int, default=50, help="Number of PCA components for visualization (>=2).")
    parser.add_argument("--k-min", type=int, default=2, help="Minimum number of clusters.")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum number of clusters.")
    parser.add_argument("--covariance-type", choices=["full", "diag"], default="full", help="Covariance type for GMM.")
    parser.add_argument("--n-init", type=int, default=5, help="Number of initializations for EM.")
    parser.add_argument("--random-state", type=int, default=None, help="Random seed.")
    parser.add_argument("--output-dir", default="figs", help="Directory to save output plots.")
    args = parser.parse_args()

    run_pipeline_summary(
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
