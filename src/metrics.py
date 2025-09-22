"""
Evaluation metrics for clustering results.

This module wraps scikit‑learn functions for common clustering metrics
including the silhouette score and adjusted Rand index. It also includes
simple helper functions to visualise soft assignments.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Computes the mean silhouette score for a clustering.

    A higher silhouette indicates more well‑separated clusters. Requires at
    least two clusters.
    """
    if len(np.unique(labels)) < 2:
        return float('nan')
    return float(silhouette_score(X, labels))


def compute_ari(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Adjusted Rand index between ground truth and predicted labels.

    ARI == 1.0 when the clustering matches the ground truth perfectly and
    has an expected value of 0.0 for random assignments.
    """
    return float(adjusted_rand_score(true_labels, pred_labels))


def compute_nmi(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Normalized mutual information between ground truth and predicted labels.
    """
    return float(normalized_mutual_info_score(true_labels, pred_labels))


def soft_to_hard(responsibilities: np.ndarray) -> np.ndarray:
    """Converts a matrix of responsibilities into hard cluster assignments.

    Each sample is assigned to the component with the highest posterior
    probability.
    """
    return np.argmax(responsibilities, axis=1)
