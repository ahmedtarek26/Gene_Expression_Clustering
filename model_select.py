"""
Utilities for selecting the number of clusters in a Gaussian mixture model
using Bayesian model selection. The primary function `select_gmm_bic` fits
Gaussian mixtures with different numbers of components and computes the
Bayesian information criterion (BIC) for each. The model with the lowest
BIC is returned alongside the full sweep of results.

This module depends on the custom `GaussianMixtureEM` implementation from
`em_gmm.py`. You can adjust the range of `K` values, number of initial
random restarts and covariance type via the function arguments.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Import without relative package prefix so that this module can be used
# when `src` is added to the Python path (e.g., via sys.path.append).
from em_gmm import GaussianMixtureEM


@dataclass
class BICResult:
    k: int
    bic: float
    log_likelihood: float
    model: GaussianMixtureEM


def select_gmm_bic(
    X: np.ndarray,
    k_range: List[int],
    covariance_type: str = 'full',
    n_init: int = 1,
    tol: float = 1e-6,
    max_iter: int = 100,
    random_state: Optional[int] = None,
) -> Tuple[GaussianMixtureEM, List[BICResult]]:
    """Runs EM for each `K` in `k_range` and computes the BIC.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    k_range : list of int
        List of numbers of components to try.
    covariance_type : {'full', 'diag'}, default='full'
        Covariance type for all candidate models.
    n_init : int, default=1
        Number of random restarts for each `K`. The best initialisation is kept.
    tol : float, default=1e-6
        Convergence tolerance for EM.
    max_iter : int, default=100
        Maximum number of EM iterations.
    random_state : Optional[int], default=None
        Random seed for reproducibility.

    Returns
    -------
    best_model : GaussianMixtureEM
        The fitted model with the lowest BIC.
    results : list of BICResult
        List of results for each `K`, including BIC values and models.
    """
    X = np.asarray(X)
    results: List[BICResult] = []
    best_bic = np.inf
    best_model: Optional[GaussianMixtureEM] = None
    for k in k_range:
        model = GaussianMixtureEM(
            n_components=k,
            covariance_type=covariance_type,
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
        model.fit(X)
        bic_value = model.bic(X)
        loglike = float(model.lower_bound_ * X.shape[0])  # average loglike * N
        results.append(BICResult(k=k, bic=bic_value, log_likelihood=loglike, model=model))
        if bic_value < best_bic:
            best_bic = bic_value
            best_model = model
    return best_model, results


def print_bic_results(results: List[BICResult]) -> None:
    """Pretty‑prints BIC results.

    Useful for quick inspection in a script or notebook.
    """
    print("K\tBIC\tLogLikelihood")
    for res in results:
        print(f"{res.k}\t{res.bic:.2f}\t{res.log_likelihood:.2f}")
