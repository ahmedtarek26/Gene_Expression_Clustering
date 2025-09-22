"""
Implementation of the Expectation–Maximisation (EM) algorithm for Gaussian
mixture models. This module defines a `GaussianMixtureEM` class that can
estimate mixtures of arbitrary dimension with either full or diagonal
covariance matrices. It provides log‑likelihood, prediction and BIC
computation utilities.

The implementation intentionally does not depend on scikit‑learn's built‑in
`GaussianMixture` so that the derivation of the EM steps can be clearly
illustrated and customised. Nevertheless, it adheres to the standard API to
facilitate comparison.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import slogdet, inv
from typing import Optional, Tuple


class GaussianMixtureEM:
    """Gaussian mixture model fitted with the EM algorithm.

    Parameters
    ----------
    n_components : int
        Number of mixture components (clusters).
    covariance_type : {'full', 'diag'}, default='full'
        Form of the covariance matrices. 'full' uses full matrices with
        off‑diagonal entries, while 'diag' constrains them to be diagonal.
        Diagonal covariances scale better in high dimensions but assume
        independence among features.
    tol : float, default=1e-6
        Convergence threshold on the change in log‑likelihood.
    max_iter : int, default=100
        Maximum number of EM iterations.
    n_init : int, default=1
        Number of initialisations to run. The best run in terms of
        log‑likelihood is kept.
    random_state : Optional[int], default=None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int,
        covariance_type: str = 'full',
        tol: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        if covariance_type not in ('full', 'diag'):
            raise ValueError("covariance_type must be 'full' or 'diag'")
        if n_components < 1:
            raise ValueError('n_components must be >= 1')
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        # Attributes initialised during fitting
        self.weights_: Optional[np.ndarray] = None  # shape (K,)
        self.means_: Optional[np.ndarray] = None    # shape (K, D)
        self.covariances_: Optional[np.ndarray] = None  # shape (K, D, D) or (K, D)
        self.n_features_in_: Optional[int] = None
        self.converged_: bool = False
        self.lower_bound_: Optional[float] = None  # final log‑likelihood

    # ------------------------------------------------------------------
    # Helpers

    def _initialize_parameters(self, X: np.ndarray, rng: np.random.Generator) -> None:
        """Randomly initialises mixture weights, means and covariances.

        Means are chosen by randomly sampling `n_components` points from `X`.
        Covariance matrices are initialised to the empirical covariance of the
        data divided by the number of components. Weights start uniform.
        """
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        # Initialise means to random subset of data
        indices = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices].copy()
        # Initialise covariance(s) to empirical covariance scaled down
        emp_cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(n_features)
        if self.covariance_type == 'full':
            self.covariances_ = np.stack([emp_cov.copy() for _ in range(self.n_components)], axis=0)
        else:
            self.covariances_ = np.tile(np.diag(emp_cov).copy(), (self.n_components, 1))
        # Initialise weights uniformly
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)

    def _estimate_log_gaussian(self, X: np.ndarray) -> np.ndarray:
        """Computes the log density of each point under each component.

        Returns
        -------
        log_prob : ndarray of shape (n_samples, n_components)
            Logarithm of the probability density p(x_n | z=k).
        """
        n_samples, n_features = X.shape
        K = self.n_components
        log_prob = np.empty((n_samples, K))
        for k in range(K):
            mean = self.means_[k]
            if self.covariance_type == 'full':
                cov = self.covariances_[k]
                sign, logdet = slogdet(cov)
                if sign <= 0:
                    raise np.linalg.LinAlgError('Covariance matrix is not positive definite')
                inv_cov = inv(cov)
                diff = X - mean
                # Mahalanobis distances squared
                mah = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + logdet + mah)
            else:  # diag
                var = self.covariances_[k]
                if np.any(var <= 0):
                    raise np.linalg.LinAlgError('Covariance diagonal has non‑positive entries')
                # log determinant of diagonal covariance
                logdet = np.sum(np.log(var))
                diff = X - mean
                mah = np.sum((diff ** 2) / var, axis=1)
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + logdet + mah)
        return log_prob

    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Performs the expectation step.

        Computes responsibilities `r` and the log‑likelihood (the
        lower bound) for the current parameters.

        Returns
        -------
        r : ndarray of shape (n_samples, n_components)
            Responsibilities: r[n, k] = p(z_n=k | x_n).
        log_likelihood : float
            Current log marginal likelihood of the data.
        """
        log_prob = self._estimate_log_gaussian(X)
        # Add log weights
        weighted_log_prob = log_prob + np.log(self.weights_)
        # Compute logsumexp along components for normalisation
        log_prob_norm = np.logaddexp.reduce(weighted_log_prob, axis=1)
        r = np.exp(weighted_log_prob - log_prob_norm[:, None])
        log_likelihood = np.mean(log_prob_norm)
        return r, log_likelihood

    def _m_step(self, X: np.ndarray, r: np.ndarray) -> None:
        """Performs the maximisation step.

        Updates the mixture weights, means and covariances based on the
        current responsibilities.
        """
        n_samples, n_features = X.shape
        Nk = np.sum(r, axis=0)  # effective number of points per component
        # Update weights
        self.weights_ = Nk / n_samples
        # Update means
        self.means_ = (r.T @ X) / Nk[:, None]
        # Update covariances
        if self.covariance_type == 'full':
            self.covariances_ = np.zeros((self.n_components, n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                # weighted empirical covariance
                cov_k = (r[:, k][:, None] * diff).T @ diff / Nk[k]
                # Add tiny value for numerical stability
                cov_k += 1e-6 * np.eye(n_features)
                self.covariances_[k] = cov_k
        else:
            self.covariances_ = np.zeros((self.n_components, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                # weighted variance per feature
                var_k = np.sum(r[:, k][:, None] * diff ** 2, axis=0) / Nk[k]
                var_k += 1e-6  # numerical stability
                self.covariances_[k] = var_k

    def fit(self, X: np.ndarray) -> 'GaussianMixtureEM':
        """Fits the model to the data using EM.

        Runs `n_init` initialisations and retains the best solution in terms of
        log‑likelihood.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        best_loglike = -np.inf
        best_params = None
        rng_global = np.random.default_rng(self.random_state)
        for init in range(self.n_init):
            rng = np.random.default_rng(rng_global.integers(np.iinfo(np.int32).max))
            self._initialize_parameters(X, rng)
            prev_loglike = -np.inf
            for iteration in range(self.max_iter):
                r, loglike = self._e_step(X)
                self._m_step(X, r)
                if np.abs(loglike - prev_loglike) < self.tol:
                    self.converged_ = True
                    break
                prev_loglike = loglike
            # Save the parameters if better
            if loglike > best_loglike:
                best_loglike = loglike
                best_params = (
                    self.weights_.copy(),
                    self.means_.copy(),
                    self.covariances_.copy(),
                    self.converged_,
                    loglike,
                )
        # Restore best run
        self.weights_, self.means_, self.covariances_, self.converged_, self.lower_bound_ = best_params
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Returns the log probability of each sample under the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        log_prob : ndarray of shape (n_samples,)
            Logarithm of the probability density for each sample.
        """
        log_prob = self._estimate_log_gaussian(X)
        weighted_log_prob = log_prob + np.log(self.weights_)
        return np.logaddexp.reduce(weighted_log_prob, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Posterior probabilities (responsibilities) for each component.
        """
        log_prob = self._estimate_log_gaussian(X)
        weighted_log_prob = log_prob + np.log(self.weights_)
        log_prob_norm = np.logaddexp.reduce(weighted_log_prob, axis=1)
        return np.exp(weighted_log_prob - log_prob_norm[:, None])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard cluster assignments via maximum a posteriori estimation.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def bic(self, X: np.ndarray) -> float:
        """Bayesian information criterion for the fitted model.

        BIC = -2 * log‑likelihood + p * log(N), where p is the number of
        free parameters and N is the number of samples. Lower BIC indicates
        better trade‑off between fit and complexity.
        """
        n_samples, n_features = X.shape
        log_likelihood = np.sum(self.score_samples(X))
        # Number of parameters: mixture weights (K-1), means (K * D), covariances
        if self.covariance_type == 'full':
            # Full covariances: K * D * (D + 1) / 2
            cov_params = self.n_components * n_features * (n_features + 1) / 2
        else:
            # Diagonal: K * D
            cov_params = self.n_components * n_features
        n_params = (self.n_components - 1) + self.n_components * n_features + cov_params
        bic = -2.0 * log_likelihood + n_params * np.log(n_samples)
        return float(bic)

