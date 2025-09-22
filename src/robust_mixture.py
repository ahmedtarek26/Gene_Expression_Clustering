"""
Mixture of Student‑t distributions fitted with the EM algorithm.

This implementation mirrors the Gaussian mixture in `em_gmm.py` but
replaces the Gaussian likelihood with a multivariate Student‑t. The heavy
tails of the *t*‑distribution provide robustness against outliers and
model mis‑specification. The degrees of freedom (``nu``) are held fixed;
optimising ``nu`` would require an additional root‑finding step and is
outside the scope of this project.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import slogdet, inv
from typing import Optional, Tuple


class StudentTMixtureEM:
    """Student‑t mixture model fitted with EM.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    nu : float, default=4.0
        Degrees of freedom of the Student‑t distribution. Smaller values
        yield heavier tails. Must be positive.
    covariance_type : {'full', 'diag'}, default='full'
        Covariance structure per component.
    tol : float, default=1e-6
        Convergence tolerance on the change in log‑likelihood.
    max_iter : int, default=100
        Maximum number of EM iterations.
    n_init : int, default=1
        Number of random restarts.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int,
        nu: float = 4.0,
        covariance_type: str = 'full',
        tol: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        if covariance_type not in ('full', 'diag'):
            raise ValueError("covariance_type must be 'full' or 'diag'")
        if nu <= 0:
            raise ValueError('nu must be positive')
        self.n_components = n_components
        self.nu = nu
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        # Attributes
        self.weights_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.converged_: bool = False
        self.lower_bound_: Optional[float] = None

    def _initialize(self, X: np.ndarray, rng: np.random.Generator) -> None:
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        # Select random subset of samples as initial means
        idx = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[idx].copy()
        emp_cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(n_features)
        if self.covariance_type == 'full':
            self.covariances_ = np.stack([emp_cov.copy() for _ in range(self.n_components)], axis=0)
        else:
            self.covariances_ = np.tile(np.diag(emp_cov).copy(), (self.n_components, 1))
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)

    def _estimate_log_student(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the log‑density and Mahalanobis distances.

        Returns
        -------
        log_pdf : ndarray of shape (n_samples, n_components)
            Logarithm of the Student‑t density.
        delta : ndarray of shape (n_samples, n_components)
            Mahalanobis distance squared per component.
        """
        n_samples, d = X.shape
        K = self.n_components
        log_pdf = np.empty((n_samples, K))
        delta = np.empty((n_samples, K))
        nu = self.nu
        for k in range(K):
            mean = self.means_[k]
            if self.covariance_type == 'full':
                cov = self.covariances_[k]
                sign, logdet = slogdet(cov)
                if sign <= 0:
                    raise np.linalg.LinAlgError('Covariance matrix is not positive definite')
                inv_cov = inv(cov)
                diff = X - mean
                mah = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
                delta[:, k] = mah
                # Log of multivariate t density
                log_pdf[:, k] = (
                    np.log(np.exp(np.math.lgamma((nu + d) / 2)))
                    - np.log(np.exp(np.math.lgamma(nu / 2)))
                    - (d / 2) * np.log(nu * np.pi)
                    - 0.5 * logdet
                    - ((nu + d) / 2) * np.log(1 + mah / nu)
                )
            else:
                var = self.covariances_[k]
                if np.any(var <= 0):
                    raise np.linalg.LinAlgError('Non positive variances')
                logdet = np.sum(np.log(var))
                diff = X - mean
                mah = np.sum((diff ** 2) / var, axis=1)
                delta[:, k] = mah
                log_pdf[:, k] = (
                    np.log(np.exp(np.math.lgamma((nu + d) / 2)))
                    - np.log(np.exp(np.math.lgamma(nu / 2)))
                    - (d / 2) * np.log(nu * np.pi)
                    - 0.5 * logdet
                    - ((nu + d) / 2) * np.log(1 + mah / nu)
                )
        return log_pdf, delta

    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """E‑step for Student‑t mixture.

        Computes responsibilities ``r`` and weight factors ``w`` used in the
        M‑step. Also returns the average log‑likelihood.
        """
        log_pdf, delta = self._estimate_log_student(X)
        weighted_log_pdf = log_pdf + np.log(self.weights_)
        # logsumexp over components
        log_prob_norm = np.logaddexp.reduce(weighted_log_pdf, axis=1)
        r = np.exp(weighted_log_pdf - log_prob_norm[:, None])
        # Weight factors for each observation/component
        w = (self.nu + self.n_features_in_) / (self.nu + delta)
        log_likelihood = np.mean(log_prob_norm)
        return r, w, log_likelihood

    def _m_step(self, X: np.ndarray, r: np.ndarray, w: np.ndarray) -> None:
        """M‑step updates for Student‑t mixture.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data matrix.
        r : ndarray of shape (n_samples, n_components)
            Responsibilities.
        w : ndarray of shape (n_samples, n_components)
            Weight factors.
        """
        n_samples, d = X.shape
        Nk = np.sum(r, axis=0)
        # Weighted responsibilities
        r_w = r * w
        Nk_w = np.sum(r_w, axis=0)
        # Update weights
        self.weights_ = Nk / n_samples
        # Update means
        self.means_ = (r_w.T @ X) / Nk_w[:, None]
        # Update covariances
        if self.covariance_type == 'full':
            self.covariances_ = np.zeros((self.n_components, d, d))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                cov_k = (r_w[:, k][:, None] * diff).T @ diff / Nk[k]
                cov_k += 1e-6 * np.eye(d)
                self.covariances_[k] = cov_k
        else:
            self.covariances_ = np.zeros((self.n_components, d))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                var_k = np.sum(r_w[:, k][:, None] * diff ** 2, axis=0) / Nk[k]
                var_k += 1e-6
                self.covariances_[k] = var_k

    def fit(self, X: np.ndarray) -> 'StudentTMixtureEM':
        """Fits the Student‑t mixture model using EM.

        Runs multiple initialisations and keeps the best in terms of log‑likelihood.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        best_loglike = -np.inf
        best_params = None
        rng_global = np.random.default_rng(self.random_state)
        for _ in range(self.n_init):
            rng = np.random.default_rng(rng_global.integers(np.iinfo(np.int32).max))
            self._initialize(X, rng)
            prev_loglike = -np.inf
            for _ in range(self.max_iter):
                r, w, loglike = self._e_step(X)
                self._m_step(X, r, w)
                if np.abs(loglike - prev_loglike) < self.tol:
                    self.converged_ = True
                    break
                prev_loglike = loglike
            if loglike > best_loglike:
                best_loglike = loglike
                best_params = (
                    self.weights_.copy(),
                    self.means_.copy(),
                    self.covariances_.copy(),
                    self.converged_,
                    loglike,
                )
        self.weights_, self.means_, self.covariances_, self.converged_, self.lower_bound_ = best_params
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Log probability of samples under the Student‑t mixture.
        """
        log_pdf, _ = self._estimate_log_student(X)
        weighted_log_pdf = log_pdf + np.log(self.weights_)
        return np.logaddexp.reduce(weighted_log_pdf, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_pdf, _ = self._estimate_log_student(X)
        weighted_log_pdf = log_pdf + np.log(self.weights_)
        log_prob_norm = np.logaddexp.reduce(weighted_log_pdf, axis=1)
        return np.exp(weighted_log_pdf - log_prob_norm[:, None])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def bic(self, X: np.ndarray) -> float:
        """Approximate BIC for Student‑t mixture.

        The parameter count matches the Gaussian case. While this is a
        simplification (the degrees of freedom are held fixed), it allows
        qualitative comparison of model complexity.
        """
        n_samples, d = X.shape
        log_likelihood = np.sum(self.score_samples(X))
        if self.covariance_type == 'full':
            cov_params = self.n_components * d * (d + 1) / 2
        else:
            cov_params = self.n_components * d
        n_params = (self.n_components - 1) + self.n_components * d + cov_params
        bic = -2.0 * log_likelihood + n_params * np.log(n_samples)
        return float(bic)
