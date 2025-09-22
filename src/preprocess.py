"""
Data loading and preprocessing utilities for the mixture model project.

This module provides functions to load CSV files, optionally remove
non‑feature columns, select highly variable genes (or features), apply
standardisation and dimensionality reduction via principal component
analysis (PCA). These steps are common in transcriptomic workflows and
help stabilise the EM algorithm in high dimensions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_features(
    path: str,
    drop_first_column: bool = True,
) -> pd.DataFrame:
    """Loads a CSV file of features.

    Parameters
    ----------
    path : str
        Path to the CSV file. The file should contain samples in rows and
        features in columns. The first column can optionally be a sample
        identifier (e.g., `sample_0`), which is dropped if `drop_first_column`
        is True.
    drop_first_column : bool, default=True
        Whether to drop the first column of the CSV. Many datasets include an
        index column that is not a feature.

    Returns
    -------
    df : pandas.DataFrame
        The loaded data as a DataFrame with samples in rows and features in
        columns.
    """
    df = pd.read_csv(path)
    if drop_first_column:
        df = df.iloc[:, 1:]
    return df


def load_labels(path: str, drop_first_column: bool = True) -> pd.Series:
    """Loads a CSV file containing sample labels.

    Parameters
    ----------
    path : str
        Path to the labels CSV file. The file should have a column with sample
        identifiers and a column with the class or subtype label.
    drop_first_column : bool, default=True
        Whether to drop the first column (sample identifiers).

    Returns
    -------
    labels : pandas.Series
        A Series of labels indexed by sample index.
    """
    df = pd.read_csv(path, header=0)
    if drop_first_column:
        df = df.iloc[:, 1:]
    # The remaining column contains the label
    return df.iloc[:, 0]


def select_top_variable_features(
    X: pd.DataFrame,
    n_features: int,
) -> pd.DataFrame:
    """Selects the `n_features` most variable columns in `X`.

    Variance is computed across samples. Selecting highly variable genes is
    common in transcriptomics to focus on informative features and reduce
    dimensionality.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame containing features.
    n_features : int
        Number of top variable features to select.

    Returns
    -------
    X_sel : pandas.DataFrame
        DataFrame containing only the selected features.
    """
    if n_features >= X.shape[1]:
        return X
    variances = X.var(axis=0)
    top_features = variances.nlargest(n_features).index
    return X[top_features]


def scale_data(X: pd.DataFrame) -> np.ndarray:
    """Standardises features to zero mean and unit variance.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def pca_reduce(
    X: np.ndarray,
    n_components: int,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, PCA]:
    """Performs principal component analysis on `X`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Standardised feature matrix.
    n_components : int
        Number of principal components to retain.
    random_state : Optional[int], default=None
        Random seed for deterministic results.

    Returns
    -------
    X_pca : ndarray of shape (n_samples, n_components)
        Reduced‑dimension representation of `X`.
    pca : sklearn.decomposition.PCA
        Fitted PCA object (contains components, explained variance, etc.).
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def load_and_preprocess(
    feature_path: str,
    label_path: Optional[str] = None,
    n_variable_genes: Optional[int] = None,
    n_pca_components: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convenience function to load features and labels and apply preprocessing.

    This function sequentially loads the data, selects highly variable genes
    (if requested), standardises the features and applies PCA (if requested).

    Parameters
    ----------
    feature_path : str
        Path to the features CSV file.
    label_path : str, optional
        Path to the labels CSV file. If None, labels are not loaded.
    n_variable_genes : int, optional
        Number of most variable genes to retain. If None, all features are
        kept.
    n_pca_components : int, optional
        Number of principal components for PCA. If None, PCA is not
        performed.
    random_state : Optional[int], default=None
        Random seed for PCA.

    Returns
    -------
    X_proc : ndarray of shape (n_samples, n_features_processed)
        Preprocessed feature matrix.
    y : ndarray of shape (n_samples,), optional
        Array of labels if `label_path` is provided, otherwise None.
    """
    df = load_features(feature_path, drop_first_column=True)
    if n_variable_genes is not None:
        df = select_top_variable_features(df, n_variable_genes)
    # Standardise features
    X = scale_data(df)
    # Dimensionality reduction
    if n_pca_components is not None:
        X, _ = pca_reduce(X, n_components=n_pca_components, random_state=random_state)
    # Load labels if provided
    y = None
    if label_path is not None:
        y = load_labels(label_path, drop_first_column=True).values
    return X, y
