# Robust Bayesian Clustering of Tumour Transcriptomes

This repository contains the code and documentation for a course project exploring unsupervised clustering of gene expression profiles using probabilistic models. The goal is to identify tumour subtypes or hidden structure in high‑dimensional transcriptomic data using Gaussian mixture models (GMMs) and principled model selection.

## Overview

- **Gaussian Mixture Model (GMM)**: We implement the Expectation–Maximisation (EM) algorithm to fit a GMM to normalised expression data. The E‑step computes responsibilities (soft cluster assignments) and the M‑step updates mixture weights, means and covariances. The procedure monotonically increases the likelihood and terminates at convergence.

- **Model Selection**: Choosing the number of clusters \(K\) is critical. We follow a Bayesian approach, computing the Bayesian Information Criterion (BIC) for different values of \(K\) and selecting the model with the highest evidence. The `model_select` module sweeps over a range of cluster numbers and reports the best model according to BIC.

- **Robust and Extended Variants**: For robustness to outliers and high‑dimensional data, the repository includes:
    - A **Student‑t mixture** model with heavier tails, fitted by a modified EM routine.
    - MAP‑EM with Gaussian–Wishart priors, adding regularisation to the means and covariances.
    - Utilities for diagonal or shrinkage covariance matrices and mixture of factor analysers (MFA) if dimensionality reduction is desired.

- **Evaluation**: We provide functions to compute log‑likelihood, BIC, silhouette scores and adjusted Rand index (ARI) when ground‑truth labels are available. Responsibilities can be visualised to assess uncertainty in cluster assignments. Optional MCMC diagnostics such as \(\hat R\) and effective sample size can be run on a small subset of parameters.

## Repository Structure

- `src/` – Python modules for preprocessing, EM algorithms, model selection, metrics and robust mixtures.
- `data/` – Folder for data files. It contains a small synthetic dataset (`synthetic_features.csv`, `synthetic_labels.csv`) for quick testing. Replace these with your own gene expression matrix (samples × features) and optional labels.
- `data/README.md` – Details on the data directory and instructions for adding your own datasets.
- `figs/` – Place generated figures (BIC curves, cluster visualisations) here.
- `notebooks/` – Jupyter notebooks for exploratory analysis and reproducing results.
- `slides/` – Presentation slides summarising the project (if applicable).

## Getting Started

1. **Install dependencies** (Python ≥ 3.8):  
   ```bash
   pip install -r requirements.txt
   ```
   The project relies on NumPy, SciPy, scikit‑learn and matplotlib.

2. **Prepare data**:  
   - Normalise your expression matrix (e.g. log‑transform and standardise) and save it as a CSV file.
   - Place the feature matrix in `data/<your_features>.csv` and optional labels in `data/<your_labels>.csv`.
   - Update the file paths in the notebooks or scripts accordingly.

3. **Run the pipeline**:
   - Use `python src/run_pipeline.py --features data/<your_features>.csv --labels data/<your_labels>.csv` to run the complete clustering pipeline. This script performs preprocessing, GMM model selection with BIC, prints an easy‑to‑read results table, and automatically saves a BIC curve and 2D scatter plot (PCA) in the `figs/` directory.
   - Adjust `--n-variable-genes`, `--n-pca-components`, `--k-min`, and `--k-max` to suit your dataset and desired complexity.

4. **Inspect results**:
   - Open the saved figures in the `figs/` folder. The BIC curve shows how model fit changes with K and highlights the optimal number of clusters. The 2D scatter plot visualises the sample clusters.
- Check the printed BIC table and ARI to see how well the clustering aligns with known subtypes (if labels are provided).


This project is for educational purposes and distributed under the MIT License. See `LICENSE` for more information (to be added).

## Acknowledgements

This work builds on lecture material from the Probabilistic Machine Learning course, including EM for mixture models, Bayesian model comparison (BIC) and MCMC diagnostics. It is designed to be compatible with a Computational Genomics project, using gene expression data as the application domain.
