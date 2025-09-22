# Data directory

This directory is intended to hold input data for your analyses.

## Synthetic data

A small synthetic dataset is included for quick testing and verification. It consists of:

- `synthetic_features.csv`: each row is a sample and each column is a feature (gene); the first column contains a sample identifier and is ignored by the preprocessing code.
- `synthetic_labels.csv`: contains the corresponding tumour subtype label for each sample.

## TCGA Pan-Cancer RNA Seq (real data)

For the actual analysis, we recommend using the TCGA Pan-Cancer “HiSeq” gene‑expression dataset (801 samples × 20 531 genes) supplied via the UCI Machine Learning Repository. Each value represents log2(RSEM+1) normalised gene expression. Follow these steps to set up the data:

1. Download the archive from the [UCI Gene Expression Cancer RNA-Seq repository](https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-seq) and extract the `data.csv` and `labels.csv` files.  
2. Create a new folder under `data/` called `TCGA-PANCAN-HiSeq-801x20531` and place `data.csv` and `labels.csv` inside it.  
3. In the project code, use `load_and_preprocess()` from `src/preprocess.py` on `data/TCGA-PANCAN-HiSeq-801x20531/data.csv` (and optionally the labels file) to prepare the data for clustering.

Do not commit large biological datasets to this repository; only metadata or scripts should be stored here.
