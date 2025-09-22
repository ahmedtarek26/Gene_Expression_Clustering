This directory is intended to hold input data for your analyses. The
repository does not include large biological datasets; instead, you can
generate synthetic data or drop your own CSV files here. A small
synthetic dataset for testing is provided in `synthetic_features.csv`
and `synthetic_labels.csv`. Each row in `synthetic_features.csv` is a
sample and each column is a feature; the first column contains a sample
identifier and will be ignored by the preprocessing code. The labels
file contains the corresponding tumour subtype for each sample.