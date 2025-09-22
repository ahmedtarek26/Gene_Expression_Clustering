eimport argparse
from preprocess import load_and_preprocess
from model_select import select_gmm_bic, print_bic_results
from metrics import compute_ari


def main():
    parser = argparse.ArgumentParser(description="Run GMM clustering with BIC model selection")
    parser.add_argument("--features", type=str, required=True, help="Path to CSV containing features (samples x genes).")
    parser.add_argument("--labels", type=str, default=None, help="Optional path to CSV with sample labels.")
    parser.add_argument("--id-col", type=int, default=0, help="Index of ID column to drop from features.")
    parser.add_argument("--n-variable-genes", type=int, default=2000, help="Number of variable genes to select.")
    parser.add_argument("--n-pca-components", type=int, default=50, help="Number of PCA components to retain.")
    parser.add_argument("--k-min", type=int, default=2, help="Minimum number of mixture components.")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum number of mixture components.")
    parser.add_argument("--covariance-type", type=str, default="full", choices=["full","diag"], help="Covariance type for GMM.")
    parser.add_argument("--n-init", type=int, default=5, help="Number of random initializations for EM.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    # Load and preprocess data
        X, y = load_and_preprocess(
        args.features,
        args.labels,
        args.n_variable_genes,
        args.n_pca_components,
        args.random_state,
    )

    # Define range of K values to evaluate
    ks = range(args.k_min, args.k_max + 1)

    # Perform model selection using BIC
    best_model, results = select_gmm_bic(
        X,
             k_range=ks,
        n_init=args.n_init,
        covariance_type=args.covariance_type,
        random_state=args.random_state,
    )

    # Display results table
    print_bic_results(results)
    print(f"Best number of components: {best_model.n_components}")
    print(f"BIC for best model: {best_model.bic(X):.2f}")

    # Evaluate clustering performance if labels are provided
    if y is not None:
        labels_pred = best_model.predict(X)
        ari = compute_ari(y, labels_pred)
        print(f"Adjusted Rand Index (ARI): {ari:.3f}")

if __name__ == "__main__":
    main()
