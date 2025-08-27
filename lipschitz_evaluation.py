import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dataloader_ids import drop_columns


def compute_density_matrix(document_vectors, weights=None):
    """
    Density matrix encoding function for NumPy arrays with weighted contributions.
    document_vectors: NumPy array of shape (n, m), where n is the number of vectors, and m is their dimension.
    weights: NumPy array of shape (n,), representing normalized weights for each vector.
    Returns the density matrix of shape (m, m) with trace normalized to 1.
    """
    # Check if weights are provided
    document_vectors = np.array(document_vectors)
    if weights is None:
        weights = np.ones(document_vectors.shape[0]) / document_vectors.shape[0]  # Uniform weights

    # Normalize the vectors
    norms = np.linalg.norm(document_vectors, axis=1, keepdims=True)
    normalized_vectors = document_vectors / norms

    # Compute the weighted density matrix
    density_matrix = np.sum(
        weights[i] * np.outer(normalized_vectors[i], normalized_vectors[i]) 
        for i in range(len(normalized_vectors))
    )

    # Normalize the trace to 1
    trace = np.trace(density_matrix)
    if trace > 0:
        density_matrix /= trace

    return density_matrix


def compute_stats_encoding(document_vectors):
    document_array = np.array(document_vectors)
    min_vec = np.min(document_array, axis=0)
    max_vec = np.max(document_array, axis=0)
    mean_vec = np.mean(document_array, axis=0)
    var_vec = np.var(document_array, axis=0)
    std_vec = np.std(document_array, axis=0)
    stats_encoding = np.concatenate((min_vec, max_vec, mean_vec, var_vec, std_vec))
    return stats_encoding

def frobenius_norm_difference(matrix_a, matrix_b):
    return np.linalg.norm(matrix_a - matrix_b, 'fro')

def euclidean_distance(vec_a, vec_b):
    return np.linalg.norm(vec_a - vec_b)

def document_distance(doc_x, doc_y):
    return np.sum(np.linalg.norm(x - y) for x, y in zip(doc_x, doc_y))

def load_and_prepare_data(dataset, noise_level, seed=None):
    """
    Load and prepare real dataset for DM and Stats encoding experiments.
    Args:
        dataset: Name of the dataset ('unsw-nb15', 'mirai').
        noise_level: Noise level to add to perturbed document vectors.
        num_trials: Number of samples to draw for evaluation.
        seed: Random seed for reproducibility.
    Returns:
        data_pairs: List of tuples where each tuple contains (base_doc, perturbed_doc).
    """
    # Set the file path based on the dataset
    if dataset == 'unsw-nb15':
        fp = 'datasets/unsw-nb15/DM_STATS_COMBINED.csv'
    elif dataset == 'mirai':
        fp = 'datasets/mirai/DM_STATS_COMBINED.csv'
    else:
        raise ValueError("Invalid dataset")

    # Load data
    data = pd.read_csv(fp).fillna(0)
    data = data.drop(columns=['frame.time_epoch', 'frame.date_time', 'dm_prob', 'dm_prob_softmax', 't_delta', 'cumulative_t_delta'], errors='ignore')

    grouped = list(data.groupby('flow_window'))

    if seed is not None:
        np.random.seed(seed)

    # Aggregate data by flow_window
    data_pairs_dm = []
    data_pairs_stats = []
    for _, group in grouped:
        group_array = group.drop(columns=['label', 'flow_window', 'dm_prob_softmax'], errors='ignore').to_numpy(dtype=np.float32)

        # Normalize the base document
        base_doc_dm = [vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec for vec in group_array]
        perturbed_doc_dm = [
            vec + np.random.normal(0, noise_level, vec.shape) for vec in base_doc_dm
        ]

        base_doc_stats = [vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec for vec in group_array]
        perturbed_doc_stats = [
            vec + np.random.normal(0, noise_level, vec.shape) for vec in base_doc_stats
        ]

        data_pairs_dm.append((base_doc_dm, perturbed_doc_dm))
        data_pairs_stats.append((base_doc_stats, perturbed_doc_stats))

    return data_pairs_dm, data_pairs_stats


def generate_synthetic_data(n, m, dim, noise_level):
    """
    Generate synthetic base and perturbed document vectors.
    Args:
        n: Number of base document vectors.
        m: Number of perturbed document vectors.
        dim: Dimensionality of vectors.
        noise_level: Level of noise to add for perturbations.

    Returns:
        base_doc, perturbed_doc: Normalized base and perturbed documents.
    """
    base_doc = [np.random.rand(dim) for _ in range(n)]
    perturbed_doc = [np.random.rand(dim) + np.random.normal(0, noise_level, dim) for _ in range(m)]

    # Normalize vectors
    base_doc = [vec / np.linalg.norm(vec) for vec in base_doc]
    perturbed_doc = [vec / np.linalg.norm(vec) for vec in perturbed_doc]

    return base_doc, perturbed_doc


def compute_metrics(base_doc_dm, perturbed_doc_dm, base_doc_stats, perturbed_doc_stats):
    """
    Compute DM and Stats encodings and their distances.
    Args:
        base_doc: Base document vectors.
        perturbed_doc: Perturbed document vectors.

    Returns:
        dm_dist, stats_dist, doc_dist: Frobenius norm, Euclidean distance, and document distances.
    """
    dm_base = compute_density_matrix(base_doc_dm)
    dm_perturbed = compute_density_matrix(perturbed_doc_dm)
    stats_base = compute_stats_encoding(base_doc_stats)
    stats_perturbed = compute_stats_encoding(perturbed_doc_stats)

    dm_doc_dist = document_distance(base_doc_dm, perturbed_doc_dm)
    stats_doc_dist = document_distance(base_doc_stats, perturbed_doc_stats)
    dm_dist = frobenius_norm_difference(dm_base, dm_perturbed)
    stats_dist = euclidean_distance(stats_base, stats_perturbed)

    return dm_dist, stats_dist, dm_doc_dist, stats_doc_dist

def run_experiment(dataset, noise_level):
    """
    Run an experiment using real data.
    Args:
        dataset: Name of the dataset.
        noise_level: Noise level for perturbations.
        num_trials: Number of Monte Carlo trials.
    Returns:
        dm_ratios, stats_ratios: Lists of LC ratios for DM and Stats methods.
    """
    data_pairs_dm, data_pairs_stats = load_and_prepare_data(dataset, noise_level, seed=42)
    dm_ratios = []
    stats_ratios = []

    for (base_doc_dm, perturbed_doc_dm), (base_doc_stats, perturbed_doc_stats) in zip(data_pairs_dm, data_pairs_stats):
        dm_dist, stats_dist, dm_doc_dist, stats_doc_dist = compute_metrics(base_doc_dm, perturbed_doc_dm, base_doc_stats, perturbed_doc_stats )

        # Calculate and store LC ratios
        if dm_doc_dist != 0:
            dm_ratios.append(dm_dist / dm_doc_dist)
        if stats_doc_dist != 0:
            stats_ratios.append(stats_dist / stats_doc_dist)

    return dm_ratios, stats_ratios



def visualize_results(dm_ratios, stats_ratios, dataset, n, m, dim, noise_level):
    """
    Visualize and save results for LC ratio distributions.
    Args:
        dm_ratios, stats_ratios: Lists of LC ratios for DM and Stats methods.
        dataset: dataset of the experiment.
        n, m, dim, noise_level: Experiment configuration.
    """
    plt.figure(figsize=(14, 6))

    # Histogram for DM and Stats LC ratios
    plt.subplot(1, 2, 1)
    sns.histplot(dm_ratios, kde=True, color='blue', label='DM Encoding', stat='density')
    sns.histplot(stats_ratios, kde=True, color='orange', label='Stats Encoding', stat='density')
    plt.xscale("log")
    plt.xlabel("Empirical Lipschitz Constant Ratio")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Distribution of Empirical Lipschitz Ratios")

    # Box Plot Comparison
    plt.subplot(1, 2, 2)
    sns.boxplot(data=[dm_ratios, stats_ratios], palette=['blue', 'orange'], notch=True)
    plt.xticks([0, 1], ['DM Encoding', 'Stats Encoding'])
    plt.ylabel("Empirical Lipschitz Constant Ratio")
    plt.title("Empirical LC Ratio Comparison")

    # Add an overarching title for the entire figure
    plt.suptitle(f"{dataset}: n = {n}, m = {m}, dim = {dim}, noise = {noise_level}", fontsize=16)

    # Adjust layout and save/show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for suptitle

    results_dir = f"results/lipschitz_ratios/Plots/{dataset}"
    os.makedirs(results_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    file_path = os.path.join(
        results_dir, f'lc_ratio_{dataset}_n{n}_m{m}_dim{dim}_noise{noise_level}.png'
    )
    plt.savefig(file_path)
    print(f"Plot saved to: {file_path}")
    plt.close()


def print_summary_stats(dm_ratios, stats_ratios, name, n, m, dim, noise_level):
    """
    Print summary statistics for an experiment.
    Args:
        dm_ratios, stats_ratios: Lists of LC ratios for DM and Stats methods.
        name, n, m, dim, noise_level: Experiment configuration.
    """
    print(f"\nExperiment: {name}")
    print(f"Parameters: n = {n}, m = {m}, dim = {dim}, noise_level = {noise_level}")
    print(f"Mean LC Ratio (DM Encoding): {np.mean(dm_ratios):.4f}")
    print(f"Mean LC Ratio (Stats Encoding): {np.mean(stats_ratios):.4f}")
    print(f"Variance LC Ratio (DM Encoding): {np.var(dm_ratios):.4f}")
    print(f"Variance LC Ratio (Stats Encoding): {np.var(stats_ratios):.4f}\n")


def save_results_to_file(dm_ratios, stats_ratios, dataset, n, m, dim, noise_level):
    """
    Save DM and Stats LC ratios to a CSV file.
    Args:
        dm_ratios, stats_ratios: Lists of LC ratios for DM and Stats methods.
        name, n, m, dim, noise_level: Experiment configuration.
    """
    results_dir = f"results/lipschitz_ratios/Ratios/{dataset}"
    os.makedirs(results_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    file_path = os.path.join(
        results_dir, f"{dataset}_n{n}_m{m}_dim{dim}_noise{noise_level}.csv"
    )
    data = {
        "DM_Ratios": dm_ratios,
        "Stats_Ratios": stats_ratios,
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Results saved to: {file_path}")


if __name__ == "__main__":

    for dataset in ['mirai', 'unsw-nb15']:
        experiments = [
            {"dataset": dataset, "n": 10, "m": 10, "dim": 10, "noise_level": 0.01, "name": "Small_Noise"},
            {"dataset": dataset, "n": 10, "m": 10, "dim": 10, "noise_level": 0.1, "name": "Moderate Noise"},
            {"dataset": dataset, "n": 10, "m": 10, "dim": 10, "noise_level": 0.5, "name": "High Noise"}
        ]

        # Monte Carlo Simulation parameters
        num_trials = 1000
        np.random.seed(42)

        # Run each experiment
        for exp in experiments:
            dm_ratios, stats_ratios = run_experiment(exp["dataset"], exp["noise_level"])
            visualize_results(dm_ratios, stats_ratios, exp["dataset"], exp["n"], exp["m"], exp["dim"], exp["noise_level"])
            print_summary_stats(dm_ratios, stats_ratios, exp["dataset"], exp["n"], exp["m"], exp["dim"], exp["noise_level"])
            save_results_to_file(dm_ratios, stats_ratios, exp["dataset"], exp["n"], exp["m"], exp["dim"], exp["noise_level"])


    for dataset in ['mirai', 'unsw-nb15']:
        fp_1 = f"results/lipschitz_ratios/Ratios/{dataset}/{dataset}_n10_m10_dim10_noise0.01.csv"
        fp_2 = f"results/lipschitz_ratios/Ratios/{dataset}/{dataset}_n10_m10_dim10_noise0.1.csv"
        fp_3 = f"results/lipschitz_ratios/Ratios/{dataset}/{dataset}_n10_m10_dim10_noise0.5.csv"

        df_1 = pd.read_csv(fp_1)
        df_2 = pd.read_csv(fp_2)
        df_3 = pd.read_csv(fp_3)

        dm_ratios_1 = df_1["DM_Ratios"]
        stats_ratios_1 = df_1["Stats_Ratios"]
        dm_ratios_2 = df_2["DM_Ratios"]
        stats_ratios_2 = df_2["Stats_Ratios"]
        dm_ratios_3 = df_3["DM_Ratios"]
        stats_ratios_3 = df_3["Stats_Ratios"]


        noise_levels = [0.01, 0.1, 0.5]
        for i, (dm_ratios, stats_ratios, noise) in enumerate(
            zip([dm_ratios_1, dm_ratios_2, dm_ratios_3], 
                [stats_ratios_1, stats_ratios_2, stats_ratios_3], 
                noise_levels)):
            plt.figure(figsize=(8, 6))
            sns.histplot(dm_ratios, kde=True, color='blue', label='DM Encoding', stat='density')
            sns.histplot(stats_ratios, kde=True, color='orange', label='Stats Encoding', stat='density')
            plt.xscale("log")
            plt.xlabel("Empirical Lipschitz Constant Ratio", fontsize=14)
            plt.ylabel("Density", fontsize=14)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(f'lipschitz_{dataset}_{noise}.png')
            plt.close()

        plt.figure(figsize=(16, 6))  # Slightly wider for better spacing

        # Histogram for DM and Stats LC ratios
        plt.subplot(1, 3, 1)
        sns.histplot(dm_ratios_1, kde=True, color='blue', label='DM Encoding', stat='density')
        sns.histplot(stats_ratios_1, kde=True, color='orange', label='Stats Encoding', stat='density')
        plt.xscale("log")
        plt.xlabel("Empirical Lipschitz Constant Ratio", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(fontsize=12)

        plt.subplot(1, 3, 2)
        sns.histplot(dm_ratios_2, kde=True, color='blue', label='DM Encoding', stat='density')
        sns.histplot(stats_ratios_2, kde=True, color='orange', label='Stats Encoding', stat='density')
        plt.xscale("log")
        plt.xlabel("Empirical Lipschitz Constant Ratio", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(fontsize=12)

        plt.subplot(1, 3, 3)
        sns.histplot(dm_ratios_3, kde=True, color='blue', label='DM Encoding', stat='density')
        sns.histplot(stats_ratios_3, kde=True, color='orange', label='Stats Encoding', stat='density')
        plt.xscale("log")
        plt.xlabel("Empirical Lipschitz Constant Ratio", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(fontsize=12)

        # Adjust layout and save/show the plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for suptitle
        plt.savefig(f'lipschitz_{dataset}.png')