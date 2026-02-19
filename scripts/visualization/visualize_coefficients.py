import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scripts.analysis.sanity_check import get_bernstein_coefficients


NUM_RULES = 2**18  # 262,144 rules


def collect_all_coefficients():
    """
    Enumerate all rules and collect:
    - Full coefficient matrix (262144 x 10)
    """
    coeff_matrix = np.zeros((NUM_RULES, 10))

    print("Collecting Bernstein coefficients for all rules...")

    for mask in tqdm(range(NUM_RULES)):
        coeff_matrix[mask] = get_bernstein_coefficients(mask)

    return coeff_matrix


def plot_global_histogram(coeff_matrix):
    """
    Histogram of all 2,621,440 coefficient values.
    """
    all_values = coeff_matrix.flatten()

    plt.figure()
    plt.hist(all_values, bins=20)
    plt.xlabel("Coefficient value (")
    plt.ylabel("Frequency")
    plt.title("Histogram of All Bernstein Coefficients")
    plt.savefig("figures/histogram_bernstein_coefficients.png", dpi=300)
    plt.show()



def plot_heatmap_sample(coeff_matrix, sample_size=2000):
    """
    Plot heatmap of a random subset of rules.
    """
    np.random.seed(0)
    indices = np.random.choice(NUM_RULES, sample_size, replace=False)
    sample = coeff_matrix[indices]

    plt.figure()
    plt.imshow(sample, aspect='auto')
    plt.colorbar()
    plt.xlabel("Coefficient Index m")
    plt.ylabel("Sampled Rule Index")
    plt.title("Heatmap of Sampled Bernstein Coefficients")
    plt.savefig("figures/heatmap_bernstein_coefficients.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    coeff_matrix = collect_all_coefficients()
    plot_global_histogram(coeff_matrix)
    plot_heatmap_sample(coeff_matrix)
