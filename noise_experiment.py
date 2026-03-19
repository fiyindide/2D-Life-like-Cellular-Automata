import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from inverse_fit import fit_rule_from_density_curve
from rule import parse, encode
from forward_map import AnalyticForwardMap


def run_measurement_noise_experiment(num_den=25, trials=50, sigma_levels=None):
    """
    Stress-tests the inverse solver by adding Gaussian noise to the density response.

    This experiment measures the 'Recovery Rate' and 'Solver Confidence' as data
    quality degrades. It simulates a scenario where density measurements are imperfect.

    Args:
        num_den (int): Number of density points sampled between 0.05 and 0.95.
        trials (int): Number of Monte Carlo simulations per noise level.
        sigma_levels (list): Standard deviations of Gaussian noise to test.

    Returns:
        pd.DataFrame: Summary statistics for each noise level.
    """
    # Initialize experimental parameters
    if sigma_levels is None:
        # Range from perfect data (0) to noise level 0.1( i.e. 10% std dev)
        sigma_levels = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    # Define Ground Truth
    rule_str = "B3/S23"  # Game of Life
    b_set, s_set = parse(rule_str)
    true_mask = encode(b_set, s_set)

    # Pre-compute the 'Perfect' Signal.
    d_vals = np.linspace(0.05, 0.95, num_den)
    fmap = AnalyticForwardMap(d_vals)
    p_clean = fmap.predict(true_mask)

    results = []
    all_collisions = Counter()  # Tracks names of rules that 'tie' or 'beat' the true rule

    print(f"Starting Noise Experiment for {rule_str}...")

    # Iterate through different levels of Sigma
    for sigma in sigma_levels:
        correct_count = 0
        margins = []

        # Monte Carlo trials to get averages
        for t in range(trials):
            # Apply Gaussian Noise: η_j ~ N(0, sigma^2)
            noise = np.random.normal(0, sigma, size=p_clean.shape)
            # Clip values to [0, 1]
            p_noisy = np.clip(p_clean + noise, 0, 1)

            # Invoke Inverse Solver: Brute-force search through all 262,144 rules
            best_mask, top_candidates = fit_rule_from_density_curve(d_vals, p_noisy, sigma=None)

            # Check for Success
            if best_mask == true_mask:
                correct_count += 1
            else:
                # Rule Collision: Record the 'Imposter' that looked better than the truth
                wrong_rule = top_candidates[0]['rule']
                all_collisions[wrong_rule] += 1

            # Quantify Solver Confidence
            margin = top_candidates[1]['score'] - top_candidates[0]['score']
            margins.append(margin)

        # Aggregate results for this specific noise level
        accuracy = correct_count / trials
        avg_margin = np.mean(margins)
        results.append({
            "sigma": sigma,
            "accuracy": accuracy,
            "margin": avg_margin
        })

        print(f"Sigma: {sigma:.3f} | Accuracy: {accuracy * 100:.1f}%")

    # If errors occurred, print the most common Rules that caused confusion.
    if all_collisions:
        print("\n--- TOP IMPOSTER RULES (COLLISIONS) ---")
        for rule, count in all_collisions.most_common(5):
            print(f"Rule {rule} tricked the solver {count} times")

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Execute experiment and collect data
    df_results = run_measurement_noise_experiment()

    # Visualization
    plt.figure(figsize=(12, 5))

    # Plot 1: Recovery Accuracy (Phase Transition)
    plt.subplot(1, 2, 1)
    plt.plot(df_results['sigma'], df_results['accuracy'], marker='o', linestyle='-', color='b')
    plt.title(r"Recovery Accuracy vs. Noise ($\sigma$)")
    plt.xlabel(r"Noise Level ($\sigma$)")
    plt.ylabel("Success Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)

    # Plot 2: Solver Margin (Identifiability Metric)
    plt.subplot(1, 2, 2)
    plt.plot(df_results['sigma'], df_results['margin'], marker='s', linestyle='--', color='r')
    plt.title(r"Solver Margin vs. Noise ($\sigma$)")
    plt.xlabel(r"Noise Level ($\sigma$)")
    plt.ylabel("Score Gap (SSE Difference)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # Save plot
    output_dir = "figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, "thesis_noise_experiment.png")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nExperiment complete. Plot saved as '{save_path}'.")
    plt.show()

