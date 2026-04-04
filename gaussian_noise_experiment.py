

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from inverse_fit import fit_rule_from_density_curve
from rule import parse, encode
from forward_map import AnalyticForwardMap


def run_gaussian_noise_experiment(num_den=25, trials=50, sigma_levels=None, rule_str="B3/S23"):
    """
    Stress-tests the inverse solver by adding Gaussian noise to the density response.

    This experiment measures Recovery Rate, True-Rule Gap, and RMSE as data
    quality degrades. It simulates a scenario where density measurements are imperfect.

    Args:
        num_den (int): Number of density points sampled between 0.05 and 0.95.
        trials (int): Number of Monte Carlo simulations per noise level.
        sigma_levels (list): Standard deviations of Gaussian noise to test.
        rule_str (str): The ground truth rule in B/S notation.

    Returns:
        pd.DataFrame: Summary statistics for each noise level.
    """

    if sigma_levels is None:
        sigma_levels = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    # Define Ground Truth
    true_mask = encode(*parse(rule_str))
    d_vals = np.linspace(0.05, 0.95, num_den)

    # Precompute analytic forward map and clean predictions once
    fmap = AnalyticForwardMap(d_vals)
    p_clean = fmap.predict(true_mask)

    results = []
    all_collisions = Counter()

    print(f"\n[Exp 1] Starting Gaussian Noise Experiment for {rule_str}...")

    for sigma in sigma_levels:
        correct_count, gaps, rmses = 0, [], []

        for t in range(trials):
            # Apply Gaussian Noise: η_j ~ N(0, sigma^2) and clip values to [0, 1]
            p_noisy = np.clip(
                p_clean + np.random.normal(0, sigma, size=p_clean.shape), 0, 1
            )

            # Invoke Inverse Solver
            best_mask, top_candidates = fit_rule_from_density_curve(d_vals, p_noisy)

            # Check for Success
            if best_mask == true_mask:
                correct_count += 1
            else:
                all_collisions[top_candidates[0]['rule']] += 1

            # True-Rule Gap: 0 when correct, positive when true rule is outcompeted
            true_rule_score = np.sum((p_noisy - p_clean) ** 2)
            best_score = top_candidates[0]['score']
            true_gap = true_rule_score - best_score
            gaps.append(true_gap)

            # RMSE: stored for use in the cross-experiment summary (Step 3)
            rmse = np.sqrt(np.mean((p_noisy - p_clean) ** 2))
            rmses.append(rmse)

        results.append({
            "level":    sigma,
            "accuracy": correct_count / trials,
            "true_gap": np.mean(gaps),
            "rmse":     np.mean(rmses)
        })
        print(f"  Sigma: {sigma:.3f} | Accuracy: {(correct_count / trials) * 100:>5.1f}% | RMSE: {np.mean(rmses):.5f}")

    return pd.DataFrame(results), all_collisions, rule_str


if __name__ == "__main__":
    target_rule = "B3/S23"
    df_results, collisions, rule_name = run_gaussian_noise_experiment(rule_str=target_rule)

    # Save results and imposter data to CSV
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    clean_rule = rule_name.replace("/", "")
    df_results.to_csv(f"{data_dir}/gaussian_noise_results_{clean_rule}.csv", index=False)
    imposter_df = pd.DataFrame(collisions.most_common(), columns=['Imposter_Rule', 'Count'])
    imposter_df.to_csv(f"{data_dir}/gaussian_noise_imposters_{clean_rule}.csv", index=False)

    # Visualization
    output_dir = "figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    axes[0].plot(df_results['level'], df_results['accuracy'], marker='o', color='teal', lw=2)
    axes[0].set_title(f"Recovery Accuracy: {rule_name}")
    axes[0].set_xlabel("Perturbation Level ($\\sigma$)")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    # True-Rule Gap
    axes[1].plot(df_results['level'], df_results['true_gap'], marker='s', ls='--', color='darkorange', lw=2)
    axes[1].set_title("True-Rule Gap (Solver Confidence)")
    axes[1].set_xlabel("Perturbation Level ($\\sigma$)")
    axes[1].set_ylabel("True-Rule Gap (SSE, Log Scale)")
    axes[1].set_yscale('symlog', linthresh=1e-10)
    axes[1].grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"gaussian_noise_experiment_{clean_rule}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\n--- Top Imposter Rules (Gaussian Noise Experiment) ---")
    for rule, count in collisions.most_common(5):
        print(f"Rule {rule}: {count} times")