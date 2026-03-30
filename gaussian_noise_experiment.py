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
        sigma_levels = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    # Define Ground Truth
    true_mask = encode(*parse(rule_str))
    d_vals = np.linspace(0.05, 0.95, num_den)
    fmap = AnalyticForwardMap(d_vals)
    p_clean = fmap.predict(true_mask)

    results = []
    all_collisions = Counter() # Tracks names of rules that 'tie' or 'beat' the true rule

    print(f"\n[Exp 1] Starting Gaussian Noise Experiment for {rule_str}...")

    # Iterate through different levels of Sigma
    for sigma in sigma_levels:
        correct_count, margins = 0, []

        # Monte Carlo trials to get averages
        for t in range(trials):
            # Apply Gaussian Noise: η_j ~ N(0, sigma^2) and clip values to [0, 1]
            p_noisy = np.clip(p_clean + np.random.normal(0, sigma, size=p_clean.shape), 0, 1)

            # Invoke Inverse Solver: Brute-force search through all 262,144 rules
            best_mask, top_candidates = fit_rule_from_density_curve(d_vals, p_noisy)

            # Check for Success
            if best_mask == true_mask:
                correct_count += 1
            else:
                # Rule Collision: Record the 'Imposter' that looked better than the truth
                all_collisions[top_candidates[0]['rule']] += 1
            # Quantify Solver Confidence
            margins.append(top_candidates[1]['score'] - top_candidates[0]['score'])

        # Aggregate results for this specific noise level
        results.append({"level": sigma, "accuracy": correct_count/trials, "margin": np.mean(margins)})
        print(f"  Sigma: {sigma:.3f} | Accuracy: {(correct_count/trials)*100:>5.1f}%")

    return pd.DataFrame(results), all_collisions, rule_str

if __name__ == "__main__":
    target_rule = "B3678/S34678"
    df_results, collisions, rule_name = run_gaussian_noise_experiment(rule_str=target_rule)

    # Save Imposter Data to CSV
    data_dir = "data"
    if not os.path.exists(data_dir): os.makedirs(data_dir)

    clean_rule = rule_name.replace("/", "")
    imposter_df = pd.DataFrame(collisions.most_common(), columns=['Imposter_Rule', 'Count'])
    imposter_df.to_csv(f"{data_dir}/gaussian_noise_imposters_{clean_rule}.csv", index=False)


    # Visualization
    output_dir = "figures"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    clean_rule = rule_name.replace("/", "")

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(df_results['level'], df_results['accuracy'], marker='o', color='teal', lw=2)
    plt.title(f"Recovery Accuracy: {rule_name}")
    plt.xlabel("Perturbation Level")
    plt.ylabel("Success Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)

    # Margin Plot
    plt.subplot(1, 2, 2)
    plt.plot(df_results['level'], df_results['margin'], marker='s', ls='--', color='darkorange', lw=2)
    plt.title("Solver Margin (Confidence Gap)")
    plt.xlabel("Perturbation Level")
    plt.ylabel("SSE Difference (Log Scale)")
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"gaussian_noise_experiment_{clean_rule}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\n--- Top Imposter Rules (Gaussian Noise Experiment) ---")
    for rule, count in collisions.most_common(5):
        print(f"Rule {rule}: {count} times")
