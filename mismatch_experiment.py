import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from scipy.ndimage import gaussian_filter
from inverse_fit import fit_rule_from_density_curve
from rule import parse, encode, Rule
from life_like import step
from forward_map import AnalyticForwardMap


def run_mismatch_experiment(N=100, num_den=25, trials=20, rule_str="B3/S23"):
    """
    Stress-tests the solver against Model Mismatch (Spatial Correlation).
    Directly violates the assumption that the initial configuration is
    generated as i.i.d. Bernoulli(d).

    Args:
        N (int): Grid size (N x N).
        num_den (int): Number of density points sampled between 0.05 and 0.95.
        trials (int): Number of Monte Carlo simulations per smoothing level.
        rule_str (str): The ground truth rule in B/S notation.

    Returns:
        pd.DataFrame: Summary statistics for each smoothing level.
    """
    sigmas = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    # Define Ground Truth
    true_mask = encode(*parse(rule_str))
    rule_obj = Rule(true_mask)
    d_vals = np.linspace(0.05, 0.95, num_den)

    # Precompute analytic forward map and clean predictions once
    fmap = AnalyticForwardMap(d_vals)
    p_clean = fmap.predict(true_mask)

    results = []
    all_collisions = Counter()

    print(f"\n[Exp 3] Starting Mismatch Experiment for {rule_str}...")

    for s in sigmas:
        correct_count, gaps, rmses = 0, [], []

        for t in range(trials):
            p_observed = []
            for d in d_vals:
                # Generate Structured Initial State
                raw = np.random.rand(N, N)

                # Apply Gaussian filter to introduce spatial correlation
                smoothed = gaussian_filter(raw, sigma=s, mode='wrap') if s > 0 else raw

                # Threshold to preserve marginal density d
                X0 = (smoothed >= np.percentile(smoothed, 100 * (1 - d))).astype(np.uint8)

                # Run one CA step and measure the resulting density
                p_observed.append(np.mean(step(X0, rule_obj)))

            p_observed = np.array(p_observed)

            # Invoke solver
            best_mask, top_candidates = fit_rule_from_density_curve(d_vals, p_observed)

            # Check for Success
            if best_mask == true_mask:
                correct_count += 1
            else:
                all_collisions[top_candidates[0]['rule']] += 1

            # True-Rule Gap: 0 when correct, positive when true rule is outcompeted
            true_rule_score = np.sum((p_observed - p_clean) ** 2)
            best_score = top_candidates[0]['score']
            true_gap = true_rule_score - best_score
            gaps.append(true_gap)

            # RMSE: stored for use in the cross-experiment summary (Step 3)
            rmse = np.sqrt(np.mean((p_observed - p_clean) ** 2))
            rmses.append(rmse)

        results.append({
            "level":    s,
            "accuracy": correct_count / trials,
            "true_gap": np.mean(gaps),
            "rmse":     np.mean(rmses)
        })
        print(f"  Sigma: {s:.2f} | Accuracy: {(correct_count / trials) * 100:>5.1f}% | RMSE: {np.mean(rmses):.5f}")

    return pd.DataFrame(results), all_collisions, rule_str


if __name__ == "__main__":
    target_rule = "B3/S23"
    df_mismatch, all_collisions, rule_name = run_mismatch_experiment(rule_str=target_rule)

    # Save results and imposter data to CSV
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    clean_rule = rule_name.replace("/", "")
    df_mismatch.to_csv(f"{data_dir}/mismatch_results_{clean_rule}.csv", index=False)
    imposter_df = pd.DataFrame(all_collisions.most_common(), columns=['Imposter_Rule', 'Count'])
    imposter_df.to_csv(f"{data_dir}/mismatch_imposters_{clean_rule}.csv", index=False)

    # Visualization
    output_dir = "figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    axes[0].plot(df_mismatch['level'], df_mismatch['accuracy'], marker='o', color='teal', lw=2)
    axes[0].set_title(f"Recovery Accuracy: {rule_name}")
    axes[0].set_xlabel(r"Smoothing Strength ($\sigma$)")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    # True-Rule Gap
    axes[1].plot(df_mismatch['level'], df_mismatch['true_gap'], marker='s', ls='--', color='darkorange', lw=2)
    axes[1].set_title("True-Rule Gap (Solver Confidence)")
    axes[1].set_xlabel(r"Smoothing Strength ($\sigma$)")
    axes[1].set_ylabel("True-Rule Gap (SSE, Log Scale)")
    axes[1].set_yscale('symlog', linthresh=1e-10)
    axes[1].grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"mismatch_experiment_{clean_rule}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\n--- Top Imposter Rules (Mismatch Experiment) ---")
    for rule, count in all_collisions.most_common(5):
        print(f"Rule {rule}: {count} times")