import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from scipy.ndimage import gaussian_filter
from inverse_fit import fit_rule_from_density_curve
from rule import parse, encode, Rule
from life_like import step

def run_mismatch_experiment(N=100, num_den=25, trials=20, rule_str="B3/S23"):
    """
        Stress-tests the solver against Model Mismatch (Spatial Correlation).
        Directly violates the assumption that the initial configuration is
        generated as i.i.d. Bernoulli(d).
    """

    # Initialize experimental parameters
    sigmas = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # Gaussian smoothing strengths
    true_mask = encode(*parse(rule_str))
    rule_obj = Rule(true_mask)
    d_vals = np.linspace(0.05, 0.95, num_den)
    results = []
    all_collisions = Counter()  # Tracks names of rules that 'tie' or 'beat' the true rule

    print(f"\n[Exp 3] Starting Mismatch Experiment for {rule_str}...")

    for s in sigmas:
        correct_count, margins = 0, []
        for t in range(trials):
            p_observed = []
            for d in d_vals:
                # Generate Structured Initial State
                raw = np.random.rand(N, N)

                # Apply Gaussian filter
                smoothed = gaussian_filter(raw, sigma=s, mode='wrap') if s > 0 else raw
                # Threshold the smoothed grid to preserve the marginal density 'd'
                X0 = (smoothed >= np.percentile(smoothed, 100 * (1 - d))).astype(np.uint8)
                # Run one CA step and measure the resulting density
                p_observed.append(np.mean(step(X0, rule_obj)))

            # Invoke solver
            best_mask, top_candidates = fit_rule_from_density_curve(d_vals, np.array(p_observed))
            # Check for Success
            if best_mask == true_mask:
                correct_count += 1
            else:
                all_collisions[top_candidates[0]['rule']] += 1
            margins.append(top_candidates[1]['score'] - top_candidates[0]['score'])

        results.append({"level": s, "accuracy": correct_count/trials, "margin": np.mean(margins)})
        print(f"  Sigma: {s:.2f} | Accuracy: {(correct_count/trials)*100:>5.1f}%")

    return pd.DataFrame(results), all_collisions, rule_str


if __name__ == "__main__":
    target_rule = "B3678/S34678"
    # Capture the three return values
    df_mismatch, all_collisions, rule_name = run_mismatch_experiment(rule_str=target_rule)

    # Save Imposter Data to CSV
    data_dir = "data"
    if not os.path.exists(data_dir): os.makedirs(data_dir)

    clean_rule = rule_name.replace("/", "")
    imposter_df = pd.DataFrame(all_collisions.most_common(), columns=['Imposter_Rule', 'Count'])
    imposter_df.to_csv(f"{data_dir}/mismatch_imposters_{clean_rule}.csv", index=False)

    # Visualization
    output_dir = "figures"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    clean_rule = rule_name.replace("/", "")

    plt.figure(figsize=(12, 5))

    # Accuracy Plot (Teal)
    plt.subplot(1, 2, 1)
    plt.plot(df_mismatch['level'], df_mismatch['accuracy'], marker='o', color='teal', lw=2)
    plt.title(f"Recovery Accuracy: {rule_name}")
    plt.xlabel(r"Smoothing Strength ($\sigma$)") # Specific label for Mismatch
    plt.ylabel("Success Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)

    # Margin Plot (Dark Orange)
    plt.subplot(1, 2, 2)
    plt.plot(df_mismatch['level'], df_mismatch['margin'], marker='s', ls='--', color='darkorange', lw=2)
    plt.title("Solver Margin (Confidence Gap)")
    plt.xlabel(r"Smoothing Strength ($\sigma$)")
    plt.ylabel("SSE Difference (Log Scale)")
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    # Save using the standardized thesis prefix
    save_path = os.path.join(output_dir, f"mismatch_experiment_{clean_rule}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\n--- Top Imposter Rules (Mismatch Experiment) ---")
    for rule, count in all_collisions.most_common(5):
        print(f"Rule {rule}: {count} times")