import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from inverse_fit import fit_rule_from_density_curve
from rule import parse, encode
from forward_map import AnalyticForwardMap


# --- EXPERIMENT 1: GAUSSIAN NOISE ---
def run_measurement_noise_experiment(num_den=25, trials=50, sigma_levels=None):
    if sigma_levels is None:
        sigma_levels = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    rule_str = "B3/S23"
    b_set, s_set = parse(rule_str)
    true_mask = encode(b_set, s_set)

    d_vals = np.linspace(0.05, 0.95, num_den)
    fmap = AnalyticForwardMap(d_vals)
    p_clean = fmap.predict(true_mask)

    results = []
    all_collisions = Counter()

    print(f"\n[Exp 1] Starting Gaussian Noise Experiment...")

    for sigma in sigma_levels:
        correct_count = 0
        margins = []
        for t in range(trials):
            noise = np.random.normal(0, sigma, size=p_clean.shape)
            p_noisy = np.clip(p_clean + noise, 0, 1)

            best_mask, top_candidates = fit_rule_from_density_curve(d_vals, p_noisy)

            if best_mask == true_mask:
                correct_count += 1
            else:
                all_collisions[top_candidates[0]['rule']] += 1

            margin = top_candidates[1]['score'] - top_candidates[0]['score']
            margins.append(margin)

        accuracy = correct_count / trials
        results.append({"sigma": sigma, "accuracy": accuracy, "margin": np.mean(margins)})
        print(f"  Sigma: {sigma:.3f} | Accuracy: {accuracy * 100:.1f}%")

    return pd.DataFrame(results), all_collisions


# --- EXPERIMENT 2: BIT-FLIP NOISE ---
def run_bitflip_experiment(num_den=25, trials=50, epsilon_levels=None):
    if epsilon_levels is None:
        epsilon_levels = [0, 0.005, 0.01, 0.02, 0.05, 0.1]

    rule_str = "B3/S23"
    b_set, s_set = parse(rule_str)
    true_mask = encode(b_set, s_set)

    d_vals = np.linspace(0.05, 0.95, num_den)
    fmap = AnalyticForwardMap(d_vals)
    p_clean = fmap.predict(true_mask)

    results = []
    all_collisions = Counter()

    print(f"\n[Exp 2] Starting Bit-Flip Experiment...")

    for eps in epsilon_levels:
        correct_count = 0
        margins = []
        for t in range(trials):
            p_noisy = p_clean * (1 - eps) + (1 - p_clean) * eps
            p_final = np.clip(p_noisy + np.random.normal(0, 0.001, size=p_noisy.shape), 0, 1)

            best_mask, top_candidates = fit_rule_from_density_curve(d_vals, p_final)

            if best_mask == true_mask:
                correct_count += 1
            else:
                all_collisions[top_candidates[0]['rule']] += 1

            margin = top_candidates[1]['score'] - top_candidates[0]['score']
            margins.append(margin)

        accuracy = correct_count / trials
        results.append({"epsilon": eps, "accuracy": accuracy, "margin": np.mean(margins)})
        print(f"  Epsilon: {eps:.3f} | Accuracy: {accuracy * 100:.1f}%")

    return pd.DataFrame(results), all_collisions


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    df_noise, col_noise = run_measurement_noise_experiment()
    df_flip, col_flip = run_bitflip_experiment()

    print("\n" + "=" * 50)
    print("DETAILED IMPOSTER ANALYSIS (TOP 5)")
    print("=" * 50)

    print("\n[Gaussian Noise Experiment]")
    for rule, count in col_noise.most_common(5):
        print(f"  Rule {rule:<10} | Tricked solver {count} times")

    print("\n[Bit-Flip Noise Experiment]")
    for rule, count in col_flip.most_common(5):
        print(f"  Rule {rule:<10} | Tricked solver {count} times")

    # --- VISUALIZATION: 4-PANEL LAYOUT ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Exp 1 Accuracy
    axes[0, 0].plot(df_noise['sigma'], df_noise['accuracy'], 'b-o')
    axes[0, 0].set_title("Exp 1: Measurement Noise (Accuracy)")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].grid(True, alpha=0.3)

    # Exp 1 Margin
    axes[0, 1].plot(df_noise['sigma'], df_noise['margin'], 'r--s')
    axes[0, 1].set_title("Exp 1: Solver Margin (Confidence)")
    axes[0, 1].set_ylabel("Score Gap (Log Scale)")
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    # Exp 2 Accuracy
    axes[1, 0].plot(df_flip['epsilon'], df_flip['accuracy'], 'g-o')
    axes[1, 0].set_title("Exp 2: Sensor Bit-Flips (Accuracy)")
    axes[1, 0].set_xlabel("Epsilon (Flip Rate)")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].grid(True, alpha=0.3)

    # Exp 2 Margin - FIXED LINE HERE
    axes[1, 1].plot(df_flip['epsilon'], df_flip['margin'], color='orange', linestyle='--', marker='s')
    axes[1, 1].set_title("Exp 2: Solver Margin (Confidence)")
    axes[1, 1].set_xlabel("Epsilon (Flip Rate)")
    axes[1, 1].set_ylabel("Score Gap (Log Scale)")
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if not os.path.exists("figures"): os.makedirs("figures")
    plt.savefig("figures/combined_experiments.png")
    plt.show()