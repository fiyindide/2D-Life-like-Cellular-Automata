import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from inverse_fit import fit_rule_from_density_curve
from rule import parse, encode
from forward_map import AnalyticForwardMap


def run_bitflip_experiment(num_den=25, trials=50, epsilon_levels=None):
    """
    Experiment 2: Observation Corruption

    This experiment tests the inverse solver's vulnerability to sensor corruption.
    Unlike measurement noise (which happens after the density is measured),
    bit-flips happen at the cell level, shifting the expected value of the density.

    Args:
        num_den (int): M density points sampled between [0.05, 0.95].
        trials (int): Number of Monte Carlo simulations per epsilon level.
        epsilon_levels (list): Probabilities of a single cell being flipped (0 to 1 or 1 to 0).
    """
    if epsilon_levels is None:
        # sensor corruption up to 10%
        epsilon_levels = [0, 0.005, 0.01, 0.02, 0.05, 0.1]

    # Initialize the 'Ground Truth' Rule (Game of Life)
    rule_str = "B3/S23"
    b_set, s_set = parse(rule_str)
    true_mask = encode(b_set, s_set)

    # Initialize the density range and the Analytic Map
    d_vals = np.linspace(0.05, 0.95, num_den)
    fmap = AnalyticForwardMap(d_vals)
    p_clean = fmap.predict(true_mask)

    results = []
    all_collisions = Counter()  # Dictionary to track which rules 'beat' the truth

    print(f"Starting Bit-Flip Experiment for {rule_str}...")

    for eps in epsilon_levels:
        correct_count = 0
        margins = []

        for t in range(trials):


            # Apply the Bit-Flip Formula: Each 'True' 1 has a (1-eps) chance of staying 1. Each 'True' 0 has an (eps) chance of becoming 1.
            p_noisy = p_clean * (1 - eps) + (1 - p_clean) * eps

            # RE-INTRODUCE SAMPLING VARIANCE
            sampling_noise = np.random.normal(0, 0.001, size=p_noisy.shape)
            p_final = np.clip(p_noisy + sampling_noise, 0, 1)


            # The solver tries to find the rule in the 262,144 library that best fits the 'p_final' curve.
            best_mask, top_candidates = fit_rule_from_density_curve(d_vals, p_final)

            # Accuracy Check
            if best_mask == true_mask:
                correct_count += 1
            else:
                # Store the most likely imposter rule
                all_collisions[top_candidates[0]['rule']] += 1

            # Margin Calculation: Measures the gap between #1 and #2 candidates.
            margin = top_candidates[1]['score'] - top_candidates[0]['score']
            margins.append(margin)

        # Store mean stats for this Epsilon level
        results.append({
            "epsilon": eps,
            "accuracy": correct_count / trials,
            "margin": np.mean(margins)
        })
        print(f"Epsilon: {eps:.3f} | Accuracy: {(correct_count / trials) * 100:.1f}%")

    # Output detailed failure report if accuracy dropped below 100%
    if all_collisions:
        print("\n--- TOP IMPOSTER RULES (BIT-FLIP COLLISION) ---")
        for rule, count in all_collisions.most_common(5):
            print(f"Rule {rule} picked {count} times")

    return pd.DataFrame(results), all_collisions


if __name__ == "__main__":
    # Run simulation and obtain results
    df_results, all_collisions = run_bitflip_experiment()

    output_dir = "figures"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # --- VISUALIZATION BLOCK ---
    plt.figure(figsize=(12, 5))

    # SUBPLOT 1: RECOVERY RATE (Accuracy Phase Transition)
    plt.subplot(1, 2, 1)
    plt.plot(df_results['epsilon'], df_results['accuracy'], marker='o', linestyle='-', color='teal')
    plt.title(r"Recovery Accuracy vs. Bit-Flip Rate ($\epsilon$)")
    plt.xlabel(r"Flip Probability ($\epsilon$)")
    plt.ylabel("Success Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)

    # SUBPLOT 2: SOLVER CONFIDENCE
    plt.subplot(1, 2, 2)
    plt.plot(df_results['epsilon'], df_results['margin'], marker='s', linestyle='--', color='darkorange')
    plt.title(r"Solver Margin vs. Flip Rate ($\epsilon$)")
    plt.xlabel(r"Flip Probability ($\epsilon$)")
    plt.ylabel("Score Gap (SSE Difference)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Final Save and Display
    save_path = os.path.join(output_dir, "thesis_bitflip_experiment_2.png")
    plt.savefig(save_path)

    print(f"\nExperiment complete. Plot saved as '{save_path}'.")
    plt.show()