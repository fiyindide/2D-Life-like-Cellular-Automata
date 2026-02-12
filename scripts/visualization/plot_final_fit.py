
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forward_map import AnalyticForwardMap
from rule import Rule, encode, parse, format_rule
from inverse_fit import fit_rule_from_density_curve
from data_gen import CONFIG, measure_one_step_density


def get_random_rule_str():
    """Generates a random Life-like rule string."""
    b = random.sample(range(9), random.randint(1, 4))
    s = random.sample(range(9), random.randint(1, 4))
    return f"B{''.join(map(str, sorted(b)))}/S{''.join(map(str, sorted(s)))}"


def plot_best_fit_results(num_random=3):
    """
    Produce plots for Game of Life and several random rules.
    """
    os.makedirs("figures", exist_ok=True)

    # 1. Define rules to process (GoL + Random)
    rules_to_plot = [CONFIG["rule_str"]]  # Starts with GoL (or whatever is in CONFIG)
    for _ in range(num_random):
        rules_to_plot.append(get_random_rule_str())

    for rule_str in rules_to_plot:
        print(f"Generating plot for: {rule_str}")

        # 2. Generate Data for this rule
        b_true, s_true = parse(rule_str)
        true_mask = encode(b_true, s_true)
        rule_obj = Rule(true_mask)

        # Simulate data on the fly so we don't need different CSV files
        df = measure_one_step_density(
            rule_obj,
            CONFIG['N'],
            CONFIG['d_values'],
            CONFIG['trials'],
            CONFIG['seed']
        )

        d_exp = df['d'].values
        p_hat = df['p_hat'].values
        sigma = df['p_std'].values

        # 3. Use solver to find the "Fitted" rule from the inverse solver
        best_idx, fitted_rule_str, best_lists, top_candidates = fit_rule_from_density_curve(d_exp, p_hat, sigma)

        # Access the score directly from the top candidate
        best_score = top_candidates[0]['score']

        # 4. Generate Analytic Curves for both
        d_model = np.linspace(0, 1, 100)
        fm = AnalyticForwardMap(d_model)
        p_true = fm.predict(true_mask)
        p_fitted = fm.predict(best_idx)

        # 5. Create Visualization
        plt.figure(figsize=(10, 7))
        plt.plot(d_model, d_model, 'k--', alpha=0.3, label="Steady State ($d_{t+1} = d_t$)")

        # Observed points
        plt.errorbar(d_exp, p_hat, yerr=sigma, fmt='ko', markersize=4,
                     capsize=3, elinewidth=1, alpha=0.7, label=f"Simulated Data (N={CONFIG['N']})")

        # Analytic curve p(d; true)
        plt.plot(d_model, p_true, color='forestgreen', linewidth=3, label=f"True Rule: {rule_str}")

        # Analytic curve p(d; fitted)
        plt.plot(d_model, p_fitted, color='crimson', linestyle='--', linewidth=2,
                 label=f"Fitted Rule: {fitted_rule_str}")

        # 6. Styling & Labels
        plt.title(f"Verification: {rule_str} vs. Fitted {fitted_rule_str}", fontsize=14)
        plt.xlabel("Initial Density ($d_t$)", fontsize=12)
        plt.ylabel("Next-Step Density ($d_{t+1}$)", fontsize=12)
        plt.legend(loc='upper left', frameon=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # 7. Save each plot uniquely
        save_path = f"figures/fit_comparison_{rule_str.replace('/', '')}.png"
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to: {save_path}")
        plt.close()


if __name__ == "__main__":
    plot_best_fit_results()