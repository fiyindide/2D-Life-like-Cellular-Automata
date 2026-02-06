import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from inverse_fit import fit_rule_from_density_curve
from forward_map import AnalyticForwardMap
from rule import Rule, parse, encode, format_rule
from data_gen import CONFIG, get_filename, measure_one_step_density


def plot_rule_fit(csv_path, true_rule_str, output_folder="figures"):
    """
    Generates a comparison plot for simulation vs. truth vs. fit.
    """
    os.makedirs(output_folder, exist_ok=True)

    # 1. Load Simulation Data
    data = pd.read_csv(csv_path)
    d_obs = data['d'].values
    p_hat = data['p_hat'].values
    sigma = data['std_dev'].values

    # 2. Run Inverse Solver to get the "Fitted" Rule
    best_mask, rule_sets, score, _ = fit_rule_from_density_curve(d_obs, p_hat, sigma)
    fitted_rule_str = format_rule(*rule_sets)

    # 3. Get the "True" Rule mask
    true_b, true_s = parse(true_rule_str)
    true_mask = encode(true_b, true_s)

    # 4. Generate Analytic Curves (100 points for smoothness)
    d_model = np.linspace(0, 1, 100)
    fm = AnalyticForwardMap(d_model)
    p_true_curve = fm.predict(true_mask)
    p_fitted_curve = fm.predict(best_mask)

    # 5. Plotting
    plt.figure(figsize=(9, 6))

    # Observed Data
    plt.errorbar(d_obs, p_hat, yerr=sigma, fmt='ko', markersize=4,
                 capsize=3, label=f"Observed (N={CONFIG['N']})", alpha=0.6)

    # True vs Fitted Curves
    plt.plot(d_model, p_true_curve, 'g-', linewidth=2.5, label=f"True: {true_rule_str}")
    plt.plot(d_model, p_fitted_curve, 'r--', linewidth=2, label=f"Fitted: {fitted_rule_str}")

    plt.title(f"Transition Map Fit: {true_rule_str}\n(Inverse Score: {score:.6f})")
    plt.xlabel("Initial Density ($d_t$)")
    plt.ylabel("Next-Step Density ($d_{t+1}$)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Save to your figures folder
    clean_name = true_rule_str.replace('/', '')
    plt.savefig(f"{output_folder}/fit_comparison_{clean_name}.png", dpi=300)
    print(f"Plot saved to {output_folder}/fit_comparison_{clean_name}.png")
    plt.close()


def generate_random_rule():
    """Generates a random Life-like rule string."""
    b = random.sample(range(9), random.randint(1, 4))
    s = random.sample(range(9), random.randint(1, 4))
    return f"B{''.join(map(str, sorted(b)))}/S{''.join(map(str, sorted(s)))}"


if __name__ == "__main__":
    # 1. First, plot the Game of Life (ensure it exists)
    gol_path = get_filename()  # Uses CONFIG["rule_str"] which is B3/S23
    if os.path.exists(gol_path):
        plot_rule_fit(gol_path, CONFIG["rule_str"])

    # 2. Generate and plot 2 random rules
    print("\n--- Generating Random Rule Fits ---")
    for _ in range(2):
        rand_rule_str = generate_random_rule()
        b, s = parse(rand_rule_str)
        rule_obj = Rule(encode(b, s))

        # Simulate data on the fly
        print(f"Simulating {rand_rule_str}...")
        df = measure_one_step_density(rule_obj, CONFIG['N'], CONFIG['d_values'], CONFIG['trials'], CONFIG['seed'])

        temp_csv = f"data/temp_rand.csv"
        df.to_csv(temp_csv, index=False)
        plot_rule_fit(temp_csv, rand_rule_str)