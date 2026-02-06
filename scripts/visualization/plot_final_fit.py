
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forward_map import AnalyticForwardMap
from rule import encode, parse, format_rule
from inverse_fit import fit_rule_from_density_curve
from data_gen import CONFIG, get_filename

def plot_best_fit_results():
    # 1. Setup and Load Data
    os.makedirs("figures", exist_ok=True)
    csv_path = get_filename()  # Dynamic path from CONFIG

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run data_gen.py first.")
        return

    data = pd.read_csv(csv_path)
    d_exp = data['d'].values
    p_hat = data['p_hat'].values
    sigma = data['std_dev'].values

    # 2. Use the solver to get the "Fitted" rule
    best_idx, (b_fit, s_fit), score, _ = fit_rule_from_density_curve(d_exp, p_hat, sigma)
    fitted_rule_str = format_rule(b_fit, s_fit)

    # 3. Get the "True" rule from CONFIG
    true_rule_str = CONFIG["rule_str"]
    b_true, s_true = parse(true_rule_str)
    true_mask = encode(b_true, s_true)

    # 4. Generate Analytic Curves
    d_model = np.linspace(0, 1, 100)
    fm = AnalyticForwardMap(d_model)
    p_true = fm.predict(true_mask)
    p_fitted = fm.predict(best_idx)

    # 5. Create the Visualization
    plt.figure(figsize=(10, 7))

    # Plot the Identity Line (Steady State)
    plt.plot(d_model, d_model, 'k--', alpha=0.3, label="Steady State ($d_{t+1} = d_t$)")

    # Plot the analytic curve p(d; true)
    plt.plot(d_model, p_true, color='forestgreen', linewidth=3, label=f"True Rule: {true_rule_str}")

    # Plot the analytic curve p(d; fitted)
    plt.plot(d_model, p_fitted, color='crimson', linestyle='--', linewidth=2, label=f"Fitted Rule: {fitted_rule_str}")

    # Plot the observed points (dj, p_hat_j)
    plt.errorbar(d_exp, p_hat, yerr=sigma, fmt='ko', markersize=4,
                 capsize=3, elinewidth=1, alpha=0.7, label=f"Simulated Data (N={CONFIG['N']})")

    # 6. Styling & Labels
    plt.title(f"Verification: {true_rule_str} vs. Fitted {fitted_rule_str}", fontsize=14)
    plt.xlabel("Initial Density ($d_t$)", fontsize=12)
    plt.ylabel("Next-Step Density ($d_{t+1}$)", fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # 7. Save and Show
    save_path = f"figures/fit_comparison_{true_rule_str.replace('/', '')}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Success! Figure saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_best_fit_results()