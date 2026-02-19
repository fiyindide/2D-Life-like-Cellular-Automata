import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)


def save_and_show(name):
    """
        Utility function to standardize plot aesthetics and handle export.
    """
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=300)
    plt.show()


def plot_recovery_vs_N(df, fixed_M=49, fixed_trials=50):
    """
        Visualizes the 'Success Rate' of the solver as a function of board size N.
        This identifies the 'Phase Transition' where the solver
        moves from failing to succeeding due to reduced statistical noise.
    """

    # Filtering for a specific 'experimental slice' to provide a clean comparison.
    subset = df[(df['M'] == fixed_M) & (df['Trials'] == fixed_trials)]
    subset = subset.sort_values('N')

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        subset['N'],
        subset['RecoveryRate'],
        yerr=subset['RecoveryStdErr'],
        marker='o',
        linestyle='-',
        capsize=5
    )

    plt.xlabel("Board Size ($N$)")
    plt.ylabel("Recovery Rate")
    plt.title(f"Recovery vs $N$ ($M={fixed_M}$, Trials={fixed_trials})")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)

    save_and_show("recovery_vs_N")



def plot_margin_vs_N(df, fixed_M=49, fixed_trials=50):
    """
        Plots the gap between the best candidate and the runner-up.
        This plot shows whether boards size makes
        the correct rule much easier to distinguish from incorrect rules.
    """

    subset = df[(df['M'] == fixed_M) & (df['Trials'] == fixed_trials)]
    subset = subset.sort_values('N')

    plt.figure(figsize=(8, 5))
    plt.plot(
        subset['N'],
        subset['AvgNormalizedMargin'],
        marker='o'
    )

    plt.xlabel("Board Size ($N$)")
    plt.ylabel("Average Normalized Margin")
    plt.title("Solver Confidence vs Board Size")
    plt.grid(True, alpha=0.3)

    save_and_show("margin_vs_N")



def plot_runtime_scaling(df):
    """
        Examines the computational overhead.
    """

    df_sorted = df.sort_values('CostMetric')

    plt.figure(figsize=(8, 5))
    plt.plot(
        df_sorted['CostMetric'],
        df_sorted['RuntimeSeconds'],
        marker='o'
    )

    plt.xlabel("Cost Metric (N × M × Trials)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Scaling with Experimental Cost")
    plt.grid(True, alpha=0.3)

    save_and_show("runtime_scaling")



def plot_margin_heatmap(df, fixed_trials=50):
    """
        It visualizes the entire parameter space at once,
        showing exactly where the solver is most robust.
    """

    subset = df[df['Trials'] == fixed_trials]

    N_vals = sorted(subset['N'].unique())
    M_vals = sorted(subset['M'].unique())

    heat = np.zeros((len(N_vals), len(M_vals)))

    for i, N in enumerate(N_vals):
        for j, M in enumerate(M_vals):
            val = subset[(subset['N'] == N) & (subset['M'] == M)]['AvgNormalizedMargin'].values
            if len(val) > 0:
                heat[i, j] = val[0]

    plt.figure(figsize=(8, 6))
    plt.imshow(heat, aspect='auto', origin='lower')
    plt.colorbar(label='Avg Normalized Margin')
    plt.xticks(range(len(M_vals)), M_vals)
    plt.yticks(range(len(N_vals)), N_vals)

    plt.xlabel("Density Points ($M$)")
    plt.ylabel("Board Size ($N$)")
    plt.title(f"Margin Heatmap (Trials={fixed_trials})")

    save_and_show("margin_heatmap")



def plot_stability_vs_cost(df):
    """
        This plot justifies the recommended parameters
        by showing the point of diminishing returns where adding more cost
        no longer significantly improves the safety margin.
        """

    plt.figure(figsize=(8, 5))
    plt.scatter(
        df['CostMetric'],
        df['AvgNormalizedMargin'],
        s=120,
        alpha=0.7
    )

    plt.xlabel("Cost Metric (N × M × Trials)")
    plt.ylabel("Avg Normalized Margin")
    plt.title("Stability vs Computational Cost")
    plt.grid(True, alpha=0.3)

    save_and_show("stability_vs_cost")



if __name__ == "__main__":

    path = "data/robustness_results.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)

        plot_recovery_vs_N(df)
        plot_margin_vs_N(df)
        plot_runtime_scaling(df)
        plot_margin_heatmap(df)
        plot_stability_vs_cost(df)

    else:
        print("Run robustness experiment first.")
