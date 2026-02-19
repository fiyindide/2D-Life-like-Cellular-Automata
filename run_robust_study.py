import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm

from rule import Rule
from data_gen import measure_one_step_density
from inverse_fit import fit_rule_from_density_curve


def run_experiment():
    """
        Automates a systematic grid search across experimental parameters to determine
        the practical limits of the inverse solver.
    """

    # Initialize a master random number generator with a fixed seed for reproducibility.
    master_rng = np.random.default_rng(42)

    # Define the experimental values
    N_values = [64, 128, 256]     # Resolution of the CA board
    M_values = [19, 49, 99]       # Density values
    TRIALS_per_density = [10, 50, 200]  # Number of independent board initializations per density point

    # Number of different random rules to test for every single condition.
    REPEATS = 20

    results = []
    total_start = time.time()

    # Nested loop to explore every combination of N, M, and trials per density.
    for N in N_values:
        for M in M_values:
            for t_per_d in TRIALS_per_density:

                condition_start = time.time()

                successes = 0
                margins = []
                normalized_margins = []

                print(f"\nCondition: N={N}, M={M}, Trials={t_per_d}")

                # Run multiple independent trials for the current set of parameters
                for _ in tqdm(range(REPEATS)):

                    # Generate a random ground-truth rule (0 to 2^18 - 1)
                    true_mask = master_rng.integers(0, 2**18)
                    rule_obj = Rule(true_mask)
                    print(f"Testing Rule: {rule_obj}")

                    # Setup the density sampling points
                    d_vals = np.linspace(0.01, 0.99, M)

                    # Generate a unique seed for the CA simulation to ensure no overlap
                    sim_seed = master_rng.integers(0, 10**9)

                    # Simulate the CA to get empirical density response (p_hat)
                    df = measure_one_step_density(
                        rule_obj,
                        N,
                        d_vals,
                        t_per_d,
                        seed=sim_seed
                    )

                    # Use the exhaustive search solver to find the best-fit rule.
                    best_idx, _, _, top_candidates = fit_rule_from_density_curve(
                        df['d'].values,
                        df['p_hat'].values,
                        df['p_std'].values
                    )

                    # Compare the recovered rule mask to the true ground-truth mask
                    if best_idx == true_mask:
                        successes += 1

                    # Calculate the gap between the best fit and the runner-up.
                    best_score = top_candidates[0]['score']
                    second_score = top_candidates[1]['score']
                    margin = second_score - best_score
                    normalized_margin = margin / (abs(best_score) + 1e-12)
                    margins.append(margin)
                    normalized_margins.append(normalized_margin)

                # Record how long this specific experimental condition took to process
                condition_runtime = time.time() - condition_start

                # Calculate the Recovery Rate (probability of success) and its Standard Error.
                p = successes / REPEATS
                stderr = np.sqrt(p * (1 - p) / REPEATS)

                # Store all data for this condition into the results list.
                results.append({
                    'N': N,
                    'M': M,
                    'Trials': t_per_d,
                    'RecoveryRate': p,
                    'RecoveryStdErr': stderr,
                    'AvgMargin': np.mean(margins),
                    'AvgNormalizedMargin': np.mean(normalized_margins),
                    'RuntimeSeconds': condition_runtime,
                    'CostMetric': N * M * t_per_d  # for computational cost
                })

    total_runtime = time.time() - total_start
    print(f"\nTotal experiment runtime: {total_runtime:.2f} seconds")

    df_results = pd.DataFrame(results)


    # Filter conditions where the solver achieved 100% accuracy.
    reliable = df_results[df_results['RecoveryRate'] == 1.0]

    # Recommended Parameter Settings
    if len(reliable) > 0:
        # Minimal: The cheapest settings (lowest CostMetric) that still result in perfect recovery.
        minimal = reliable.sort_values('CostMetric').iloc[0]

        # Robust: The settings that provide the widest gap (Margin) between the truth and the runner-up.
        robust = reliable.sort_values('AvgNormalizedMargin', ascending=False).iloc[0]

        print("\n Recommended Parameters....")
        print("\nMinimal Working Configuration:")
        print(minimal[['N', 'M', 'Trials', 'RuntimeSeconds']])

        print("\nMost Robust Configuration (Max Margin):")
        print(robust[['N', 'M', 'Trials', 'AvgNormalizedMargin']])

    # Save
    os.makedirs("data", exist_ok=True)
    df_results.to_csv("data/robustness_results.csv", index=False)

    print("\nResults saved to data/robustness_results.csv")

    return df_results


if __name__ == "__main__":
    run_experiment()
