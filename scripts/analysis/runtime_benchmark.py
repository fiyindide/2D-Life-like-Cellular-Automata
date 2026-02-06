import time
import numpy as np
import pandas as pd
from inverse_fit import fit_rule_from_density_curve
from rule import Rule, format_rule
from data_gen import CONFIG, get_filename


def run_benchmark():
    """
        Evaluates the performance of the inverse solver by measuring the
        time required to search the entire 18-bit Life-like rule space.
    """

    # Load dataset
    csv_path = get_filename()
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Run data_gen.py first.")
        return

    d = data['d'].values
    p_hat = data['p_hat'].values
    sigma = data['std_dev'].values

    num_rules = 2 ** 18  # 262,144
    M = len(d)  # Number of density points

    print(f"Target Rule: {CONFIG['rule_str']}")
    print(f"Density Points (M): {M}")
    print(f"Total Candidates: {num_rules:,}")
    print("----------------------------------")

    # Start Timing
    start_time = time.perf_counter()

    # Execute the exhaustive search across the rule space
    best_mask, rule_sets, score, top_results = fit_rule_from_density_curve(d, p_hat, sigma)

    end_time = time.perf_counter()

    # Calculate Metrics
    total_runtime = end_time - start_time
    avg_time_per_candidate = total_runtime / num_rules

    # Report Results
    print(f"\n[Benchmark Results]")
    print(f"Total Runtime: {total_runtime:.4f} seconds")
    print(f"Avg Time per Candidate: {avg_time_per_candidate * 1e6:.4f} microseconds")

    print("\nTop Candidates")
    print(f"{'Rank':<6} | {'Rule':<12} | {'Score (SSE)':<12}")

    for i, res in enumerate(top_results):
        # Format rule mask into standard B/S notation for the final report
        rule_name = str(Rule(res['mask']))
        print(f"{i + 1:<6} | {rule_name:<12} | {res['score']:<12.6f}")


if __name__ == "__main__":
    run_benchmark()