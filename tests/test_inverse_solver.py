
import numpy as np
import pandas as pd
import os
import pytest
from inverse_fit import fit_rule_from_density_curve
from forward_map import AnalyticForwardMap
from rule import Rule, encode, decode, parse, format_rule  # Use existing logic
from data_gen import CONFIG, get_filename


def test_exact_recovery():
    """
    Validation Test 1: Verifies that the solver can recover a rule perfectly
    when there is zero noise in the data.
    """
    print("\n[Test 1] Running Exact Recovery (Noise-Free)")
    d = np.linspace(0.1, 0.9, 20)
    sigma = np.ones_like(d) * 0.01

    # Select a test rule: B36/S23 (HighLife)
    true_rule_mask = encode([3, 6], [2, 3])

    # Generate ground-truth data using the analytic map
    fm = AnalyticForwardMap(d)
    p_perfect = fm.predict(true_rule_mask)

    # Run inverse solver for unweighted least square
    best_mask, rule_sets, score, _ = fit_rule_from_density_curve(d, p_perfect)
    print(f"SSE Mode Score: {score:.2e}")


    # Verification
    assert best_mask == true_rule_mask, f"Failed! Expected {true_rule_mask}, got {best_mask}"
    assert score < 1e-12, f"Error too high for exact data: {score}"

    # print the recovered rule in string format
    print(f"Recovered Rule: {format_rule(*rule_sets)}\n")


def test_simulated_recovery():
    """
    Validation Test 2: Simulated Recovery (WSSE).
    Uses empirical data from CSV, which includes 'std_dev'. This activates
    the Weighted Least Squares route to handle simulation noise.
    """
    csv_path = get_filename()
    b, s = parse(CONFIG["rule_str"])
    target_mask = encode(b, s)

    if not os.path.exists(csv_path):
        pytest.skip(f"Data not found. Run data_gen.py first.")

    data = pd.read_csv(csv_path)

    # Run inverse solver
    best_mask, rule_sets, best_score, top_results = fit_rule_from_density_curve(
        data['d'].values,
        data['p_hat'].values,
        data['std_dev'].values
    )

    print("\n Top 5 Inverse Solver Candidates ")
    print(f"{'Rank':<6} | {'Rule':<12} | {'Score (WSSE)':<12}")
    print("-" * 40)

    # Rank the top candidates from best match to least match
    for i, res in enumerate(top_results):
        rank = i + 1

        rule_obj = Rule(res['mask'])
        rule_name = str(rule_obj)

        score = res['score']
        marker = " <--- (Best Match!)" if res['mask'] == target_mask else ""

        # Print results in a table
        print(f"{rank:<6} | {rule_name:<12} | {score:<12.6f}{marker}")

    assert best_mask == target_mask, f"Failed! Expected {CONFIG['rule_str']}, found {top_results[0]['rule']}"


if __name__ == "__main__":
    test_exact_recovery()
    test_simulated_recovery()