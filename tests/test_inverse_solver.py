
import numpy as np
import pandas as pd
import os
import pytest
import random
from inverse_fit import fit_rule_from_density_curve, fit_from_standard_df
from forward_map import AnalyticForwardMap
from rule import Rule, encode, decode, parse, format_rule
from data_gen import CONFIG, get_filename



def test_exact_inversion():
    """
    Validation Test 1: Verifies that the solver can recover a rule perfectly
    when there is zero noise in the data.
    """
    print("\n" + "=" * 50)
    print("[Test 1] Running Exact Recovery (Noise-Free)")
    print("=" * 50)

    # Setup Ground Truth
    d = np.linspace(0.1, 0.9, 20)

    # Generate a random rule mask
    true_b = random.sample(range(9), random.randint(1, 4))
    true_s = random.sample(range(9), random.randint(1, 4))


    true_rule_mask = encode(true_b, true_s)
    ground_rule_str = format_rule(decode(true_rule_mask)[0], decode(true_rule_mask)[1])

    # Generate perfect data
    fm = AnalyticForwardMap(d)
    p_perfect = fm.predict(true_rule_mask)

    # Run Solver
    best_mask, top_candidates = fit_rule_from_density_curve(d, p_perfect)

    # Extract SSE score from the best candidate
    best_score = top_candidates[0]['score']
    recovered_str = top_candidates[0]['rule']

    # Print comparative Output
    print(f"Ground Rule:    {ground_rule_str}")
    print(f"Recovered Rule: {recovered_str}")
    print(f"SSE Mode Score: {best_score:.2e}")

    # Verification Assertions
    assert best_mask == true_rule_mask, f"Failed! Expected {true_rule_mask}, got {best_mask}"
    assert best_score < 1e-12, f"Error too high for exact data: {best_score}"

def test_simulated_recovery():
    """
    Validation Test 2: Simulated Recovery (WSSE).
    Uses empirical data from CSV to handle simulation noise.
    P:S: You must run data_gen.py first
    """
    csv_path = get_filename()
    if not os.path.exists(csv_path):
        print("Please run data_gen.py first.")
        return

    b, s = parse(CONFIG["rule_str"])
    target_mask = encode(b, s)

    # Load standardized dataset
    df = pd.read_csv(csv_path)

    results = fit_from_standard_df(df)

    print(f"\nStandardized Recovery Results:")
    print(f"(i)   Best Mask: {results['best_mask']}")
    print(f"(ii)  Best Rule: {results['best_rule_str']}")
    print(f"(iii) Best Score: {results['best_score']:.6f}")
    print(f"(iv)  Top-K Strings:")
    for res in results['top_k']:
        marker = " <--- (Best Match!)" if res['mask'] == target_mask else ""
        print(f"      Rank {res['rank']}: {res['rule']} (Score: {res['score']:.6f}{marker})")

    # Assert success based on target mask from CONFIG
    b, s = parse(CONFIG["rule_str"])
    target_mask = encode(b, s)
    assert results['best_mask'] == target_mask, f"Failed! Expected {CONFIG['rule_str']}, found {best_str}"



def test_analytic_precision():
    print("\n[Test 2] Analytic Forward-Map Precision (GoL)")
    d = np.linspace(0, 1, 101)
    fm = AnalyticForwardMap(d)
    gol_mask = encode([3], [2, 3])
    p_analytic = fm.predict(gol_mask)

    p_expected = 28 * (d ** 3) * ((1 - d) ** 5) * (3 - d)
    max_diff = np.max(np.abs(p_analytic - p_expected))

    assert max_diff < 1e-12
    print(f"PASSED: Max Difference {max_diff:.2e}")

if __name__ == "__main__":
    test_exact_inversion()
    test_analytic_precision()
    test_simulated_recovery()


