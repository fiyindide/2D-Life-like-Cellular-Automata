import numpy as np
from forward_map import AnalyticForwardMap
from rule import encode


def test_analytic_vs_closed_form():
    """
        Verification Test: Analytic validation using Game of Life

        This test ensures that the precomputed lookup tables in AnalyticForwardMap
        correctly implement the analytic one-step density map.
    """
    # Define range of densities
    d_values = np.linspace(0, 1, 101)

    forward_map = AnalyticForwardMap(d_values)

    # Define birth and survival sets for B3 / S23 rule mask
    B_SET = [3]
    S_SET = [2,3]

    # convert the B3/S23 rule into an 18-bit integer mask.
    gol_mask = encode(B_SET, S_SET)

    # Use the precomputed lookup tables to retrieve the predicted next-step
    p_analytic = forward_map.predict(gol_mask)

    # Compute p_infinity using the closed-form expression
    d = d_values
    p_inf = 28 * (d ** 3) * ((1 - d) ** 5) * (3 - d)

    # Calculate the Absolute Error between p_analytic and p_inf
    max_diff = np.max(np.abs(p_analytic - p_inf))
    print(f"Max absolute difference: {max_diff:.2e}")

    # asserts that the maximum absolute difference is below a tight tolerance 1e-12
    assert max_diff < 1e-12, f"Analytic mismatch! Difference: {max_diff}"

    # Boundary Check to confirm next-step density is 0 for d = 0 and d = 1
    print(f"Density at 0.0: {forward_map.predict(gol_mask)[0]:.4f}")
    print(f"Density at 1.0: {forward_map.predict(gol_mask)[-1]:.4f}")




if __name__ == "__main__":
    test_analytic_vs_closed_form()