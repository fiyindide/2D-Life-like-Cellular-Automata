import numpy as np
from forward_map import AnalyticForwardMap
from rule import decode, format_rule, bool_to_list

def fit_rule_from_density_curve(d, p_hat, sigma=None):
    """
    Searches over all 2^18 Life-like rules to find the best fit to observed
    density data using precomputed lookup tables.

    Args:
        d (np.ndarray): Density values {d_j}
        p_hat (np.ndarray): Measured values {p_hat_j}
        sigma (np.ndarray, optional): Empirical standard deviations {sigma_j}

    Returns:
        tuple: (best_rule_mask, (B, S) sets, objective_value, top_k_candidates)
    """
    top_k = 5  # number of top candidates to be observed
    epsilon = 1e-6   # Regularization constant to prevent division by zero

    forward_map = AnalyticForwardMap(d)
    num_rules = 2 ** 18
    scores = np.zeros(num_rules)

    # Iterate through every possible 18-bit rule mask
    for mask in range(num_rules):
        # Retrieve the (1 x num_den) prediction vector using the lookup tables
        p_pred = forward_map.predict(mask)
        # Calculate Objective Value (SSE vs WSSE)
        if sigma is not None:
            # Weighted Least Squares (WSSE)
            diff = ((p_hat - p_pred) / (sigma + epsilon)) ** 2
        else:
            # Unweighted Least Squares (SSE)
            diff = (p_hat - p_pred) ** 2

        # Update the score for the specific rule
        scores[mask] = np.sum(diff)

    best_idx = np.argmin(scores)
    top_indices = np.argsort(scores)[:top_k]

    top_candidates = []
    for idx in top_indices:
        b_bool, s_bool = decode(idx)
        top_candidates.append({
            "rank": len(top_candidates) + 1,
            "mask": int(idx),
            "rule": format_rule(b_bool, s_bool),
            "b_list": bool_to_list(b_bool),
            "s_list": bool_to_list(s_bool),
            "score": float(scores[idx])
        })

    best_b_bool, best_s_bool = decode(best_idx)
    return (
        best_idx,
        format_rule(best_b_bool, best_s_bool),
        (bool_to_list(best_b_bool), bool_to_list(best_s_bool)),
        top_candidates
    )