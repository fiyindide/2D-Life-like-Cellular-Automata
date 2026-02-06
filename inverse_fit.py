import numpy as np
from forward_map import AnalyticForwardMap
from rule import decode


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
    top_k = 5   # number of top candidates to be observed
    epsilon = 1e-6   # Regularization constant to prevent division by zero

    # This precomputes the PB and PS tables (512 x num_den) with the observed densities
    forward_map = AnalyticForwardMap(d)

    num_rules = 2 ** 18  # 262,144 rules

    # initialize the error score of each rule with '0'
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


    # Find the corresponding index (mask) of the least error score
    best_idx = np.argmin(scores)

    # Sort all 262,144 rule scores and select the 'top_k' best matches (lowest error first)
    top_indices = np.argsort(scores)[:top_k]

    top_candidates = []
    for idx in top_indices:
        b_set, s_set = decode(idx)

        # Construct a dictionary for each candidate rule for easy reporting
        top_candidates.append({
            "mask": int(idx),
            "rule": f"B{''.join(map(str, b_set))}/S{''.join(map(str, s_set))}",
            "score": float(scores[idx])
        })

    # Decouple the winning mask/rule into Birth and Survival sets
    best_b, best_s = decode(best_idx)

    return best_idx, (best_b, best_s), scores[best_idx], top_candidates
