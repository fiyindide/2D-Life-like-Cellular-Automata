import numpy as np


def get_bernstein_coefficients(mask):
    """
    performs a computational "sanity check" to verify the identifiability of
    Life-like cellular automata rules. It ensures that no two distinct 18-bit rule
    masks produce the exact same density response curve.
    """
    # Decode 18-bit mask into Birth and Survival indicator bits
    b = [(mask >> i) & 1 for i in range(9)]   # Bits 0-8
    s = [(mask >> (i + 9)) & 1 for i in range(9)]   # Bits 9-17

    # Initialize the boundary of the bernstein coefficient
    alpha = np.zeros(10)

    # Boundary coefficients
    alpha[0] = float(b[0])
    alpha[9] = float(s[8])

    # Interior mixture coefficients (m = 1 to 8)
    for m in range(1, 9):
        alpha[m] = ((9 - m) / 9.0) * b[m] + (m / 9.0) * s[m - 1]

    return tuple(np.round(alpha, 10))


def run_collision_diagnostic():
    num_rules = 2 ** 18  # 262,144 rules
    seen_vectors = {}
    collisions = []

    print(f"Starting diagnostic for {num_rules:,} rules...")

    for mask in range(num_rules):
        alpha_vec = get_bernstein_coefficients(mask)

        if alpha_vec in seen_vectors:
            collisions.append((mask, seen_vectors[alpha_vec], alpha_vec))
        else:
            seen_vectors[alpha_vec] = mask

    if not collisions:
        print("Success: No collisions found. Every rule has a unique coefficient vector.")
    else:
        print(f"Diagnostic Failed: Found {len(collisions)} collisions.")


if __name__ == "__main__":
    run_collision_diagnostic()