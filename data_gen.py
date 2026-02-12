import numpy as np
import pandas as pd
from life_like import step
from rule import Rule
from rule import parse, encode
import os


def sample_bernoulli_grid(N, d, rng):
    """
    Samples an N x N grid where each cell is 1 with probability d.
    """
    grid = (rng.random((N, N)) < d).astype(np.uint8)
    return grid


def measure_one_step_density(rule, N, d_list, trials, seed):
    """
    Measures the fraction of alive cells after one-step, p_hat(d), for a given rule.
    Includes standardized metadata columns in the output.
    """

    rng = np.random.default_rng(seed)
    results = []

    for d in d_list:
        p_hat_trial_densities = []
        for trial in range(trials):
            # 1. Sample initial grid
            grid = sample_bernoulli_grid(N, d, rng)

            # 2. Apply one step of the rule
            next_grid = step(grid, rule)

            # 3. Record fraction of alive cells
            p_hat = np.mean(next_grid)
            p_hat_trial_densities.append(p_hat)

        # Calculate statistics across trials
        results.append({
            'd': d,                 # Initial density
            'p_hat': np.mean(p_hat_trial_densities), # Measured next-step density
            'p_std': np.std(p_hat_trial_densities),   # Empirical standard deviation
            'N': N,                 # Grid size
            'trials': trials,       # Number of trials
            'seed': seed,           # Random seed
            'rule_mask': rule.mask,  # The 18-bit integer mask
            'boundary': "periodic"  # Explicit boundary column
        })

    # Convert to DataFrame
    return pd.DataFrame(results)

# Centralized Configuration
CONFIG = {
    "N": 256,
    "trials": 50,
    "seed": 42,
    "rule_str": "B3/S23",
    "d_values": np.linspace(0.05, 0.95, 19)
}

def get_filename():
    clean_rule = CONFIG["rule_str"].replace('/', '')
    return f"data/life_{clean_rule}_N{CONFIG['N']}_trials{CONFIG['trials']}.csv"

if __name__ == "__main__":
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)

    # 1. Access values via CONFIG
    b, s = parse(CONFIG["rule_str"])
    rule = Rule(encode(b, s))

    # 2. Run simulation using CONFIG values
    print(f"Generating data for {CONFIG['rule_str']}...")
    df = measure_one_step_density(
        rule,
        CONFIG["N"],
        CONFIG["d_values"],
        CONFIG["trials"],
        CONFIG["seed"]
    )

    # 3. Save file
    filename = get_filename()
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")