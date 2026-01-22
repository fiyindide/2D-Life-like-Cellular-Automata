import numpy as np
import pandas as pd
from life_like import step
from rule import Rule


def sample_bernoulli_grid(N, d, rng):
    """
    Samples an N x N grid where each cell is 1 with probability d.
    """
    grid = (rng.random((N, N)) < d).astype(np.uint8)
    return grid


def measure_one_step_density(rule, N, d_list, trials, seed):
    """
    Measures the fraction of alive cells after one-step, p_hat(d), for a given rule.
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
            'd': d,
            'p_hat': np.mean(p_hat_trial_densities),
            'std_dev': np.std(p_hat_trial_densities)
        })

    # Convert to DataFrame
    return pd.DataFrame(results)


if __name__ == "__main__":
    from rule import parse, encode

    # Configuration
    N = 256
    trials = 50
    seed = 42
    rule_str = "B3/S23"
    # Generate 19 evenly spaced values between 0.05 and 0.95 (inclusive)
    d_values = np.linspace(0.05, 0.95, 19)

    # Setup rule
    b, s = parse(rule_str)
    rule = Rule(encode(b, s))

    # Generate data
    df = measure_one_step_density(rule, N, d_values, trials, seed)

    # Save data in a csv file
    filename = f"data/life_{rule_str.replace('/', '')}_N{N}_trials{trials}.csv"
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")