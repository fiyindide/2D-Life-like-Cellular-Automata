import numpy as np
from rule import Rule, encode, parse
from data_gen import measure_one_step_density


def analytic_p_life(d):
    """Mean-field one-step prediction for Game of Life (B3/S23)."""
    return 28 * d**3 * (1 - d)**5 * (3 - d)


def run_meanfield_experiment(
    N=256,
    trials=100,
    seed=42,
    d_list=None,
):
    """
    Runs the mean-field experiment for Game of Life and
    returns (d_list, df, p_theoretical).
    """
    if d_list is None:
        d_list = np.linspace(0.05, 0.95, 19)

    # Setup rule
    b, s = parse("B3/S23")
    rule = Rule(encode(b, s))

    # Simulation
    df = measure_one_step_density(rule, N, d_list, trials, seed)

    # Theory
    p_theoretical = analytic_p_life(d_list)

    return d_list, df, p_theoretical
