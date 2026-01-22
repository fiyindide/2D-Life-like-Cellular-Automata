import numpy as np
from meanfield_life import run_meanfield_experiment


def test_meanfield_life():
    d_list, df, p_theoretical = run_meanfield_experiment()

    p_simulated = df["p_hat"].values

    max_error = np.max(np.abs(p_simulated - p_theoretical))
    rmse = np.sqrt(np.mean((p_simulated - p_theoretical) ** 2))

    assert max_error < 0.01, f"Max error {max_error:.5f} exceeds tolerance"
    assert rmse < 0.01, f"RMSE {rmse:.5f} exceeds tolerance"
