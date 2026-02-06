import os
import numpy as np
import matplotlib.pyplot as plt
from forward_map import AnalyticForwardMap
from rule import encode


def generate_transition_plot():
    # 1. Setup Directories
    os.makedirs("figures", exist_ok=True)

    # 2. Generate Data
    d_values = np.linspace(0, 1, 200)
    forward_map = AnalyticForwardMap(d_values)

    # Encode Game of Life (B3/S23)
    gol_mask = encode([3], [2, 3])
    p_pred = forward_map.predict(gol_mask)

    # 3. Create Visualization
    plt.figure(figsize=(10, 7))

    # Plot the 'Identity Line'
    plt.plot(d_values, d_values, color='black', linestyle='--', alpha=0.6, label="Steady State ($d_{t+1} = d_t$)")

    # Plot the Analytic Prediction for GoL
    plt.plot(d_values, p_pred, color='crimson', linewidth=2.5, label="GoL Analytic Map (B3/S23)")

    # 4. Styling the Plot
    plt.title("Game of Life: Mean Field Theory State Transition Map", fontsize=14)
    plt.xlabel("Current Density ($d_t$)", fontsize=12)
    plt.ylabel("Predicted Next Density ($d_{t+1}$)", fontsize=12)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(loc='upper left')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # # 5. Save and Show
    plt.savefig("figures/gol_transition_map.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    generate_transition_plot()