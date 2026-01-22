import os
import matplotlib.pyplot as plt
from meanfield_life import run_meanfield_experiment


def main():
    os.makedirs("figures", exist_ok=True)

    d_list, df, p_theoretical = run_meanfield_experiment()

    plt.figure(figsize=(8, 6))
    plt.plot(d_list, p_theoretical, "r-", label="Mean-field theory")
    plt.errorbar(
        df["d"],
        df["p_hat"],
        yerr=df["std_dev"],
        fmt="ko",
        label="Simulation",
        markersize=4,
    )

    plt.xlabel("Initial density d")
    plt.ylabel("Next-step density p")
    plt.title("Mean-field check: Game of Life (B3/S23)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig("figures/meanfield_check.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
