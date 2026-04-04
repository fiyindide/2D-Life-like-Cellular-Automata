import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from forward_map import AnalyticForwardMap
from rule import parse, encode
from gaussian_noise_experiment import run_gaussian_noise_experiment
from bitflip_experiment import run_bitflip_experiment
from mismatch_experiment import run_mismatch_experiment

RULES = [
    "B3/S23",         # Game of Life, baseline
    "B36/S23",        # HighLife, one bit away from GoL (nearest-neighbour stress test)
    "B3678/S34678",   # Day & Night, symmetric rule with dense survival set
    "B2/S",           # No survival
    "B1/S1",          # Birth and survival only at k=1
    "B45678/S2345",   # Dense rule, many birth/survival conditions
]

# Experiment settings
GAUSSIAN_KWARGS = dict(num_den=25, trials=30,
                       sigma_levels=[0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
BITFLIP_KWARGS  = dict(N=100, num_den=25, trials=30,
                       epsilon_levels=[0, 0.005, 0.01, 0.02, 0.05, 0.1])
MISMATCH_KWARGS = dict(N=100, num_den=25, trials=20)

DATA_DIR   = "data"
FIGURE_DIR = "figures"
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Styles for the three experiment types on the summary plot
EXP_STYLE = {
    "gaussian": dict(color="teal",       marker="o", ls="-",  label="Gaussian Noise"),
    "bitflip":  dict(color="darkorange", marker="s", ls="--", label="Bit-Flip"),
    "mismatch": dict(color="steelblue",  marker="^", ls=":",  label="Model Mismatch"),
}

# Colours for imposter bar charts (one per experiment type)
IMPOSTER_COLORS = {
    "gaussian": "teal",
    "bitflip":  "darkorange",
    "mismatch": "steelblue",
}

# Maps experiment key to the file prefix used when saving imposter CSVs
IMPOSTER_FILE_PREFIX = {
    "gaussian": "gaussian_noise",
    "bitflip":  "bitflip",
    "mismatch": "mismatch",
}


def check_density_curves(rules=RULES, num_den=25):
    """
    Plots the analytic density curve p(d; B, S) for each rule
    We check the density curves at the beginning of the experiment to ensure that
    each selected rule possessed a unique and non-degenerate "density fingerprint"
    before attempting to recover them.
    """
    d_vals = np.linspace(0.05, 0.95, num_den)
    fig, ax = plt.subplots(figsize=(9, 5))

    for rule_str in rules:
        mask = encode(*parse(rule_str))
        fmap = AnalyticForwardMap(d_vals)
        p = fmap.predict(mask)
        ax.plot(d_vals, p, marker='o', markersize=3, lw=1.5, label=rule_str)

    ax.set_title("Analytic Density Curves — Sanity Check")
    ax.set_xlabel("Initial Density $d$")
    ax.set_ylabel("One-Step Alive Probability $p(d)$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "density_curve_check.png")
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"[Check] Density curves saved to {path}")



def run_all(rules=RULES, force_rerun=False):
    """
    Runs all three experiments for every rule in `rules`,  save results to csv files
    If a results CSV already exists and force_rerun=False, loads from cache.

    Returns:
        dict: { rule_str: { "gaussian": df, "bitflip": df, "mismatch": df } }
    """
    all_results = {}

    for rule_str in rules:
        clean = rule_str.replace("/", "")
        all_results[rule_str] = {}

        # Gaussian noise
        path = f"{DATA_DIR}/gaussian_noise_results_{clean}.csv"
        if os.path.exists(path) and not force_rerun:
            print(f"Loading gaussian results for {rule_str}")
            all_results[rule_str]["gaussian"] = pd.read_csv(path)
        else:
            df, collisions, _ = run_gaussian_noise_experiment(
                rule_str=rule_str, **GAUSSIAN_KWARGS
            )
            df.to_csv(path, index=False)
            imp = pd.DataFrame(collisions.most_common(),
                               columns=["Imposter_Rule", "Count"])
            imp.to_csv(f"{DATA_DIR}/gaussian_noise_imposters_{clean}.csv", index=False)
            all_results[rule_str]["gaussian"] = df

        # Bit-flip
        path = f"{DATA_DIR}/bitflip_results_{clean}.csv"
        if os.path.exists(path) and not force_rerun:
            print(f"Loading bitflip results for {rule_str}")
            all_results[rule_str]["bitflip"] = pd.read_csv(path)
        else:
            df, collisions, _ = run_bitflip_experiment(
                rule_str=rule_str, **BITFLIP_KWARGS
            )
            df.to_csv(path, index=False)
            imp = pd.DataFrame(collisions.most_common(),
                               columns=["Imposter_Rule", "Count"])
            imp.to_csv(f"{DATA_DIR}/bitflip_imposters_{clean}.csv", index=False)
            all_results[rule_str]["bitflip"] = df

        # Mismatch
        path = f"{DATA_DIR}/mismatch_results_{clean}.csv"
        if os.path.exists(path) and not force_rerun:
            print(f"Loading mismatch results for {rule_str}")
            all_results[rule_str]["mismatch"] = pd.read_csv(path)
        else:
            df, collisions, _ = run_mismatch_experiment(
                rule_str=rule_str, **MISMATCH_KWARGS
            )
            df.to_csv(path, index=False)
            imp = pd.DataFrame(collisions.most_common(),
                               columns=["Imposter_Rule", "Count"])
            imp.to_csv(f"{DATA_DIR}/mismatch_imposters_{clean}.csv", index=False)
            all_results[rule_str]["mismatch"] = df

    return all_results



def interpolate_to_common_rmse(all_results, rmse_grid=None):
    """
    Each experiment produces (rmse, accuracy, true_gap) pairs, but the rmse
    values differ across rules and experiment types. This function interpolates
    all curves onto a shared rmse_grid so we can compute mean +/- std.

    Returns:
        dict: {
            exp_name: {
                "accuracy_mat": (n_rules x n_grid) array,
                "true_gap_mat": (n_rules x n_grid) array,
                "rmse_grid":    1D array
            }
        }
    """
    if rmse_grid is None:
        all_rmse = []
        for rule_str, exps in all_results.items():
            for exp_name, df in exps.items():
                all_rmse.extend(df["rmse"].tolist())
        rmse_grid = np.linspace(0, np.percentile(all_rmse, 95), 50)

    exp_names = ["gaussian", "bitflip", "mismatch"]
    interpolated = {}

    for exp_name in exp_names:
        acc_rows, gap_rows = [], []
        for rule_str, exps in all_results.items():
            df = exps[exp_name].sort_values("rmse")
            rmse_vals = df["rmse"].values
            acc_vals  = df["accuracy"].values
            gap_vals  = df["true_gap"].values

            acc_interp = np.interp(rmse_grid, rmse_vals, acc_vals)
            gap_interp = np.interp(rmse_grid, rmse_vals, gap_vals)

            acc_rows.append(acc_interp)
            gap_rows.append(gap_interp)

        interpolated[exp_name] = {
            "accuracy_mat": np.array(acc_rows),
            "true_gap_mat": np.array(gap_rows),
            "rmse_grid":    rmse_grid,
        }

    return interpolated


def plot_summary(interpolated, rules=RULES):
    """
    Produces two summary figures on the common RMSE scale:
      Figure 1: Mean accuracy +/- std across rules, one line per experiment type
      Figure 2: Mean true-rule gap +/- std across rules, one line per experiment type
    """
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for exp_name, style in EXP_STYLE.items():
        data      = interpolated[exp_name]
        rmse_grid = data["rmse_grid"]
        acc_mat   = data["accuracy_mat"]
        gap_mat   = data["true_gap_mat"]

        acc_mean = acc_mat.mean(axis=0)
        acc_std  = acc_mat.std(axis=0)
        gap_mean = gap_mat.mean(axis=0)
        gap_std  = gap_mat.std(axis=0)

        # Accuracy summary
        ax1.plot(rmse_grid, acc_mean,
                 color=style["color"], marker=style["marker"],
                 ls=style["ls"], lw=2, label=style["label"], markevery=5)
        ax1.fill_between(rmse_grid,
                         np.clip(acc_mean - acc_std, 0, 1),
                         np.clip(acc_mean + acc_std, 0, 1),
                         color=style["color"], alpha=0.15)

        # True-rule gap summary  (lower std band clipped to 0)
        ax2.plot(rmse_grid, gap_mean,
                 color=style["color"], marker=style["marker"],
                 ls=style["ls"], lw=2, label=style["label"], markevery=5)
        ax2.fill_between(rmse_grid,
                         np.clip(gap_mean - gap_std, 0, None),
                         gap_mean + gap_std,
                         color=style["color"], alpha=0.15)

    n = len(rules)
    ax1.set_title(f"Recovery Accuracy vs. RMSE\nMean $\\pm$ std across {n} rules")
    ax1.set_xlabel("RMSE of Observed vs. Clean Curve")
    ax1.set_ylabel("Recovery Accuracy")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    path1 = os.path.join(FIGURE_DIR, "summary_accuracy_vs_rmse.png")
    fig1.savefig(path1, dpi=300)
    print(f"[Plot] Saved {path1}")

    ax2.set_title(f"True-Rule Gap vs. RMSE\nMean $\\pm$ std across {n} rules")
    ax2.set_xlabel("RMSE of Observed vs. Clean Curve")
    ax2.set_ylabel("True-Rule Gap (SSE)")
    ax2.set_yscale("symlog", linthresh=1e-10)
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.2)
    fig2.tight_layout()
    path2 = os.path.join(FIGURE_DIR, "summary_truegap_vs_rmse.png")
    fig2.savefig(path2, dpi=300)
    print(f"[Plot] Saved {path2}")

    plt.show()


def print_summary_table(all_results):
    """
    Prints a compact table with four columns per rule x experiment:

      perfect_threshold  — max RMSE where accuracy is still 1.0 (safe zone ends)
      failure_threshold  — min RMSE where accuracy first hits 0.0 (dead zone begins)
      transition_width   — gap between the two; narrow = sharp collapse (bit-flip),
                           wide = gradual decay (Gaussian noise);
                           '>range' means total failure was never reached
      min_accuracy       — lowest accuracy observed across all perturbation levels

    """
    rows = []
    for rule_str, exps in all_results.items():
        for exp_name, df in exps.items():
            df_sorted = df.sort_values("rmse")

            perfect = df_sorted[df_sorted["accuracy"] >= 1.0]
            perfect_threshold = round(perfect["rmse"].max(), 5) \
                                if not perfect.empty else 0.0

            failed = df_sorted[df_sorted["accuracy"] <= 0.0]
            failure_threshold = round(failed["rmse"].min(), 5) \
                                if not failed.empty else None

            if failure_threshold is not None:
                transition_width = round(failure_threshold - perfect_threshold, 5)
            else:
                transition_width = ">range"

            rows.append({
                "rule":              rule_str,
                "experiment":        exp_name,
                "perfect_threshold": perfect_threshold,
                "failure_threshold": failure_threshold,
                "transition_width":  transition_width,
                "min_accuracy":      round(df["accuracy"].min(), 3),
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(f"{DATA_DIR}/summary_table.csv", index=False)

    print("\n--- Summary Table ---")
    print("Columns: perfect_threshold = last RMSE at 100% accuracy | "
          "failure_threshold = first RMSE at 0% accuracy | "
          "transition_width = failure - perfect (narrow = sharp collapse, "
          ">range = total failure never reached)\n")
    print(summary_df.to_string(index=False))



def plot_imposter_analysis(rules=RULES, top_n=5):
    """
    For each rule, produces a single figure with three horizontal bar charts
    side by side — one per experiment type — showing the top-N imposter rules
    ranked by how many times they beat the true rule.

    This directly answers:
      - Which impostor rules become nearest competitors under each noise type?
      - Why does recovery fail? (the identity of the impostor reveals the
        structural reason — e.g. a Hamming-1 neighbour beating under Gaussian
        noise vs. a qualitatively different rule winning under mismatch)
    """
    exp_names  = ["gaussian", "bitflip", "mismatch"]
    exp_labels = {
        "gaussian": "Gaussian Noise",
        "bitflip":  "Bit-Flip",
        "mismatch": "Model Mismatch",
    }

    for rule_str in rules:
        clean = rule_str.replace("/", "")
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Top-{top_n} Impostor Rules — True Rule: {rule_str}",
                     fontsize=13, fontweight='bold')

        for ax, exp_name in zip(axes, exp_names):

            # Use the correct file prefix for each experiment type
            file_prefix = IMPOSTER_FILE_PREFIX[exp_name]
            path = f"{DATA_DIR}/{file_prefix}_imposters_{clean}.csv"

            # Load imposter CSV if it exists
            if not os.path.exists(path):
                ax.text(0.5, 0.5, "No data\n(file not found)",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='grey')
                ax.set_title(exp_labels[exp_name])
                ax.axis('off')
                continue

            imp_df = pd.read_csv(path)

            if imp_df.empty:
                ax.text(0.5, 0.5, "Perfect recovery\n(no imposters)",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='grey')
                ax.set_title(exp_labels[exp_name])
                ax.axis('off')
                continue

            # Take top-N imposters
            top = imp_df.head(top_n).copy()

            # Plot horizontal bars — longest bar = most frequent impostor
            bars = ax.barh(top['Imposter_Rule'], top['Count'],
                           color=IMPOSTER_COLORS[exp_name], alpha=0.85,
                           edgecolor='white')

            # Annotate count on each bar
            for bar, count in zip(bars, top['Count']):
                ax.text(bar.get_width() + 0.3,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(count)), va='center', ha='left', fontsize=9)

            ax.set_title(exp_labels[exp_name], fontsize=11)
            ax.set_xlabel("Failure Count")
            ax.invert_yaxis()  # highest count at the top
            ax.grid(True, axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        save_path = os.path.join(FIGURE_DIR, f"imposters_{clean}.png")
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"[Plot] Saved {save_path}")




if __name__ == "__main__":

    # Sanity-check density curves first
    print(" Checking density curves...")
    check_density_curves()

    # Run all experiments (or load from cache)
    print("\n Running experiments across all rules...")
    all_results = run_all(rules=RULES, force_rerun=False)

    # Interpolate onto common RMSE grid
    print("\n Interpolating onto common RMSE grid...")
    interpolated = interpolate_to_common_rmse(all_results)

    # Plot summary figures
    print("\n[Step 3] Plotting summary figures...")
    plot_summary(interpolated, rules=RULES)

    # Print summary table
    print("\n[Step 4] Summary table...")
    print_summary_table(all_results)

    # Per-rule imposter analysis
    print("\n[Step 5] Plotting per-rule imposter analysis...")
    plot_imposter_analysis(rules=RULES, top_n=5)