
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import convolve2d
from inverse_fit import fit_rule_from_density_curve
from rule import parse, encode, Rule, format_rule, decode
from life_like import step


def generate_correlated_state(N, d, smoothing_radius=1):
    """
    Generates an NxN grid with spatial correlation (clumping) while maintaining a target density.
    This simulates 'Model Mismatch' by violating the i.i.d. assumption.
    """
    # Create initial white noise (completely independent random values)
    raw_noise = np.random.rand(N, N)

    if smoothing_radius > 0:
        # Determine kernel size for local averaging based on the radius
        size = int(2 * np.ceil(smoothing_radius) + 1)
        # Create a uniform square kernel (box filter)
        kernel = np.ones((size, size))

        # Apply 2D convolution to smooth the noise and wrap the boundaries
        smoothed = convolve2d(raw_noise, kernel, mode='same', boundary='wrap')
    else:
        # If radius is 0, keep the independent white noise (i.i.d. case)
        smoothed = raw_noise

    # To maintain the specific target density 'd', we threshold the smoothed values.
    # We find the percentile cutoff so that exactly d% of cells become 'Alive' (1).
    cutoff = np.percentile(smoothed, 100 * (1 - d))

    # Return binary grid: 1 where value is above cutoff, 0 otherwise
    return (smoothed >= cutoff).astype(np.uint8)


def run_model_mismatch_experiment(N=100, num_den=25, trials=20):
    """
    Core experiment loop: Tests the solver's ability to recover B3/S23 from 'clumpy' grids.
    """
    # Define granular smoothing radii to observe the 'failure cliff'
    radii = [0, 0.25, 0.5, 0.75, 1.0]

    # Setup the 'Ground Truth' rule (Game of Life)
    rule_str = "B3/S23"
    b_set, s_set = parse(rule_str)
    true_mask = encode(b_set, s_set)
    rule_obj = Rule(true_mask)

    # Sample density points across the range [0.05, 0.95]
    d_vals = np.linspace(0.05, 0.95, num_den)
    results = []

    print(f"\n Starting Model Mismatch Experiment...")

    # Iterate through each correlation level (smoothing radius)
    for r in radii:
        correct_count = 0
        total_margin = 0
        imposter_counts = Counter()

        for t in range(trials):
            p_observed = []
            for d in d_vals:
                # Generate the correlated (clumpy) initial state
                X0_raw = generate_correlated_state(N, d, smoothing_radius=r)

                # Ensure data type is compatible with the life_like engine
                X0 = X0_raw.astype(np.uint8)

                # Advance the grid by one CA step (Numerical Truth)
                X1 = step(X0, rule_obj)

                # Measure the resulting density (the observed outcome)
                p_observed.append(np.mean(X1))

            p_observed = np.array(p_observed)

            # Apply the Solver
            best_mask, top_candidates = fit_rule_from_density_curve(d_vals, p_observed)

            # Calculate Solver Confidence (Margin between best and second-best guess)
            if len(top_candidates) > 1:
                margin = top_candidates[1]['score'] - top_candidates[0]['score']
            else:
                margin = 0
            total_margin += margin

            # Track success vs failure
            if best_mask == true_mask:
                correct_count += 1
            else:
                # Record which 'Imposter' rule the solver confidently picked instead
                imposter_counts[top_candidates[0]['rule']] += 1

        # Average metrics across all trials for this specific radius
        accuracy = correct_count / trials
        avg_margin = total_margin / trials
        top_imp = imposter_counts.most_common(1)[0][0] if imposter_counts else "None"

        results.append({
            "radius": r,
            "accuracy": accuracy,
            "margin": avg_margin,
            "top_imposter": top_imp
        })

        print(f"  r={r:.2f} | Acc: {accuracy * 100:>5.1f}% | Confidence: {avg_margin:.2e} | Imposter: {top_imp}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Execute the experiment
    df_mismatch = run_model_mismatch_experiment()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Recovery Accuracy
    color_acc = 'tab:purple'
    ax1.plot(df_mismatch['radius'], df_mismatch['accuracy'],
             color=color_acc, marker='o', lw=2.5, label='Accuracy')
    ax1.set_xlabel('Smoothing Radius (r)', fontsize=11)
    ax1.set_ylabel('Recovery Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Rule Recovery Rate', fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Solver Confidence (Margin)
    color_marg = 'tab:gray'
    ax2.plot(df_mismatch['radius'], df_mismatch['margin'],
             color=color_marg, linestyle='--', marker='s', lw=2, label='Confidence (SSE Diff)')
    ax2.set_xlabel('Smoothing Radius (r)', fontsize=11)
    ax2.set_ylabel('Solver Margin (Log Scale)', fontsize=11, fontweight='bold')
    ax2.set_title('Solver Confidence', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    plt.tight_layout()

    # Ensure output directory exists before saving
    output_dir = "figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, "thesis_mismatch_experiment_3.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nExperiment complete. Plot saved to: {save_path}")
    plt.show()

    # Display qualitative imposter analysis
    print("\n--- Imposter Analysis ---")
    print(df_mismatch[['radius', 'accuracy', 'top_imposter']])