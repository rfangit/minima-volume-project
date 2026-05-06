"""
Reproduce paper Figure 4 plots for the overnight nanoGPT shakespeare run.

Mirrors the relevant cells of the per-experiment
`analyze_experiments_data.ipynb` notebooks (e.g. the modulo arithmetic
example), driven by `minima_volume.analysis_funcs`. Single-seed (no error
bands), so per-run traces and the central-tendency curve coincide.

Outputs:
  overnight_run/analysis/<data_L>/data_level_vs_log_volume_avg_errbar.png
      → fig 4 TOP analog: log volume vs training dataset size,
        in the fixed landscape with `base+L` examples.
  overnight_run/analysis/log_volumes_vs_data_levels.png
      → fig 4 BOTTOM analog: lines per model, log volume across
        landscape sizes, with cliff at L > M from cutoffs.json.

Usage (run from `experiments/shakespeare_char/`):
    .venv/bin/python analyze_overnight.py --loss-value 0.5
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from minima_volume.analysis_funcs import (
    multiple_minima_fixed_landscape,
    fixed_landscape_minima_labels,
    plot_fixed_landscape_minima_pair,
    model_volume_across_targets,
    append_cutoff_points,
    varying_landscape_minima_labels,
    plot_minima_volume_vs_data_level,
    save_results_dict_npz,
)

PROBLEM_NAME = "Shakespeare-char nanoGPT"
SEED_DIR_NAME = "overnight_run"  # treated as one experiment folder
DATASET_QUANTITIES = [0, 200, 950, 4950, 19950]
BASE_TRAIN_SIZE = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-value", type=float, default=0.5,
                        help="Loss threshold (must match a directory loss_<th>/).")
    args = parser.parse_args()
    loss_value = args.loss_value

    here = Path(__file__).resolve().parent
    os.chdir(here)  # base_dir defaults relative to cwd

    seed_dir = here / SEED_DIR_NAME
    base_output_dir = seed_dir / "analysis"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    experiment_folders = [SEED_DIR_NAME]
    data_modifications = [f"data_{q}" for q in DATASET_QUANTITIES]
    model_data_sizes = list(DATASET_QUANTITIES)
    base_train_size = BASE_TRAIN_SIZE
    base_shift = base_train_size  # for "data_*" type

    print(f"problem={PROBLEM_NAME}  loss_value={loss_value}")
    print(f"seed_dir={seed_dir}")
    print(f"data_modifications={data_modifications}")
    print(f"model_data_sizes={model_data_sizes}")

    # === FIG 4 TOP: fixed-landscape plots (one per landscape) ===========
    for dm in data_modifications:
        save_dir = base_output_dir / dm
        save_dir.mkdir(parents=True, exist_ok=True)

        labels = fixed_landscape_minima_labels(dm, base_train_size)
        all_mdl, all_log_rn, all_test_loss = multiple_minima_fixed_landscape(
            experiment_folders, dm, loss_value
        )
        if not all_mdl:
            print(f"  [{dm}] no results — skipping")
            continue

        landscape_size = base_train_size + int(dm.split("_")[1])
        title = (f"{PROBLEM_NAME} Minima Volumes\n"
                 f"In {landscape_size:,} Example Loss Landscape")
        natural_label = f"Minima (Trained On {landscape_size:,} Examples)"

        plot_fixed_landscape_minima_pair(
            all_mdl, all_log_rn,
            xlabel=labels["xlabel"], ylabel="Log Volume",
            title=title,
            log_scale=False,
            ranking=False,
            alpha=0.7,
            output_dir=str(save_dir),
            filename="data_level_vs_log_volume_avg_errbar",
            show_plot=False,
            plot_average=True,
            average_style="errorbar",
            central_tendency="mean",
            plot_x_error=True,
            xlabel_size=18, ylabel_size=18, title_size=18,
            legend_size=13, tick_size=12,
            base_shift=base_shift,
            background_colors=None,
            natural_minima_loc="first",
            natural_label=natural_label,
            other_label="Minima (Larger Datasets)",
            natural_marker="^",
            other_marker="o",
        )
        print(f"  [{dm}] wrote fixed-landscape plot")

    # === FIG 4 BOTTOM: across-landscapes plot ===========================
    labels = varying_landscape_minima_labels(data_modifications[0], base_train_size)
    results_dict = model_volume_across_targets(
        target_model_data_levels=model_data_sizes,
        loss_value=loss_value,
        experiment_folders=experiment_folders,
    )
    results_with_cutoff = append_cutoff_points(
        results_dict, threshold=loss_value, base_dir=str(here)
    )
    save_results_dict_npz(results_with_cutoff,
                          str(base_output_dir / "volumes_across_datasets.npz"))

    plot_minima_volume_vs_data_level(
        results_dict=results_with_cutoff,
        data_type=labels["data_type"],
        base_train_size=base_train_size,
        xlabel=labels["xlabel"],
        ylabel="Log Volume",
        suptitle=f"{PROBLEM_NAME}",
        title="Minima Volumes Across Datasets",
        log_scale=False,
        alpha=0.6,
        plot_average=True,
        output_dir=str(base_output_dir),
        filename="log_volumes_vs_data_levels",
        xlabel_size=18, ylabel_size=18, title_size=18, suptitle_size=18,
        legend_size=14, legend_title_fontsize=14,
    )
    plt.close("all")
    print(f"\nWrote analysis to {base_output_dir}")


if __name__ == "__main__":
    main()
