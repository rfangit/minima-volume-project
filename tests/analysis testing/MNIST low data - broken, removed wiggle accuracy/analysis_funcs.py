# =====================
# Imports
# =====================
import json
from pathlib import Path
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# =====================
# Folder & Results Utilities
# =====================

def grab_folder_names(base_dir=None):
    """
    Finds all model_* folders in the given base directory.
    Collects data/poison/noise subfolders from the first model folder.

    Returns:
        experiment_folders (list[str]): List of model_* folder names.
        data_modification_folders (list[str]): List of data_*/poison_*/noise_* folders from the first model folder.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    else:
        base_dir = Path(base_dir)

    # List all folders starting with 'model_'
    experiment_folders = [f.name for f in base_dir.iterdir() if f.is_dir() and f.name.startswith("model_")]

    # Collect subfolders from the first model folder
    data_modification_folders = []
    if experiment_folders:
        first_model_folder = base_dir / experiment_folders[0]
        data_modification_folders = [
            f.name
            for f in first_model_folder.iterdir()
            if f.is_dir() and f.name.startswith(("data", "poison", "noise"))
        ]

    return experiment_folders, data_modification_folders

def load_volume_results(model_folder, data_modification, loss_value):
    """
    Load results.json from a specific model/data modification/loss folder.
    Returns:
        model_data_levels, log_exp_r_n_values, test_loss_values
    All lists are sorted by model_data_levels.
    """
    results_path = Path(model_folder) / data_modification / f"loss_{loss_value}" / "results.json"
    
    if not results_path.exists():
        print(f"Warning: {results_path} does not exist!")
        return None, None, None

    with open(results_path, "r") as f:
        results = json.load(f)

    log_exp_r_n_values = results["log_exp_r_n_values"]
    test_loss_values = results["test_loss_values"]
    model_data_levels = results["model_data_levels"]

    # Sort all three lists by model_data_levels
    sorted_tuples = sorted(zip(model_data_levels, log_exp_r_n_values, test_loss_values), key=lambda x: x[0])
    mdl_sorted, log_sorted, loss_sorted = zip(*sorted_tuples)

    return list(mdl_sorted), list(log_sorted), list(loss_sorted)

def multiple_minima_fixed_landscape(experiment_folders, target_data_modification, loss_value, selected_folders=None):
    """
    Collect results for a given data landscape across experiment folders.
    Used to analyze minima volumes for a fixed data level.

    Returns:
        tuple: (all_model_data_levels, all_log_exp_r_n, all_test_loss)
    """
    if selected_folders is not None:
        folders_to_use = [f for f in experiment_folders if f in selected_folders]
    else:
        folders_to_use = experiment_folders

    all_model_data_levels = []
    all_log_exp_r_n = []
    all_test_loss = []

    for model_folder in folders_to_use:
        mdl, log_rn, test_loss = load_volume_results(model_folder, target_data_modification, loss_value)
        if mdl is None:
            continue
        all_model_data_levels.append(mdl)
        all_log_exp_r_n.append(log_rn)
        all_test_loss.append(test_loss)

    return all_model_data_levels, all_log_exp_r_n, all_test_loss

# =====================
# Ranking Utilities
# =====================

def rank_lists(x, y, ranking=False):
    """
    If ranking=True, convert x and y into their rank order.
    Uses method='average' to handle ties fairly.
    """
    if ranking:
        x = rankdata(x, method="average").tolist()
        y = rankdata(y, method="average").tolist()
    return x, y

def compute_average_ranks(all_x_ranks, all_y_ranks):
    """
    Compute the average and variance of y-ranks at each x-rank.
    """
    x_reference = all_x_ranks[0]  # assume all x-ranks are identical
    y_matrix = np.vstack(all_y_ranks)  # shape: (num_models, num_points)

    avg = np.mean(y_matrix, axis=0)
    var = np.var(y_matrix, axis=0)

    return x_reference, avg, var

# =====================
# Plotting Functions
# =====================

def plot_data_level_vs_log_vol(all_model_data_levels, all_log_exp_r_n,
                               xlabel="model_data_levels", ylabel="log_exp_r_n_values",
                               log_scale=False, ranking=False, alpha=0.5, save_path=None):
    """
    Plot model data levels vs log volume of minima (optionally ranking and/or log-scale).
    """
    title = "Model Data Levels vs " + ("Ranks of Log Volume" if ranking else "Log Volume")
    xlabel_plot = "Ranked " + xlabel if ranking else xlabel
    ylabel_plot = "Ranked " + ylabel if ranking else ylabel

    plt.figure(figsize=(6, 5))
    all_x_ranks = []
    all_y_ranks = []

    for mdl, log_rn in zip(all_model_data_levels, all_log_exp_r_n):
        mdl_prep, log_prep = rank_lists(mdl, log_rn, ranking=ranking)
        plt.plot(mdl_prep, log_prep, marker="o", alpha=alpha)
        if ranking:
            all_x_ranks.append(mdl_prep)
            all_y_ranks.append(log_prep)

    if ranking and all_x_ranks:
        x_ref, avg_y, var_y = compute_average_ranks(all_x_ranks, all_y_ranks)
        plt.errorbar(x_ref, avg_y, yerr=np.sqrt(var_y), fmt='-o', color='red', label='Average ± Std')
        plt.legend()

    plt.xlabel(xlabel_plot)
    plt.ylabel(ylabel_plot)
    if log_scale and not ranking:
        plt.yscale("log")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    if ranking and all_x_ranks:
        return x_ref, avg_y, var_y

# =====================
# Plotting Test Loss
# =====================

def plot_data_level_vs_test_loss(all_model_data_levels, all_test_loss,
                                 xlabel="model_data_levels", ylabel="test_loss_values",
                                 log_scale=False, ranking=False, alpha=0.5, save_path=None):
    """
    Plot model data levels vs test loss (optionally ranking and/or log-scale).
    """
    title = "Model Data Levels vs " + ("Ranks of Test Loss" if ranking else "Test Loss Values")
    xlabel_plot = "Ranked " + xlabel if ranking else xlabel
    ylabel_plot = "Ranked " + ylabel if ranking else ylabel

    plt.figure(figsize=(6, 5))
    all_x_ranks = []
    all_y_ranks = []

    for mdl, test_loss in zip(all_model_data_levels, all_test_loss):
        mdl_prep, test_prep = rank_lists(mdl, test_loss, ranking=ranking)
        plt.plot(mdl_prep, test_prep, marker="o", alpha=alpha)
        if ranking:
            all_x_ranks.append(mdl_prep)
            all_y_ranks.append(test_prep)

    if ranking and all_x_ranks:
        x_ref, avg_y, var_y = compute_average_ranks(all_x_ranks, all_y_ranks)
        plt.errorbar(x_ref, avg_y, yerr=np.sqrt(var_y), fmt='-o', color='red', label='Average ± Std')
        plt.legend()

    plt.xlabel(xlabel_plot)
    plt.ylabel(ylabel_plot)
    if log_scale and not ranking:
        plt.yscale("log")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    if ranking and all_x_ranks:
        return x_ref, avg_y, var_y

# =====================
# Plotting Log Volume vs Test Loss
# =====================

def plot_log_vol_vs_test_loss(all_log_exp_r_n, all_test_loss,
                              xlabel="log_exp_r_n_values", ylabel="test_loss_values",
                              log_scale=False, ranking=False, alpha=0.5, save_path=None):
    """
    Plot log volume vs test loss (optionally ranking and/or log-scale).
    """
    title = ("Ranks of " if ranking else "") + "Log Volume vs Test Loss"
    xlabel_plot = "Ranked " + xlabel if ranking else xlabel
    ylabel_plot = "Ranked " + ylabel if ranking else ylabel

    plt.figure(figsize=(6, 5))
    all_x_ranks = []
    all_y_ranks = []

    for log_rn, test_loss in zip(all_log_exp_r_n, all_test_loss):
        sorted_pairs = sorted(zip(log_rn, test_loss), key=lambda x: x[0])
        log_sorted, test_sorted = zip(*sorted_pairs)
        log_prep, test_prep = rank_lists(log_sorted, test_sorted, ranking=ranking)
        plt.plot(log_prep, test_prep, marker="o", alpha=alpha)
        if ranking:
            all_x_ranks.append(log_prep)
            all_y_ranks.append(test_prep)

    if ranking and all_x_ranks:
        x_ref, avg_y, var_y = compute_average_ranks(all_x_ranks, all_y_ranks)
        plt.errorbar(x_ref, avg_y, yerr=np.sqrt(var_y), fmt='-o', color='red', label='Average ± Std')
        plt.legend()

    plt.xlabel(xlabel_plot)
    plt.ylabel(ylabel_plot)
    if log_scale and not ranking:
        plt.yscale("log")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    if ranking and all_x_ranks:
        return x_ref, avg_y, var_y

# =====================
# Volume Across Landscape Levels
# =====================

def model_volume_across_landscape_levels(target_model_data_level, loss_value, experiment_folder, base_dir=None):
    """
    Analyze the volume of a model trained with a target data level across different data landscapes.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    else:
        base_dir = Path(base_dir)

    experiment_path = base_dir / experiment_folder
    if not experiment_path.exists():
        print(f"Experiment folder not found: {experiment_path}")
        return [], [], []

    results_list = []

    for data_mod_folder in sorted(experiment_path.iterdir()):
        if not data_mod_folder.is_dir():
            continue

        target_folder = data_mod_folder / f"loss_{loss_value}"
        results_file = target_folder / "results.json"
        if not results_file.exists():
            continue

        try:
            with open(results_file, "r") as f:
                results = json.load(f)

            model_data_levels = results.get("model_data_levels", [])
            if target_model_data_level in model_data_levels:
                idx = model_data_levels.index(target_model_data_level)
                log_exp_rn = results.get("log_exp_r_n_values", [])[idx]
                test_loss = results.get("test_loss_values", [])[idx]

                parts = data_mod_folder.name.split("_")
                data_level = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else None
                results_list.append((data_level, log_exp_rn, test_loss))

        except Exception as e:
            print(f"Error reading {results_file}: {e}")

    results_list.sort(key=lambda x: (x[0] is None, x[0]))
    loss_landscape_data_levels = [r[0] for r in results_list]
    all_log_exp = [r[1] for r in results_list]
    all_test_loss = [r[2] for r in results_list]

    return all_log_exp, all_test_loss, loss_landscape_data_levels


def model_volume_multiple_experiments(target_model_data_level, loss_value, experiment_folders, base_dir=None):
    """
    Analyze model volume across different landscapes, for multiple experiments
    """
    experiment_results = []

    for exp_folder in experiment_folders:
        log_exp, test_loss, data_levels = model_volume_across_landscape_levels(
            target_model_data_level=target_model_data_level,
            loss_value=loss_value,
            experiment_folder=exp_folder,
            base_dir=base_dir
        )

        if log_exp:  # Only append if results exist
            experiment_results.append({
                "experiment": exp_folder,
                "log_exp": log_exp,
                "test_loss": test_loss,
                "data_levels": data_levels,
            })

    print(f"Collected results for {len(experiment_results)} experiments")
    return experiment_results

# =====================
# Plot Minima Volume vs Data Level
# =====================

def plot_minima_volume_vs_data_level(
    results_dict,
    xlabel="Loss Landscape Data Level",
    ylabel="Log Volume",
    alpha=0.7,
    show_average=False,
    plot_only_average=False
):
    """
    Plot how minima volumes change as the data level changes.

    Args:
        results_dict (dict): {target_model_data_level: experiment_results_list}.
        show_average (bool): Whether to plot average ± std shading.
        plot_only_average (bool): If True, individual curves are ignored.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    color_cycle = cycle(plt.cm.tab10.colors)
    level_to_color = {level: next(color_cycle) for level in results_dict.keys()}

    for level, exp_results in results_dict.items():
        color = level_to_color[level]
        log_exp_matrix = np.array([exp_data["log_exp"] for exp_data in exp_results])
        data_levels = exp_results[0]["data_levels"]

        if not plot_only_average:
            for exp_data in exp_results:
                ax.plot(exp_data["data_levels"], exp_data["log_exp"], marker="o", color=color, alpha=alpha)

        if show_average:
            mean_y = np.mean(log_exp_matrix, axis=0)
            std_y = np.std(log_exp_matrix, axis=0)
            ax.plot(data_levels, mean_y, color=color, linewidth=2.5, label=f"Level {level} (avg)")
            ax.fill_between(data_levels, mean_y - std_y, mean_y + std_y, color=color, alpha=0.2)
        elif plot_only_average:
            ax.plot([], [], color=color, marker="o", linestyle="", label=f"Level {level}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(title="Target Model Data Level", fontsize=8)
    plt.show()
