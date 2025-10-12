# ============================================================
# FUNCTION DEPENDENCIES (overview of this module)
# ============================================================
#
# SECTION 1: RAW DATA GATHERING
#   grab_folder_names(base_dir=None)
#       → standalone
#   load_volume_results(model_folder, data_modification, loss_value)
#       → standalone
#   load_base_train_size(experiment_folder)
#       → standalone
#
# SECTION 2: RANKING UTILITIES
#   rank_lists(x, y)
#       → standalone
#   compute_average_ranks(all_x_ranks, all_y_ranks)
#       → depends on rank_lists (indirectly, since you must feed ranked lists)
#
# SECTION 3: FIXED LANDSCAPE ANALYSIS
#   multiple_minima_fixed_landscape(experiment_folders, target_data_modification, loss_value, selected_folders=None)
#       → calls load_volume_results
#
# SECTION 4: VARYING LANDSCAPE ANALYSIS (with cutoffs)
#   find_cutoff_index(experiment_folder, target_data_level, threshold)
#       → standalone (reads cutoffs.json)
#   append_cutoff_points(results_by_target, threshold, base_dir=None)
#       → calls find_cutoff_index
#   model_volume_varying_landscape(target_model_data_level, loss_value, experiment_folder, base_dir=None)
#       → standalone (reads results.json inside data_modification folders)
#   model_volume_multiple_experiments(target_model_data_level, loss_value, experiment_folders, base_dir=None)
#       → calls model_volume_across_landscape_levels
#   model_volume_across_targets(target_model_data_levels, loss_value, experiment_folders, base_dir=None)
#       → calls model_volume_multiple_experiments
#
# SECTION 5: LABELS + PLOTTING
#   fixed_landscape_minima_labels(data_modification_folder, base_train_size=None)
#       → standalone (uses only string logic)
#   varying_landscape_minima_labels(data_modification_folder, base_train_size=None)
#       → standalone (uses only string logic)
#   plot_fixed_landscape_minima_pair(...)
#       → generic plotting, not called by other functions
#   plot_minima_volume_vs_data_level(...)
#       → generic plotting, not called by other functions
#
# ============================================================
# Quick notes:
# - Fixed landscape family → multiple_minima_fixed_landscape, fixed_landscape_minima_labels
# - Varying landscape family → model_volume_*, append_cutoff_points, varying_landscape_minima_labels
# - Ranking utilities (rank_lists, compute_average_ranks) are general helpers
# - Plotting functions are independent utilities
# ============================================================


# =====================
# Imports
# =====================
import os
import json
from pathlib import Path
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import rankdata

# ============================================================
# SECTION 1: RAW DATA GATHERING FROM FOLDERS
# ============================================================

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

    experiment_folders = [f.name for f in base_dir.iterdir() if f.is_dir() and f.name.startswith("model_")]

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

    sorted_tuples = sorted(zip(model_data_levels, log_exp_r_n_values, test_loss_values), key=lambda x: x[0])
    mdl_sorted, log_sorted, loss_sorted = zip(*sorted_tuples)

    return list(mdl_sorted), list(log_sorted), list(loss_sorted)

def load_perturb_probs(model_folder, data_modification, loss_value):
    """
    Load results.json from a specific model/data modification/loss folder.
    Returns:
        subset_sizes, radii_rank_probabilities_full, radii_rank_probabilities 
    """
    results_path = Path(model_folder) / data_modification / f"loss_{loss_value}" / "results.json"
    
    if not results_path.exists():
        print(f"Warning: {results_path} does not exist!")
        return None, None, None

    with open(results_path, "r") as f:
        results = json.load(f)

    subset_sizes_values = results["subset_sizes"]
    radii_rank_probabilities_values = results["radii_rank_probabilities"]
    radii_rank_probabilities_full_levels = results["radii_rank_probabilities_full"]

    return list(subset_sizes_values), list(radii_rank_probabilities_values), list(radii_rank_probabilities_full_levels)

def load_param_num(experiment_folder, data_modification, loss_value):
    """
    Load num_params from results.json in a specific experiment/data modification/loss folder.
    
    Parameters:
        experiment_folder (str): Path to the experiment folder
        data_modification (str): Data modification folder name
        loss_value (float): Loss value to look for
        
    Returns:
        int: Number of parameters, or None if not found
    """
    results_path = Path(experiment_folder) / data_modification / f"loss_{loss_value}" / "results.json"
    
    if not results_path.exists():
        print(f"Warning: {results_path} does not exist!")
        return None

    try:
        with open(results_path, "r") as f:
            results = json.load(f)
        
        # Check if num_params exists in the results
        if "num_params" in results:
            return results["num_params"]
        else:
            print(f"Warning: 'num_params' not found in {results_path}")
            return None
            
    except (json.JSONDecodeError, KeyError, IOError) as e:
        print(f"Error reading {results_path}: {e}")
        return None

def load_base_train_size(experiment_folder: str) -> int:
    """
    Loads the base_train_size from cutoffs.json inside the given experiment/data folder.
    """
    cutoff_path = Path(experiment_folder) / "cutoffs" / "cutoffs.json"
    if not cutoff_path.exists():
        raise FileNotFoundError(f"cutoffs.json not found at {cutoff_path}")

    with open(cutoff_path, "r") as f:
        cutoff_results = json.load(f)

    if not cutoff_results:
        raise ValueError("cutoffs.json is empty")

    first_key = next(iter(cutoff_results))
    base_train_size = cutoff_results[first_key].get("base_train_size")

    if base_train_size is None:
        raise KeyError(f"'base_train_size' not found in {first_key}")

    print(f"First entry: {first_key}, base_train_size = {base_train_size}")
    return base_train_size

def list_additional_data(experiment_folder: str):
    """
    Collects all additional_data values from cutoffs.json in an experiment folder.

    Args:
        experiment_folder (str): Path to the experiment folder containing cutoffs/cutoffs.json.

    Returns:
        list[int]: All additional_data values found in the file.
    """
    cutoff_path = Path(experiment_folder) / "cutoffs" / "cutoffs.json"
    if not cutoff_path.exists():
        raise FileNotFoundError(f"cutoffs.json not found at {cutoff_path}")

    with open(cutoff_path, "r") as f:
        cutoff_results = json.load(f)

    additional_data_values = []
    for model_key, model_data in cutoff_results.items():
        if "additional_data" in model_data:
            additional_data_values.append(model_data["additional_data"])
        else:
            print(f"⚠️ Skipping {model_key}: no additional_data")

    return additional_data_values

def save_results_dict_npz(results_dict, filepath):
    """
    Save results_dict into a compressed .npz file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Save using object array with pickle
    np.savez_compressed(filepath, results_dict=results_dict)
    print(f"results_dict saved to {filepath}")


def load_results_dict_npz(filepath):
    """
    Load results_dict from a compressed .npz file.
    """
    data = np.load(filepath, allow_pickle=True)
    results_dict = data["results_dict"].item()  # unpack object
    print(f"results_dict loaded from {filepath}")
    return results_dict

# ============================================================
# SECTION 2: RANKING UTILITIES
# ============================================================

def rank_lists(x, y):
    """
    Convert x and y into their rank order.
    Uses method='average' to handle ties fairly.
    """
    x_ranked = rankdata(x, method="average").tolist()
    y_ranked = rankdata(y, method="average").tolist()
    return x_ranked, y_ranked


def compute_average_ranks(all_x_ranks, all_y_ranks):
    """
    Compute the average and variance of y-ranks at each x-rank.
    """
    x_reference = all_x_ranks[0]
    y_matrix = np.vstack(all_y_ranks)

    avg = np.mean(y_matrix, axis=0)
    var = np.var(y_matrix, axis=0)

    return x_reference, avg, var


# ============================================================
# SECTION 3: FIXED LANDSCAPE FUNCTIONS
# ============================================================

def multiple_minima_fixed_landscape(experiment_folders, target_data_modification, loss_value, selected_folders=None):
    """
    Goes through multiple experiment folders, looking for a fixed data landscape.
    """
    if selected_folders is not None:
        folders_to_use = [f for f in experiment_folders if f in selected_folders]
    else:
        folders_to_use = experiment_folders

    all_model_data_levels, all_log_exp_r_n, all_test_loss = [], [], []

    for model_folder in folders_to_use:
        mdl, log_rn, test_loss = load_volume_results(model_folder, target_data_modification, loss_value)
        if mdl is None:
            continue
        all_model_data_levels.append(mdl)
        all_log_exp_r_n.append(log_rn)
        all_test_loss.append(test_loss)

    return all_model_data_levels, all_log_exp_r_n, all_test_loss

def multiple_minima_fixed_landscape_perturb_probs(experiment_folders, target_data_modification, loss_value, selected_folders=None):
    """
    Goes through multiple experiment folders, looking for a fixed data landscape, and the probability of the correct largest ranking
    versus the number of perturbations needed
    """
    if selected_folders is not None:
        folders_to_use = [f for f in experiment_folders if f in selected_folders]
    else:
        folders_to_use = experiment_folders

    all_subset_sizes, all_perturb_probs, all_perturb_probs_full = [], [], []
    
    for model_folder in folders_to_use:
        subset_sizes, perturb_probs, perturb_probs_full = load_perturb_probs(model_folder, target_data_modification, loss_value)
        all_subset_sizes.append(subset_sizes)
        all_perturb_probs.append(perturb_probs)
        all_perturb_probs_full.append(perturb_probs_full)

    return all_subset_sizes, all_perturb_probs, all_perturb_probs_full

# ============================================================
# SECTION 4: NON-FIXED LANDSCAPE FUNCTIONS (WITH CUTOFFS)
# ============================================================

def find_cutoff_index(experiment_folder: str, target_data_level: int, threshold: float) -> int:
    """
    Finds the index where the loss curve of a given model first exceeds a threshold.
    """
    cutoff_path = Path(experiment_folder) / "cutoffs" / "cutoffs.json"
    if not cutoff_path.exists():
        raise FileNotFoundError(f"cutoffs.json not found at {cutoff_path}")

    with open(cutoff_path, "r") as f:
        cutoff_results = json.load(f)

    model_key = f"Model_{target_data_level}"
    if model_key not in cutoff_results:
        raise KeyError(f"{model_key} not found in cutoffs.json")

    loss_curve = cutoff_results[model_key].get("loss_curve")
    if loss_curve is None:
        raise ValueError(f"No 'loss_curve' found for {model_key}")

    for idx, val in enumerate(loss_curve):
        if val > threshold:
            print(f"{model_key}: first exceedance at index {idx} (value={val}, threshold={threshold})")
            return idx

    raise ValueError(f"{model_key}: loss curve never exceeds threshold={threshold}")


def append_cutoff_points(results_by_target, threshold, base_dir=None):
    """
    Append cutoff points to experiment results for each target data level.
    """
    base_dir = Path(base_dir) if base_dir else Path.cwd()

    for target_level, exp_list in results_by_target.items():
        for exp_data in exp_list:
            exp_folder = base_dir / exp_data["experiment"]

            try:
                cutoff_idx = find_cutoff_index(exp_folder, target_level, threshold)

                exp_data["data_levels"].append(cutoff_idx)
                exp_data["log_exp"].append(0)

                print(f"Appended cutoff for {exp_data['experiment']} | "
                      f"target={target_level}, cutoff_idx={cutoff_idx}")

            except Exception as e:
                print(f"⚠️ Skipped {exp_data['experiment']} (target={target_level}): {e}")

    return results_by_target


def model_volume_varying_landscape(target_model_data_level, loss_value, experiment_folder, base_dir=None):
    """
    Analyze how a particular model changes its volume over different landscapes.
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
    Analyze model volume across different landscapes, for multiple experiments.
    """
    experiment_results = []

    for exp_folder in experiment_folders:
        log_exp, test_loss, data_levels = model_volume_varying_landscape(
            target_model_data_level=target_model_data_level,
            loss_value=loss_value,
            experiment_folder=exp_folder,
            base_dir=base_dir
        )
        if log_exp:
            experiment_results.append({
                "experiment": exp_folder,
                "log_exp": log_exp,
                "test_loss": test_loss,
                "data_levels": data_levels,
            })

    print(f"Collected results for {len(experiment_results)} experiments")
    return experiment_results

def model_volume_across_targets(target_model_data_levels, loss_value, experiment_folders, base_dir=None):
    """
    Run volume analysis for multiple target data levels across multiple experiments.
    Iterates through target_model_data_levels in sorted order.
    """
    results_by_target = {}
    for target_level in sorted(target_model_data_levels):
        experiment_results = model_volume_multiple_experiments(
            target_model_data_level=target_level,
            loss_value=loss_value,
            experiment_folders=experiment_folders,
            base_dir=base_dir
        )
        results_by_target[target_level] = experiment_results
    return results_by_target

# ============================================================
# SECTION 5: LABEL GENERATION + PLOTTING
# ============================================================

def fixed_landscape_minima_labels(data_modification_folder: str, base_train_size: int):
    """
    Generate axis labels and plot titles depending on the type of data modification.
    """
    if data_modification_folder.startswith("data_"):
        # Extract number after "data_"
        try:
            data_size = int(data_modification_folder.split("_")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid data_modification_folder format: {data_modification_folder}")

        xlabel = "Training Dataset Size"
        title_volume = f"Volumes in Landscape with {base_train_size + data_size} Data"
        title_loss = f"Test Loss Vs Volumes (Landscape W/ {base_train_size + data_size} Data)"

    elif data_modification_folder.startswith(("poison", "noise")):
        xlabel = "Additional Poisoned Examples"
        title_volume = (
            f"Poisoned Dataset Size vs Log Volume "
            f"(Initial {base_train_size} Correct Points)"
        )
        title_loss = (
            f"Poisoned Dataset Size vs Test Loss "
            f"(Initial {base_train_size} Correct Points)"
        )

    else:
        raise ValueError(f"Unknown data modification type: {data_modification_folder}")

    return {"xlabel": xlabel, "title_volume": title_volume, "title_loss": title_loss}


def varying_landscape_minima_labels(data_modification_folder: str, base_train_size: int):
    """
    Generate axis labels and plot titles for varying landscape experiments.

    Args:
        data_modification_folder (str): The folder name (e.g. "data_0", "poison_10", "noise_50").
        base_train_size (int): The size of the base training set (used for poisoned/noise datasets).

    Returns:
        dict: A dictionary with 'xlabel', 'title_volume', 'title_loss', and 'data_type'.
    """
    # Figure out prefix
    if data_modification_folder.startswith("data"):
        data_type = "data"
        xlabel = "# Of Examples In Loss Landscape"
        title_volume = "Dataset Size Vs Log Volume"
        title_loss = "Dataset Size Vs Test Loss"

    elif data_modification_folder.startswith("poison"):
        data_type = "poison"
        xlabel = "# Of Poisoned Examples In Loss Landscape"
        title_volume = f"Poisoned Dataset Size Vs Log Volume (Initial {base_train_size} Correct Points)"
        title_loss = f"Poisoned Dataset Size Vs Test Loss (Initial {base_train_size} Correct Points)"

    elif data_modification_folder.startswith("noise"):
        data_type = "noise"
        xlabel = "# Of Noisy Examples In Loss Landscape"
        title_volume = f"Noisy Dataset Size Vs Log Volume (Initial {base_train_size} Correct Points)"
        title_loss = f"Noisy Dataset Size Vs Test Loss (Initial {base_train_size} Correct Points)"

    else:
        raise ValueError(f"Unknown data modification type: {data_modification_folder}")

    return {
        "xlabel": xlabel,
        "title_volume": title_volume,
        "title_loss": title_loss,
        "data_type": data_type
    }

def plot_fixed_landscape_minima_pair(
    all_x, all_y,
    xlabel="x", ylabel="y",
    log_scale=False, ranking=False, alpha=0.5,
    output_dir="analysis", filename="random_directions",
    title=None, sort_x=False, plot_average=False,
    average_style="errorbar",       # "errorbar" or "shaded"
    central_tendency="mean",        # "mean" or "median"
    plot_x_error=False,             # NEW: include x error bars if True
    show_plot=True,
    xlabel_size=12, ylabel_size=12, title_size=14,
    legend_size=12, tick_size=12,
    base_shift=0,
    show_average_spread_label = True,
    # Special plotting parameters:
    background_colors=None,         # list of colors for per-run lines (len = len(all_x))
    natural_minima_loc = "first",
    natural_label=None,             # legend label for "natural" minima (first point)
    other_label=None,               # legend label for all other points
    natural_marker="o",             # marker style for natural minima
    other_marker="o",               # marker style for others
    figsize=None,                   # NEW: optional figure size override
    seed_count=True,
):
    """
    General-purpose plotting function for relationships between model data.
    Can optionally plot an average (mean or median) across runs, with either
    error bars or a shaded region for spread.

    Args:
        all_x (list[list[float]]): X-axis data for multiple runs.
        all_y (list[list[float]]): Y-axis data for multiple runs.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        log_scale (bool): Apply log scaling on y-axis (only if not ranking).
        ranking (bool): Rank-transform data before plotting.
        alpha (float): Line transparency.
        output_dir (str): Directory to save plot.
        filename (str): Base filename (without extension).
        title (str|None): Plot title. If None, auto-generated.
        sort_x (bool): If True, sort (x, y) pairs by x before plotting.
        plot_average (bool): If True, plot a thicker line showing the average across runs.
        average_style (str): "errorbar" (default) or "shaded" region for spread visualization.
        central_tendency (str): "mean" (default, uses mean±std) or "median" (uses median±IQR).
        plot_x_error (bool): If True, also include x-error bars when using average_style="errorbar".
        xlabel_size (int): Font size for x-axis label.
        ylabel_size (int): Font size for y-axis label.
        title_size (int): Font size for title.
        base_shift (float): Optional constant to add to all x-values. Used for data level plots.
    """
    # Title logic
    if title is None:
        title = f"{ylabel} vs {xlabel}" if not ranking else f"Ranks of {ylabel} vs {xlabel}"

    xlabel_plot = "Ranked " + xlabel if ranking else xlabel
    ylabel_plot = "Ranked " + ylabel if ranking else ylabel

    fig = plt.figure(figsize=figsize if figsize is not None else (6, 5), constrained_layout=True)
    ax = plt.gca()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,2))    
    
    processed_x, processed_y = [], []

    for i, (x_vals, y_vals) in enumerate(zip(all_x, all_y)):
        x_vals = [x + base_shift for x in x_vals]

        if sort_x:
            pairs = sorted(zip(x_vals, y_vals), key=lambda p: p[0])
            x_vals, y_vals = zip(*pairs)

        if ranking:
            x_vals, y_vals = rank_lists(x_vals, y_vals)

        # Debugging: check for NaNs/Infs BEFORE plotting
        x_array_check = np.array(x_vals)
        y_array_check = np.array(y_vals)
        if np.isnan(x_array_check).any():
            print(f"⚠️ NaN detected in x_vals: {x_array_check}")
        if np.isnan(y_array_check).any():
            print(f"⚠️ NaN detected in y_vals: {y_array_check}")
        if np.isinf(x_array_check).any():
            print(f"⚠️ Inf detected in x_vals: {x_array_check}")
        if np.isinf(y_array_check).any():
            print(f"⚠️ Inf detected in y_vals: {y_array_check}")

        # Use custom colors if provided
        color = background_colors[i] if background_colors is not None else None
        plt.plot(x_vals, y_vals, marker="o", linestyle="--", alpha=alpha, color=color)

        processed_x.append(x_vals)
        processed_y.append(y_vals)
    
    if plot_average and processed_x:
        x_array = np.array(processed_x)
        y_array = np.array(processed_y)
    
        if central_tendency == "median":
            x_ref = np.median(x_array, axis=0)
            center_y = np.median(y_array, axis=0)
            y_low = np.percentile(y_array, 25, axis=0)
            y_high = np.percentile(y_array, 75, axis=0)
            x_low = np.percentile(x_array, 25, axis=0)
            x_high = np.percentile(x_array, 75, axis=0)
            label_center = "Median"
            label_spread = "IQR (25–75%)"
            color = "black"
        else:
            x_ref = np.mean(x_array, axis=0)
            center_y = np.mean(y_array, axis=0)
            y_std = np.std(y_array, axis=0)
            x_std = np.std(x_array, axis=0)
            y_low, y_high = center_y - y_std, center_y + y_std
            x_low, x_high = x_ref - x_std, x_ref + x_std
            label_center = "Mean"
            label_spread = "1 Std"
            color = "black"

        if seed_count:
            num_seeds = len(processed_x)
            label_spread += f" ({num_seeds} Seeds)"
    
        if average_style == "shaded":
            plt.plot(x_ref, center_y, color=color, linewidth=3, linestyle=":",
                     label=label_center)
            plt.fill_between(x_ref, y_low, y_high, color=color, alpha=0.2,
                             label=label_spread if show_average_spread_label else None)
            
        elif average_style == "errorbar":
            # --- Step 1: plot the connecting line first ---
            plt.plot(x_ref, center_y, color=(0, 0, 0), linewidth=2, linestyle="-", alpha=0.9)
        
            # --- Step 2: prepare errors and masks ---
            xerr = np.vstack([x_ref - x_low, x_high - x_ref]) if plot_x_error else None
            yerr = np.vstack([center_y - y_low, y_high - center_y])
        
            # --- Step 3: determine natural point location ---
            # user specifies: natural_miima_loc = "first" or "last"
            if natural_minima_loc == "first":
                natural_idx = np.argmin(x_ref)
            elif natural_minima_loc == "last":
                natural_idx = np.argmax(x_ref)
            else:
                raise ValueError("natural_miima_loc must be 'first' or 'last'")
        
            mask_natural = np.zeros_like(x_ref, dtype=bool)
            mask_natural[natural_idx] = True
            mask_rest = ~mask_natural
        
            # --- Step 4: plot natural (highlighted) point ---
            plt.errorbar(
                x_ref[mask_natural], center_y[mask_natural],
                xerr=xerr[:, mask_natural] if xerr is not None else None,
                yerr=yerr[:, mask_natural],
                fmt=natural_marker, linestyle="none",
                markersize=13,
                mfc='red',       # marker fill
                mec='black',     # marker edge
                mew=1.5,
                ecolor='black',  # error bar color
                elinewidth=2,
                capsize=4,
                capthick=2,
                zorder=5,
                label=(natural_label if natural_label is not None 
                       else (f"{label_center} ± {label_spread}" if show_average_spread_label else None))
            )
        
            # --- Step 5: plot remaining points ---
            plt.errorbar(
                x_ref[mask_rest], center_y[mask_rest],
                xerr=xerr[:, mask_rest] if xerr is not None else None,
                yerr=yerr[:, mask_rest],
                fmt=other_marker, linestyle="none",
                markersize=10,
                mfc='red',
                mec='black',
                mew=1.5,
                ecolor='black',
                elinewidth=2,
                capsize=4,
                capthick=2,
                zorder=4,
                label=(other_label if other_label is not None else None)
            )
        elif average_style == "none":
            plt.plot(x_ref, center_y, color=color, linewidth=2, linestyle=":",
                     label=label_center)

        # --- Step 5: create legend entries manually ---
        legend_handles = []
        legend_labels = []

        if natural_label is not None:
            legend_handles.append(Line2D([0], [0], marker='^', color='w',
                                         markerfacecolor='red', markeredgecolor='black',
                                         markersize=10, mew=1.5))
            legend_labels.append(natural_label)
        
        if other_label is not None:
            legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor='red', markeredgecolor='black',
                                         markersize=10, mew=1.5))
            legend_labels.append(other_label)
            
       # Create a dummy plot that's completely invisible
        dummy_fig, dummy_ax = plt.subplots()
        dummy_err = dummy_ax.errorbar(
            [0], [0],
            xerr=0.2, yerr=0.2,  # Use scalar values instead of lists for simplicity
            fmt='o',
            markersize=1,
            mfc='red',
            mec='black',
            mew=1,
            ecolor='black',
            elinewidth=1,
            capsize=3,
            capthick=2,
            linestyle="none"
        )
        
        # Get the handle and close the dummy figure
        legend_handle = dummy_err
        plt.close(dummy_fig)
        
        legend_handles.append(legend_handle)
        legend_labels.append(f"{label_center} ± {label_spread}" if show_average_spread_label else label_center)

        plt.legend(handles=legend_handles,
           labels=legend_labels,
           fontsize=legend_size)

    plt.xlabel(xlabel_plot, fontsize=xlabel_size)
    plt.ylabel(ylabel_plot, fontsize=ylabel_size)

    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    if log_scale:
        plt.xscale("log")

    plt.title(title, fontsize=title_size)
    plt.grid(True)

    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)

        save_name = filename
        if log_scale:
            save_name += "_log"
        if ranking:
            save_name += "_ranked"
        if plot_average:
            save_name += "_avg"
        save_name += ".png"

        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    if plot_average:
        return x_ref, center_y, y_low, y_high

def plot_individual_traces(minima_results, minima_trained_on_additional_data_level, base_shift, color, alpha, sort_x):
    """
    Plot volumes for minima trained with a given amount of additional data.

    Args:
        minima_results (list[dict]): Each entry represents one minima and contains:
            {
                "data_levels": list[int|float],  # Data levels used when probing the minima
                "log_exp": list[float],          # Corresponding log(volume) values
            }

        minima_trained_on_additional_data_level (int|float):
            The amount of *additional data* used when this minima was originally trained.
            (Used for highlighting its "own" training point in the plot.)

        base_shift (int):
            Constant shift applied to x-values, used when converting relative data levels
            to absolute dataset sizes (e.g., when base_train_size is nonzero).
        color (tuple):
        alpha (float):
        sort_x (bool):
    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]:
            all_x, all_y — Lists containing x and y arrays for every minima trace.
    """
    all_x, all_y = [], []

    # Ensure all traces have consistent length (shortest minima determines)
    min_len = min(len(m["log_exp"]) for m in minima_results)
    print(f"Plotting minima trained with {minima_trained_on_additional_data_level} additional data points")

    for minima in minima_results:
        x_vals = [x + base_shift for x in minima["data_levels"][:min_len]]
        y_vals = minima["log_exp"][:min_len]

        # Sort by x for consistent plotting
        if sort_x:
            x_vals, y_vals = zip(*sorted(zip(x_vals, y_vals), key=lambda p: p[0]))

        # Plot the trace
        plt.plot(x_vals, y_vals, marker="o", color=color, alpha=alpha)
        all_x.append(x_vals)
        all_y.append(y_vals)

        # Highlight the point corresponding to the minima’s own training data level
        training_point = minima_trained_on_additional_data_level + base_shift
        if training_point in x_vals:
            idx = x_vals.index(training_point)
            plt.scatter(
                training_point, y_vals[idx],
                color=color,
                s=50,
                alpha=alpha,
                zorder=3
            )

    return all_x, all_y


def plot_average_curve(minima_results, minima_trained_on_additional_data_level, base_shift, color, sort_x, central_tendency):
    """
    Plot the average (mean or median) curve across all minima trained with a certain data level.

    Args:
        minima_results (list[dict]):
            List of minima result dictionaries, each containing:
                {
                    "data_levels": list[int|float],
                    "log_exp": list[float],
                }

        minima_trained_on_additional_data_level (int|float):
            Amount of additional data used to train this group of minima.
            (Used for highlighting the training-level point.)

        base_shift (int):
        color (tuple):
        sort_x (bool):
        central_tendency (str): Either "mean" or "median" — determines which statistic to plot as the central line.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            data_levels, center_y — the x and y coordinates of the averaged curve.
    """
    # Truncate all traces to match the shortest one
    min_len = min(len(m["log_exp"]) for m in minima_results)
    log_exp_matrix = np.array([m["log_exp"][:min_len] for m in minima_results])
    data_levels = np.array([x + base_shift for x in minima_results[0]["data_levels"][:min_len]])

    # Compute central tendency and spread
    if central_tendency == "median":
        center_y = np.median(log_exp_matrix, axis=0)
        y_low = np.percentile(log_exp_matrix, 25, axis=0)
        y_high = np.percentile(log_exp_matrix, 75, axis=0)
    else:
        center_y = np.mean(log_exp_matrix, axis=0)
        y_std = np.std(log_exp_matrix, axis=0)
        y_low = center_y - y_std
        y_high = center_y + y_std

    # Optional sorting
    if sort_x:
        data_levels, center_y, y_low, y_high = map(np.array, zip(*sorted(
            zip(data_levels, center_y, y_low, y_high),
            key=lambda p: p[0]
        )))

    # Plot mean/median line and shaded area
    plt.plot(data_levels, center_y, color=color, linewidth=2.5)
    plt.fill_between(data_levels, y_low, y_high, color=color, alpha=0.2)

    return data_levels, center_y

def plot_minima_volume_vs_data_level(
    results_dict,
    xlabel="Loss Landscape Data Level",
    ylabel="Log Volume",
    title=None,
    suptitle=None,
    alpha=0.7,
    log_scale=False,
    output_dir="imgs/volume_plots",
    filename="minima_volume",
    sort_x=False,
    plot_average=False,
    plot_only_average=False,
    show_plot=True,
    central_tendency="mean",
    xlabel_size=12, ylabel_size=12, title_size=14, suptitle_size=18,
    legend_size=12, legend_title_fontsize=12, tick_size = 12, legend_loc="best",
    show_legend=True,
    data_type=None,
    base_train_size=None,
    xlim=None,
    ylim=None,
    yticks=None,
):
    """
    Plot the minima volumes (log volume) versus the data level they were evaluated on.

    -------------------------------------------------------------------------------
    Expected Format of `results_dict`:
    -------------------------------------------------------------------------------
    `results_dict` is a dictionary where:
        - **Keys** are the additional data levels used to train each minima.
          Example: 0, 50, 100, ... (number of extra training examples used for minima).

        - **Values** are lists of experiment results, where each experiment result
          is itself a dictionary representing one minima. Each minima dictionary has:
              {
                  "experiment": "model_0_data_10",       # Name or identifier of the minima
                  "log_exp": [923312.2196997597, 0],     # List of log(exp(r/n)) values for each data level, last is usually 0
                  "test_loss": [2.2788678636550905],     # List of test loss values (not always used here)
                  "data_levels": [0, 62]                 # Data levels where this minima was evaluated, last is where it exceeds threshold
              }

    For example:
        results_dict[0] → list of 10 minima trained on *0 additional data*.
        Each minima entry shows how its volume/log_exp changes across data_levels.

    -------------------------------------------------------------------------------
    How the function works:
    -------------------------------------------------------------------------------
    1. For each `minima_trained_on_additional_data_level` (the dict key),
       it gathers all corresponding minima results.
    2. Each minima trace (`log_exp` vs `data_levels`) is plotted individually.
    3. Optionally, a mean or median curve across all minima is plotted with shading.
    4. The point corresponding to the minima’s own training level is highlighted.
    5. The figure is optionally saved and/or displayed.
    -------------------------------------------------------------------------------
    """

    # --- Figure setup ---
    if title is None:
        title = f"{ylabel} vs {xlabel}"

    fig = plt.figure(figsize=(6, 5), constrained_layout=True)
    ax = plt.gca()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,2))
    
    color_cycle = cycle(plt.cm.tab10.colors)
    level_to_color = {level: next(color_cycle) for level in results_dict.keys()}

    found_minima_vol, found_minima_dataset = [], []

    # Legend title depends on whether we're plotting data experiments
    legend_title = "Trained on:" if data_type else "Minima Trained on Data Level"
    base_shift = base_train_size if data_type == "data" else 0

    # --- Iterate over all minima training levels ---
    for minima_trained_on_additional_data_level, exp_results in results_dict.items():
        color = level_to_color[minima_trained_on_additional_data_level]

        # Label for legend
        if data_type is None:
            label_str = f"Level {minima_trained_on_additional_data_level}"
        elif data_type == "data":
            if base_train_size is None:
                raise ValueError("base_train_size must be provided when data_type='data'")
            label_str = f"{minima_trained_on_additional_data_level + base_train_size:,} Examples"
        elif data_type in {"poison", "noise"}:
            label_str = f"Incorrect Points: {minima_trained_on_additional_data_level}"
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        # --- Plot the minima found with this data level. ---
        if not plot_only_average:
            plot_individual_traces(
                exp_results,
                minima_trained_on_additional_data_level,
                base_shift,
                color,
                alpha,
                sort_x
            )

        # --- Plot average ± spread for minima trained on a certain amount of data ---
        if plot_average:
            data_levels, center_y = plot_average_curve(
                exp_results,
                minima_trained_on_additional_data_level,
                base_shift,
                color,
                sort_x,
                central_tendency
            )

            plt.plot([], [], color=color, linewidth=2.5, label=label_str)

            # Highlight point corresponding to where minima was trained
            training_point = minima_trained_on_additional_data_level + base_shift
            if training_point in data_levels:
                idx = np.where(data_levels == training_point)[0]
                if len(idx):
                    idx = idx[0]
                    plt.scatter(
                        training_point,
                        center_y[idx],
                        color=color,
                        s=140,
                        edgecolors="black",
                        linewidths=1.0,
                        zorder=4
                    )
                    found_minima_dataset.append(training_point)
                    found_minima_vol.append(center_y[idx])
        elif plot_only_average:
            plt.plot([], [], color=color, marker="o", linestyle="", label=label_str)

    # --- Axis and title formatting ---
    plt.xlabel(xlabel, fontsize=xlabel_size)
    plt.ylabel(ylabel, fontsize=ylabel_size)
    if log_scale:
        plt.xscale("log")

    if suptitle:
        plt.suptitle(suptitle, fontsize=suptitle_size, y=1.01)
    plt.title(title, fontsize=title_size)
    plt.grid(True)

    # --- Tickmark sizes ---
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    
    # --- Legend and bounds ---
    if show_legend:
        plt.legend(
            title=legend_title,
            fontsize=legend_size,
            title_fontsize=legend_title_fontsize,
            loc=legend_loc
        )

    if yticks is not None:
        plt.yticks(yticks)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    # --- Save and show ---
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        save_name = filename
        if plot_average:
            save_name += "_avg"
        if plot_only_average:
            save_name += "_onlyavg"
        save_path = os.path.join(output_dir, f"{save_name}.png")
        plt.savefig(save_path)

    if show_plot:
        plt.show()
    plt.close()

    return found_minima_vol, found_minima_dataset

