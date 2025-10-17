# Standard library
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

"""
Data Processing and Analysis Utilities for Perturbation Experiments

This module provides tools for:
- Saving and loading perturbation experiment results
- Processing and analyzing wiggle test results
- Computing various metrics on perturbation radii
- Visualizing relationships between metrics

Functions:

* `save_perturbations`
* `load_perturbations`
* `create_small_data`
* `grab_param_num`
* `extract_and_store_train_loss`
* `extract_and_sort_data`
* `loss_threshold`
* `process_wiggle_results`
* `accuracy_threshold`
* `process_wiggle_accuracy_results`
* `compute_log_power_mean`
* `compute_power_mean`
* `compute_median`
* `compute_product`
* `plot_pair_metrics`
* `plot_metric_distributions`
* `check_training_explanation`
* `count_disordered_volumes`
* `get_ranks`
* `volume_analysis_pipeline`

"""

"""
The structure of our data dictionary is as follows

{
'wiggle_results': list of dictionaries of a wiggle_result. The number in this list is equal to the number of perturbations
Each dictionary contains results for a single perturbation.
{
'loss':
'coefficients':
'accs':
'perturbation_seed':
'perturbation_norm':
}
'model': PyTorch model used in analysis (state_dict will be saved)

Then additional kwargs, typically:

'additional_data':
'model_trained_data':
'dataset_type':
'base_dataset_size': 
'test_loss':
'test_accuracy':
'num_params':
}

When loading a set of models from a dataset:
We have the name of the model, followed by an above data dictionary.
Eg, poison_0 is the name of our model, and then we can access poison_0's wiggle_results or related.

When using process_wiggle_results:
Each model_name gets two new entries, r_vals, and violation_coeffs.
These entries are lists of the size of the number of perturbations.

For identifying large perturbations:
Going through the list of models, we inspect the r_vals and look for the top k values.
We obtain the indices related to these top vectors, and then inside of the models wiggle_results,
We retrieve the values of the seeds that gave rise to these large perturbations.
"""


# --------------------------
# File I/O Operations
# --------------------------

def save_perturbations(wiggle_results: list, 
                      model: torch.nn.Module,
                      output_dir: str = "imgs/swiss/random_dirs", 
                      filename: str = "random_directions.npz", 
                      **kwargs) -> None:
    """
    Save perturbation analysis results with complete metadata.
    
    Args:
        wiggle_results: List of dictionaries containing wiggle test results
        Each dictionary is of the form
        {
        'loss':
        'coefficients':
        'accs': #if this is in the metrics
        'perturbation_seed':
        'perturbation_norm':
        }
        model: PyTorch model used in analysis (state_dict will be saved)
        output_dir: Directory to save results (default: "imgs/swiss/random_dirs")
        filename: Name of output file (default: "random_directions.npz")
        **kwargs: Additional key-value pairs to be saved in the output file
        Typically:
        'additional_data':
        'model_trained_data':
        'dataset_type':
        'base_dataset_size': 
        'test_loss':
        'test_accuracy':
        'num_params':
        
    Saves:
        NPZ file containing all results and metadata with compression
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Move model to CPU and get state_dict
    original_device = next(model.parameters()).device
    model_cpu = model.cpu()
    model_state_cpu = model_cpu.state_dict()
    model.to(original_device)  # Restore original device
    
    save_dict = {
        'wiggle_results': wiggle_results,
        'model_state': model_state_cpu,
    }
    save_dict.update(kwargs)
    
    np.savez_compressed(output_path / filename, **save_dict)

def load_data_files(file_list, directory=''):
    """
    Load multiple NPZ files into a dictionary with filenames as keys.
    
    Args:
        file_list: List of filenames to load
        directory: Path where files are located (default: current directory)
    
    Returns:
        Dictionary mapping filenames (without .npz) to file contents
    """
    data_dict = {}
    
    for filename in file_list:
        filepath = Path(directory) / filename
        try:
            # Load the NPZ file
            npz_data = np.load(filepath, allow_pickle=True)
            
            # Convert to regular dictionary
            file_data = {key: npz_data[key] for key in npz_data.files}
            
            # Use filename without extension as key
            key = Path(filename).stem
            data_dict[key] = file_data
            
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    
    return data_dict

def create_small_data(data_dictionary: dict, subset_size: int = 10) -> dict:
    """
    Create a reduced-size version of loaded data by subsampling large arrays.
    
    Args:
        data_dictionary: Original dictionary from load_data_files()
        subset_size: Number of elements to keep (default: 10)
        
    Returns:
        Dictionary with same structure but smaller arrays
    """
    small_data = {}
    for filename, file_data in data_dictionary.items():
        small_data[filename] = {
            key: value[:subset_size] if key in ['wiggle_results', 'perturb_norms'] else value
            for key, value in file_data.items()
        }
    return small_data

# --------------------------
# Data Processing Functions
# --------------------------

def extract_and_store_train_loss(data_dictionary):
    """
    Extracts the base train loss value from each model's wiggle results
    and stores it back in the data dictionary under the key 'train_loss'.
    
    Args:
        data_dictionary: The main data dictionary to modify
        
    Returns:
        Dictionary mapping model names to their first loss values
    """
    loss_dict = {}
    
    for model_name, model_data in data_dictionary.items():
        try:
            # Get the first loss value from the first wiggle result
            first_loss = model_data['wiggle_results'][0]['loss'][0]
            loss_dict[model_name] = first_loss
            
            # Add the loss value back to the data dictionary
            data_dictionary[model_name]['train_loss'] = first_loss
            
        except (KeyError, IndexError, TypeError) as e:
            print(f"Warning: Could not extract loss for {model_name}: {str(e)}")
            loss_dict[model_name] = None
            data_dictionary[model_name]['train_loss'] = None
    
    return loss_dict

# Helper function to extract single values from arrays
def extract_value(value):
    # Check if it's a numpy array with a single element
    if hasattr(value, 'item') and hasattr(value, 'size') and value.size == 1:
        return value.item()
    # Check if it's a PyTorch tensor with a single element
    elif hasattr(value, 'item') and hasattr(value, 'numel') and value.numel() == 1:
        return value.item()
    # Otherwise return as-is
    else:
        return value

def extract_data(data_dict, x_axis_key, y_axis_key):
    """
    Extract data from a dictionary based on specified keys.
    
    Args:
        data_dict: Dictionary containing the processed data
        x_axis_key: Key for the x-axis values (e.g., 'poison_amount')
        y_axis_key: Key for the corresponding y-axis values (e.g., 'r_vals')
    
    Returns:
        Tuple of (x_values, y_values) in original order
    """
    # Initialize storage
    x_values = []
    y_values = []
    
    # Collect data for each model
    for model_data in data_dict.values():
        if x_axis_key in model_data and y_axis_key in model_data:
            x_val = extract_value(model_data[x_axis_key])
            y_val = extract_value(model_data[y_axis_key])
            x_values.append(x_val)
            y_values.append(y_val)
    
    return x_values, y_values
    
def loss_threshold(loss_values: list, threshold: float) -> int:
    """
    Find the first index where loss exceeds given threshold.
    
    Args:
        loss_values: List/array of loss values
        threshold: Threshold value to check against
        
    Returns:
        Index of first violation, or length of array if none found
    """
    loss_array = np.asarray(loss_values)
    above_threshold = np.where(loss_array > threshold)[0]
    return above_threshold[0] if len(above_threshold) > 0 else len(loss_array)

def process_wiggle_results(data_dictionary, threshold=0.1):
    """
    Process wiggle results across all models in data_dictionary to find threshold violations
    and compute perturbation radii. Adds 'r_vals' to each model's entry.
    
    Args:
        data_dictionary: Dictionary containing loaded data for multiple models
        threshold: Loss threshold value to check against
        
    Returns:
        Modified data_dictionary with added 'r_vals' for each model
    """
    for model_name, model_data in data_dictionary.items():
        coefficients = []
        perturb_radii = []
        
        # Skip if this model doesn't have wiggle results
        if 'wiggle_results' not in model_data:
            continue
            
        num_results = len(model_data['wiggle_results'])
        
        for i in range(num_results):
            loss = model_data['wiggle_results'][i]['loss']
            #print (loss)
            
            # Find first index where loss exceeds threshold
            violation_idx = loss_threshold(loss, threshold)

            # Check if not at end of array.
            # If it's at the end of the array, that means it never crossed a threshold, and don't add anything
            if violation_idx < len(loss): 
                coeff = model_data['wiggle_results'][i]['coefficients'][violation_idx]
                coefficients.append(coeff)
                
                # Compute perturbed radius and store
                radius = coeff * model_data['wiggle_results'][i]['perturbation_norm']
                perturb_radii.append(radius)
        
        # Add the computed radii to the model's data
        model_data['r_vals'] = perturb_radii
        model_data['violation_coeffs'] = coefficients
    
    return data_dictionary

# Accuracy Variants
def accuracy_threshold(accuracy_values, threshold):
    """
    Find the index of the first element in accuracy_values that drops below the threshold.
    """
    # Convert to numpy array if not already
    accuracy_array = np.asarray(accuracy_values)
    below_threshold = np.where(accuracy_array < threshold)[0]
    # Return first index if any exist, otherwise None
    return below_threshold[0] if len(below_threshold) > 0 else len(accuracy_array)

def process_wiggle_accuracy_results(data_dictionary, threshold=0.8):
    """
    Process wiggle results across all models in data_dictionary to find accuracy threshold violations
    and compute perturbation radii. Adds 'r_vals' to each model's entry.
    
    Args:
        data_dictionary: Dictionary containing loaded data for multiple models
        threshold: Accuracy threshold value to check against (accuracy drops below this)
        
    Returns:
        Modified data_dictionary with added 'r_vals' for each model
    """
    for model_name, model_data in data_dictionary.items():
        coefficients = []
        perturb_radii = []
        
        # Skip if this model doesn't have wiggle results
        if 'wiggle_results' not in model_data:
            continue
            
        num_results = len(model_data['wiggle_results'])
        
        for i in range(num_results):
            accs = model_data['wiggle_results'][i]['accs']            
            # Find first index where accs exceeds threshold
            violation_idx = loss_threshold(accs, threshold)

            # Check if not at end of array.
            # If it's at the end of the array, that means it never crossed the threshold, and don't add anything
            if violation_idx < len(accs): 
                coeff = model_data['wiggle_results'][i]['coefficients'][violation_idx]
                coefficients.append(coeff)
                
                # Compute perturbed radius and store
                radius = coeff * model_data['wiggle_results'][i]['perturbation_norm']
                perturb_radii.append(radius)
        processed_full_data = process_wiggle_results(data_dictionary, threshold=0.15)

        # Add the computed radii to the model's data
        model_data['accuracy_r_vals'] = perturb_radii
        model_data['violation_coeffs'] = coefficients
    
    return data_dictionary

# --------------------------
# Metric Computation
# --------------------------

def compute_log_power_mean(radii_list: List[List[float]], 
                          exponent: float = 1.0,
                          max_perturbs: Optional[int] = None) -> List[float]:
    """
    Compute log(<r^n>) for each sublist in radii_list using log-sum-exp for stability.
    Handles zeros safely.

    Args:
        radii_list: List of lists of radii (each sublist is a group of samples).
        exponent: Power n (can be large).
        max_perturbs: Optional maximum number of vectors to consider from each sublist.

    Returns:
        List of log-means: log( mean( r^n ) ) for each sublist.
        (If all radii are zero, returns -inf.)
    """
    results = []
    for radii in radii_list:
        if max_perturbs is not None:
            radii = radii[:max_perturbs]
        radii = np.array(radii, dtype=np.float64)
        
        # mask out zeros
        nonzero = radii > 0
        count_total = len(radii)
        count_nonzero = np.count_nonzero(nonzero)

        if count_nonzero == 0:
            # all radii = 0 → mean is 0 → log-mean = -inf
            results.append(-np.inf)
            continue

        logs = exponent * np.log(radii[nonzero])
        max_log = np.max(logs)

        # log of sum over nonzero entries
        log_sum_nonzero = max_log + np.log(np.sum(np.exp(logs - max_log)))

        # convert to log-mean over all entries (including zeros)
        log_mean = log_sum_nonzero - np.log(count_total)

        results.append(log_mean)

    return results

def compute_power_mean(radii_list: List[List[float]], 
                      exponent: float = 1.0,
                      max_perturbs: Optional[int] = None) -> List[float]:
    """
    Compute power mean (r^n) for each sublist of radii.
    
    Args:
        radii_list: List of lists containing radius values
        exponent: Power to raise radii to (default: 1.0 = arithmetic mean)
        max_perturbs: Optional maximum number of vectors to consider from each sublist.
        
    Returns:
        List of power means for each sublist
    """
    results = []
    for radii in radii_list:
        if max_perturbs is not None:
            radii = radii[:max_perturbs]
        results.append(np.mean(np.power(radii, exponent)))
    return results

def compute_max_radius(radii_list: List[List[float]],
                      max_perturbs: Optional[int] = None) -> List[float]:
    """
    Compute the maximum radius value for each sublist of radii.
    
    Args:
        radii_list: List of lists containing radius values
        max_perturbs: Optional maximum number of vectors to consider from each sublist.
        
    Returns:
        List of maximum radius values for each sublist
    """
    results = []
    for radii in radii_list:
        if max_perturbs is not None:
            radii = radii[:max_perturbs]
        results.append(np.max(radii))
    return results

def compute_median(radii_list: List[List[float]],
                  max_perturbs: Optional[int] = None) -> List[float]:
    """Compute median radius for each sublist in radii_list."""
    results = []
    for radii in radii_list:
        if max_perturbs is not None:
            radii = radii[:max_perturbs]
        results.append(np.median(radii))
    return results

def compute_product(radii_list: List[List[float]],
                   max_perturbs: Optional[int] = None) -> List[float]:
    """Compute product of all radii in each sublist."""
    results = []
    for radii in radii_list:
        if max_perturbs is not None:
            radii = radii[:max_perturbs]
        results.append(np.prod(radii))
    return results

# --------------------------
# Subset Rankings
# --------------------------

def generate_random_subset(current_lists, subset_size):
    """Generate a random subset of lists using shared indices."""
    if not current_lists:
        return []
    
    list_length = len(current_lists[0])
    if any(len(lst) != list_length for lst in current_lists):
        raise ValueError("All lists must have the same length.")
    if subset_size > list_length:
        raise ValueError(f"Subset size {subset_size} cannot exceed list length {list_length}.")
    
    random_indices = random.sample(range(list_length), subset_size)
    return [[lst[i] for i in random_indices] for lst in current_lists]


def get_ranking(lists):
    """Return the ranking (indices sorted by max value, descending)."""
    max_values = [max(lst) for lst in lists]
    return sorted(range(len(max_values)), key=lambda i: max_values[i], reverse=True)

def get_top_element(lists):
    """Return the index of the list with the maximum value."""
    max_values = [max(lst) for lst in lists]
    return max_values.index(max(max_values))

def ranking_stability(current_lists, subset_size, num_trials, check_full_ranking=True):
    """
    Calculate probability that random subsets produce the same ranking as the full data.
    
    Args:
        current_lists (list[list]): Lists of values
        subset_size (int): Size of random subsets
        num_trials (int): Number of random subsets to test
        check_full_ranking (bool): If True, check entire ranking matches.
                                  If False, check only top element matches.
    
    Returns:
        float: Probability of matching ranking
    """
    if check_full_ranking:
        original_result = get_ranking(current_lists)
    else:
        original_result = get_top_element(current_lists)
    
    matches = 0
    
    for _ in range(num_trials):
        subset_lists = generate_random_subset(current_lists, subset_size)
        
        if check_full_ranking:
            subset_result = get_ranking(subset_lists)
        else:
            subset_result = get_top_element(subset_lists)
        
        if subset_result == original_result:
            matches += 1
    
    return matches / num_trials

# --- Glue Function ---

def ranking_stability_curve(current_lists, num_trials, check_full_ranking=True, 
                           subset_sizes=None, max_subset_size=None):
    """
    Compute ranking stability probabilities for specified subset sizes.
    
    Args:
        current_lists (list[list]): Lists of values (all must have same length)
        num_trials (int): Number of random subsets per size
        check_full_ranking (bool): If True, check entire ranking matches.
                                  If False, check only top element matches.
        subset_sizes (list[int], optional): Specific subset sizes to test
        max_subset_size (int, optional): Maximum subset size to test (used if subset_sizes is None)
    
    Returns:
        tuple: (subset_sizes, probabilities)
    """
    list_length = len(current_lists[0])
    
    # Determine which subset sizes to test
    if subset_sizes is not None:
        # Use provided subset sizes, but filter out invalid ones
        valid_sizes = [size for size in subset_sizes if 1 <= size <= list_length]
        if not valid_sizes:
            raise ValueError("No valid subset sizes provided")
        subset_sizes = sorted(set(valid_sizes))  # Remove duplicates and sort
    else:
        # Generate default subset sizes
        if max_subset_size is None:
            max_subset_size = list_length
        else:
            max_subset_size = min(max_subset_size, list_length)
        subset_sizes = range(1, max_subset_size + 1)
    
    probabilities = [
        ranking_stability(current_lists, subset_size, num_trials, check_full_ranking)
        for subset_size in subset_sizes
    ]
    
    return list(subset_sizes), probabilities

# --------------------------
# Visualization
# --------------------------

def plot_pair_metrics(metric1_values: list,
                     metric2_values: list,
                     xlabel: str = "Metric Value 1",
                     ylabel: str = "Metric Value 2",
                     title: str = "",
                     log_scale: bool = False,
                     save_path: str = None,
                     display: bool = True,
                     xlim: tuple = None,
                     ylim: tuple = None,
                     connect_dots: bool = True,
                     label: str = 'Data',
                     show_best_fit: bool = False): 
    """
    Plot of a metric vs model additional data levels with connected dots, sorted by x-axis values
    
    Args:
        metric1_values: List of x-axis values
        metric2_values: List of corresponding y-axis values
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title
        log_scale: Whether to use log scale for y-axis
        save_path: Path to save the plot (if None, plot is displayed but not saved)
        display: Whether to display the plot (default: True)
        connect_dots: Whether to connect the dots with lines (default: True)
        label: Label for the data series in the legend (default: 'Data')
        show_best_fit: Whether to show linear best fit line (default: False)
    """
    # Sort both lists together based on metric1_values (x-axis)
    sorted_pairs = sorted(zip(metric1_values, metric2_values))
    sorted_metric1, sorted_metric2 = zip(*sorted_pairs)
    
    plt.figure(figsize=(8, 5))
    
    # Plot dots, optionally connected by lines using sorted values
    marker_style = 'o-' if connect_dots else 'o'
    plt.plot(sorted_metric1, sorted_metric2, marker_style, color='blue', 
             markersize=8, linewidth=2, label=label)
    
    # Add linear best fit line if requested
    if show_best_fit:
        # Calculate linear regression (y = mx + b)
        coefficients = np.polyfit(sorted_metric1, sorted_metric2, 1)
        slope, intercept = coefficients
        
        # Calculate R² value
        y_pred = np.polyval(coefficients, sorted_metric1)
        ss_res = np.sum((sorted_metric2 - y_pred) ** 2)
        ss_tot = np.sum((sorted_metric2 - np.mean(sorted_metric2)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Create best fit line
        fit_line = np.polyval(coefficients, sorted_metric1)
        plt.plot(sorted_metric1, fit_line, '--', color='red', linewidth=3, 
                 label=f'Linear Fit')
    
    # Formatting
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    if log_scale:
        plt.yscale('log')

    # Set axis limits if provided
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
        
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Display the plot only if requested
    if display:
        plt.show()
    else:
        plt.close()

def plot_metric_distributions(categories: list, 
                            metric_values: list, 
                            category_name: str = "Category",
                            metric_name: str = "Metric",
                            base_shift: int = 0,
                            title: str = "Metric Distributions by Category",
                            save_path: str = None,
                            display: bool = True) -> None:
                            
    """
    Plot separate histogram distributions of metrics for each category with consistent scaling.
    Categories are sorted before plotting to ensure ordered display.
    
    Args:
        categories: List of category labels (e.g., poison levels)
        metric_values: List of metric value arrays for each category (must match categories length)
        category_name: Name for the category variable (for axis labels)
        metric_name: Name for the metric being plotted (for axis labels)
        title: Overall plot title
        save_path: Path to save the plot (if None, plot is displayed but not saved)
    """
    # Input validation
    if len(categories) != len(metric_values):
        raise ValueError("categories and metric_values must have the same length")
    
    # Filter out empty metric arrays and pair with categories
    valid_data = [(cat, vals) for cat, vals in zip(categories, metric_values) if len(vals) > 0]
    if not valid_data:
        raise ValueError("No valid metric values to plot")
    
    # Sort by category to ensure ordered display
    valid_data.sort(key=lambda x: x[0])
    sorted_categories, sorted_metric_values = zip(*valid_data)
    
    # Get global min/max for consistent x-axis
    all_metrics = np.concatenate(sorted_metric_values)
    global_min = np.min(all_metrics)
    global_max = np.max(all_metrics)
    x_range = (global_min, global_max * 1.05)  # Add 5% padding
    
    # Create common bins
    bins = np.linspace(x_range[0], x_range[1], 30)
    
    # Create subplots
    fig, axes = plt.subplots(len(valid_data), 1, 
                           figsize=(10, 3 * len(valid_data)),
                           sharex=True)
    
    if len(valid_data) == 1:  # Handle single subplot case
        axes = [axes]
    
    # Colormap for consistent coloring
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(sorted_categories), vmax=max(sorted_categories))
    
    # Plot each distribution in sorted order
    for ax, (category, values) in zip(axes, valid_data):
        # Plot histogram
        counts, _, patches = ax.hist(values, bins=bins, alpha=0.7, 
                                   density=True,
                                   color=cmap(norm(category)))
        
        # Calculate statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        max_val = np.max(values)  # NEW: Calculate maximum value
        q1, q3 = np.percentile(values, [25, 75])
        
        # Add statistical markers
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                  label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle=':', linewidth=1.5,
                  label=f'Median: {median_val:.2f}')
        ax.axvline(max_val, color='green', linestyle='-', linewidth=1.5,  # NEW: Add maximum line
                  label=f'Max: {max_val:.2f}')
        ax.axvspan(q1, q3, color='gray', alpha=0.2, label='Interquartile Range')
        
        # Format subplot
        ax.set_title(f'{category_name} {category + base_shift} (n={len(values)})', pad=10)
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Set common labels
    axes[-1].set_xlabel(metric_name, fontsize=12)
    plt.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Display the plot only if requested
    if display:
        plt.show()
    else:
        plt.close()

# --------------------------
# Summary Functions
# --------------------------

def check_training_explanation(model_data_levels, log_exp_r_n):
    """
    Check if largest log_exp_r_n element = smallest model_data_levels element.    
    """
    max_log_index = log_exp_r_n.index(max(log_exp_r_n))
    min_poison_index = model_data_levels.index(min(model_data_levels))
    return max_log_index == min_poison_index

def count_disordered_volumes(test_loss, log_r_n):
    """
    Count violations where test_loss decreases instead of increasing
    when sorted by log_r_n (highest to smallest).
    """
    # Pair the lists and sort by log_r_n in descending order
    paired = list(zip(log_r_n, test_loss))
    paired.sort(key=lambda x: x[0], reverse=True)  # Sort by log_r_n, highest first
    
    # Extract sorted test_loss values, count violations
    sorted_test_loss = [loss for _, loss in paired]
    
    violations = 0
    for i in range(len(sorted_test_loss) - 1):
        if sorted_test_loss[i + 1] < sorted_test_loss[i]:
            violations += 1
    
    return violations

def get_ranks(lst):
    """
    Returns the ranks of elements in a list, where rank 1 is the largest element.

    Returns:
        List of ranks where each element's rank represents its position
        when sorted in descending order
    """
    arr = np.array(lst)
    # Get indices that would sort the array in descending order
    sorted_indices = np.argsort(-arr)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(lst) + 1)
    return ranks.tolist()

# --------------------------
# Top K Perturbations
# --------------------------

def get_top_perturbations_single_model(model_data, k=50):
    """
    For a single model's data, find the top k perturbations with largest r_vals,
    and return the r_vals, indices, and seeds.
    
    Args:
        model_data (dict): The data dictionary for a single model
        k (int): Number of top perturbations to extract
        
    Returns:
        tuple: (top_r_vals, top_indices, top_seeds) - all sorted by r_val descending
    """
    # Check if model has the required data
    if 'r_vals' not in model_data or 'wiggle_results' not in model_data:
        return [], [], []
    
    r_vals = model_data['r_vals']
    
    # Get indices of top k r_vals (largest first)
    top_indices = sorted(range(len(r_vals)), key=lambda i: r_vals[i], reverse=True)[:k]
    
    # Extract the corresponding values
    top_r_vals = [r_vals[i] for i in top_indices]
    top_seeds = [model_data['wiggle_results'][i]['perturbation_seed'] for i in top_indices]
    
    return top_r_vals, top_indices, top_seeds

# --------------------------
# Final Analysis Pipeline
# --------------------------

def analyze_and_plot_model_landscape(directory, loss_threshold, acc_threshold, verbose=True,
                                     display_options=None, top_k=50, save_paths_dict=None,
                                    max_perturbs = None):
    """
    Analyze model landscape by processing wiggle data and generating various plots.

    Args:
        directory (str): Directory path where input data is stored
        loss_threshold (float): Threshold value for processing wiggle loss results
        acc_threshold (float): Threshold value for processing wiggle accuracy results
        verbose (bool): If True, print detailed verification and evaluation messages
        display_options (dict): Which plots to display (not save)
        top_k (int): Number of top perturbations
        save_paths_dict (dict): Dictionary specifying paths where plots should be saved.
        max_perturbs (int, optional): Maximum number of perturbations to consider for each minima.

    Returns:
        dict: Dictionary containing computed metrics and processed data
    """
    
    if display_options is None:
        display_options = {'loss_plots': True, 'accuracy_plots': True}

    # === Validate user-specified paths ===
    required_keys = [
        "log_volume", "log_volume_generalization", "model_modification_vs_test_loss",
        "average_radius_loss", "radius_histogram_loss",
    ]
    if acc_threshold is not None:  # only require accuracy-related keys if accuracy is used
        required_keys += [
            "average_radius_acc", "radius_histogram_acc",
            "log_volume_acc", "log_volume_generalization_acc"
        ]
    if save_paths_dict is None:
        raise ValueError("You must specify save_paths_dict dict with keys: " + ", ".join(required_keys))
    missing = [k for k in required_keys if k not in save_paths_dict]
    if missing:
        raise ValueError(f"Missing required plot paths: {missing}")
    
    # Load data dictionary from directory
    # --- collect all .npz files ---
    files_to_load = [f for f in os.listdir(directory) if f.endswith(".npz")]
    data_dictionary = load_data_files(files_to_load, directory=directory)
    
    if verbose:
        print(f"Loaded data dictionary with {len(data_dictionary)} models from {directory}")
    
    # Acquire number of parameters
    model_key = list(data_dictionary.keys())[0]
    num_params = data_dictionary[model_key]['num_params']
    loss_landscape_data_param = data_dictionary[model_key]['additional_data']
    base_dataset_size = data_dictionary[model_key]['base_dataset_size']
    print ("The size of the base dataset is ", base_dataset_size)

    # Acquire type
    dataset_type = data_dictionary[model_key]['dataset_type']
    additional_param_level = f"{dataset_type} Level"

    # Base train loss for comparison with our threshold
    loss_values = extract_and_store_train_loss(data_dictionary)

    if verbose:
        print ("The number of model parameters is ", num_params)
        print (f"The loss landscape is for {dataset_type} trained with {loss_landscape_data_param}")
        for model, loss in loss_values.items():
            print(f"{model}: {loss}")

    #############################################################
    ######### LOSS ANALYSIS #########
    #############################################################
    
    processed_full_data = process_wiggle_results(data_dictionary, threshold=loss_threshold)
    model_data_levels, radii_list = extract_data(
        data_dict=processed_full_data,
        x_axis_key='model_trained_data',
        y_axis_key='r_vals'
    )

    train_loss, test_loss = extract_data(
        data_dict=data_dictionary,
        x_axis_key='train_loss',
        y_axis_key='test_loss'
    )

    _, test_accs = extract_data(
        data_dict=data_dictionary,
        x_axis_key='train_loss',
        y_axis_key='test_accs'
    )

    # ========== COMPUTE TOP K PERTURBATIONS ==========
    # Initialize lists to store top k perturbations across all models
    top_r_vals_all_models = []
    top_r_val_seeds_all_models = []
    
    # Process each model to get top k perturbations
    for model_name, model_data in data_dictionary.items():
        top_r_vals, top_indices, top_seeds = get_top_perturbations_single_model(model_data, k=top_k)
        top_r_vals_all_models.append(top_r_vals)
        top_r_val_seeds_all_models.append(top_seeds)
        
        if verbose:
            print(f"\nTop perturbations for {model_name}:")
            print(f"Top 5 r_vals: {top_r_vals[:5]}")
            print(f"Top 5 seeds: {top_seeds[:5]}")
    # ========== END TOP K ==========
    
    # Compute metrics for loss analysis
    power_mean_1 = compute_power_mean(radii_list, exponent=1.0, max_perturbs = max_perturbs) 
    max_radii = compute_max_radius(radii_list, max_perturbs = max_perturbs)
    log_exp_r_n = compute_log_power_mean(radii_list, exponent=num_params, max_perturbs = max_perturbs) 
    #print ("The shape of radii are:")
    #print (len(radii_list))
    #print (len(radii_list[0]))
    
    #############################################################
    ######### PLOTTING ##########
    #############################################################

    # Main Plots
    plot_pair_metrics(
        metric1_values=model_data_levels,
        metric2_values=log_exp_r_n,
        xlabel=additional_param_level,
        ylabel="Log Perturbation Radii^n (r^n)",
        title="Log Volume vs "+additional_param_level,
        save_path=save_paths_dict["log_volume"],
        display=True
    )

    print ("Test loss ", test_loss)
    print ("log_exp_r_n ", log_exp_r_n)

    plot_pair_metrics(
        metric1_values=test_loss,
        metric2_values=log_exp_r_n,
        xlabel="Test Loss",
        ylabel="Log Volume",
        title="Test Loss vs Log Volume",
        save_path=save_paths_dict["log_volume_generalization"],
        display=True
    )

    plot_pair_metrics(
        metric1_values=model_data_levels,
        metric2_values=test_loss,
        xlabel=additional_param_level,
        ylabel="Test Loss",
        title="Test Loss vs "+additional_param_level,
        save_path=save_paths_dict["model_modification_vs_test_loss"],
        display=True
    )
    
    # Loss analysis plots
    plot_pair_metrics( 
        metric1_values=model_data_levels,
        metric2_values=power_mean_1,
        xlabel=additional_param_level,
        ylabel="Mean Perturbation Radius (r)",
        title="<r> vs "+additional_param_level,
        save_path=save_paths_dict["average_radius_loss"],
        display=display_options['loss_plots']
    )

    plot_metric_distributions(
        categories=model_data_levels,
        metric_values=radii_list,
        category_name=additional_param_level,
        metric_name="Perturbation Radius",
        base_shift = base_dataset_size,
        title="Radius Distributions by "+additional_param_level,
        save_path=save_paths_dict["radius_histogram_loss"],
        display=display_options['loss_plots']
    )

    #############################################################
    ######### ACCURACY ANALYSIS #########
    #############################################################
    if acc_threshold is not None:
        processed_acc = process_wiggle_accuracy_results(data_dictionary, threshold=acc_threshold)
        model_data_levels_acc, accuracy_list = extract_data(
            data_dict=processed_acc,
            x_axis_key='model_trained_data',
            y_axis_key='accuracy_r_vals'
        )
    
        # Compute metrics for accuracy analysis
        power_mean_1_acc = compute_power_mean(accuracy_list, exponent=1.0, max_perturbs = max_perturbs)  # Regular mean
        log_exp_r_n_acc = compute_log_power_mean(accuracy_list, exponent=num_params, max_perturbs = max_perturbs) 
        
        # Accuracy analysis
        plot_pair_metrics( 
            metric1_values=model_data_levels_acc,
            metric2_values=power_mean_1_acc,
            xlabel=additional_param_level,
            ylabel="Mean Perturbation Radius (r)",
            title="<r> vs "+additional_param_level,
            save_path=save_paths_dict["average_radius_acc"],
            display=display_options['accuracy_plots']
        )
    
        plot_metric_distributions(
            categories=model_data_levels_acc,
            metric_values=accuracy_list,
            category_name=additional_param_level,
            metric_name="Perturbation Radius",
            title="Radius Distributions by "+additional_param_level,
            save_path=save_paths_dict["radius_histogram_acc"],
            display=display_options['accuracy_plots']
        )
    
        plot_pair_metrics(
            metric1_values=model_data_levels_acc,
            metric2_values=log_exp_r_n_acc,
            xlabel=additional_param_level,
            ylabel="Log Perturbation Radii^n (r^n)",
            title="Log Volume vs "+additional_param_level,
            save_path=save_paths_dict["log_volume_acc"],
            display=display_options['accuracy_plots']
        )
    
        plot_pair_metrics(
            metric1_values=test_loss,
            metric2_values=log_exp_r_n_acc,
            xlabel="Test Loss",
            ylabel="Log Volume",
            title="Test Loss vs Log Volume",
            save_path=save_paths_dict["log_volume_generalization_acc"],
            display=display_options['accuracy_plots']
        )

    #############################################################
    # Evaluation and results to be saved in results.json
    #############################################################
    
    # Calculate ranks for all arrays
    data_level_ranks = get_ranks(model_data_levels)
    test_loss_ranks = get_ranks(test_loss)
    log_exp_r_n_ranks = get_ranks(log_exp_r_n)
    log_exp_r_n_acc_ranks = get_ranks(log_exp_r_n_acc)
    
    # Print ranks for verification only if verbose is True
    if verbose:
        print("Ranks for quick verification:")
        print(f"data_level ranks: {data_level_ranks}")
        print(f"test_loss ranks: {test_loss_ranks}")
        print(f"log_exp_r_n ranks: {log_exp_r_n_ranks}")
        print(f"log_exp_r_n_acc ranks: {log_exp_r_n_acc_ranks}")
    
    # Calculate evaluation metrics
    indices_match = check_training_explanation(model_data_levels, log_exp_r_n)
    violation_count = count_disordered_volumes(test_loss, log_exp_r_n)    
    
    # Print evaluation results only if verbose is True
    if verbose:
        print("\nLoss:\nDoes it predict the minimas which are actually trained? ", indices_match)
        print("How many violations does it have, when ordering the log volumes? ", violation_count)

    if acc_threshold is not None:
        indices_match_acc = check_training_explanation(model_data_levels, log_exp_r_n_acc)
        violation_count_acc = count_disordered_volumes(test_loss, log_exp_r_n_acc)  
        if verbose:
            print("\nAccuracy:\nDoes it predict the minimas which are actually trained? ", indices_match_acc)
            print("How many violations does it have, when ordering the log volumes? ", violation_count_acc)
    
    # Create rank_results dictionary (secondary evaluations)
    rank_results = {
        "loss_predicts_minimas": bool(check_training_explanation(model_data_levels, log_exp_r_n)),
        "loss_ordering_violations": int(count_disordered_volumes(test_loss, log_exp_r_n)),
        "test_loss_ranks": test_loss_ranks,
        "log_exp_r_n_ranks": log_exp_r_n_ranks,
        "data_level_ranks": data_level_ranks,
    }

    if acc_threshold is not None:
        log_exp_r_n_acc_ranks = get_ranks(log_exp_r_n_acc)
        rank_results.update({
            "accuracy_predicts_minimas": bool(check_training_explanation(model_data_levels, log_exp_r_n_acc)),
            "accuracy_ordering_violations": int(count_disordered_volumes(test_loss, log_exp_r_n_acc)),
            "log_exp_r_n_acc_ranks": log_exp_r_n_acc_ranks,
        })
    
    # Create data_results dictionary (primary data summaries)
    data_results = {
        "test_loss_values": test_loss,
        "log_exp_r_n_values": log_exp_r_n,
        "log_exp_r_n_acc_values": log_exp_r_n_acc,
        "model_data_levels": model_data_levels,
        "loss_landscape_data_param": loss_landscape_data_param.item(),
        "train_loss_values": train_loss,
        "num_params": num_params.item(),  # Extract single value from numpy array
        "dataset_type": dataset_type.item(),  # Extract string from numpy array
        "top_r_vals": top_r_vals_all_models,  # List of lists: 5x50 for 5 models
        "top_r_val_seeds": top_r_val_seeds_all_models,  # List of lists: 5x50 for 5 models
        "test_acc_values": test_accs,
    }

    # Combine both dictionaries for saving
    results = {**data_results, **rank_results}
    
    # Save to file
    results_file_path = os.path.join(save_paths_dict["results.json"], "results.json")
    with open(results_file_path, "w") as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"Results saved to {results_file_path}")
    
    return results