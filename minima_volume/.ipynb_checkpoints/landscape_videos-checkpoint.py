# Standard library
import glob
import os
from pathlib import Path
import subprocess
import re

# Third-party
import numpy as np
import torch
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

# External
from minima_volume.perturb_funcs import ModelPerturber

"""
landscape_videos.py

A comprehensive toolkit for computing, visualizing, and animating neural network loss landscapes.
This module provides functions to analyze how loss surfaces evolve with varying amounts of training data.

Key features:
- 2D and 3D loss landscape visualization with logarithmic scaling
- Comparative analysis between multiple model parameterizations  
- Video generation showing landscape evolution across data amounts
- Support for PyTorch models and custom perturbation vectors

Typical workflow:
1. Compute perturbation vectors between reference and target models
2. Generate loss landscapes on grid coordinates
3. Visualize as 2D contour plots or 3D surfaces
4. Create animations showing landscape changes with increasing data

Example:
    >>> from landscape_videos import compute_loss_landscape_3models, plot_2d_loss_landscape
    >>> loss_grid, perturbations = compute_loss_landscape_3models(model_a, model_b, model_c, grid, x, y, loss_fn)
    >>> plot_2d_loss_landscape(C1_grid, C2_grid, loss_grid, -1, 1, additional_data=1000)
"""

###############################################################################
# MODEL COMPARISON AND PERTURBATION FUNCTIONS
###############################################################################

def compute_parameter_difference(model_1, model_2):
    """
    Compute the perturbation needed to make model_1 equal to model_2.
    Returns a dictionary where keys are parameter names and values are the differences.
    """
    perturbation_dict = {}
    
    # Get parameters from both models
    params_1 = dict(model_1.named_parameters())
    params_2 = dict(model_2.named_parameters())
    
    # Check if models have the same architecture
    if set(params_1.keys()) != set(params_2.keys()):
        raise ValueError("Models have different parameter names/architectures")
    
    # Compute differences
    for name in params_1.keys():
        param_1 = params_1[name]
        param_2 = params_2[name]
        
        # Check shapes match
        if param_1.shape != param_2.shape:
            raise ValueError(f"Parameter {name} has different shapes: {param_1.shape} vs {param_2.shape}")
        
        # Compute difference (what to add to model_1 to get model_2)
        perturbation_dict[name] = param_2.data - param_1.data
    
    return perturbation_dict

###############################################################################
# LOSS LANDSCAPE COMPUTATION FUNCTIONS
###############################################################################

def loss_landscape_2d(model, points, labels, loss_fn, perturbation_vectors,
                     grid_coefficients, output_dir="imgs/swiss/2d_perturb", 
                     filename="2d_perturbation_results.npz"):
    """
    2D perturbation analysis across two random directions.
    
    Args:
        model: Trained PyTorch model
        points: Input data (tensor)
        labels: Target labels (tensor)
        loss_fn: Loss function
        perturbation_vectors: List of two perturbation vectors
        grid_coefficients: Tuple of (C1_grid, C2_grid) meshgrid arrays
        output_dir: Output directory path
        filename: Name of the output NPZ file
    
    Returns:
        Tuple of (loss_grid, C1_values, C2_values)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Unpack the grid
    C1, C2 = grid_coefficients
    grid_points = C1.shape[0]  # Assuming square grid
    
    # Prepare model perturber
    perturber = ModelPerturber(model)
    
    # Compute loss grid
    loss_grid = np.zeros((grid_points, grid_points))

    for i in range(grid_points):
        for j in range(grid_points):
            # Apply combined perturbation
            combined_perturbation = {}
            
            for name in perturbation_vectors[0].keys():
                dir1 = perturbation_vectors[0][name]
                dir2 = perturbation_vectors[1][name]
                
                # Scale by coefficients from the pre-computed grid
                scaled_dir1 = dir1 * C1[i,j]
                scaled_dir2 = dir2 * C2[i,j]
                combined_perturbation[name] = scaled_dir1 + scaled_dir2
            
            perturber.apply_perturbation(combined_perturbation)
            
            # Compute loss
            with torch.no_grad():
                logits = model(points)
                loss = loss_fn(logits, labels)
                loss_grid[i,j] = loss.item()
            
            # Reset model
            perturber.reset()
    
    # Save results
    np.savez(output_path / filename,
            C1_values=C1,
            C2_values=C2,
            loss_grid=loss_grid)
    
    return loss_grid, C1, C2

    
def compute_loss_landscape_3models(model_0, model_1, model_2, grid_coefficients, 
                                 x_train, y_train, loss_fn, output_dir="test", 
                                 filename="loss_landscape.npz"):
    """
    Compute loss landscape for three models using the first as reference.
    
    Args:
        model_0: Reference model (origin at [0,0])
        model_1: First target model (will be at [1,0] in perturbation space)
        model_2: Second target model (will be at [0,1] in perturbation space)
        grid_coefficients: Tuple of (C1_grid, C2_grid) meshgrid arrays
        x_train: Input data tensor
        y_train: Target labels tensor
        loss_fn: Loss function
        output_dir: Output directory for saving results
        filename: Output filename
    
    Returns:
        Tuple of (loss_grid, perturbation_vectors)
        where perturbation_vectors = [vec_to_model_1, vec_to_model_2]
    """
    # Compute perturbation vectors
    perturbation_vector_1 = compute_parameter_difference(model_0, model_1)
    perturbation_vector_2 = compute_parameter_difference(model_0, model_2)
    perturbation_vectors = [perturbation_vector_1, perturbation_vector_2]
    
    # Generate loss landscape
    loss_grid, C1_grid, C2_grid = loss_landscape_2d(
        model=model_0,
        points=x_train,
        labels=y_train,
        loss_fn=loss_fn,
        perturbation_vectors=perturbation_vectors,
        grid_coefficients=grid_coefficients,
        output_dir=output_dir,
        filename=filename
    )
    
    return loss_grid, perturbation_vectors

def compute_loss_landscapes_varying_data(model_0, model_1, model_2, grid_coefficients, 
                                       x_base, y_base, x_additional, y_additional, 
                                       additional_data_amounts, loss_fn, output_dir="test", 
                                       base_filename="loss_landscape"):
    """
    Compute loss landscapes for varying amounts of additional data.
    
    Args:
        model_0: Reference model (origin at [0,0])
        model_1: First target model (will be at [1,0] in perturbation space)
        model_2: Second target model (will be at [0,1] in perturbation space)
        grid_coefficients: Tuple of (C1_grid, C2_grid) meshgrid arrays
        x_base: Base input data tensor
        y_base: Base target labels tensor
        x_additional: Additional input data tensor
        y_additional: Additional target labels tensor
        additional_data_amounts: List of integers specifying how much additional data to use
        loss_fn: Loss function
        output_dir: Output directory for saving results
        base_filename: Base filename for output files
    
    Returns:
        List of loss_grids for each additional_data amount, in the same order as input list
    """
    loss_grids = []
    
    # Compute perturbation vectors once (they don't depend on the data)
    perturbation_vector_1 = compute_parameter_difference(model_0, model_1)
    perturbation_vector_2 = compute_parameter_difference(model_0, model_2)
    perturbation_vectors = [perturbation_vector_1, perturbation_vector_2]
    
    print(f"=== Computing loss landscapes for {len(additional_data_amounts)} data amounts ===")
    
    for i, additional_data in enumerate(additional_data_amounts):
        print(f"Processing additional data amount {i+1}/{len(additional_data_amounts)}: {additional_data}")
        
        # Create evaluation dataset with specified amount of additional data
        if additional_data == 0:
            x_landscape = x_base
            y_landscape = y_base
        else:
            # Ensure we don't try to use more data than available
            actual_additional = min(additional_data, len(x_additional))
            x_landscape = torch.cat([x_base, x_additional[:actual_additional]], dim=0)
            y_landscape = torch.cat([y_base, y_additional[:actual_additional]], dim=0)
        
        print(f"  Dataset size: {len(x_landscape)} samples")
        
        # Generate filename with index
        filename = f"{base_filename}_index_{i}_amount_{additional_data}.npz"
        
        # Generate loss landscape
        loss_grid, _, _ = loss_landscape_2d(
            model=model_0,
            points=x_landscape,
            labels=y_landscape,
            loss_fn=loss_fn,
            perturbation_vectors=perturbation_vectors,
            grid_coefficients=grid_coefficients,
            output_dir=output_dir,
            filename=filename
        )
        
        loss_grids.append(loss_grid)
        print(f"  ✓ Completed and saved as {filename}")
    
    return loss_grids

###############################################
# New Better Computation Function
###############################################

def compute_loss_landscape_3models_varying_data(
    model_0,
    model_1,
    model_2,
    grid_coefficients,
    x_base,
    y_base,
    x_additional,
    y_additional,
    additional_data_amounts,
    loss_fn_per_sample,
    output_dir="test",
    base_filename="loss_landscape"
):
    """
    Efficiently compute loss landscapes for three models over varying amounts of data
    using a single forward pass per grid point and per-sample loss computation.

    Args:
        model_0, model_1, model_2: Models to compare (model_0 is the reference/origin)
        grid_coefficients: Tuple of (C1_grid, C2_grid) meshgrid arrays
        x_base, y_base: Base dataset
        x_additional, y_additional: Additional data to append
        additional_data_amounts: List of integers specifying how much additional data to use
        loss_fn_per_sample: Loss function that returns per-sample loss values
        output_dir: Where to save .npz results
        base_filename: Base filename for saved files

    Returns:
        loss_grids: List of loss grids, one for each additional data amount
        C1, C2: The coefficient grids
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    C1_grid, C2_grid = grid_coefficients
    grid_points = C1_grid.shape[0]
    
    # Compute perturbation vectors once
    perturbation_vector_1 = compute_parameter_difference(model_0, model_1)
    perturbation_vector_2 = compute_parameter_difference(model_0, model_2)
    perturbation_vectors = [perturbation_vector_1, perturbation_vector_2]

    # Prepare perturber
    perturber = ModelPerturber(model_0)

    # Precompute largest dataset
    max_additional = max(additional_data_amounts)
    x_landscape = torch.cat([x_base, x_additional[:max_additional]], dim=0)
    y_landscape = torch.cat([y_base, y_additional[:max_additional]], dim=0)

    # Preallocate one loss grid per data amount
    loss_grid_list = [
        np.zeros((grid_points, grid_points), dtype=np.float32)
        for _ in additional_data_amounts
    ]

    print(f"=== Computing loss landscapes for {len(additional_data_amounts)} data amounts ===")

    for i in range(grid_points):
        for j in range(grid_points):
            # Compute perturbation at (i, j)
            combined_perturbation = {}
            for name in perturbation_vectors[0].keys():
                dir1 = perturbation_vectors[0][name]
                dir2 = perturbation_vectors[1][name]
                combined_perturbation[name] = dir1 * C1_grid[i, j] + dir2 * C2_grid[i, j]

            # Apply perturbation
            perturber.apply_perturbation(combined_perturbation)

            with torch.no_grad():
                logits = model_0(x_landscape)
                per_sample_losses = loss_fn_per_sample(logits, y_landscape)  # shape [N]

                # Compute average losses for each data amount
                for k, additional_data in enumerate(additional_data_amounts):
                    total_count = len(x_base) + additional_data
                    avg_loss = per_sample_losses[:total_count].mean().item()
                    loss_grid_list[k][i, j] = avg_loss

            # Reset model before next perturbation
            perturber.reset()

    # Save results
    for k, additional_data in enumerate(additional_data_amounts):
        filename = f"{base_filename}_index_{k}_amount_{additional_data}.npz"
        np.savez(output_path / filename, C1_values=C1_grid, C2_values=C2_grid, loss_grid=loss_grid_list[k])
        print(f"  ✓ Saved loss landscape for amount={additional_data} as {filename}")

    return loss_grid_list, C1_grid, C2_grid

###############################################################################
# VISUALIZATION FUNCTIONS
###############################################################################

# Format the colorbar tick labels to show 3 significant digits
def format_func(value, tick_number):
    if value == 0:
        return "0"
    # Use scientific notation for very small or very large numbers
    if value < 0.001 or value >= 1000:
        return f"{value:.2e}"
    # For numbers between 0.001 and 1000, use 3 significant digits
    return f"{value:.3g}"

def plot_2d_loss_landscape_3model(C1_grid, C2_grid, loss_grid, span_low, span_high, additional_data=0, 
                                  model_labels=None, vmin=None, vmax=None, figsize=(8, 6), 
                                  cmap='viridis', output_path=None, show=True):
    """
    Plot 2D loss landscape with logarithmic color scale.
    
    Args:
        C1: Meshgrid for first coefficient
        C2: Meshgrid for second coefficient  
        loss_grid: 2D array of loss values
        span_low: Lower bound of grid range
        span_high: Upper bound of grid range
        additional_data: Amount of additional data used for this landscape
        model_labels: List of three labels for [origin, model_1, model_2]. If None, uses default labels.
        vmin: Minimum value for color scale (values below this will use darkest color)
        vmax: Maximum value for color scale (if None, uses max of this loss_grid)
        figsize: Tuple specifying figure size (width, height) in inches. Default: (8, 6)
        output_path: Path to save the plot (optional)
        show: Whether to display the plot (default: True)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use default labels if not provided
    if model_labels is None:
        model_labels = ['Original Model (0,0)', 'Model 1 (1,0)', 'Model 2 (0,1)']
    
    # Determine color scale limits
    if vmin is None:
        vmin = loss_grid[loss_grid > 0].min()
    if vmax is None:
        vmax = loss_grid.max()
    
    # Clip values below vmin to vmin to ensure they use the darkest color
    clipped_loss_grid = np.clip(loss_grid, vmin, None)
    
    # Use logarithmic levels for contour plot
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 50)

    contour = ax.contourf(C1_grid, C2_grid, clipped_loss_grid, levels=levels, cmap=cmap, 
                          norm=LogNorm(vmin=vmin, vmax=vmax), extend='both')
    
    # Create colorbar with formatted labels (3 significant digits)
    cbar = plt.colorbar(contour, label='Loss (log scale)')    
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

    # Mark the key points with custom labels
    ax.scatter([0], [0], color='red', s=100, marker='o', label=model_labels[0])
    ax.scatter([1], [0], color='blue', s=100, marker='s', label=model_labels[1])
    ax.scatter([0], [1], color='green', s=100, marker='^', label=model_labels[2])

    ax.set_xlabel('Coefficient for Model 1 perturbation')
    ax.set_ylabel('Coefficient for Model 2 perturbation')
    ax.set_title(f'2D Loss Landscape (Logarithmic Color Scale)\n(Additional Data = {additional_data})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_3d_loss_landscape(C1_grid, C2_grid, loss_grid, span_low, span_high, additional_data=0, 
                          model_labels=None, figsize=(12, 9), output_path=None, show=True):
    """
    Plot 3D loss landscape with logarithmic z-axis.
    
    Args:
        C1: Meshgrid for first coefficient
        C2: Meshgrid for second coefficient
        loss_grid: 2D array of loss values  
        span_low: Lower bound of grid range
        span_high: Upper bound of grid range
        additional_data: Amount of additional data used for this landscape
        model_labels: List of three labels for [origin, model_1, model_2]. If None, uses default labels.
        figsize: Tuple specifying figure size (width, height) in inches. Default: (12, 9)
        output_path: Path to save the plot (optional)
        show: Whether to display the plot (default: True)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Use default labels if not provided
    if model_labels is None:
        model_labels = ['Original', 'Model 1', 'Model 2']

    # Apply logarithmic transformation to the loss values
    epsilon = 1e-10
    log_loss_grid = np.log(loss_grid + epsilon)

    surf = ax.plot_surface(C1_grid, C2_grid, log_loss_grid, cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)

    # Calculate grid indices for key points
    N = C1_grid.shape[0]
    origin_idx = N // 2
    model1_idx = int((1 - span_low) / (span_high - span_low) * (N - 1))
    model2_idx = int((1 - span_low) / (span_high - span_low) * (N - 1))

    # Mark the key points with custom labels
    ax.scatter([0], [0], [log_loss_grid[origin_idx, origin_idx]], 
               color='red', s=100, marker='o', label=model_labels[0])
    ax.scatter([1], [0], [log_loss_grid[model1_idx, origin_idx]], 
               color='blue', s=100, marker='s', label=model_labels[1])
    ax.scatter([0], [1], [log_loss_grid[origin_idx, model2_idx]], 
               color='green', s=100, marker='^', label=model_labels[2])

    ax.set_xlabel('Coefficient for Model 1')
    ax.set_ylabel('Coefficient for Model 2')
    ax.set_zlabel('Log(Loss)')
    ax.set_title(f'3D Loss Landscape (Logarithmic Scale)\n(Additional Data = {additional_data})')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Log(Loss)')
    ax.legend()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)

###############################################################################
# ANIMATION AND VIDEO FUNCTIONS
###############################################################################

# Function to create video from images using only matplotlib
def create_video_ffmpeg(image_folder, output_video_path, fps=2):
    """
    Create video from images sorted by index, using ffmpeg with a concat file list.
    No extra libraries required.
    """
    image_folder = Path(image_folder)

    # Sort by index
    def extract_index(path):
        m = re.search(r'index_(\d+)', path.stem)
        return int(m.group(1)) if m else float('inf')

    image_files = sorted(
        [f for f in image_folder.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')],
        key=extract_index
    )

    if not image_files:
        raise FileNotFoundError(f"No images found in {image_folder}")

    print(f"Creating video from {len(image_files)} images (sorted by index)...")

    # Build ffmpeg concat file list
    concat_file = image_folder / "ffmpeg_input.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for img in image_files:
            # escape single quotes for ffmpeg compatibility
            f.write(f"file '{img.as_posix()}'\n")

    # Run ffmpeg with concat demuxer
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-r", str(fps),
        "-i", str(concat_file),
        "-pix_fmt", "yuv420p",
        str(output_video_path)
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"✓ Video saved to {output_video_path}")

    # Optional: remove the concat file afterwards
    concat_file.unlink(missing_ok=True)

###############################################################################
# DATA LOADING AND PROCESSING FUNCTIONS
###############################################################################

def load_computed_loss_landscapes(output_dir, indexes_to_load):
    """
    Simple function to load loss landscapes by index.
    
    Args:
        output_dir: Directory where the NPZ files are stored
        indexes_to_load: List of indexes to load (e.g., [0, 1, 2, 3])
    
    Returns:
        tuple: (loss_grids_list, C1_grid, C2_grid)
        where loss_grids_list is a list of loss grids in the same order as indexes_to_load
    """
    output_path = Path(output_dir)
    loss_grids_list = []
    C1_grid, C2_grid = None, None
    
    for index in indexes_to_load:
        # Use pathlib for pattern matching
        pattern = output_path / f"landscape_index_{index}_amount_*.npz"
        files = list(output_path.glob(f"landscape_index_{index}_amount_*.npz"))
        
        if not files:
            raise FileNotFoundError(f"No file found for index {index} in {output_dir}")
        
        if len(files) > 1:
            print(f"Warning: Multiple files found for index {index}, using first one: {files[0].name}")
        
        # Load the file
        data = np.load(files[0])
        
        # Store the grid from the first file
        if C1_grid is None:
            C1_grid = data['C1_values']
            C2_grid = data['C2_values']
        
        # Add the loss grid to our list
        loss_grids_list.append(data['loss_grid'])
        
        print(f"✓ Loaded index {index}: {files[0].name}")
    
    print(f"Successfully loaded {len(loss_grids_list)} loss landscapes")
    return loss_grids_list, C1_grid, C2_grid

###############################################################################
# LANDSCAPE ANALYSIS AND UTILITY FUNCTIONS
###############################################################################

def select_evenly_spaced_models(loaded_models, loaded_additional_data, n_models=3):
    """
    Select evenly spaced models based on their additional data values.
    
    Args:
        loaded_models (list): List of loaded model objects
        loaded_additional_data (list): List of additional data values for each model
        n_models (int): Number of models to select (default: 3)
        
    Returns:
        tuple: (selected_models, selected_additional_data) - lists of selected models and their additional data values
        
    Raises:
        ValueError: If fewer models are available than requested
    """
    # Check if we have enough models
    if len(loaded_models) < n_models:
        raise ValueError(f"Requested {n_models} models, but only {len(loaded_models)} available")
    
    if len(loaded_models) != len(loaded_additional_data):
        raise ValueError("Number of models and additional data values must be equal")
    
    # Pair models with their additional data and sort by additional data
    paired = sorted(zip(loaded_models, loaded_additional_data), key=lambda x: x[1])
    
    # Get evenly spaced indices
    indices = np.linspace(0, len(paired) - 1, n_models, dtype=int)
    
    # Extract selected models and their additional data
    selected_models = [paired[i][0] for i in indices]
    selected_additional_data = [paired[i][1] for i in indices]
    
    print(f"\nSelected {n_models} evenly spaced models:")
    for i, (model, data_value) in enumerate(zip(selected_models, selected_additional_data)):
        print(f"Model {i+1}: additional_data = {data_value}")
    
    return selected_models, selected_additional_data

def calculate_global_loss_bounds(loss_grids_list):
    """
    Calculate global min and max values across all loss landscapes for consistent scaling.
    
    Args:
        loss_grids_list (list): List of 2D arrays containing loss values from different landscapes
        
    Returns:
        tuple: (global_vmin_smallest, global_vmin_largest, global_vmax)
            - global_vmin_smallest: Smallest non-zero minimum across all landscapes
            - global_vmin_largest: Largest non-zero minimum across all landscapes  
            - global_vmax: Maximum value across all landscapes
    """
    global_vmin_smallest = float('inf')
    global_vmin_largest = float('-inf')
    global_vmax = float('-inf')
    
    for loss_grid in loss_grids_list:
        # Find non-zero minimum in current landscape
        current_min = loss_grid[loss_grid > 0].min()
        current_max = loss_grid.max()
        
        # Update global bounds
        global_vmin_smallest = min(global_vmin_smallest, current_min)
        global_vmin_largest = max(global_vmin_largest, current_min)
        global_vmax = max(global_vmax, current_max)
    
    print(f"Global min across selected landscapes (smallest value): {global_vmin_smallest:.6f}")
    print(f"Global min across selected landscapes (largest value): {global_vmin_largest:.6f}")
    print(f"Global max across ALL landscapes: {global_vmax:.6f}")
    
    return global_vmin_smallest, global_vmin_largest, global_vmax