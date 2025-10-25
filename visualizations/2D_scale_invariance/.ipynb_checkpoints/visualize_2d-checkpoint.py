import matplotlib.pyplot as plt
import numpy as np

def plot_parameter_space_with_arrows(loss_fn, model, perturbation_directions, r_values, 
                                    x_min=-5, x_max=5, y_min=-5, y_max=5, 
                                    fig_size=(8, 8), filename=None):
    """
    Plots the 2D parameter space with minima regions, model position, and perturbation arrows.
    
    Args:
        loss_fn: Loss function object that contains minima parameters
        model: The model to plot (must have get_parameter_values() method)
        perturbation_directions: List of perturbation direction dictionaries
        r_values: List of radii for each perturbation direction
        x_min, x_max: X-axis plot limits
        y_min, y_max: Y-axis plot limits
        fig_size: Figure size
        filename: If provided, saves plot to this filename
    """
    _setup_base_plot(loss_fn, x_min, x_max, y_min, y_max, fig_size)
    
    # Plot model position
    model_params = model.get_parameter_values()
    plt.plot(model_params[0], model_params[1], 'go', markersize=10, label="Model position")

    # Plot perturbation directions as arrows
    for i, perturb in enumerate(perturbation_directions):
        if i < len(r_values):  # Ensure we have a radius for this direction
            direction = np.array([
                perturb['params.0'].item() * r_values[i],
                perturb['params.1'].item() * r_values[i]
            ])
            
            plt.arrow(model_params[0], model_params[1], 
                     direction[0], direction[1],
                     head_width=0.1, head_length=0.15, 
                     fc='purple', ec='purple', alpha=0.7)

    _finalize_plot(filename)

def plot_parameter_space_with_polygon(loss_fn, model, perturbation_directions, r_values,
                                     x_min=-5, x_max=5, y_min=-5, y_max=5,
                                     fig_size=(8, 8), filename=None):
    """
    Plots the 2D parameter space with minima regions, model position, and perturbation polygon.
    
    Args:
        loss_fn: Loss function object that contains minima parameters
        model: The model to plot (must have get_parameter_values() method)
        perturbation_directions: List of perturbation direction dictionaries
        r_values: List of radii for each perturbation direction
        x_min, x_max: X-axis plot limits
        y_min, y_max: Y-axis plot limits
        fig_size: Figure size
        filename: If provided, saves plot to this filename
    """
    _setup_base_plot(loss_fn, x_min, x_max, y_min, y_max, fig_size)
    
    # Plot model position
    origin = model.get_parameter_values()
    plt.plot(origin[0], origin[1], 'go', markersize=10, label="Model position")

    # Calculate endpoints for polygon
    endpoints = []
    for i, perturb in enumerate(perturbation_directions):
        if i < len(r_values):  # Ensure we have a radius for this direction
            direction = np.array([
                perturb['params.0'].item(),
                perturb['params.1'].item()
            ])
            endpoint = origin + direction * r_values[i]
            endpoints.append(endpoint)
    
    if endpoints:
        endpoints = np.array(endpoints)
        # Plot the polygon
        plt.fill(endpoints[:, 0], endpoints[:, 1], color='blue', alpha=0.2, label="Perturbation region")
        plt.plot(endpoints[:, 0], endpoints[:, 1], 'b-', alpha=0.5)

    _finalize_plot(filename)

def _setup_base_plot(loss_fn, x_min, x_max, y_min, y_max, fig_size):
    """Helper function to set up the base plot with minima regions"""
    params = loss_fn.get_minima_parameters()

    # Parameters for the red curve (a1, w1)
    a1 = params['minima_wide_loc']
    w1 = params['wide_width']

    print ("Wide loc ", a1)

    # Parameters for the blue curve (a2, w2)
    a2 = params['minima_sharp_loc']
    w2 = params['sharp_width']

    # Create x values (avoid x=0)
    x_pos = np.linspace(1e-10, x_max, 500)  # Positive x
    x_neg = np.linspace(x_min, -1e-10, 500)  # Negative x

    # Calculate bounds for red curve (a1, w1)
    red_upper_pos = (a1 + w1)/x_pos
    red_lower_pos = (a1 - w1)/x_pos
    red_upper_neg = (a1 + w1)/x_neg
    red_lower_neg = (a1 - w1)/x_neg

    # Calculate bounds for blue curve (a2, w2)
    blue_upper_pos = (a2 + w2)/x_pos
    blue_lower_pos = (a2 - w2)/x_pos
    blue_upper_neg = (a2 + w2)/x_neg
    blue_lower_neg = (a2 - w2)/x_neg

    # Create the plot
    plt.figure(figsize=fig_size)

    # Fill the red region (a1 ± 2w1)
    plt.fill_between(x_pos, red_lower_pos, red_upper_pos, color='red', alpha=0.3, label=f'Wide minima: {a1:.2f} ± {w1:.2f}')
    plt.fill_between(x_neg, red_lower_neg, red_upper_neg, color='red', alpha=0.3)

    # Fill the blue region (a2 ± 2w2)
    plt.fill_between(x_pos, blue_lower_pos, blue_upper_pos, color='blue', alpha=0.3, label=f'Sharp minima: {a2:.2f} ± {w2:.2f}')
    plt.fill_between(x_neg, blue_lower_neg, blue_upper_neg, color='blue', alpha=0.3)

    # Plot the boundary lines for both regions
    plt.plot(x_pos, red_upper_pos, 'r-', linewidth=1, alpha=0.5)
    plt.plot(x_pos, red_lower_pos, 'r-', linewidth=1, alpha=0.5)
    plt.plot(x_neg, red_upper_neg, 'r-', linewidth=1, alpha=0.5)
    plt.plot(x_neg, red_lower_neg, 'r-', linewidth=1, alpha=0.5)

    plt.plot(x_pos, blue_upper_pos, 'b-', linewidth=1, alpha=0.5)
    plt.plot(x_pos, blue_lower_pos, 'b-', linewidth=1, alpha=0.5)
    plt.plot(x_neg, blue_upper_neg, 'b-', linewidth=1, alpha=0.5)
    plt.plot(x_neg, blue_lower_neg, 'b-', linewidth=1, alpha=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

def _finalize_plot(filename=None):
    """Helper function to finalize plot settings and save if needed"""
    # Plot settings
    plt.xlabel('Parameter 1', fontsize=14)
    plt.ylabel('Parameter 2', fontsize=14)
    plt.title('Parameter Space with Minima Regions', fontsize=16)
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    if filename is not None:
        plt.savefig(filename)