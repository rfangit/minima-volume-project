# Standard library
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode


"""
Advanced Visualization Tools for Loss Landscape Analysis

This module provides specialized visualization functions for:
- 2D contour plots of perturbation landscapes
- 3D surface plots of loss landscapes
- Interactive 3D visualizations using Plotly

Key Features:
- High-quality static visualizations for publications
- Interactive 3D plots for exploratory analysis
- Consistent styling across all plot types
- Flexible saving options for all visualizations
"""

# --------------------------
# 2D Visualization Functions
# --------------------------

def plot_perturbation_contour(loss_grid, C1, C2, title="Perturbation Landscape (Contour)", 
                              output_path=None, figsize=(8, 6), cmap='viridis', show=True):
    """
    Create a publication-quality contour plot of a loss landscape.
    
    Args:
        loss_grid: 2D array of loss values (numpy.ndarray)
        C1: Meshgrid coordinates for first perturbation direction (numpy.ndarray)
        C2: Meshgrid coordinates for second perturbation direction (numpy.ndarray)
        title: Plot title string (default: "Perturbation Landscape (Contour)")
        output_path: File path to save figure (str, optional)
        figsize: Figure dimensions in inches (tuple, default: (8, 6))
        cmap: Matplotlib colormap name (str, default: 'viridis')
        show: Whether to display the figure (bool, default: True)
    """
    fig = plt.figure(figsize=figsize)
    
    # Logarithmic contour levels
    levels = np.logspace(np.log10(loss_grid.min()), np.log10(loss_grid.max()), 20)
    
    # Filled contour plot
    contour = plt.contourf(C1, C2, loss_grid, levels=levels, norm=colors.LogNorm(), cmap=cmap)
    plt.contour(C1, C2, loss_grid, levels=levels, colors='k', linewidths=0.5, alpha=0.3)
    
    plt.colorbar(contour, label='Log Loss')
    plt.xlabel("Direction 1 Coefficient")
    plt.ylabel("Direction 2 Coefficient")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------
# 3D Visualization Functions
# --------------------------

def plot_perturbation_3d(loss_grid, C1, C2, title="Perturbation Landscape (3D)", 
                         output_path=None, figsize=(8, 6), cmap='viridis', elev=15, azim=-60, show=True):
    """
    Create a 3D surface plot of a loss landscape with contour projection.
    
    Args:
        loss_grid: 2D array of loss values (numpy.ndarray)
        C1: Meshgrid coordinates for first direction (numpy.ndarray)
        C2: Meshgrid coordinates for second direction (numpy.ndarray)
        title: Plot title string (default: "Perturbation Landscape (3D)")
        output_path: File path to save figure (str, optional)
        figsize: Figure dimensions in inches (tuple, default: (8, 6))
        cmap: Matplotlib colormap name (str, default: 'viridis')
        elev: Elevation angle in degrees (int, default: 15)
        azim: Azimuth angle in degrees (int, default: -60)
        show: Whether to display the figure (bool, default: True)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    log_loss = np.log10(loss_grid + 1e-10)
    surf = ax.plot_surface(C1, C2, log_loss, cmap=cmap, rstride=1, cstride=1,
                           edgecolor='none', linewidth=0, alpha=0.9, antialiased=True)
    
    levels = np.linspace(log_loss.min(), log_loss.max(), 12)
    ax.contour(C1, C2, log_loss, zdir='z', levels=levels, offset=log_loss.min()-0.1,
               colors='k', alpha=0.4, linewidths=0.8)
    
    ax.set_xlabel("Direction 1 Coefficient")
    ax.set_ylabel("Direction 2 Coefficient")
    ax.set_zlabel("Log10(Loss)")
    ax.set_title(title)
    
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    
    cbar = fig.colorbar(surf, ax=ax)
    cbar.set_label('Loss (log scale)')
    
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)

# --------------------------
# Interactive Visualization
# --------------------------

def plot_interactive_3d_surface(loss_grid, C1, C2, title="Interactive Loss Landscape"):
    """
    Create an interactive 3D visualization of loss landscape using Plotly.
    
    Args:
        loss_grid: 2D array of loss values (numpy.ndarray)
        C1: Meshgrid coordinates for first direction (numpy.ndarray)
        C2: Meshgrid coordinates for second direction (numpy.ndarray)
        title: Plot title string (default: "Interactive Loss Landscape")
        
    Returns:
        plotly.graph_objects.Figure: Interactive figure object
        
    Note:
        Requires Plotly installed. Best viewed in Jupyter notebooks.
    """
    # Initialize notebook mode if in Jupyter
    init_notebook_mode(connected=True)
    
    # Create interactive surface plot
    fig = go.Figure(data=[
        go.Surface(
            x=C1,
            y=C2,
            z=loss_grid,
            colorscale='Viridis',
            contours_z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project_z=True
            )
        )
    ])
    
    # Configure layout for optimal viewing
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Direction 1',
            yaxis_title='Direction 2',
            zaxis_title='Loss (log scale)',
            zaxis_type='log',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)  # Initial viewing angle
            )
        ),
        width=800,
        height=600,
        autosize=False,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    return fig