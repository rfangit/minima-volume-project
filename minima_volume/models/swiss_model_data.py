"""
Model + dataset definition for the Swiss Roll problem.
Contains all Swiss-specific components that can be swapped out by other models.
"""

# === Imports ===


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Less Important Imports (For non-main functionality)
import time
import glob
from pathlib import Path
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation

# External Imports
from minima_volume.perturb_funcs import ModelPerturber

# --------------------------
# Data Generation
# --------------------------

def two_spirals(n_points, noise=0.5, seed=None):
    """Generate two interlocking spirals dataset."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    n_rotations = 1.7
    n = torch.sqrt(torch.rand(n_points, 1, generator=gen)) * n_rotations * (2 * np.pi)
    
    noise_tensor1 = torch.randn(n_points, 1, generator=gen) * noise
    noise_tensor2 = torch.randn(n_points, 1, generator=gen) * noise

    d1x = -1.5 * torch.cos(n) * n + noise_tensor1
    d1y = 1.5 * torch.sin(n) * n + noise_tensor2

    points = torch.cat((torch.cat((d1x, d1y), dim=1),
                        torch.cat((-d1x, -d1y), dim=1)), dim=0)
    labels = torch.cat((torch.zeros(n_points), torch.ones(n_points)), dim=0)

    return points, labels

def get_dataset(base_data_size, dataset_quantities, test_dataset_size, noise=0.3, extra_pts=1000, dataset_type="data", seed=0):
    """
    Generate base and test datasets in a standardized way.
    Returns:
        x_base, y_base, x_test, y_test
    """
    # Compute number of points needed
    if dataset_type in ["poison", "data"]:
        num_pts_generate = base_data_size + max(dataset_quantities) + extra_pts
    else:
        num_pts_generate = base_data_size #for noise, we don't need to generate as many points!

    # Base dataset
    x_base, y_base = two_spirals(n_points=num_pts_generate, noise=noise, seed=seed)
    # Test dataset
    x_test, y_test = two_spirals(n_points=test_dataset_size, noise=noise, seed=seed+1)
    
    return x_base, y_base, x_test, y_test

# --------------------------
# Model Definition
# --------------------------
class SwissMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
        
def get_model(input_dim=2, hidden_dims=[32]*5, output_dim=1, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = SwissMLP(input_dim, hidden_dims, output_dim)
    return model.to(device)
    
# --------------------------
# Loss + Metrics
# --------------------------

# Create the loss instance once.
_criterion = nn.BCEWithLogitsLoss()
_criterion_sample = nn.BCEWithLogitsLoss(reduction='none')


def loss_fn(logits, labels):
    return _criterion(logits.squeeze(), labels)

def loss_fn_per_sample(logits, labels):
    return _criterion_sample(logits.squeeze(), labels)

def get_loss_fn():
    return loss_fn

def get_loss_fn_per_sample():
    return loss_fn_per_sample

def accuracy_fn(logits, labels):
    preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
    return (preds == labels).sum().item() / len(labels)

def get_additional_metrics():
    """
    Return dictionary of metric functions.
    Keys are names, values are callables: fn(logits, labels).
    """
    #the accuracy key HAS to be accs for future code to work
    return {
        "accs": accuracy_fn 
    }

# --------------------------
# Summary Analysis
# --------------------------
# If this model doesn't have a special visualization function, leave this as return none

def verify_model_results(
    all_models,
    x_base_train, y_base_train,
    x_additional, y_additional,
    x_test, y_test,
    dataset_quantities,
    dataset_type
):
    """Rebuild datasets on the fly for visualization, matching training setup"""

    colors = plt.cm.tab10.colors
    plt.figure(figsize=(14, 5))

    # 1. Training curves comparison
    # -----------------------------
    # Loss
    plt.subplot(1, 2, 1)
    for i, model_data in enumerate(all_models):
        label_suffix = f"{model_data['additional_data']} {dataset_type}"
        plt.plot(model_data['train_loss'], 
                 color=colors[i], 
                 label=f"Train ({label_suffix})")
        plt.plot(model_data['test_loss'], 
                 '--', color=colors[i],
                 label=f"Test ({label_suffix})")
    plt.title('Training History - Loss', pad=20)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Accuracy
    plt.subplot(1, 2, 2)
    for i, model_data in enumerate(all_models):
        label_suffix = f"{model_data['additional_data']} {dataset_type}"
        plt.plot(model_data['train_accs'], 
                 color=colors[i],
                 label=f"Train ({label_suffix})")
        plt.plot(model_data['test_accs'], 
                 '--', color=colors[i],
                 label=f"Test ({label_suffix})")
    plt.title('Training History - Accuracy', pad=20)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()

    # 2. Individual decision boundaries
    # ---------------------------------
    for i, (additional_data, model_data) in enumerate(zip(dataset_quantities, all_models)):
        model = model_data['model']
        device = next(model.parameters()).device

        # Reconstruct dataset for this quantity
        x_train = torch.cat([x_base_train, x_additional[:additional_data]], dim=0)
        y_train = torch.cat([y_base_train, y_additional[:additional_data]], dim=0)

        # Build plotting grid
        all_x = torch.cat([x_train.cpu(), x_test.cpu()])
        x_min, x_max = all_x[:,0].min()-1, all_x[:,0].max()+1
        y_min, y_max = all_x[:,1].min()-1, all_x[:,1].max()+1

        xx, yy = torch.meshgrid(
            torch.linspace(x_min, x_max, 500),
            torch.linspace(y_min, y_max, 500),
            indexing='ij'
        )
        grid = torch.stack([xx.ravel(), yy.ravel()], dim=1).to(device)

        with torch.no_grad():
            logits = model(grid)
            probs = torch.sigmoid(logits).reshape(xx.shape).cpu()

        # Plot
        plt.figure(figsize=(6, 6), constrained_layout=True)
        plt.contourf(xx.numpy(), yy.numpy(), probs.numpy(), levels=50, cmap='bwr', alpha=0.6)
        
        # Clean + additional training data
        plt.scatter(
            x_base_train[:, 0].cpu(), x_base_train[:, 1].cpu(),
            c=y_base_train.cpu(), cmap='bwr', edgecolor='k',
            s=60, alpha=0.9, label='Base Train'
        )
        
        if additional_data > 0:
            # Determine label based on dataset_type
            if dataset_type == 'poison':
                add_label = 'Poison'
            else:
                add_label = f'Extra ({dataset_type}, {additional_data})'
        
            plt.scatter(
                x_additional[:additional_data, 0].cpu(),
                x_additional[:additional_data, 1].cpu(),
                marker='x', color='black', s=120,
                linewidths=3, alpha=1,
                label=add_label
            )

        # Title / annotation
        final_test_acc = model_data['test_accs'][-1]
        plt.title(
            f"Base Dataset: {len(x_base_train)} Examples\n"
            f"Additional {dataset_type.capitalize()}: {additional_data} Examples\n"
            f"Final Test Accuracy: {final_test_acc:.2%}",
            pad=20,
            fontsize=15
        )
        plt.xlabel("Feature 1", fontsize = 20)
        plt.ylabel("Feature 2", fontsize = 20)

        # Custom ticks
        plt.xticks([-10, 0, 10], fontsize=15)
        plt.yticks([-10, 0, 10], fontsize=15)

        plt.legend(loc='upper left', fontsize = 15)

        # Add annotation about proportion of additional data
        #if additional_data > 0:
        #    extra_percent = 100 * additional_data / len(x_train)
        #    plt.annotate(f"{extra_percent:.1f}% {dataset_type}",
        #                 xy=(0.05, 0.05), xycoords='axes fraction',
        #                 bbox=dict(boxstyle='round', fc='white', ec='black'))

        plt.tight_layout()
        plt.show()


####################################################
## UNUSED CODE 
####################################################
# Not used in the current main pipeline, but useful to remember exists
# for the swiss roll

def plot_decision_boundary(model, x, y, title="Decision Boundary", save_path=None):
    """
    Plot the decision boundary of a binary classifier on 2D data.
    
    Args:
        model: Trained PyTorch model
        x: Input features tensor of shape [N, 2]
        y: Target labels tensor of shape [N]
        title: Plot title
        save_path: If provided, saves plot to this path instead of displaying
    """
    device = next(model.parameters()).device
    x = x.to(device)
    
    x_min, x_max = x[:,0].min()-1, x[:,0].max()+1
    y_min, y_max = x[:,1].min()-1, x[:,1].max()+1
    
    xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 300),
                           torch.linspace(y_min, y_max, 300), indexing='ij')
    grid = torch.stack([xx.ravel(), yy.ravel()], dim=1).to(device)
    
    with torch.no_grad():
        logits = model(grid)
        probs = torch.sigmoid(logits).reshape(xx.shape).cpu()
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx.numpy(), yy.numpy(), probs.numpy(), levels=50, cmap='bwr', alpha=0.6)
    plt.scatter(x[:,0].cpu(), x[:,1].cpu(), c=y.cpu(), cmap='bwr', edgecolor='k')
    plt.colorbar(label='Class Probability')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
    else:
        plt.show()

def create_perturbation_animation(image_folder='imgs/swiss/decision_gif_1', gif_name='perturbation_animation.gif', save=True):
    """
    Creates an animation from a sequence of PNGs and returns a FuncAnimation for interactive display.
    
    Args:
        image_folder: Folder containing PNG files.
        gif_name: Name to save the gif as, in the same folder.
        save: Whether to save the gif as a file.
        
    Returns:
        ani: matplotlib.animation.FuncAnimation object.
    """
    image_folder = Path(image_folder)
    image_files = sorted(
        glob.glob(str(image_folder / "swiss_perturb_*.png")),
        key=lambda x: int(Path(x).stem.split('_')[-1])
    )
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_folder} matching pattern 'swiss_perturb_1_*.png'")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    img_obj = ax.imshow(mpimg.imread(image_files[0]))
    ax.axis('off')
    ax.set_title("Perturbation Animation")

    def update(frame):
        img_obj.set_data(mpimg.imread(image_files[frame]))
        ax.set_title(f"Frame {frame}/{len(image_files)-1}")
        return [img_obj]

    ani = FuncAnimation(fig, update, frames=len(image_files), interval=200, blit=True)
    plt.close()

    if save:
        start_time = time.time()
        gif_path = image_folder / gif_name
        ani.save(gif_path, writer='pillow', fps=5)
        print(f"GIF saved to {gif_path} in {time.time() - start_time:.2f} seconds")

    return ani
    
# --------------------------
# Model Analysis Functions
# --------------------------

def wiggle_swiss(model, points, labels, loss_fn, perturbation_vector, coefficients, 
                 folder="imgs/swiss/decision_gif", plot=True, save_results=True, seed = None):
    """
    Evaluates model sensitivity by applying scaled perturbations and tracking loss/decision boundaries.
    
    Performs a 1D sweep through perturbation space, generating:
    - Loss curves across perturbation scales
    - Optional visualization of decision boundary evolution, for a 2D problem
    
    Args:
        model: Trained PyTorch model to analyze
        points: Data, input tensor (shape [N, D]), where D = 2 for our 2D decision problem
        labels: Target labels tensor (shape [N])
        loss_fn: Differentiable loss function (e.g., BCEWithLogitsLoss)
        perturbation_vector: Dictionary {param_name: tensor} contaiing the given perturbations effects on each model parameter
        coefficients: series of coefficients to multiply the perturbation
        folder: Output directory for saved results
        plot: Whether to generate decision boundary plots (default: True)
        save_results: Whether to save results to disk (default: True)
        
    Returns:
        Dictionary containing:
        - 'losses': array of losses for each perturbation
        - 'coefficients': array of coefficients used (same as input)
        - 'perturbations': list of perturbation directions used
        
    File Outputs (when save_results=True):
        - loss_curve.npz: Contains C_values and losses
        - decision_gif/*.png: Sequence of decision boundary plots (if plot=True)

    """
    folder = Path(folder)
    if seed is not None:
        folder = Path(str(folder) + f"_{seed}")
    if save_results:
        folder.mkdir(parents=True, exist_ok=True)

    perturber = ModelPerturber(model)
    losses = []
    
    for i, coeff in enumerate(coefficients):
        perturbation = {
            name: perturbation_vector[name] * coeff
            for name in perturbation_vector
        }

        perturber.apply_perturbation(perturbation)

        # Compute loss
        with torch.no_grad():
            logits = model(points)
            loss = loss_fn(logits.squeeze(), labels.float())
            losses.append(loss.item())

        # Optionally plot and save decision boundary
        if plot and save_results:
            plot_filename = folder / f"swiss_perturb_{i:03d}.png"
            if seed is not None:
                plot_filename = folder / f"swiss_perturb_{i:03d}.png"
            plot_decision_boundary(
                model, points, labels,
                title=f"Perturbation Scale: {coeff:.3f}",
                save_path=plot_filename
            )
            
        perturber.reset()

    # Save results if requested
    results = {        
                'losses': np.array(losses),
                'coefficients': coefficients,
                'perturbation_vector': perturbation_vector,
    }

    if save_results:
        torch.save(results, folder / "perturbation_results.pt")

    return results
