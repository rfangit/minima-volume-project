"""
Model + dataset definition for the Swiss Roll problem.
Contains all Swiss-specific components that can be swapped out by other models.
"""

# === Imports ===


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Less Important Imports (For non-main functionality
import time
import glob
from pathlib import Path

# MNIST SPECIFIC
import torch.nn.functional as F
from torchvision import datasets, transforms

# Import config
from .config import DATA_DIRS

# --------------------------
# Data Generation
# --------------------------

def get_dataset(device):
    data_dir = DATA_DIRS["MNIST"]
    
    # Define transform (convert to tensor + normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load dataset (not a tensor yet, just a Dataset object)
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(data_dir, train=False, transform=transform)
    
    # Convert entire dataset into big tensors
    train_images = torch.stack([img for img, _ in train_dataset])   # shape: [60000, 1, 28, 28]
    train_labels = torch.tensor([label for _, label in train_dataset])  # shape: [60000]
    
    test_images  = torch.stack([img for img, _ in test_dataset])    # shape: [10000, 1, 28, 28]
    test_labels  = torch.tensor([label for _, label in test_dataset])   # shape: [10000]

    x_base = train_images.to(device)
    y_base = train_labels.to(device)
    x_test = test_images.to(device)
    y_test = test_labels.to(device)

    return x_base, y_base, x_test, y_test

# --------------------------
# Model Definition
# --------------------------
class MNIST_MLP(nn.Module):
    def __init__(self, hidden_dims=None, seed=None):
        super(MNIST_MLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]  # default hidden layer sizes

        self.hidden_dims = hidden_dims

        if seed is not None:
            self.seed = seed
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self._initialize_layers()
        else:
            self._initialize_layers()

    def _initialize_layers(self):
        layers = []
        input_dim = 28 * 28
        for h in self.hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 10))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.model(x)

def get_model(hidden_dims=None, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = MNIST_MLP(hidden_dims=hidden_dims, seed=seed)
    return model.to(device)
    
# --------------------------
# Loss + Metrics
# --------------------------

# Create the loss instance once.
_criterion = nn.CrossEntropyLoss()
_criterion_sample = nn.CrossEntropyLoss(reduction='none')

def loss_fn(logits, labels):
    return _criterion(logits, labels)

def loss_fn_per_sample(logits, labels):
    return _criterion_sample(logits, labels)

def get_loss_fn():
    return loss_fn

def get_loss_fn_per_sample():
    return loss_fn_per_sample

def accuracy_fn(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()

def get_additional_metrics():
    """
    Return dictionary of metric functions.
    Keys are names, values are callables: fn(logits, labels).
    """
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
    """
    Verify MNIST models trained on different dataset sizes.

    For each model, displays 5 images from:
    - Base training set
    - Additional data
    - Test set

    Shows true labels and model predictions.
    """
    device = next(all_models[0]['model'].parameters()).device
    num_samples = 5  # number of images to display from each set

    for additional_data, model_data in zip(dataset_quantities, all_models):
        model = model_data['model']
        model.eval()

        # Construct datasets for this model
        x_train = torch.cat([x_base_train, x_additional[:additional_data]], dim=0)
        y_train = torch.cat([y_base_train, y_additional[:additional_data]], dim=0)

        # Sample images
        indices_base = torch.randperm(len(x_base_train))[:num_samples]
        indices_add = torch.randperm(additional_data)[:num_samples] if additional_data > 0 else []
        indices_test = torch.randperm(len(x_test))[:num_samples]

        samples = {
            "Base Train": (x_base_train[indices_base], y_base_train[indices_base]),
            f"Additional ({dataset_type})": (x_additional[indices_add], y_additional[indices_add]) if additional_data > 0 else None,
            "Test": (x_test[indices_test], y_test[indices_test])
        }

        for set_name, data in samples.items():
            if data is None:
                continue
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1)

            # Plot images in a row
            plt.figure(figsize=(12, 2))
            for i in range(len(imgs)):
                plt.subplot(1, num_samples, i+1)
                plt.imshow(imgs[i].cpu().squeeze(), cmap='gray')
                plt.title(f"T:{labels[i].item()}\nP:{preds[i].item()}")
                plt.axis('off')
            plt.suptitle(f"{set_name} Samples ({additional_data} extra)")
            plt.tight_layout()
            plt.show()

