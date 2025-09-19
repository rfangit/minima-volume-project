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

# --------------------------
# Data Generation
# --------------------------

def get_dataset(device):
    data_dir = r"L:\Programming\ARC\minima_volume_project\MNIST"
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
class FlexibleMNIST_CNN(nn.Module):
    def __init__(
        self,
        conv_channels=None,  # e.g. [32, 64]
        kernel_size=3,
        fc_dims=None,       # e.g. [128]
        seed=None
    ):
        super(FlexibleMNIST_CNN, self).__init__()
        
        self.conv_channels = conv_channels or [32, 64]
        self.kernel_size = kernel_size
        self.fc_dims = fc_dims or [128]

        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self._initialize_layers()
        else:
            self._initialize_layers()

    def _initialize_layers(self):
        layers = []
        in_channels = 1  # MNIST is grayscale
        
        # Build convolutional layers
        for out_channels in self.conv_channels:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, 2)

        # Determine flattened size after conv + pooling
        dummy_input = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            conv_out = self.pool(self.conv_layers(dummy_input))
            flat_dim = conv_out.view(1, -1).shape[1]

        # Build fully connected layers
        fc_layers = []
        input_dim = flat_dim
        for h in self.fc_dims:
            fc_layers.append(nn.Linear(input_dim, h))
            fc_layers.append(nn.ReLU())
            input_dim = h
        fc_layers.append(nn.Linear(input_dim, 10))  # Output layer
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

def get_model(conv_channels=None, fc_dims=None, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = FlexibleMNIST_CNN(
        conv_channels=conv_channels,
        fc_dims=fc_dims,
        seed=seed
    )
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

