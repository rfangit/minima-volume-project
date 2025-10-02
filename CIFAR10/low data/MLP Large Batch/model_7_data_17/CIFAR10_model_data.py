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
    data_dir = r"L:\Programming\ARC\minima_volume_project\CIFAR10"
    
    # CIFAR-10 normalization values (mean, std for R, G, B channels)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),  # per-channel mean
            std=(0.2470, 0.2435, 0.2616)   # per-channel std
        )
    ])
    
    # Load dataset
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(data_dir, train=False, transform=transform)
    
    # Convert entire dataset into big tensors
    train_images = torch.stack([img for img, _ in train_dataset])   # shape: [50000, 3, 32, 32]
    train_labels = torch.tensor([label for _, label in train_dataset])  # shape: [50000]
    
    test_images  = torch.stack([img for img, _ in test_dataset])    # shape: [10000, 3, 32, 32]
    test_labels  = torch.tensor([label for _, label in test_dataset])   # shape: [10000]

    x_base = train_images.to(device)
    y_base = train_labels.to(device)
    x_test = test_images.to(device)
    y_test = test_labels.to(device)

    return x_base, y_base, x_test, y_test

# --------------------------
# Model Definition
# --------------------------

class CIFAR10_MLP(nn.Module):
    def __init__(self, hidden_dims=None, seed=None):
        super(CIFAR10_MLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]  # slightly larger default hidden layers for higher-dim input

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
        input_dim = 3 * 32 * 32  # <-- CHANGED for CIFAR-10 (RGB images)
        for h in self.hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 10))  # CIFAR-10 has 10 classes
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten to [batch, 3072]
        return self.model(x)

def get_model(hidden_dims=None, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = CIFAR10_MLP(hidden_dims=hidden_dims, seed=seed)
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
# CIFAR-10 normalization values
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
CIFAR_STD  = torch.tensor([0.2470, 0.2435, 0.2616])

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def verify_model_results(
    all_models,
    x_base_train, y_base_train,
    x_additional, y_additional,
    x_test, y_test,
    dataset_quantities,
    dataset_type
):
    """
    Verify models trained on CIFAR-10 data.

    Displays 5 images from each dataset for each model,
    with true labels and model predictions shown as class names.
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
                img = imgs[i].cpu().clone()
                # Unnormalize
                img = img * CIFAR_STD[:, None, None] + CIFAR_MEAN[:, None, None]
                img = img.permute(1, 2, 0).numpy()  # [H,W,C]
                img = img.clip(0, 1)

                plt.subplot(1, num_samples, i+1)
                plt.imshow(img)

                true_class = CIFAR10_CLASSES[labels[i].item()]
                pred_class = CIFAR10_CLASSES[preds[i].item()]
                plt.title(f"T:{labels[i].item()} ({true_class})\nP:{preds[i].item()} ({pred_class})", fontsize=8)
                plt.axis('off')

            plt.suptitle(f"{set_name} Samples ({additional_data} extra)")
            plt.tight_layout()
            plt.show()