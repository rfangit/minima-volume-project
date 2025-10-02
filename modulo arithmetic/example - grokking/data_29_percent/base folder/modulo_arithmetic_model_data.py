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

import random

def generate_modulo_addition_table(n):
    """
    Generates a table for modular addition
    """
    table = np.zeros((n, n), dtype=int)
    for a in range(n):
        for b in range(n):
            table[a, b] = (a + b) % n
    return table

def generate_dataset_from_table(table):
    """
    Turns a modular addition table into a list of triplets [a, b, a + b mod n ]
    """
    c = table.shape[0]
    dataset = []
    for a in range(c):
        for b in range(c):
            result = table[a, b]
            dataset.append([a, b, result])
    return dataset

def one_hot_encode_triplet(triplet, n, noise_scale=0.00):
    """
    One-hot encodes a triplet [a, b, output] into input and target tensors.

    Args:
        triplet: list or tuple of [a, b, (a + b) % c]
        c: modulus, determines one-hot vector size
        noise_scale: standard deviation of added Gaussian noise

    Returns:
        input_tensor: shape (2, n), one-hot encodings of a and b (with noise)
        target_tensor: shape (n,), one-hot encoding of output (with noise)
    """
    a, b, out = triplet

    # One-hot vectors
    a_vec = torch.zeros(n)
    b_vec = torch.zeros(n)
    out_vec = torch.zeros(n)

    a_vec[a] = 1.0
    b_vec[b] = 1.0
    out_vec[out] = 1.0

    # Add small Gaussian noise
    if noise_scale > 0:
        a_vec += noise_scale * torch.randn(n)
        b_vec += noise_scale * torch.randn(n)
        out_vec += noise_scale * torch.randn(n)

    # Optional: clip negatives (optional, depends on whether you want pure noise)
    # a_vec = torch.clip(a_vec, 0, None)
    # b_vec = torch.clip(b_vec, 0, None)
    # out_vec = torch.clip(out_vec, 0, None)

    # Stack inputs
    input_tensor = torch.stack([a_vec, b_vec])  # Shape (2, c)
    target_tensor = out_vec  # Shape (c,)

    return input_tensor, target_tensor

def encode_full_dataset(dataset, n):
    """
    Encodes a dataset of modulo addition triplets into one-hot encoded tensors.
    Flattens the input vectors [a, b] into a tensor of dim [dim(a) + dim(b)]
    """
    encoded_inputs = []
    encoded_targets = []
    for sample in dataset:
        input_tensor, target_tensor = one_hot_encode_triplet(sample, n)
        flattened_input = input_tensor.flatten() 
        encoded_inputs.append(flattened_input)
        encoded_targets.append(target_tensor)
    # Convert lists of tensors into single batched tensors
    stacked_inputs = torch.stack(encoded_inputs)  # Shape: [batch_size, 2*n]
    stacked_targets = torch.stack(encoded_targets)  # Shape: [batch_size, n]
    
    return stacked_inputs, stacked_targets

def get_dataset(modulus, device):
    """
    Returns the full modulo addition dataset as one-hot encoded PyTorch tensors.

    Args:
        modulus (int): The modulus for addition table.
        device (torch.device): The device to put tensors on.
        noise_scale (float): Gaussian noise to add to one-hot vectors.

    Returns:
        x_base: tensor of shape [modulus**2, 2*modulus]
        y_base: tensor of shape [modulus**2, modulus]
        x_test: same as x_base (full dataset)
        y_test: same as y_base (full dataset)
    """
    # Generate the addition table and dataset
    table = generate_modulo_addition_table(modulus)
    dataset = generate_dataset_from_table(table)

    # Encode entire dataset
    x_encoded, y_encoded = encode_full_dataset(dataset, modulus)

    # Optionally move to device
    x_encoded = x_encoded.to(device)
    y_encoded = y_encoded.to(device)

    # Return everything as both "train" and "test" (entire dataset)
    return x_encoded, y_encoded, x_encoded, y_encoded

# --------------------------
# Model Definition
# --------------------------
class StandardMLP(nn.Module):
    def __init__(self, N, hidden_dims, seed=None, weight_scale=None):
        """
        Args:
            N: Output dimension
            hidden_dims: List of hidden layer dimensions (e.g., [M] or [M1, M2, M3])
            seed: Random seed for reproducibility
            weight_scale: Multiplicative factor for all weights
        """
        super().__init__()
        
        # Set seed if specified
        if seed is not None:
            torch.manual_seed(seed)
        
        # Create layer dimensions [2N -> hidden_dims[0] -> ... -> N]
        dims = [2*N] + hidden_dims + [N]
        self.layers = nn.ModuleList()
        
        # Build hidden layers
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.activation = nn.ReLU()
        
        # Apply weight scaling if specified
        if weight_scale is not None:
            self._scale_weights(weight_scale)

    def _scale_weights(self, scale_factor):
        """Multiply all weights by a scale factor"""
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.data.mul_(scale_factor)
                if layer.bias is not None:
                    layer.bias.data.mul_(scale_factor)

    def forward(self, x):
        # Pass through all layers except last
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        # Final layer (no activation)
        return self.layers[-1](x)

def get_model(N=97, hidden_dims=None, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = StandardMLP(N=N, hidden_dims=hidden_dims, seed=seed)
    return model.to(device)
    
# --------------------------
# Loss + Metrics
# --------------------------

# Create the loss instance once.
_criterion = nn.MSELoss()
_criterion_sample = nn.MSELoss(reduction='none')

def loss_fn(logits, labels):
    return _criterion(logits, labels)

def loss_fn_per_sample(logits, labels):
    # MSE per element, then sum over classes for one scalar per sample
    per_element_loss = _criterion_sample(logits, labels)  # shape [batch_size, num_classes]
    per_sample_loss = per_element_loss.mean(dim=1)        # or sum(dim=1)
    return per_sample_loss  # shape [batch_size]

def get_loss_fn():
    return loss_fn

def get_loss_fn_per_sample():
    return loss_fn_per_sample

def accuracy_fn(logits, labels):
    with torch.no_grad():  # Disable gradient tracking
        preds = logits.argmax(dim=1)
        labels = labels.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
    return accuracy

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
    return None

