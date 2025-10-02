# Standard library
import copy
import glob
import os

# Third-party
import numpy as np
import torch

"""
Dataset and Model Utilities

This module provides utility functions for dataset manipulation, model saving/loading, and generating noisy/poisoned datasets for machine learning experiments.

## Available Functions

### Dataset Preparation Functions:
- `get_base_indices`: Select deterministic base indices for dataset sampling
- `get_additional_indices`: Get additional indices that don't overlap with excluded indices
- `swap_labels_to_incorrect`: Modify labels to be incorrect (for poisoning)
- `generate_noisy_dataset`: Create noisy data with similar statistics as base dataset
- `prepare_datasets`: Main function to prepare training and additional datasets

### Save/Load Functions:
- `save_dataset`: Save dataset components and metadata to file
- `save_model`: Save model state and metadata
- `load_dataset`: Load previously saved dataset
- `load_model`: Load model into an initialized instance

These utilities are particularly useful for:
- Creating controlled experiments with clean/poisoned/noisy data
- Managing dataset splits with reproducibility
- Saving and loading experiment artifacts
- Working with both NumPy and PyTorch data structures

"""

def get_base_indices(x_base, y_base, n_samples, random_seed=42):
    """
    Get deterministic base indices - always returns the same result for same parameters. Works with numpy or pytorch.
    """
    np.random.seed(random_seed)
    total_samples = len(x_base)
    if n_samples > total_samples:
        raise ValueError(f"n_samples ({n_samples}) cannot exceed total samples ({total_samples})")
    return np.random.choice(total_samples, size=n_samples, replace=False)

def get_additional_indices(x_base, y_base, excluded_indices, n_additional, random_seed=42):
    """
    Get additional indices that do not overlap with the excluded indices. Works with numpy or pytorch.
    """
    np.random.seed(random_seed)
    total_samples = len(x_base)
    
    # Remove excluded indices
    all_indices = np.arange(total_samples)
    available_indices = np.setdiff1d(all_indices, excluded_indices)
    
    if n_additional > len(available_indices):
        raise ValueError(f"n_additional ({n_additional}) cannot exceed available samples ({len(available_indices)})")

    return np.random.choice(available_indices, size=n_additional, replace=False)

def swap_labels_to_incorrect(y_base, y_additional, random_seed=42):
    """
    Swap all labels in y_additional to make them incorrect.
    
    Parameters:
    y_base: original data tensor (used to find all labels)
    y_additional: tensor of labels to be modified
    random_seed: optional seed for reproducibility
    
    Returns:
    y_additional_swapped: tensor with all incorrect labels
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # All possible classes
    unique_labels = torch.unique(y_base)
    num_classes = len(unique_labels)
    
    if num_classes < 2:
        raise ValueError("Need at least 2 classes for label swapping")
    
    # Map labels to indices [0..num_classes-1] for easy math
    label_to_index = {label.item(): i for i, label in enumerate(unique_labels)}
    indices = torch.tensor([label_to_index[l.item()] for l in y_additional], device=y_additional.device)
    
    # Pick a random offset in [1, num_classes-1] for each label
    random_offsets = torch.randint(1, num_classes, (len(y_additional),), device=y_additional.device)
    
    # Compute swapped indices (ensures different label)
    swapped_indices = (indices + random_offsets) % num_classes
    
    # Map back to label values
    y_swapped = unique_labels[swapped_indices]
    return y_swapped

def generate_noisy_dataset(x_base, y_base, num_samples=None, random_seed=42):
    """
    Generate a noisy dataset with the same shape, mean, and variance as x_base.
    Labels are randomly generated from the set of y_base labels.
    
    Parameters:
    x_base: torch.Tensor, shape [N, ...]
        Original training inputs
    y_base: torch.Tensor, shape [N]
        Original training labels
    num_samples: int, optional
        Number of noisy samples to generate (default = len(x_base))
    random_seed: int, optional
        Random seed for reproducibility
        
    Returns:
    X_noisy: torch.Tensor
        Generated noisy inputs
    y_noisy: torch.Tensor
        Random labels
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    if num_samples is None:
        num_samples = len(x_base)
    
    # Compute per-dataset mean and std
    mean = x_base.mean()
    std = x_base.std()
    
    # Generate Gaussian noise with same mean + variance
    X_noisy = torch.randn((num_samples,) + x_base.shape[1:], device=x_base.device) * std + mean
    
    # Unique classes from y_train
    classes = torch.unique(y_base)
    num_classes = len(classes)
    
    # Random labels
    rand_indices = torch.randint(0, num_classes, (num_samples,), device=y_base.device)
    y_noisy = classes[rand_indices]
    
    return X_noisy, y_noisy

def prepare_datasets(x_base, y_base, dataset_type, dataset_quantities,
                     base_data_size, data_seed=42, seed_1 = None, seed_2 = None):
    """
    Prepare training and additional datasets depending on dataset_type.

    Parameters
    ----------
    x_base : torch.Tensor
        Full dataset features (shape: [N, ...])
    y_base : torch.Tensor
        Full dataset labels (shape: [N])
    dataset_type : str
        One of {"data", "poison", "noise"}
    dataset_quantities : list[int]
        List of dataset sizes, max determines how many additional samples to generate
    base_data_size : int
        Number of base samples to select for training
    data_seed : int, optional
        Random seed for reproducibility
    seed_1, seed_2: random seeds for swapping labels and getting additional indices

    Returns
    -------
    x_base_train : torch.Tensor
        Base training inputs
    y_base_train : torch.Tensor
        Base training labels
    x_additional : torch.Tensor
        Additional inputs (depending on dataset_type)
    y_additional : torch.Tensor
        Additional labels (depending on dataset_type)
    """
    # ----------------------------
    # 1. Select base training set
    # ----------------------------
    base_indices = get_base_indices(x_base=x_base, y_base=y_base,
                                    n_samples=base_data_size,
                                    random_seed=data_seed)
    x_base_train = x_base[base_indices]
    y_base_train = y_base[base_indices]

    # ----------------------------
    # 2. Prepare additional dataset
    # ----------------------------
    max_additional_indices = max(dataset_quantities)

    seed_1 = seed_1 or data_seed
    seed_2 = seed_2 or data_seed
        
    if dataset_type == "data":
        additional_indices = get_additional_indices(x_base=x_base, y_base=y_base,
                                                    excluded_indices=base_indices,
                                                    n_additional=max_additional_indices,
                                                    random_seed=seed_1)
        x_additional = x_base[additional_indices]
        y_additional = y_base[additional_indices]

    elif dataset_type == "poison":
        additional_indices = get_additional_indices(x_base=x_base, y_base=y_base,
                                                    excluded_indices=base_indices,
                                                    n_additional=max_additional_indices,
                                                    random_seed=seed_1)
        x_additional = x_base[additional_indices]
        y_additional_unpoisoned = y_base[additional_indices]
        y_additional = swap_labels_to_incorrect(
            y_base=y_base,
            y_additional=y_additional_unpoisoned,
            random_seed=data_seed
        )

    elif dataset_type == "noise":
        x_additional, y_additional = generate_noisy_dataset(
            x_base=x_base,
            y_base=y_base,
            num_samples=max_additional_indices,
            random_seed=seed_1
        )

    else:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'")

    return x_base_train, y_base_train, x_additional, y_additional


# -----------------------------
# Generating Imbalanced Classes
# -----------------------------

def _select_indices_with_class_imbalance(y_data, available_indices, n_samples, target_labels, target_ratio, random_seed):
    """
    Internal helper function to select indices with class imbalance.
    
    Parameters:
    y_data: labels for the entire dataset
    available_indices: indices available for selection
    n_samples: number of samples to select
    target_labels: list of target labels to prioritize
    target_ratio: proportion of samples from target labels
    random_seed: random seed for reproducibility
    
    Returns:
    selected_indices: indices of selected samples
    """
    np.random.seed(random_seed)
    
    # Convert to numpy arrays if they're torch tensors
    if torch.is_tensor(y_data):
        y_data_np = y_data.cpu().numpy()
    else:
        y_data_np = y_data
    
    if torch.is_tensor(target_labels):
        target_labels_np = target_labels.cpu().numpy()
    else:
        target_labels_np = np.array(target_labels)
    
    # Separate available indices into target and non-target
    target_mask = np.isin(y_data_np[available_indices], target_labels_np)
    target_available_indices = available_indices[target_mask]
    non_target_available_indices = available_indices[~target_mask]
    
    # Calculate how many samples to take from each group
    n_target = min(int(n_samples * target_ratio), len(target_available_indices))
    n_non_target = n_samples - n_target
    
    if n_non_target > len(non_target_available_indices):
        # If we don't have enough non-target samples, take more target samples
        n_target = n_samples - len(non_target_available_indices)
        n_non_target = len(non_target_available_indices)
    
    # Randomly select from each group
    selected_target_indices = np.random.choice(target_available_indices, size=n_target, replace=False)
    selected_non_target_indices = np.random.choice(non_target_available_indices, size=n_non_target, replace=False)
    
    # Combine and return
    selected_indices = np.concatenate([selected_target_indices, selected_non_target_indices])
    np.random.shuffle(selected_indices)  # Shuffle to mix target and non-target
    
    return selected_indices

def get_base_indices_class(x_base, y_base, n_samples, target_labels, target_ratio=0.5, random_seed=42):
    """
    Get deterministic base indices with class imbalance - prioritizes target labels.
    
    Parameters:
    x_base: base data
    y_base: base labels
    n_samples: total number of samples to select
    target_labels: list of target labels to prioritize
    target_ratio: proportion of samples that should come from target labels (0-1)
    random_seed: random seed for reproducibility
    
    Returns:
    selected_indices: indices of selected samples
    """
    total_samples = len(x_base)
    
    if n_samples > total_samples:
        raise ValueError(f"n_samples ({n_samples}) cannot exceed total samples ({total_samples})")
    
    # All indices are available for base selection
    all_indices = np.arange(total_samples)
    
    return _select_indices_with_class_imbalance(
        y_data=y_base,
        available_indices=all_indices,
        n_samples=n_samples,
        target_labels=target_labels,
        target_ratio=target_ratio,
        random_seed=random_seed
    )

def get_additional_indices_class(x_base, y_base, excluded_indices, n_additional, 
                                target_labels, target_ratio=0.5, random_seed=42):
    """
    Get additional indices with class imbalance that do not overlap with excluded indices.
    
    Parameters:
    x_base: base data
    y_base: base labels
    excluded_indices: indices to exclude from selection
    n_additional: number of additional samples to select
    target_labels: list of target labels to prioritize
    target_ratio: proportion of samples that should come from target labels (0-1)
    random_seed: random seed for reproducibility
    
    Returns:
    selected_indices: indices of selected additional samples
    """
    total_samples = len(x_base)
    
    # Remove excluded indices
    all_indices = np.arange(total_samples)
    available_indices = np.setdiff1d(all_indices, excluded_indices)
    
    if n_additional > len(available_indices):
        raise ValueError(f"n_additional ({n_additional}) cannot exceed available samples ({len(available_indices)})")
    
    return _select_indices_with_class_imbalance(
        y_data=y_base,
        available_indices=available_indices,
        n_samples=n_additional,
        target_labels=target_labels,
        target_ratio=target_ratio,
        random_seed=random_seed
    )

def prepare_class_imbalanced_dataset(x_base, y_base, target_labels, dataset_quantities,
                                    base_data_size, target_ratio=0.5, data_seed=42, 
                                    seed_1=None, base_imbalanced = True, additional_imbalanced=True):
    """
    Prepare training and additional datasets with optional class imbalance for additional data.
    
    Parameters
    ----------
    x_base : torch.Tensor
        Full dataset features (shape: [N, ...])
    y_base : torch.Tensor
        Full dataset labels (shape: [N])
    target_labels : list or torch.Tensor
        List of target labels to prioritize
    dataset_quantities : list[int]
        List of dataset sizes, max determines how many additional samples to generate
    base_data_size : int
        Number of base samples to select for training
    target_ratio : float, optional
        Proportion of samples that should come from target labels (default: 0.5)
    data_seed : int, optional
        Random seed for reproducibility
    seed_1: random seed for getting additional indices
    additional_imbalanced : bool, optional
        Whether to apply class imbalance to additional data (default: True)
        If False, uses the original get_additional_indices function

    Returns
    -------
    x_base_train : torch.Tensor
        Base training inputs
    y_base_train : torch.Tensor
        Base training labels
    x_additional : torch.Tensor
        Additional inputs
    y_additional : torch.Tensor
        Additional labels
    """
    # ----------------------------
    # 1. Select base training set with class imbalance
    # ----------------------------
    if base_imbalanced:
        base_indices = get_base_indices_class(
            x_base=x_base, 
            y_base=y_base,
            n_samples=base_data_size,
            target_labels=target_labels,
            target_ratio=target_ratio,
            random_seed=data_seed
        )
    else:
        base_indices = get_base_indices(x_base=x_base, y_base=y_base,
                                n_samples=base_data_size,
                                random_seed=data_seed)
    x_base_train = x_base[base_indices]
    y_base_train = y_base[base_indices]

    # ----------------------------
    # 2. Prepare additional dataset
    # ----------------------------
    max_additional_indices = max(dataset_quantities)
    seed_1 = seed_1 or data_seed
    
    if additional_imbalanced:
        # Use class-imbalanced selection for additional data
        additional_indices = get_additional_indices_class(
            x_base=x_base, 
            y_base=y_base,
            excluded_indices=base_indices,
            n_additional=max_additional_indices,
            target_labels=target_labels,
            target_ratio=target_ratio,
            random_seed=seed_1
        )
    else:
        # Use original balanced selection for additional data
        additional_indices = get_additional_indices(
            x_base=x_base,
            y_base=y_base,
            excluded_indices=base_indices,
            n_additional=max_additional_indices,
            random_seed=seed_1
        )
    
    x_additional = x_base[additional_indices]
    y_additional = y_base[additional_indices]

    return x_base_train, y_base_train, x_additional, y_additional
    
# --------------------------
# Misc Data Cleaning
# --------------------------

def tensor_to_list(val, key_path=""):
    """
    Convert a tensor or nested structure of tensors to Python types.
    Logs a message if a conversion occurs.
    """
    if torch.is_tensor(val):
        if val.ndim == 0:
            print(f"[DEBUG] Converted scalar tensor at '{key_path}' to float")
            return val.item()
        else:
            print(f"[DEBUG] Converted tensor at '{key_path}' to list of floats")
            return val.detach().cpu().tolist()
    elif isinstance(val, (list, tuple)):
        return [tensor_to_list(v, f"{key_path}[{i}]") for i, v in enumerate(val)]
    else:
        return val  # already a Python type or NumPy array

# --------------------------
# File I/O Operations
# --------------------------

def save_dataset(folder="results/", filename="dataset.pt",
                 x_base_train=None, y_base_train=None,
                 x_additional=None, y_additional=None,
                 x_test=None, y_test=None,
                 dataset_quantities=None, dataset_type="data"):
    """
    Save a dataset (base, additional, test, metadata) to a file.
    """

    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, filename)

    dataset_dict = {
        "x_base_train": x_base_train.cpu() if x_base_train is not None else None,
        "y_base_train": y_base_train.cpu() if y_base_train is not None else None,
        "x_additional": x_additional.cpu() if x_additional is not None else None,
        "y_additional": y_additional.cpu() if y_additional is not None else None,
        "x_test": x_test.cpu() if x_test is not None else None,
        "y_test": y_test.cpu() if y_test is not None else None,
        "dataset_quantities": dataset_quantities,
        "dataset_type": dataset_type
    }

    torch.save(dataset_dict, save_path)
    print(f"✅ Dataset saved to {save_path}")

def save_model(folder="results/", filename="model.pt", model=None, train_loss=None, 
               train_accs=None, test_loss=None, test_accs=None, 
               additional_data=None, dataset_type=None,
               **kwargs  # ✅ Allow arbitrary extra keys 
              ):
    """
    Save a model with complete training metadata.
    """
    
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, filename)

    model_dict = {
        "state_dict": model.state_dict(),
        "train_loss": train_loss,
        "train_accs": train_accs,
        "test_loss": test_loss,
        "test_accs": test_accs,
        "additional_data": additional_data,
        "dataset_type": dataset_type,
        "model_class": model.__class__.__name__  # Save model class name for reference
    }
    # ✅ Merge in additional keys
    model_dict.update(kwargs)
    
    torch.save(model_dict, save_path)
    print(f"✅ Model saved to {save_path}")

def load_model(model, folder="results/models", filename="model.pt", device="cpu"):
    """
    Load a saved model with complete training metadata.

    Args:
        model: an instance of your model class (already initialized)
        folder: directory where the model file is saved. This file contains model + metadata
        filename: model filename
        device: torch device ("cpu" or "cuda")
    
    Returns:
        model: model with loaded weights (on specified device)
        model_data: dictionary containing all saved metadata
    """
    load_path = os.path.join(folder, filename)
    model_dict = torch.load(load_path, map_location=device)

    model.load_state_dict(model_dict["state_dict"])
    model.to(device)

    # Return everything except the state_dict as metadata
    model_data = {k: v for k, v in model_dict.items() if k != "state_dict"}

    # Example
    #model_data = {
    #    "train_loss": model_dict.get("train_loss", None),
    #    "train_accs": model_dict.get("train_accs", None),
    #    "test_loss": model_dict.get("test_loss", None),
    #    "test_accs": model_dict.get("test_accs", None),
    #    "additional_data": model_dict.get("additional_data", None),
    #    "dataset_type": model_dict.get("dataset_type", None),
    #    "model_class": model_dict.get("model_class", None)
    #}

    print(f"✅ Model loaded into provided instance from {load_path}")
    return model, model_data

def load_dataset(folder="results", filename="dataset.pt", device="cpu"):
    """
    Load a dataset saved by save_dataset.
    Moves tensors to the specified device.
    """
    load_path = os.path.join(folder, filename)
    dataset_dict = torch.load(load_path, map_location=device)

    # Move tensors to device if they exist
    for key in ["x_base_train", "y_base_train", "x_additional", "y_additional", "x_test", "y_test"]:
        if dataset_dict[key] is not None:
            dataset_dict[key] = dataset_dict[key].to(device)

    print(f"✅ Dataset loaded from {load_path}")
    return dataset_dict

def load_models_and_data(model_template=None, target_dir=".", device="cpu"):
    """
    Load all models and dataset from the specified directory.
    
    Returns:
        tuple: (loaded_models, loaded_model_data, loaded_dataset)
    """    
    print(f"Looking for models and dataset in: {target_dir}")

    # Find all files that start with "model_" in the target directory
    model_files = glob.glob(os.path.join(target_dir, "model_*"))
    
    print(f"Found {len(model_files)} model files:")
    for file in model_files:
        print(f"  - {os.path.basename(file)}")

    # Lists to store loaded models and model data
    loaded_models = []
    loaded_model_data = []  # Changed from loaded_additional_data

    # Load each model
    for model_file in model_files:
        # Create a copy of the model template for each model
        if model_template is None:
            print ("ERROR! You need a model!")
        else:
            # Create a deep copy of the model template
            model = copy.deepcopy(model_template)
        
        # Load the model - now returns model_data instead of just additional_data
        model_loaded, model_data = load_model(
            model, 
            folder=str(target_dir),
            filename=os.path.basename(model_file),
            device=device
        )
        
        # Store the results
        loaded_models.append(model_loaded)
        loaded_model_data.append(model_data)  # Store complete model data
        
        print(f"Successfully loaded: {os.path.basename(model_file)}")

    # Print the list of model data loaded
    print("\nModel data loaded from all models:")
    for i, data in enumerate(loaded_model_data):
        print(f"Model {i} ({os.path.basename(model_files[i])}):")
        print(f"  - Additional data: {data.get('additional_data', 'N/A')}")
        print(f"  - Dataset type: {data.get('dataset_type', 'N/A')}")
        print(f"  - Training accuracies: {len(data.get('train_accs', [])) if data.get('train_accs') else 0} entries")
        print(f"  - Test accuracies: {len(data.get('test_accs', [])) if data.get('test_accs') else 0} entries")

    # Load the dataset
    print("\nLoading dataset...")

    # Candidate dataset names
    candidate_names = ["dataset.pt", "data.pt", "training_data.pt"]
    dataset_filename = None
    for name in candidate_names:
        path = os.path.join(target_dir, name)
        if os.path.exists(path):
            dataset_filename = name
            print(f"Using dataset file: {name}")
            break

    if dataset_filename is None:
        raise FileNotFoundError(f"No dataset file found in {target_dir}")

    loaded_dataset = load_dataset(
        folder=target_dir,
        filename=dataset_filename,
        device=device
    )
    
    return loaded_models, loaded_model_data, loaded_dataset