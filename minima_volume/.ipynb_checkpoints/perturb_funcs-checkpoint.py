# Standard library
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Any

# Third-party
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Local package imports
from minima_volume.volume_funcs import save_perturbations

"""
PyTorch Model Perturbation and Loss Landscape Analysis Toolkit

This module provides tools for analyzing neural networks through controlled parameter 
perturbations and loss landscape visualization.

Key Features:
1. Weight Perturbation:
   - ModelPerturber: Class for safe weight modifications with automatic reset
   - Random direction generation with optional filter-wise normalization
   - Normalization calculation

2. Wiggle:
   - Applies a perturbation with varying coefficients to a model, and evaluates loss
   - Returns a dictionary of the form:
        'loss': np.array(loss),
        'coefficients': coefficients,
        'perturbations': perturbations,

3. Perturbation device transfer (moving perturbations to CPU or GPU)

Typical Workflow:
1. Generate perturbations (Optionally filter normalize)
2. Apply perturbations while tracking loss/accuracy (wiggle automatically uses ModelPerturber class)

Example: (assuming you have a model)
    >>> random_perturbs = generate_random_directions(model, num_directions=num_directions, seed=seed)
    >>> random_perturb_norms = [perturbation_norm(perturb_vector) for perturb_vector in random_perturbs]
    >>> norm_dict = filtnorm(list(model.named_parameters()))
    >>> filt_norm_perturb_vectors = [filternorm_perturbation_vectors(perturb_vector, norm_dict)
                                     for perturb_vector in random_perturbs]
    >>> all_results = [wiggle(model, points, labels, loss_fn, perturbations, coefficients) 
                       for perturbations in filt_norm_perturb_vectors]

Note:
- All perturbations are applied in-place; use ModelPerturber.reset() to recover original weights
- For reproducible results, set random seeds (both PyTorch and NumPy)
- Default perturbation modes handle Linear/Conv2D/BatchNorm layers automatically

Author: Raymond
Date: 2025-08
"""

"""
Notes:
Should make all file storage methods consistent.
Here, 2D loss landscapes get saved in one way
that may not be how I save wiggle, which is annoying?
"""

class ModelPerturber:
    def __init__(self, model):
        self.model = model
        # Pre-store parameters for faster access
        self.param_list = list(model.named_parameters())
        self.original_state = [p.detach().clone() for _, p in self.param_list]
    
    def apply_perturbation(self, perturbation_dict):
        """Faster by avoiding dictionary lookups in inner loop"""
        with torch.no_grad():
            for name, param in self.param_list:
                if name in perturbation_dict:
                    param.add_(perturbation_dict[name])
    
    def reset(self):
        """Faster by using pre-stored lists"""
        with torch.no_grad():
            for (_, param), original in zip(self.param_list, self.original_state):
                param.copy_(original)
                
def filtnorm(named_parameters, perturb_list=['weight', 'bias']):
    """
    Return filter norms as a dictionary for specified parameters.
    
    Args:
        named_parameters: Iterator of (name, parameter) tuples
        perturb_list: List of parameter types to process (e.g., ['weight', 'bias'])
                     Only parameters whose names contain these strings will be processed
    Returns:
        Dictionary {param_name: norm_tensor} containing only requested parameters
    """
    norm_dict = {}
    
    for name, param in named_parameters:
        param = param.detach()
        shape = param.shape
        
        # Check if this parameter type should be processed
        should_process = any([p in name.split('.') for p in perturb_list])
        if should_process:
            if len(shape) == 2:  # Linear layer
                f = []
                for i in range(shape[0]):  # output units
                    filt = param[i]
                    norm = torch.norm(filt)
                    f.append(torch.ones_like(filt) * norm)
                stacked = torch.stack(f, dim=0)
                
            elif len(shape) == 4:  # Conv2D
                f = []
                for i in range(shape[0]):  # output filters
                    filt = param[i]
                    norm = torch.norm(filt)
                    f.append(torch.ones_like(filt) * norm)
                stacked = torch.stack(f, dim=0)
                
            elif len(shape) == 1:  # Bias/BatchNorm
                stacked = torch.ones_like(param)
                
            else:
                print(f"Unsupported parameter shape {shape} for {name}")
                stacked = torch.ones_like(param)  # Fallback
            
            norm_dict[name] = stacked
    
    return norm_dict

def generate_random_perturbation(model, perturb_list=['weight', 'bias'], seed=None):
    """
    Generate random perturbation direction for specified model parameters.
    
    Args:
        model: PyTorch model
        perturb_list: List of parameter types to perturb (e.g., ['weight', 'bias'])
        seed: Random seed for reproducibility
    Returns:
        Dictionary {param_name: perturbation_tensor} containing only requested parameters
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    perturbation = {}
    for name, param in model.named_parameters():
        # Check if this parameter type should be perturbed
        should_perturb = any([p in name.split('.') for p in perturb_list])
        if should_perturb:            
            perturbation[name] = torch.randn_like(param)

    return perturbation

def generate_random_directions(model, num_directions=2, perturb_list=['weight', 'bias'], seed=None):
    """
    Generates multiple random perturbation directions.
    """
    return [
        generate_random_perturbation(model, perturb_list = perturb_list, seed=seed + i if seed is not None else None)
        for i in range(num_directions)
    ]

def perturbation_norm(perturbation):
    """
    Finds the L2 norm of a single perturbation
    Returns: norm
    """
    # Flatten all perturbations into a single vector
    perturb_vector = torch.cat([p.flatten() for p in perturbation.values()])
    total_norm = torch.norm(perturb_vector, p=2)
    return total_norm

def filternorm_perturbation_vectors(perturb_vectors, norm_dict):
    """
    Multiplies perturbations according to the filter normalization dictionary
    """
    filter_normalized_vectors = {
        name: perturb_vectors[name] * norm_dict[name]
        for name in perturb_vectors  # or norm_dict, since they have same keys
    }
    return filter_normalized_vectors

def move_perturbation_dict(perturbation_dict, device='cpu'):
    """
    Move all tensors in a perturbation dictionary to the specified device.
    
    Args:
        perturbation_dict: Dictionary of {param_name: perturbation_tensor}
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
    
    Returns:
        Dictionary with all tensors moved to the target device
    """
    moved_dict = {}
    for param_name, tensor in perturbation_dict.items():
        if torch.is_tensor(tensor):
            moved_dict[param_name] = tensor.to(device)
        else:
            # Handle non-tensor values (though shouldn't occur in perturbations)
            moved_dict[param_name] = tensor
    
    return moved_dict

def wiggle_evaluator(
    model,
    model_perturber,
    points: torch.Tensor,
    labels: torch.Tensor,
    coefficients: torch.Tensor,
    perturbation_dict: Dict[str, torch.Tensor],
    metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    batch_size: int = None, 
) -> Dict[str, Any]:
    """
    Generic evaluation loop for sweeping perturbations, now with optional batching.
    
    Args:
        model: The PyTorch model to evaluate.
        model_perturber: Object with apply_perturbation() and reset().
        points: Input tensor [N, D, ...].
        labels: Target tensor [N].
        coefficients: Iterable of perturbation coefficients.
        perturbation_dict: Dict of base perturbation vectors.
        metrics: Dict of {metric_name: function(logits, labels) -> scalar Tensor}.
        batch_size: Optional batch size for forward pass. If None, use full dataset.
    
    Returns:
        results: Dict with arrays of metric values for each coefficient.
    """
    device = points.device
    results = {name: torch.zeros(len(coefficients), device=device) for name in metrics}
    previous_coeff = 0.0

    with torch.no_grad():
        for i, coeff in enumerate(coefficients):
            # Apply incremental perturbation
            delta_coeff = coeff - previous_coeff
            inc_perturbs = {k: v * delta_coeff for k, v in perturbation_dict.items()}
            model_perturber.apply_perturbation(inc_perturbs)
            previous_coeff = coeff

            # Forward pass (with batching if requested)
            if batch_size is None:
                logits = model(points)
                for name, fn in metrics.items():
                    results[name][i] = fn(logits, labels)
            else:
                metric_sums = {name: 0.0 for name in metrics}
                num_samples = points.shape[0]

                for start in range(0, num_samples, batch_size):
                    end = start + batch_size
                    batch_points = points[start:end]
                    batch_labels = labels[start:end]

                    logits = model(batch_points)
                    for name, fn in metrics.items():
                        metric_sums[name] += fn(logits, batch_labels).item() * len(batch_points)

                # Average over full dataset
                for name in metrics:
                    results[name][i] = metric_sums[name] / num_samples

        model_perturber.reset()

    return {name: tensor.cpu().numpy() for name, tensor in results.items()} | {
        "coefficients": coefficients
    }

def analyze_wiggles_metrics(
    model_list, 
    x_base_train, y_base_train, 
    x_additional, y_additional,
    x_test, y_test, 
    dataset_quantities, 
    dataset_type, 
    metrics,       
    coefficients,
    num_directions=3000,
    perturbation_seed=0,
    base_output_dir="tests/", 
    device=None,
    batch_size=None 
):
    if device is None:
        device = x_base_train.device
    x_base_train, y_base_train = x_base_train.to(device), y_base_train.to(device)
    x_additional, y_additional = x_additional.to(device), y_additional.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # Precompute perturbations
    test_model = model_list[0]['model'].to(device)
    seed_list = [perturbation_seed + i for i in range(num_directions)]

    # Preallocate
    random_perturbs = [None] * num_directions
    random_perturb_norms = [None] * num_directions
    
    # Enumerate over seeds
    for idx, seed in enumerate(seed_list):
        random_perturb = generate_random_perturbation(
            test_model, 
            perturb_list=['weight', 'bias'], 
            seed=seed
        )
        random_perturb_norm = perturbation_norm(random_perturb)
        random_perturbs[idx] = random_perturb
        random_perturb_norms[idx] = random_perturb_norm

    num_params = sum(t.numel() for t in random_perturbs[0].values())
    print("The number of parameters of the perturbation is", num_params)

    # Loop through dataset sizes
    for additional_data in dataset_quantities:
        # Create output directory for this dataset
        if base_output_dir:
            output_dir = f"{base_output_dir}/{dataset_type}_{additional_data}"
        else:
            output_dir = f"{dataset_type}_{additional_data}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        x_train = torch.cat([x_base_train, x_additional[:additional_data]], dim=0)
        y_train = torch.cat([y_base_train, y_additional[:additional_data]], dim=0)
        print(f"Testing on {dataset_type} with {additional_data} samples - {num_directions} directions")

        for model_data in model_list:
            model_trained_data = model_data['additional_data']
            print(f"Testing model trained on {model_trained_data} additional data.")
            if model_trained_data >= additional_data:
                # Retrieve test performance for all metrics dynamically
                test_performance = {}
                for metric_name in metrics.keys():
                    test_key = f"test_{metric_name}"
                    #print (test_key)
                    #print (model_data[test_key][0:5])
                    if test_key in model_data:
                        test_performance[metric_name] = model_data[test_key][-1]
                    else:
                        test_performance[metric_name] = None  # or skip if you prefer
                
                # Print test performance
                for metric_name, value in test_performance.items():
                    if value is not None:
                        print(f"{metric_name.capitalize()}: {value:.4f}")
                    else:
                        print(f"{metric_name.capitalize()}: N/A")

                model = model_data['model'].to(device)
                norm_dict = filtnorm(list(model.named_parameters()))
                perturber = ModelPerturber(model)

                start_time = time.time()
                all_results = []

                for idx, (seed, perturb, perturb_norm) in enumerate(zip(seed_list, random_perturbs, random_perturb_norms)):
                    filt_norm_perturb_vectors = filternorm_perturbation_vectors(perturb, norm_dict)
                    wiggle_result = wiggle_evaluator(
                        model=model,
                        model_perturber=perturber,
                        points=x_train,
                        labels=y_train,
                        metrics=metrics, 
                        perturbation_dict=filt_norm_perturb_vectors,
                        coefficients=coefficients,
                        batch_size=batch_size,
                    )
                    
                    wiggle_result.update({
                        'perturbation_seed': seed,
                        'perturbation_norm': float(perturb_norm.item()),
                    })
                    all_results.append(wiggle_result)

                elapsed_time = time.time() - start_time
                print(f"Wiggle completed in {elapsed_time:.2f} seconds "
                      f"for {dataset_type} model trained with {model_trained_data} samples")

                save_perturbations(
                    wiggle_results=all_results, 
                    model=model, 
                    output_dir=output_dir,
                    filename=f"{dataset_type}_{model_trained_data}.npz",
                    additional_data=additional_data,  # integer for the amount used in this landscape
                    model_trained_data=model_trained_data,  # the integer for the amount of additional data trained on
                    dataset_type=dataset_type,  # string, data/noise/poison
                    base_dataset_size=len(x_base_train),  # integer
                    **{f"test_{key}": value for key, value in test_performance.items()},
                    num_params=num_params  # integer
                )
                print(f"Saved to {output_dir}\n")


##########################
## For Large Models
##########################
def wiggle_evaluator_large(
    model,
    model_perturber,
    points: torch.Tensor,
    labels: torch.Tensor,
    coefficients: torch.Tensor,
    perturbation_dict: Dict[str, torch.Tensor],
    metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    batch_size: int = None,
    timeit: bool = False,
) -> tuple[Dict[str, Any], float, float]:
    """
    Memory-optimized version of wiggle_evaluator for large models.

    If timeit=True, returns (results, total_perturb_time, total_forward_time)
    otherwise returns (results, 0.0, 0.0).
    """
    results = {name: torch.zeros(len(coefficients), device="cpu") for name in metrics}
    previous_coeff = 0.0

    total_perturb_time = 0.0
    total_forward_time = 0.0

    with torch.no_grad():
        for i, coeff in enumerate(coefficients):
            delta_coeff = coeff - previous_coeff

            t0 = time.time()
            inc_perturbs = {k: v * delta_coeff for k, v in perturbation_dict.items()}
            model_perturber.apply_perturbation(inc_perturbs)
            if timeit:
                total_perturb_time += time.time() - t0

            previous_coeff = coeff

            t1 = time.time()
            if batch_size is None:
                logits = model(points)
                for name, fn in metrics.items():
                    value = fn(logits, labels)
                    if torch.is_tensor(value):
                        value = value.detach().cpu().item()
                    results[name][i] = float(value)
            else:
                metric_sums = {name: 0.0 for name in metrics}
                num_samples = points.shape[0]
                for start in range(0, num_samples, batch_size):
                    end = start + batch_size
                    batch_points = points[start:end]
                    batch_labels = labels[start:end]
                    logits = model(batch_points)
                    for name, fn in metrics.items():
                        value = fn(logits, batch_labels)
                        if torch.is_tensor(value):
                            value = value.detach().cpu().item()
                        metric_sums[name] += float(value) * len(batch_points)

                for name in metrics:
                    results[name][i] = metric_sums[name] / num_samples
            if timeit:
                total_forward_time += time.time() - t1

        model_perturber.reset()

    return (
        {name: tensor.numpy() for name, tensor in results.items()} | {"coefficients": coefficients},
        total_perturb_time,
        total_forward_time,
    )


def analyze_wiggles_metrics_large(
    model_list,
    x_base_train, y_base_train,
    x_additional, y_additional,
    x_test, y_test,
    dataset_quantities,
    dataset_type,
    metrics,
    coefficients,
    num_directions=3000,
    perturbation_seed=0,
    base_output_dir="tests/",
    device=None,
    batch_size=None,
    timeit: bool = False,
):
    if device is None:
        device = x_base_train.device
    x_base_train, y_base_train = x_base_train.to(device), y_base_train.to(device)
    x_additional, y_additional = x_additional.to(device), y_additional.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    seed_list = [perturbation_seed + i for i in range(num_directions)]

    for additional_data in dataset_quantities:
        output_dir = f"{base_output_dir}/{dataset_type}_{additional_data}" if base_output_dir else f"{dataset_type}_{additional_data}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        x_train = torch.cat([x_base_train, x_additional[:additional_data]], dim=0)
        y_train = torch.cat([y_base_train, y_additional[:additional_data]], dim=0)
        print(f"Testing on {dataset_type} with {additional_data} samples - {num_directions} directions")

        for model_data in model_list:
            model_trained_data = model_data['additional_data']
            print(f"Testing model trained on {model_trained_data} additional data.")
            if model_trained_data >= additional_data:
                test_performance = {
                    metric_name: model_data.get(f"test_{metric_name}", [None])[-1]
                    for metric_name in metrics.keys()
                }

                for metric_name, value in test_performance.items():
                    print(f"{metric_name.capitalize()}: {value:.4f}" if value is not None else f"{metric_name.capitalize()}: N/A")

                model = model_data['model'].to(device)
                norm_dict = filtnorm(list(model.named_parameters()))
                num_params = sum(t.numel() for t in norm_dict.values())
                perturber = ModelPerturber(model)

                start_time = time.time()
                all_results = []
                total_perturb_time = 0.0
                total_forward_time = 0.0

                for seed in seed_list:
                    perturb = generate_random_perturbation(
                        model, perturb_list=['weight', 'bias'], seed=seed
                    )
                    perturb_norm = perturbation_norm(perturb)
                    filt_norm_perturb_vectors = filternorm_perturbation_vectors(perturb, norm_dict)

                    wiggle_result, p_time, f_time = wiggle_evaluator_large(
                        model=model,
                        model_perturber=perturber,
                        points=x_train,
                        labels=y_train,
                        metrics=metrics,
                        perturbation_dict=filt_norm_perturb_vectors,
                        coefficients=coefficients,
                        batch_size=batch_size,
                        timeit=timeit,
                    )

                    total_perturb_time += p_time
                    total_forward_time += f_time

                    wiggle_result.update({
                        'perturbation_seed': seed,
                        'perturbation_norm': float(perturb_norm.item()),
                    })
                    all_results.append(wiggle_result)

                    del perturb, perturb_norm, filt_norm_perturb_vectors, wiggle_result
                    torch.cuda.empty_cache()

                elapsed_time = time.time() - start_time
                print(f"Wiggle completed in {elapsed_time:.2f} seconds "
                      f"for {dataset_type} model trained with {model_trained_data} samples")

                if timeit:
                    print(f"  Total perturbation time: {total_perturb_time:.2f} s")
                    print(f"  Total forward/metric time: {total_forward_time:.2f} s")

                save_perturbations(
                    wiggle_results=all_results,
                    model=model,
                    output_dir=output_dir,
                    filename=f"{dataset_type}_{model_trained_data}.npz",
                    additional_data=additional_data,
                    model_trained_data=model_trained_data,
                    dataset_type=dataset_type,
                    base_dataset_size=len(x_base_train),
                    **{f"test_{key}": value for key, value in test_performance.items()},
                    num_params=num_params,
                )
                print(f"Saved to {output_dir}\n")

                del model
                torch.cuda.empty_cache()

##############################################
# Fixed dataset, alternative variation
##############################################

def analyze_wiggle_metrics_fixed_dataset(
    model_list, 
    x_input, y_input,
    x_test, y_test, 
    dataset_type,
    variation_metric_key,
    metrics,       
    coefficients,
    num_directions=3000,
    perturbation_seed=0,
    device=None,
    batch_size=None 
):
    # Analyzes volume for models, with a fixed dataset
    # supports flexible keys, eg, batch size or train epochs
    if device is None:
        device = x_base_train.device
    x_train, y_train = x_input.to(device), y_input.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # Precompute perturbations
    test_model = model_list[0]['model'].to(device)
    seed_list = [perturbation_seed + i for i in range(num_directions)]

    # Preallocate
    random_perturbs = [None] * num_directions
    random_perturb_norms = [None] * num_directions
    
    # Enumerate over seeds
    for idx, seed in enumerate(seed_list):
        random_perturb = generate_random_perturbation(
            test_model, 
            perturb_list=['weight', 'bias'], 
            seed=seed
        )
        random_perturb_norm = perturbation_norm(random_perturb)
        random_perturbs[idx] = random_perturb
        random_perturb_norms[idx] = random_perturb_norm

    num_params = sum(t.numel() for t in random_perturbs[0].values())
    print("The number of parameters of the perturbation is", num_params)

    # Loop through dataset sizes
    output_dir = f"{dataset_type}_0"

    print(f"Testing on dataset size {len(x_input)} - {num_directions} directions")

    for model_data in model_list:
        model_trained_data = model_data[variation_metric_key]
        print(f"Testing model {str(variation_metric_key)} {model_trained_data}")
        test_performance = {}
        for metric_name in metrics.keys():
            test_key = f"test_{metric_name}"
            test_performance[metric_name] = model_data[test_key][-1]
            
            # Print test performance
            for metric_name, value in test_performance.items():
                if value is not None:
                    print(f"{metric_name.capitalize()}: {value:.4f}")
                else:
                    print(f"{metric_name.capitalize()}: N/A")

            model = model_data['model'].to(device)
            norm_dict = filtnorm(list(model.named_parameters()))
            perturber = ModelPerturber(model)

            start_time = time.time()
            all_results = []

            for idx, (seed, perturb, perturb_norm) in enumerate(zip(seed_list, random_perturbs, random_perturb_norms)):
                filt_norm_perturb_vectors = filternorm_perturbation_vectors(perturb, norm_dict) 
                wiggle_result = wiggle_evaluator(
                    model=model,
                    model_perturber=perturber,
                    points=x_train,
                    labels=y_train,
                    metrics=metrics, 
                    perturbation_dict=filt_norm_perturb_vectors,
                    coefficients=coefficients,
                    batch_size=batch_size,
                )
                
                wiggle_result.update({
                    'perturbation_seed': seed,
                    'perturbation_norm': float(perturb_norm.item()),
                })
                all_results.append(wiggle_result)

            elapsed_time = time.time() - start_time
            print(f"Wiggle completed in {elapsed_time:.2f} seconds "
                  f"for {dataset_type} model trained with {model_trained_data} samples")

            # For save perturbations, we need 
            # 'wiggle_results'
            # 'model
            # rest are optional kwargs
            save_perturbations(
                wiggle_results=all_results, 
                model=model, 
                output_dir=output_dir,
                filename=f"{dataset_type}_{model_trained_data}.npz",
                additional_data=0, #placeholder
                model_trained_data=model_trained_data,  # the integer that denotes the model variation value
                dataset_type=dataset_type,  # string, data/noise/poison
                base_dataset_size=len(x_train),  # integer
                **{f"test_{key}": value for key, value in test_performance.items()},
                num_params=num_params  # integer
            )
            print(f"Saved to {output_dir}\n")

##############################################
# No Filter Normalization
##############################################

def analyze_wiggle_metrics_no_filternorm(
    model_list, 
    x_base_train, y_base_train, 
    x_additional, y_additional,
    x_test, y_test, 
    dataset_quantities, 
    dataset_type, 
    metrics,       
    coefficients,
    num_directions=3000,
    perturbation_seed=0,
    base_output_dir="tests/", 
    device=None,
    batch_size=None 
):
    if device is None:
        device = x_base_train.device
    x_base_train, y_base_train = x_base_train.to(device), y_base_train.to(device)
    x_additional, y_additional = x_additional.to(device), y_additional.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # Precompute perturbations
    test_model = model_list[0]['model'].to(device)
    seed_list = [perturbation_seed + i for i in range(num_directions)]

    # Preallocate
    random_perturbs = [None] * num_directions
    random_perturb_norms = [None] * num_directions
    
    # Enumerate over seeds
    for idx, seed in enumerate(seed_list):
        random_perturb = generate_random_perturbation(
            test_model, 
            perturb_list=['weight', 'bias'], 
            seed=seed
        )
        random_perturb_norm = perturbation_norm(random_perturb)
        random_perturbs[idx] = random_perturb
        random_perturb_norms[idx] = random_perturb_norm

    num_params = sum(t.numel() for t in random_perturbs[0].values())
    print("The number of parameters of the perturbation is", num_params)

    # Loop through dataset sizes
    for additional_data in dataset_quantities:
        # Create output directory for this dataset
        if base_output_dir:
            output_dir = f"{base_output_dir}/{dataset_type}_{additional_data}"
        else:
            output_dir = f"{dataset_type}_{additional_data}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        x_train = torch.cat([x_base_train, x_additional[:additional_data]], dim=0)
        y_train = torch.cat([y_base_train, y_additional[:additional_data]], dim=0)
        print(f"Testing on {dataset_type} with {additional_data} samples - {num_directions} directions")

        for model_data in model_list:
            model_trained_data = model_data['additional_data']
            print(f"Testing model trained on {model_trained_data} additional data.")
            if model_trained_data >= additional_data:
                # Retrieve test performance for all metrics dynamically
                test_performance = {}
                for metric_name in metrics.keys():
                    test_key = f"test_{metric_name}"
                    #print (test_key)
                    #print (model_data[test_key][0:5])
                    if test_key in model_data:
                        test_performance[metric_name] = model_data[test_key][-1]
                    else:
                        test_performance[metric_name] = None  # or skip if you prefer
                
                # Print test performance
                for metric_name, value in test_performance.items():
                    if value is not None:
                        print(f"{metric_name.capitalize()}: {value:.4f}")
                    else:
                        print(f"{metric_name.capitalize()}: N/A")

                model = model_data['model'].to(device)
                #norm_dict = filtnorm(list(model.named_parameters())) # NOT NEEDED
                perturber = ModelPerturber(model)

                start_time = time.time()
                all_results = []

                for idx, (seed, perturb, perturb_norm) in enumerate(zip(seed_list, random_perturbs, random_perturb_norms)):
                    # filt_norm_perturb_vectors = filternorm_perturbation_vectors(perturb, norm_dict) # NOT NEEDED
                    wiggle_result = wiggle_evaluator(
                        model=model,
                        model_perturber=perturber,
                        points=x_train,
                        labels=y_train,
                        metrics=metrics, 
                        perturbation_dict=perturb,
                        coefficients=coefficients,
                        batch_size=batch_size,
                    )
                    
                    wiggle_result.update({
                        'perturbation_seed': seed,
                        'perturbation_norm': float(perturb_norm.item()),
                    })
                    all_results.append(wiggle_result)

                elapsed_time = time.time() - start_time
                print(f"Wiggle completed in {elapsed_time:.2f} seconds "
                      f"for {dataset_type} model trained with {model_trained_data} samples")

                save_perturbations(
                    wiggle_results=all_results, 
                    model=model, 
                    output_dir=output_dir,
                    filename=f"{dataset_type}_{model_trained_data}.npz",
                    additional_data=additional_data,  # integer for the amount used in this landscape
                    model_trained_data=model_trained_data,  # the integer for the amount of additional data trained on
                    dataset_type=dataset_type,  # string, data/noise/poison
                    base_dataset_size=len(x_base_train),  # integer
                    **{f"test_{key}": value for key, value in test_performance.items()},
                    num_params=num_params  # integer
                )
                print(f"Saved to {output_dir}\n")

##############################################
# Functions related to perturbations
##############################################

def cumulative_average_loss_curve(model, x_data, y_data, loss_fn, batch_size=1000):
    """
    Compute the cumulative average loss as a function of the number of samples used.

    Args:
        model (torch.nn.Module): PyTorch model to evaluate.
        x_data (torch.Tensor): Input data, shape [N, ...].
        y_data (torch.Tensor): Target labels, shape [N].
        loss_fn (callable): A loss function that returns per-sample losses,
                            e.g., a wrapper around nn.CrossEntropyLoss.
        batch_size (int): Batch size for evaluation.

    Returns:
        cumulative_avg_loss (torch.Tensor): Tensor of length N where each entry i
                                            is the average loss over samples 0..i.
    Notes:
        Assumes loss_fn returns **per-sample losses**, not averaged.
    """
    model.eval()
    n_samples = x_data.size(0)
    per_sample_losses = torch.zeros(n_samples)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            x_batch = x_data[start:end]
            y_batch = y_data[start:end]
            logits = model(x_batch)
            losses = loss_fn(logits, y_batch)
            # Ensure per-sample losses
            if losses.dim() == 0:
                losses = losses.expand(len(x_batch))
            per_sample_losses[start:end] = losses

    # Compute cumulative average loss
    cumulative_sum = torch.cumsum(per_sample_losses, dim=0)
    cumulative_avg_loss = cumulative_sum / torch.arange(1, n_samples + 1, dtype=torch.float)

    return cumulative_avg_loss