import numpy as np
import torch

"""
PyTorch Training Utilities

This module provides standardized training loops and evaluation functions for PyTorch models.
It deals with classification tasks, and computes accuracies alongside loss.

Key Components:
1. Core Functions:
   - train_one_epoch: Single epoch training with batch processing
   - evaluate: Model validation/testing with metrics calculation
   - train: Complete training loop with epoch management

2. Metrics Tracked:
   - Training loss/accuracy per epoch
   - Test loss/accuracy per epoch

Typical Workflow:
1. Initialize model, optimizer, and loss function
2. Call train() with your data and hyperparameters:
    >>> loss = train(model, x_train, y_train, x_test, y_test,
    ...               criterion, optimizer, epochs=50)
3. Analyze returned metrics or extend the training loop

Example:
    >>> model = MLP(input_dim=2, hidden_dims=[64, 64], output_dim=1)
    >>> criterion = nn.BCEWithLogitsLoss()
    >>> optimizer = optim.Adam(model.parameters(), lr=1e-3)
    >>> train_loss, train_acc, test_loss, test_acc = train(
    ...     model, x_train, y_train, x_test, y_test,
    ...     criterion, optimizer, epochs=100)

Note:
- Handles both single-output (binary) and multi-output classification automatically
- For binary classification, uses sigmoid thresholding at 0.5
- For multi-class, uses argmax prediction
- All inputs should be torch.Tensor objects
- Batches are processed without gradient accumulation (call .zero_grad() each batch)

Author: Raymond
Date: 2025-08
Version: 0.1
"""

# ----- Train/Test Functions -----
def train_one_epoch(model, x_train, y_train, loss_fn, metrics, optimizer, 
                    batch_size=1000, randperm_indices=True):
    model.train()
    total_loss = 0
    metric_sums = {name: 0.0 for name in metrics} if metrics else {}
    n_samples = x_train.size(0)
    
    # Use random permutation if requested, otherwise sequential batching
    if randperm_indices:
        indices = torch.randperm(n_samples)
    else:
        indices = torch.arange(n_samples)

    for i in range(0, n_samples, batch_size):
        idx = indices[i:i+batch_size]
        x_batch, y_batch = x_train[idx], y_train[idx]

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0) 
        
        # Compute additional metrics
        if metrics:
            for name, fn in metrics.items():
                metric_sums[name] += fn(logits, y_batch) * x_batch.size(0)

    avg_loss = (total_loss / n_samples)
    metric_results = {name: metric_sums[name]/n_samples for name in metric_sums} if metrics else {}
    return avg_loss, metric_results

@torch.no_grad()
def evaluate(model, x_test, y_test, loss_fn, metrics, batch_size=1000):
    model.eval()
    total_loss = 0
    metric_sums = {name: 0.0 for name in metrics} if metrics else {}
    n_samples = x_test.size(0)
    
    for i in range(0, n_samples, batch_size):
        x_batch = x_test[i:i+batch_size]
        y_batch = y_test[i:i+batch_size]
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        total_loss += loss.item() * x_batch.size(0) 
        
        # Compute additional metrics
        if metrics:
            for name, fn in metrics.items():
                metric_sums[name] += fn(logits, y_batch) * x_batch.size(0)

    avg_loss = (total_loss / n_samples)
    metric_results = {name: metric_sums[name]/n_samples for name in metric_sums} if metrics else {}
    return avg_loss, metric_results

# ----- Main Training Loop -----
def train(model, x_train, y_train, x_test, y_test, loss_fn, metrics, optimizer, 
          epochs, batch_size=1000, verbose_every=1, randperm_indices=True):
    train_loss_arr, test_loss_arr = [], []
    train_metrics_history = []  # list of dicts per epoch
    test_metrics_history = []

    for epoch in range(epochs):
        train_loss, train_metrics = train_one_epoch(
            model, x_train, y_train, loss_fn, metrics, optimizer, 
            batch_size=batch_size, randperm_indices=randperm_indices
        )
        test_loss, test_metrics = evaluate(model, x_test, y_test, loss_fn, metrics, batch_size=batch_size)

        train_loss_arr.append(train_loss)
        test_loss_arr.append(test_loss)
        train_metrics_history.append(train_metrics)
        test_metrics_history.append(test_metrics)

        if (epoch + 1) % verbose_every == 0 or epoch == 0 or epoch == epochs - 1:
            metrics_str = ""
            if metrics:
                metrics_str = " | " + " | ".join([f"{name} Train {train_metrics[name]:.4f} Test {test_metrics[name]:.4f}" 
                                                   for name in metrics])
            print(f"Epoch {epoch+1}/{epochs}: Train Loss {train_loss:.4f} | Test Loss {test_loss:.4f}{metrics_str}")

    return train_loss_arr, train_metrics_history, test_loss_arr, test_metrics_history

# ------------------------------
# Sharpness Aware Minimization
# ------------------------------

# For SAM, we want to find a region with the smallest loss in a local area
# We first find the largest loss in a local area by gradient descent and looking
# for the perturbation that should result in the largest loss
# Then at this perturbation, we compute the loss and look for the smallest direction
# Clever way of using gradient descent.
def train_one_epoch_sam(model, x_train, y_train, loss_fn, metrics, optimizer, 
                        batch_size=1000, randperm_indices=True, rho=0.05, adaptive=False):
    model.train()
    total_loss = 0
    metric_sums = {name: 0.0 for name in metrics} if metrics else {}
    n_samples = x_train.size(0)
    
    # Random permutation if requested
    indices = torch.randperm(n_samples) if randperm_indices else torch.arange(n_samples)

    for i in range(0, n_samples, batch_size):
        idx = indices[i:i+batch_size]
        x_batch, y_batch = x_train[idx], y_train[idx]

        # ----- First forward-backward pass -----
        # First calculate the gradient, then move
        # in a direction that INCREASES the loss
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()

        # ----- Perturb weights -----
        # Correct gradient norm calculation
        if adaptive:
            # Flatten all scaled gradients into one big vector
            scaled_grads = torch.cat([(torch.abs(p) * p.grad).flatten() 
                                     for p in model.parameters() if p.grad is not None])
            grad_norm = scaled_grads.norm(p=2)
        else:
            # Flatten all gradients into one big vector  
            grads = torch.cat([p.grad.flatten() 
                              for p in model.parameters() if p.grad is not None])
            grad_norm = grads.norm(p=2)
            
        scale = rho / (grad_norm + 1e-12)
        e_ws = []
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    e_ws.append(None)
                    continue
                e_w = (p.abs() if adaptive else 1.0) * p.grad * scale
                p.add_(e_w)     # perturb weight
                e_ws.append(e_w)

        # ----- Second forward-backward pass -----
        # At the new weights with increased loss
        # we compute the loss again.
        optimizer.zero_grad()
        logits_perturbed = model(x_batch)
        loss_perturbed = loss_fn(logits_perturbed, y_batch)
        loss_perturbed.backward()

        # ----- Restore original weights -----
        # We return back to our original and move in the direction
        # that minimizes the maximum loss
        with torch.no_grad():
            for p, e_w in zip(model.parameters(), e_ws):
                if e_w is not None:
                    p.sub_(e_w)

        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)

        # Compute metrics on the first (unperturbed) forward pass
        if metrics:
            for name, fn in metrics.items():
                metric_sums[name] += fn(logits.detach(), y_batch) * x_batch.size(0)

    avg_loss = total_loss / n_samples
    metric_results = {name: metric_sums[name]/n_samples for name in metric_sums} if metrics else {}
    return avg_loss, metric_results

# Identical to before
def train_sam(model, x_train, y_train, x_test, y_test, loss_fn, metrics, optimizer, 
              epochs, batch_size=1000, verbose_every=1, randperm_indices=True, rho=0.05, adaptive=False):
    train_loss_arr, test_loss_arr = [], []
    train_metrics_history = []
    test_metrics_history = []

    for epoch in range(epochs):
        train_loss, train_metrics = train_one_epoch_sam(
            model, x_train, y_train, loss_fn, metrics, optimizer, 
            batch_size=batch_size, randperm_indices=randperm_indices,
            rho=rho, adaptive=adaptive
        )
        test_loss, test_metrics = evaluate(model, x_test, y_test, loss_fn, metrics, batch_size=batch_size)

        train_loss_arr.append(train_loss)
        test_loss_arr.append(test_loss)
        train_metrics_history.append(train_metrics)
        test_metrics_history.append(test_metrics)

        if (epoch + 1) % verbose_every == 0 or epoch == 0 or epoch == epochs - 1:
            metrics_str = ""
            if metrics:
                metrics_str = " | " + " | ".join([f"{name} Train {train_metrics[name]:.4f} Test {test_metrics[name]:.4f}" 
                                                   for name in metrics])
            print(f"[SAM] Epoch {epoch+1}/{epochs}: Train Loss {train_loss:.4f} | Test Loss {test_loss:.4f}{metrics_str}")

    return train_loss_arr, train_metrics_history, test_loss_arr, test_metrics_history
