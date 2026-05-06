"""
Train nanoGPT (char-shakespeare) at varying training-set sizes and save
checkpoints + dataset to models_and_data/, ready for the perturbation +
volume-estimation stages.

Mirrors experiments/CIFAR10/template - CNN/base folder/Train Low Test Models.ipynb
but as a script (so the 18 runs can be sharded across GPUs by a launcher).

Run nanogpt/data/shakespeare_char/prepare.py once before this. From the
repo root:

    python nanogpt/data/shakespeare_char/prepare.py
    python "experiments/shakespeare_char/base folder/train_models.py"
"""
import copy

import torch
import torch.optim as optim

from minima_volume.dataset_funcs import prepare_datasets, save_dataset, save_model
from minima_volume.train_funcs import train
from minima_volume.models import nanogpt_model_data as model_module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Configuration
# ============================================================================
data_seed = 1
model_seed = 1

# A "sample" is one block_size-length window of token ids (see
# nanogpt_model_data.py). Token-count equivalents:
#   50  windows ≈  12.8K tokens   (just the base set)
#   200 windows ≈  51.2K tokens
#   1000 windows ≈ 256K tokens
#   3000 windows ≈ 768K tokens   (~70% of shakespeare's ~1.1M-char train.bin)
base_data_size = 50
dataset_type = "data"
dataset_quantities = [0, 150, 450, 950, 2950]  # extras on top of base_data_size

# Training: fully overfit the (small) fixed window set.
# 10M-param GPT vs <=3000 windows is heavily overparameterized -- the
# bottleneck is iters, not capacity. AdamW + constant lr=1e-3 reaches
# train loss <1e-3 on shakespeare-char in this regime.
epochs = 2000
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-1

# Pool sizes for x_base / x_test that prepare_datasets will draw subsets from.
n_train_windows = max(dataset_quantities) + base_data_size + 100
n_test_windows = 256

save_generated_dataset = True
save_generated_models = True
output_folder = "models_and_data"


def main():
    # -- data ---------------------------------------------------------------
    x_base, y_base, x_test, y_test = model_module.get_dataset(
        device=device,
        n_train_windows=n_train_windows,
        n_test_windows=n_test_windows,
        seed=data_seed,
    )

    # -- model template ----------------------------------------------------
    model_template = model_module.get_model(device=device, seed=model_seed)
    loss_fn = model_module.get_loss_fn()
    other_metrics = model_module.get_additional_metrics()

    n_params = sum(p.numel() for p in model_template.parameters())
    print(f"GPT params: {n_params:,}  block_size={x_base.shape[1]}  device={device}")

    # -- dataset splits ----------------------------------------------------
    x_base_train, y_base_train, x_additional, y_additional = prepare_datasets(
        x_base=x_base,
        y_base=y_base,
        dataset_type=dataset_type,
        dataset_quantities=dataset_quantities,
        base_data_size=base_data_size,
        data_seed=data_seed,
    )
    x_base_train, y_base_train = x_base_train.to(device), y_base_train.to(device)
    x_additional, y_additional = x_additional.to(device), y_additional.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # -- training loop -----------------------------------------------------
    all_models = []
    for additional_data in dataset_quantities:
        x_train = torch.cat([x_base_train, x_additional[:additional_data]], dim=0)
        y_train = torch.cat([y_base_train, y_additional[:additional_data]], dim=0)

        torch.manual_seed(model_seed)
        model = copy.deepcopy(model_template)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        bs = min(len(x_train), batch_size)
        print(f"\n--- Training with {additional_data} additional samples "
              f"(N={len(x_train)} windows, batch={bs}) ---")

        train_loss, train_metrics_hist, test_loss, test_metrics_hist = train(
            model=model,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            loss_fn=loss_fn,
            metrics=other_metrics,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=bs,
            verbose_every=max(epochs // 10, 1),
        )

        train_metrics_dict = {}
        test_metrics_dict = {}
        if train_metrics_hist:
            for metric_name in train_metrics_hist[0].keys():
                train_metrics_dict[f"train_{metric_name}"] = [m[metric_name] for m in train_metrics_hist]
                test_metrics_dict[f"test_{metric_name}"] = [m[metric_name] for m in test_metrics_hist]

        all_models.append({
            "model": model,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "additional_data": additional_data,
            "dataset_type": dataset_type,
            **train_metrics_dict,
            **test_metrics_dict,
        })

        del x_train, y_train
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -- verify ------------------------------------------------------------
    model_module.verify_model_results(
        all_models=all_models,
        x_base_train=x_base_train, y_base_train=y_base_train,
        x_additional=x_additional, y_additional=y_additional,
        x_test=x_test, y_test=y_test,
        dataset_quantities=dataset_quantities,
        dataset_type=dataset_type,
    )

    # -- save --------------------------------------------------------------
    if save_generated_dataset:
        save_dataset(
            folder=output_folder, filename="dataset.pt",
            x_base_train=x_base_train, y_base_train=y_base_train,
            x_additional=x_additional, y_additional=y_additional,
            x_test=x_test, y_test=y_test,
            dataset_quantities=dataset_quantities,
            dataset_type=dataset_type,
        )
    if save_generated_models:
        for model_data in all_models:
            save_model(
                folder=output_folder,
                filename=f"model_additional_{model_data['additional_data']}.pt",
                model=model_data["model"],
                train_loss=model_data["train_loss"],
                train_accs=model_data["train_accs"],
                test_loss=model_data["test_loss"],
                test_accs=model_data["test_accs"],
                additional_data=model_data["additional_data"],
                dataset_type=model_data["dataset_type"],
            )


if __name__ == "__main__":
    main()
