"""
Stage 2: load the trained nanoGPT models, generate random parameter
perturbations and record loss along each direction.

Mirrors experiments/CIFAR10/template - CNN/base folder/Random Perturbs.ipynb.
Uses analyze_wiggles_metrics_large since the GPT (~10M params) is much
bigger than the image-classification baselines: it regenerates each
direction on demand and clears CUDA cache between, instead of holding all
500 directions in memory.

Run after train_models.py from this directory:

    cd "experiments/shakespeare_char/base folder"
    python train_models.py
    python random_perturbs.py
"""
import numpy as np
import torch

from minima_volume.dataset_funcs import load_models_and_data, tensor_to_list
from minima_volume.perturb_funcs import analyze_wiggles_metrics_large
from minima_volume.models import nanogpt_model_data as model_module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Configuration
# ============================================================================
perturbation_seed = 1
num_directions = 500     # paper standard; shard via launcher when scaling out
N = 100
coefficients = np.linspace(0, 1, N) ** 2  # quadratic spacing concentrates resolution near origin

# Per-coefficient eval batch size. The whole training set is passed to
# wiggle_evaluator, but loss is averaged in chunks of this size to stay
# within memory. None = single forward over the full set.
batch_size = 64

base_output_dir = ""  # results land in <cwd>/<dataset_type>_<additional>/...


def main():
    # -- rebuild a fresh model template (architecture only) ----------------
    model_template = model_module.get_model(device=device, seed=0)
    loss_fn = model_module.get_loss_fn()
    other_metrics = model_module.get_additional_metrics()

    # -- load saved models + dataset ---------------------------------------
    target_dir = "models_and_data"
    loaded_models, loaded_model_data, loaded_dataset = load_models_and_data(
        model_template=model_template,
        target_dir=target_dir,
        device=device,
    )
    dataset_type = loaded_dataset["dataset_type"]
    dataset_quantities = loaded_dataset["dataset_quantities"]
    print(f"dataset_type={dataset_type}  quantities={dataset_quantities}")

    all_models = [
        {
            "model": model,
            **{
                k: tensor_to_list(model_data[k], key_path=k)
                for k in ["train_loss", "train_accs", "test_loss", "test_accs",
                          "additional_data", "dataset_type"]
            },
        }
        for model, model_data in zip(loaded_models, loaded_model_data)
    ]

    x_base_train = loaded_dataset["x_base_train"].to(device)
    y_base_train = loaded_dataset["y_base_train"].to(device)
    x_additional = loaded_dataset["x_additional"].to(device)
    y_additional = loaded_dataset["y_additional"].to(device)

    # -- run perturbation sweep -------------------------------------------
    analyze_wiggles_metrics_large(
        model_list=all_models,
        x_base_train=x_base_train, y_base_train=y_base_train,
        x_additional=x_additional, y_additional=y_additional,
        dataset_quantities=dataset_quantities,
        dataset_type=dataset_type,
        metrics={"loss": loss_fn, **other_metrics},
        coefficients=coefficients,
        num_directions=num_directions,
        perturbation_seed=perturbation_seed,
        base_output_dir=base_output_dir,
        device=device,
        batch_size=batch_size,
        timeit=True,
    )


if __name__ == "__main__":
    main()
