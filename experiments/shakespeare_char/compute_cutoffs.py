"""
Volume Cutoff for the overnight nanoGPT run.

Mirrors `Volume Cutoff.ipynb` from the modulo arithmetic / MNIST experiments:
for each saved model, evaluates a cumulative-average per-window loss over the
full training pool (base + max additional windows) and writes
`overnight_run/cutoffs/cutoffs.json`. This is the artifact that
`analysis_funcs.find_cutoff_index` consumes to draw the L > M cliff in the
"Minima Volumes Across Datasets" plot.
"""
import json
import os
from pathlib import Path

import torch

from minima_volume.dataset_funcs import load_models_and_data, tensor_to_list
from minima_volume.perturb_funcs import cumulative_average_loss_curve
from minima_volume.models import nanogpt_model_data as model_module

OUTPUT_DIR = Path(__file__).resolve().parent / "overnight_run"
SEED_DIR = OUTPUT_DIR  # single-seed: treat overnight_run itself as the seed dir
BATCH_SIZE = 64  # block_size=256 → 64*256=16K tokens/batch fits comfortably


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    template = model_module.get_model(device=device, seed=0)
    loaded_models, loaded_md, loaded_ds = load_models_and_data(
        model_template=template,
        target_dir=str(SEED_DIR / "models_and_data"),
        device=device,
    )
    all_models = [
        {
            "model": m,
            **{k: tensor_to_list(d[k], key_path=k) for k in
               ["additional_data", "dataset_type"]},
        }
        for m, d in zip(loaded_models, loaded_md)
    ]

    x_base = loaded_ds["x_base_train"].to(device)
    y_base = loaded_ds["y_base_train"].to(device)
    x_add = loaded_ds["x_additional"].to(device)
    y_add = loaded_ds["y_additional"].to(device)
    dataset_quantities = loaded_ds["dataset_quantities"]
    max_additional = max(dataset_quantities)

    x_full = torch.cat([x_base, x_add[:max_additional]], dim=0)
    y_full = torch.cat([y_base, y_add[:max_additional]], dim=0)
    base_train_size = len(x_base)
    print(f"x_full shape: {tuple(x_full.shape)}, base_train_size={base_train_size}")

    loss_fn = model_module.get_loss_fn_per_sample()

    cutoff_results = {}
    for md in all_models:
        m = md["model"]
        ad = md["additional_data"]
        print(f"  computing cumulative loss curve for Model_{ad}")
        curve = cumulative_average_loss_curve(m, x_full, y_full, loss_fn,
                                              batch_size=BATCH_SIZE)
        cutoff_results[f"Model_{ad}"] = {
            "additional_data": int(ad),
            "base_train_size": int(base_train_size),
            "indices": list(range(1, len(curve) + 1)),
            "loss_curve": curve.tolist(),
        }

    out_dir = SEED_DIR / "cutoffs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cutoffs.json"
    with open(out_path, "w") as f:
        json.dump(cutoff_results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
