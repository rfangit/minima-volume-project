"""
End-to-end smoke run of the nanoGPT volume pipeline with a tiny config.

Goals:
  1. Catch plumbing bugs (filter-norm on tied weights, dataset shape,
     analyze_wiggles_metrics_large compatibility) before scaling.
  2. Give a real wall-clock breakdown -- train vs perturb vs estimate --
     so we can tell where to spend optimisation effort. CPU-bound python
     overhead in the wiggle inner loop is a likely suspect at this model
     size; running tiny first tells us.

Tiny config: ~150K-param GPT, 50 train windows, 2 dataset sizes, 5
perturbation directions, 20 coefficients. Should finish in under a minute
on a single H100.

Run:
    .venv/bin/python experiments/shakespeare_char/smoke_test.py
"""
import copy
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from minima_volume.dataset_funcs import (
    load_models_and_data,
    prepare_datasets,
    save_dataset,
    save_model,
    tensor_to_list,
)
from minima_volume.perturb_funcs import analyze_wiggles_metrics_large
from minima_volume.train_funcs import train
from minima_volume.volume_funcs import analyze_and_plot_model_landscape
from minima_volume.models import nanogpt_model_data as model_module


# -- Tiny config --------------------------------------------------------------
SMOKE_GPT_KWARGS = dict(
    n_layer=2, n_head=2, n_embd=64, dropout=0.0, bias=False, block_size=64,
)
BASE_DATA_SIZE = 10
DATASET_QUANTITIES = [0, 20]
EPOCHS = 200
TRAIN_BATCH_SIZE = 16
LR = 1e-3
NUM_DIRECTIONS = 5
N_COEFFS = 20
EVAL_BATCH_SIZE = 32
LOSS_THRESHOLDS = [0.5, 0.1]
ACC_THRESHOLDS = [0.95, 0.85]


def _stage(name):
    """Tiny timing helper printing 'STAGE: name took Xs'."""
    class _T:
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n=== {name} ===")
            return self

        def __exit__(self, *_):
            print(f"=== {name} done in {time.time() - self.t0:.2f}s ===")
    return _T()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    smoke_root = Path(tempfile.mkdtemp(prefix="nanogpt_smoke_"))
    print(f"smoke workdir: {smoke_root}")
    cwd_old = Path.cwd()
    try:
        models_dir = smoke_root / "models_and_data"
        models_dir.mkdir(parents=True)

        # --- Stage 1: train -------------------------------------------------
        with _stage("train"):
            x_base, y_base, x_test, y_test = model_module.get_dataset(
                device=device,
                n_train_windows=BASE_DATA_SIZE + max(DATASET_QUANTITIES) + 20,
                n_test_windows=64,
                block_size=SMOKE_GPT_KWARGS["block_size"],
                seed=1,
            )
            template = model_module.get_model(device=device, seed=1, **SMOKE_GPT_KWARGS)
            n_params = sum(p.numel() for p in template.parameters())
            print(f"  GPT params: {n_params:,}")

            x_base_train, y_base_train, x_additional, y_additional = prepare_datasets(
                x_base=x_base, y_base=y_base,
                dataset_type="data",
                dataset_quantities=DATASET_QUANTITIES,
                base_data_size=BASE_DATA_SIZE,
                data_seed=1,
            )

            loss_fn = model_module.get_loss_fn()
            other_metrics = model_module.get_additional_metrics()

            all_models = []
            for additional in DATASET_QUANTITIES:
                x_train = torch.cat([x_base_train, x_additional[:additional]], dim=0)
                y_train = torch.cat([y_base_train, y_additional[:additional]], dim=0)
                model = copy.deepcopy(template)
                opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-1, betas=(0.9, 0.95))
                tr_loss, tr_metrics, te_loss, te_metrics = train(
                    model=model,
                    x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test,
                    loss_fn=loss_fn, metrics=other_metrics,
                    optimizer=opt,
                    epochs=EPOCHS,
                    batch_size=min(len(x_train), TRAIN_BATCH_SIZE),
                    verbose_every=EPOCHS,  # only print last epoch
                )
                all_models.append({
                    "model": model,
                    "train_loss": tr_loss, "test_loss": te_loss,
                    "train_accs": [m["accs"] for m in tr_metrics],
                    "test_accs": [m["accs"] for m in te_metrics],
                    "additional_data": additional,
                    "dataset_type": "data",
                })

            save_dataset(
                folder=str(models_dir), filename="dataset.pt",
                x_base_train=x_base_train, y_base_train=y_base_train,
                x_additional=x_additional, y_additional=y_additional,
                x_test=x_test, y_test=y_test,
                dataset_quantities=DATASET_QUANTITIES,
                dataset_type="data",
            )
            for md in all_models:
                save_model(
                    folder=str(models_dir),
                    filename=f"model_additional_{md['additional_data']}.pt",
                    model=md["model"],
                    train_loss=md["train_loss"], train_accs=md["train_accs"],
                    test_loss=md["test_loss"], test_accs=md["test_accs"],
                    additional_data=md["additional_data"],
                    dataset_type=md["dataset_type"],
                )

        # --- Stage 2: perturbation sweep ------------------------------------
        # analyze_wiggles_metrics_large saves to cwd-relative dirs, so chdir
        # into the smoke workdir for the rest.
        import os
        os.chdir(smoke_root)

        with _stage("perturb"):
            template2 = model_module.get_model(device=device, seed=0, **SMOKE_GPT_KWARGS)
            loaded_models, loaded_md, loaded_ds = load_models_and_data(
                model_template=template2, target_dir=str(models_dir), device=device,
            )
            all_models2 = [
                {
                    "model": m,
                    **{k: tensor_to_list(d[k], key_path=k) for k in
                       ["train_loss", "train_accs", "test_loss", "test_accs",
                        "additional_data", "dataset_type"]},
                }
                for m, d in zip(loaded_models, loaded_md)
            ]
            coefficients = np.linspace(0, 1, N_COEFFS) ** 2
            analyze_wiggles_metrics_large(
                model_list=all_models2,
                x_base_train=loaded_ds["x_base_train"].to(device),
                y_base_train=loaded_ds["y_base_train"].to(device),
                x_additional=loaded_ds["x_additional"].to(device),
                y_additional=loaded_ds["y_additional"].to(device),
                dataset_quantities=DATASET_QUANTITIES,
                dataset_type="data",
                metrics={"loss": loss_fn, **other_metrics},
                coefficients=coefficients,
                num_directions=NUM_DIRECTIONS,
                perturbation_seed=1,
                base_output_dir="",
                device=device,
                batch_size=EVAL_BATCH_SIZE,
                timeit=True,
            )

        # --- Stage 3: volume estimation -------------------------------------
        with _stage("estimate"):
            for d in sorted(p.name for p in smoke_root.iterdir()
                            if p.is_dir() and p.name.startswith("data_")):
                for loss_th, acc_th in zip(LOSS_THRESHOLDS, ACC_THRESHOLDS):
                    save_paths = {
                        "log_volume": f"{d}/loss_{loss_th}/log_volume.png",
                        "log_volume_generalization": f"{d}/loss_{loss_th}/gen.png",
                        "model_modification_vs_test_loss": f"{d}/loss_{loss_th}/test_vs_data.png",
                        "average_radius_loss": f"{d}/loss_{loss_th}/avg.png",
                        "radius_histogram_loss": f"{d}/loss_{loss_th}/hist.png",
                        "average_radius_acc": f"{d}/acc_{acc_th}/avg.png",
                        "radius_histogram_acc": f"{d}/acc_{acc_th}/hist.png",
                        "log_volume_acc": f"{d}/acc_{acc_th}/log_volume.png",
                        "log_volume_generalization_acc": f"{d}/acc_{acc_th}/gen.png",
                        "results.json": f"{d}/loss_{loss_th}",
                    }
                    analyze_and_plot_model_landscape(
                        directory=d,
                        loss_threshold=loss_th,
                        acc_threshold=acc_th,
                        verbose=False,
                        display_options={"loss_plots": False, "accuracy_plots": False},
                        save_paths_dict=save_paths,
                    )
                    print(f"  {d} loss={loss_th}: results.json written")

        print("\n✅ smoke run completed.")
    finally:
        import os
        os.chdir(cwd_old)
        # leave the dir for inspection; comment the next line to keep it
        shutil.rmtree(smoke_root, ignore_errors=True)


if __name__ == "__main__":
    main()
