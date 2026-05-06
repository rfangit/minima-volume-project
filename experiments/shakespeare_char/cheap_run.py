"""
First real-scale single-GPU end-to-end run for the nanoGPT volume sweep.

A reduced-cost version of the paper config: full 10M-param GPT, three
training-set sizes, 100 perturbation directions. Goal is to (a) confirm
the H100 forward-pass timing extrapolation and (b) produce the first
real results.json on this branch.

Outputs go under experiments/shakespeare_char/cheap_run/ — that path is
hard-coded so the template configs in base folder/ stay clean.

Run from repo root:

    .venv/bin/python experiments/shakespeare_char/cheap_run.py
"""
import copy
import os
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


# --- Config ----------------------------------------------------------------
DATA_SEED = 1
MODEL_SEED = 1

BASE_DATA_SIZE = 50
DATASET_QUANTITIES = [0, 150, 450]      # extras on top of base; -> 50, 200, 500 windows total
DATASET_TYPE = "data"

EPOCHS = 1500
TRAIN_BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-1

NUM_DIRECTIONS = 100
N_COEFFS = 100
# Max coefficient on the perturbation walk. Paper used max=1.0 for ~100K-param
# CNNs; the 10M-param GPT's basin lives in roughly c <= 0.012 (measured from
# the first run), so we cap at 0.05 to put the crossings near idx ~45 of 100
# while preserving the quadratic-spacing pattern that concentrates resolution
# near the crossing region.
MAX_COEFF = 0.05
PERTURBATION_SEED = 1
EVAL_BATCH_SIZE = 64

# If models_and_data/ already exists, reuse it instead of retraining.
# Training is deterministic given the seeds, so this is just a time-save
# when iterating on perturbation/estimation parameters.
SKIP_TRAIN_IF_EXISTS = True

LOSS_THRESHOLDS = [0.5, 0.1, 0.05]
ACC_THRESHOLDS = [0.95, 0.9, 0.85]

OUTPUT_DIR = Path(__file__).resolve().parent / "cheap_run"


def _stage(name):
    class _T:
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n{'='*70}\n=== {name} ===\n{'='*70}")
            return self
        def __exit__(self, *_):
            print(f"\n=== {name} done in {time.time() - self.t0:.1f}s ===")
    return _T()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  output_dir: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cwd_old = Path.cwd()

    # ----------------------------------------------------------------------
    # Stage 1: train (skip if checkpoints already exist)
    # ----------------------------------------------------------------------
    models_dir = OUTPUT_DIR / "models_and_data"
    expected_ckpts = [models_dir / "dataset.pt"] + [
        models_dir / f"model_additional_{q}.pt" for q in DATASET_QUANTITIES
    ]
    skip_train = SKIP_TRAIN_IF_EXISTS and all(p.exists() for p in expected_ckpts)

    loss_fn = model_module.get_loss_fn()
    other_metrics = model_module.get_additional_metrics()

    if skip_train:
        print(f"\n=== train skipped (reusing checkpoints in {models_dir}) ===")
    else:
        with _stage("train"):
            x_base, y_base, x_test, y_test = model_module.get_dataset(
                device=device,
                n_train_windows=BASE_DATA_SIZE + max(DATASET_QUANTITIES) + 100,
                n_test_windows=256,
                seed=DATA_SEED,
            )
            template = model_module.get_model(device=device, seed=MODEL_SEED)
            n_params = sum(p.numel() for p in template.parameters())
            print(f"GPT params: {n_params:,}  block_size={x_base.shape[1]}")

            x_base_train, y_base_train, x_additional, y_additional = prepare_datasets(
                x_base=x_base, y_base=y_base,
                dataset_type=DATASET_TYPE,
                dataset_quantities=DATASET_QUANTITIES,
                base_data_size=BASE_DATA_SIZE,
                data_seed=DATA_SEED,
            )

            all_models = []
            for additional in DATASET_QUANTITIES:
                x_train = torch.cat([x_base_train, x_additional[:additional]], dim=0)
                y_train = torch.cat([y_base_train, y_additional[:additional]], dim=0)

                torch.manual_seed(MODEL_SEED)
                model = copy.deepcopy(template)
                opt = optim.AdamW(
                    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95),
                )
                bs = min(len(x_train), TRAIN_BATCH_SIZE)
                print(f"\n--- training additional={additional}  N={len(x_train)}  batch={bs} ---")
                t0 = time.time()
                tr_loss, tr_metrics, te_loss, te_metrics = train(
                    model=model,
                    x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test,
                    loss_fn=loss_fn, metrics=other_metrics,
                    optimizer=opt,
                    epochs=EPOCHS,
                    batch_size=bs,
                    verbose_every=max(EPOCHS // 5, 1),
                )
                print(f"   ({time.time() - t0:.1f}s)")
                all_models.append({
                    "model": model,
                    "train_loss": tr_loss, "test_loss": te_loss,
                    "train_accs": [m["accs"] for m in tr_metrics],
                    "test_accs": [m["accs"] for m in te_metrics],
                    "additional_data": additional,
                    "dataset_type": DATASET_TYPE,
                })
                torch.cuda.empty_cache()

            models_dir.mkdir(exist_ok=True)
            save_dataset(
                folder=str(models_dir), filename="dataset.pt",
                x_base_train=x_base_train, y_base_train=y_base_train,
                x_additional=x_additional, y_additional=y_additional,
                x_test=x_test, y_test=y_test,
                dataset_quantities=DATASET_QUANTITIES,
                dataset_type=DATASET_TYPE,
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

    # ----------------------------------------------------------------------
    # Stage 2: perturbation sweep
    # analyze_wiggles_metrics_large saves to cwd-relative dirs.
    # ----------------------------------------------------------------------
    os.chdir(OUTPUT_DIR)
    try:
        with _stage("perturb"):
            template2 = model_module.get_model(device=device, seed=0)
            loaded_models, loaded_md, loaded_ds = load_models_and_data(
                model_template=template2, target_dir="models_and_data", device=device,
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
            coefficients = (np.linspace(0, 1, N_COEFFS) ** 2) * MAX_COEFF
            analyze_wiggles_metrics_large(
                model_list=all_models2,
                x_base_train=loaded_ds["x_base_train"].to(device),
                y_base_train=loaded_ds["y_base_train"].to(device),
                x_additional=loaded_ds["x_additional"].to(device),
                y_additional=loaded_ds["y_additional"].to(device),
                dataset_quantities=DATASET_QUANTITIES,
                dataset_type=DATASET_TYPE,
                metrics={"loss": loss_fn, **other_metrics},
                coefficients=coefficients,
                num_directions=NUM_DIRECTIONS,
                perturbation_seed=PERTURBATION_SEED,
                base_output_dir="",
                device=device,
                batch_size=EVAL_BATCH_SIZE,
                timeit=True,
            )

        # ------------------------------------------------------------------
        # Stage 3: volume estimation
        # ------------------------------------------------------------------
        with _stage("estimate"):
            for d in sorted(p.name for p in Path(".").iterdir()
                            if p.is_dir() and p.name.startswith(("data_", "poison_", "noise_"))):
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
                    try:
                        analyze_and_plot_model_landscape(
                            directory=d,
                            loss_threshold=loss_th,
                            acc_threshold=acc_th,
                            verbose=False,
                            display_options={"loss_plots": False, "accuracy_plots": False},
                            save_paths_dict=save_paths,
                        )
                        print(f"  {d}/loss_{loss_th}: results.json written")
                    except Exception as e:
                        print(f"  {d}/loss_{loss_th}: ERROR {e}")
    finally:
        os.chdir(cwd_old)

    print(f"\n✅ cheap real-scale run complete. Artifacts under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
