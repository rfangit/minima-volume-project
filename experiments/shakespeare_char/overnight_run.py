"""
Multi-GPU overnight nanoGPT volume sweep.

Paper-scale single-seed run aimed at reproducing the volume-vs-data
findings for language models: 400x training-set span (50 -> 20000
windows), 200 perturbation directions, full 10M-param GPT.

Parallelism layout:
  - Stage 1 (train): 5 dataset sizes -> one subprocess per GPU 0..4 in
    parallel. Bottleneck is the 20000-window run (longest).
  - Stage 2 (perturb): 8 subprocesses on GPUs 0..7, each owning a slice
    of the 200 perturbation seeds. Each writes its (model, landscape)
    .npz files into shard_<i>/.
  - Stage 3 (merge): concatenates each (landscape, model) pair's
    wiggle_results across the 8 shards into the canonical
    data_<L>/data_<M>.npz layout.
  - Stage 4 (estimate): runs analyze_and_plot_model_landscape over
    all loss/accuracy threshold pairs, writes results.json + plots.

Resumable: existing checkpoints in models_and_data/ skip training.
Existing shard outputs are overwritten by re-runs (analyze_wiggles_*
itself overwrites the per-pair .npz). The merge + estimate stages are
idempotent.

Run from repo root:
    .venv/bin/python experiments/shakespeare_char/overnight_run.py

Or just one stage:
    .venv/bin/python experiments/shakespeare_char/overnight_run.py --stage merge
"""
import argparse
import copy
import os
import subprocess
import sys
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
from minima_volume.train_funcs import train as train_loop
from minima_volume.volume_funcs import analyze_and_plot_model_landscape
from minima_volume.models import nanogpt_model_data as model_module


# --- Config -----------------------------------------------------------------
# Per-seed knobs are env-overridable so multiseed_run.py can spawn this
# script with different (seed, output dir, GPU pool) without code edits.
def _envcfg(name, default, cast=str):
    val = os.environ.get(f"OVN_{name}")
    return cast(val) if val is not None else default


DATA_SEED = _envcfg("DATA_SEED", 1, int)
MODEL_SEED = _envcfg("MODEL_SEED", 1, int)
BASE_DATA_SIZE = 50

# 50, 250, 1000, 5000, 20000 windows total -- 400x span (paper used 1000x for MNIST)
DATASET_QUANTITIES = [0, 200, 950, 4950, 19950]
DATASET_TYPE = "data"

EPOCHS = _envcfg("EPOCHS", 800, int)
TRAIN_BATCH_SIZE = 64
LR = float(_envcfg("LR", 1e-3, float))
WEIGHT_DECAY = 1e-1
# Anneal AdamW LR to this floor over EPOCHS epochs to break through the
# constant-LR SGD noise floor that pins q=4950/19950 above zero. Set to 0
# (or the same as LR) to disable cosine and recover constant-LR behavior.
COSINE_MIN_LR = float(_envcfg("COSINE_MIN_LR", 0.0, float))

NUM_DIRECTIONS = _envcfg("NUM_DIRECTIONS", 200, int)
N_COEFFS = _envcfg("N_COEFFS", 100, int)
MAX_COEFF = 0.05
# Bumped per-seed by multiseed_run so two seeds don't share direction seeds.
PERTURBATION_SEED_BASE = _envcfg("PERTURBATION_SEED_BASE", 1, int)
EVAL_BATCH_SIZE = 64

# Wider threshold ladder than cheap_run -- larger landscapes may not converge
# below 0.1, so we want a few rungs above to still produce useful radii.
LOSS_THRESHOLDS = [2.0, 1.0, 0.5, 0.1, 0.05]
ACC_THRESHOLDS = [0.5, 0.7, 0.85, 0.9, 0.95]

NUM_TRAIN_GPUS = _envcfg("NUM_TRAIN_GPUS", 5, int)  # one per dataset size
NUM_PERTURB_GPUS = _envcfg("NUM_PERTURB_GPUS", 8, int)
# Physical GPU index to add when spawning workers; lets us pin a run to GPUs 2..7.
GPU_OFFSET = _envcfg("GPU_OFFSET", 0, int)

OUTPUT_DIR = Path(_envcfg("OUTPUT_DIR",
                          str(Path(__file__).resolve().parent / "overnight_run")))


# --- Worker entrypoints (run inside subprocess with CUDA_VISIBLE_DEVICES set) -

def stage_train_one(quantity: int):
    device = torch.device("cuda")
    print(f"[train q={quantity}] device={device}")

    x_base, y_base, x_test, y_test = model_module.get_dataset(
        device=device,
        n_train_windows=BASE_DATA_SIZE + max(DATASET_QUANTITIES) + 100,
        n_test_windows=256,
        seed=DATA_SEED,
    )
    x_base_train, y_base_train, x_additional, y_additional = prepare_datasets(
        x_base=x_base, y_base=y_base,
        dataset_type=DATASET_TYPE,
        dataset_quantities=DATASET_QUANTITIES,
        base_data_size=BASE_DATA_SIZE,
        data_seed=DATA_SEED,
    )
    x_train = torch.cat([x_base_train, x_additional[:quantity]], dim=0).to(device)
    y_train = torch.cat([y_base_train, y_additional[:quantity]], dim=0).to(device)

    torch.manual_seed(MODEL_SEED)
    model = model_module.get_model(device=device, seed=MODEL_SEED)
    n_params = sum(p.numel() for p in model.parameters())
    opt = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95),
    )
    scheduler = None
    if COSINE_MIN_LR > 0 and COSINE_MIN_LR < LR:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS, eta_min=COSINE_MIN_LR,
        )
    bs = min(len(x_train), TRAIN_BATCH_SIZE)

    loss_fn = model_module.get_loss_fn()
    other_metrics = model_module.get_additional_metrics()

    print(f"[train q={quantity}] N={len(x_train)} batch={bs} params={n_params:,} "
          f"lr={LR} cosine_min={COSINE_MIN_LR if scheduler else 'off'}")
    t0 = time.time()
    tr_loss, tr_metrics, te_loss, te_metrics = train_loop(
        model=model,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        loss_fn=loss_fn, metrics=other_metrics, optimizer=opt,
        epochs=EPOCHS,
        batch_size=bs,
        verbose_every=max(EPOCHS // 10, 1),
        scheduler=scheduler,
    )
    print(f"[train q={quantity}] training done in {time.time()-t0:.1f}s, "
          f"final train_loss={tr_loss[-1]:.4f}")

    models_dir = OUTPUT_DIR / "models_and_data"
    models_dir.mkdir(parents=True, exist_ok=True)
    # Only one worker (q=0) writes dataset.pt to avoid races; identical content anyway.
    if quantity == 0:
        save_dataset(
            folder=str(models_dir), filename="dataset.pt",
            x_base_train=x_base_train, y_base_train=y_base_train,
            x_additional=x_additional, y_additional=y_additional,
            x_test=x_test, y_test=y_test,
            dataset_quantities=DATASET_QUANTITIES,
            dataset_type=DATASET_TYPE,
        )
    save_model(
        folder=str(models_dir),
        filename=f"model_additional_{quantity}.pt",
        model=model,
        train_loss=tr_loss, train_accs=[m["accs"] for m in tr_metrics],
        test_loss=te_loss, test_accs=[m["accs"] for m in te_metrics],
        additional_data=quantity,
        dataset_type=DATASET_TYPE,
    )
    print(f"[train q={quantity}] saved")


def stage_perturb_one(shard_id: int):
    device = torch.device("cuda")
    K = NUM_DIRECTIONS // NUM_PERTURB_GPUS
    seed_base = PERTURBATION_SEED_BASE + shard_id * K
    n_dirs = K + (NUM_DIRECTIONS - K * NUM_PERTURB_GPUS if shard_id == NUM_PERTURB_GPUS - 1 else 0)

    print(f"[perturb shard={shard_id}] device={device} seeds=[{seed_base}, {seed_base+n_dirs-1}]  ({n_dirs} dirs)")

    template = model_module.get_model(device=device, seed=0)
    loaded_models, loaded_md, loaded_ds = load_models_and_data(
        model_template=template,
        target_dir=str(OUTPUT_DIR / "models_and_data"),
        device=device,
    )
    all_models = [
        {
            "model": m,
            **{k: tensor_to_list(d[k], key_path=k) for k in
               ["train_loss", "train_accs", "test_loss", "test_accs",
                "additional_data", "dataset_type"]},
        }
        for m, d in zip(loaded_models, loaded_md)
    ]
    coefficients = (np.linspace(0, 1, N_COEFFS) ** 2) * MAX_COEFF
    loss_fn = model_module.get_loss_fn()
    other_metrics = model_module.get_additional_metrics()

    shard_dir = OUTPUT_DIR / f"shard_{shard_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    analyze_wiggles_metrics_large(
        model_list=all_models,
        x_base_train=loaded_ds["x_base_train"].to(device),
        y_base_train=loaded_ds["y_base_train"].to(device),
        x_additional=loaded_ds["x_additional"].to(device),
        y_additional=loaded_ds["y_additional"].to(device),
        dataset_quantities=DATASET_QUANTITIES,
        dataset_type=DATASET_TYPE,
        metrics={"loss": loss_fn, **other_metrics},
        coefficients=coefficients,
        num_directions=n_dirs,
        perturbation_seed=seed_base,
        base_output_dir=str(shard_dir),
        device=device,
        batch_size=EVAL_BATCH_SIZE,
        timeit=True,
    )
    print(f"[perturb shard={shard_id}] DONE")


# --- Orchestration --------------------------------------------------------

def _spawn_subprocess(self_arg: str, val: int, gpu_id: int, log_path: Path):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log = open(log_path, "w")
    p = subprocess.Popen(
        [sys.executable, __file__, self_arg, str(val)],
        env=env, stdout=log, stderr=subprocess.STDOUT,
        # Detach into a new session so the children survive even if the
        # orchestrator's controlling terminal goes away.
        start_new_session=True,
    )
    return p, log


def stage_train_orchestrator():
    """Train each dataset size; never put two workers on the same GPU.

    With NUM_TRAIN_GPUS < len(DATASET_QUANTITIES) (e.g. 3 GPUs / 5 sizes),
    a per-GPU thread pulls from a shared queue. Longest-processing-time-first
    ordering keeps the q=19950 monster on its own GPU end-to-end.
    """
    import threading

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    job_queue = sorted(DATASET_QUANTITIES, reverse=True)  # largest first
    queue_lock = threading.Lock()
    failed = []
    failed_lock = threading.Lock()

    gpu_lo, gpu_hi = GPU_OFFSET, GPU_OFFSET + NUM_TRAIN_GPUS - 1
    print(f"[train] {len(job_queue)} jobs on GPUs {gpu_lo}..{gpu_hi} (LPT-first)")

    def gpu_worker(gpu: int):
        while True:
            with queue_lock:
                if not job_queue:
                    return
                q = job_queue.pop(0)
            log_path = OUTPUT_DIR / f"train_q{q}.log"
            p, log = _spawn_subprocess("--train-one", q, gpu, log_path)
            print(f"[train] q={q} -> cuda:{gpu}, log={log_path.name}, pid={p.pid}")
            rc = p.wait()
            log.close()
            print(f"[train] q={q} (cuda:{gpu}) exit={rc}")
            if rc != 0:
                with failed_lock:
                    failed.append(q)

    threads = [threading.Thread(target=gpu_worker, args=(GPU_OFFSET + i,))
               for i in range(NUM_TRAIN_GPUS)]
    for t in threads: t.start()
    for t in threads: t.join()

    if failed:
        print(f"[train] FAILED: {failed} -- aborting")
        sys.exit(1)


def stage_perturb_orchestrator():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    gpu_lo, gpu_hi = GPU_OFFSET, GPU_OFFSET + NUM_PERTURB_GPUS - 1
    print(f"[perturb] spawning {NUM_PERTURB_GPUS} workers on GPUs {gpu_lo}..{gpu_hi}")
    for i in range(NUM_PERTURB_GPUS):
        gpu = GPU_OFFSET + i
        log_path = OUTPUT_DIR / f"perturb_shard_{i}.log"
        p, log = _spawn_subprocess("--perturb-one", i, gpu, log_path)
        procs.append((p, log, i))
        print(f"[perturb] shard_{i} -> cuda:{gpu}, log={log_path.name}, pid={p.pid}")

    failed = []
    for p, log, i in procs:
        rc = p.wait()
        log.close()
        print(f"[perturb] shard_{i} exit={rc}")
        if rc != 0:
            failed.append(i)
    if failed:
        print(f"[perturb] WARN: failed shards: {failed} -- continuing to merge what we have")


def stage_merge():
    """Concat each (landscape, model) pair's wiggle_results across all shards
    into the canonical data_<L>/data_<M>.npz layout that volume_estimation expects."""
    print("[merge] combining shard outputs")
    for landscape in DATASET_QUANTITIES:
        ld_name = f"{DATASET_TYPE}_{landscape}"
        for model_id in DATASET_QUANTITIES:
            if model_id < landscape:
                continue
            mf = f"{DATASET_TYPE}_{model_id}.npz"
            shard_files = []
            for i in range(NUM_PERTURB_GPUS):
                p = OUTPUT_DIR / f"shard_{i}" / ld_name / mf
                if p.exists():
                    shard_files.append(p)
            if not shard_files:
                print(f"[merge] {ld_name}/{mf}: NO SHARD OUTPUT -- skipping")
                continue
            all_wr = []
            metadata = {}
            for sf in shard_files:
                npz = np.load(sf, allow_pickle=True)
                all_wr.extend(list(npz["wiggle_results"]))
                if not metadata:
                    metadata = {k: npz[k] for k in npz.files if k != "wiggle_results"}
            out_dir = OUTPUT_DIR / ld_name
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dir / mf,
                wiggle_results=np.array(all_wr, dtype=object),
                **metadata,
            )
            print(f"[merge] {ld_name}/{mf}: {len(all_wr)} dirs from {len(shard_files)} shards")


def stage_estimate():
    cwd_old = Path.cwd()
    os.chdir(OUTPUT_DIR)
    try:
        landscape_dirs = sorted(
            p.name for p in Path(".").iterdir()
            if p.is_dir() and p.name.startswith(("data_", "poison_", "noise_"))
        )
        for d in landscape_dirs:
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
                    print(f"[estimate] {d}/loss_{loss_th}")
                except Exception as e:
                    print(f"[estimate] {d}/loss_{loss_th}: ERROR {e}")
    finally:
        os.chdir(cwd_old)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-one", type=int, default=None,
                        help="Worker mode: train a single model with this many additional samples.")
    parser.add_argument("--perturb-one", type=int, default=None,
                        help="Worker mode: run shard ID's portion of the perturbation sweep.")
    parser.add_argument("--stage", default="all",
                        choices=["all", "train", "perturb", "merge", "estimate"])
    args = parser.parse_args()

    if args.train_one is not None:
        stage_train_one(args.train_one); return
    if args.perturb_one is not None:
        stage_perturb_one(args.perturb_one); return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_top = time.time()

    if args.stage in ("all", "train"):
        models_dir = OUTPUT_DIR / "models_and_data"
        expected = [models_dir / "dataset.pt"] + [
            models_dir / f"model_additional_{q}.pt" for q in DATASET_QUANTITIES
        ]
        if all(p.exists() for p in expected):
            print(f"[train] reusing checkpoints in {models_dir}")
        else:
            t0 = time.time()
            stage_train_orchestrator()
            print(f"[train] elapsed {time.time()-t0:.1f}s")

    if args.stage in ("all", "perturb"):
        t0 = time.time()
        stage_perturb_orchestrator()
        print(f"[perturb] elapsed {time.time()-t0:.1f}s")

    if args.stage in ("all", "merge"):
        stage_merge()

    if args.stage in ("all", "estimate"):
        stage_estimate()

    print(f"\n=== overnight_run complete in {time.time()-t_top:.1f}s ===")


if __name__ == "__main__":
    main()
