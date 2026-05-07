"""
Spawn multiple parameterized overnight_run.py invocations in parallel.

Each entry in SEED_CONFIGS describes one (model_seed, data_seed) run pinned
to a contiguous GPU range. The orchestrator just sets `OVN_*` env vars and
launches `overnight_run.py` per entry. After all seeds finish, it runs
`compute_cutoffs.py` against each seed dir.

Resumable in the same way as overnight_run.py: rerunning skips training
when checkpoints exist; perturb/merge/estimate are idempotent.

Run from repo root:
    .venv/bin/python experiments/shakespeare_char/multiseed_run.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPT_ROOT = HERE / "cosine_run"  # new dir; preserves the constant-LR overnight_run/
OVERNIGHT = HERE / "overnight_run.py"
COMPUTE_CUTOFFS = HERE / "compute_cutoffs.py"

# 3 cosine seeds in parallel, 2 GPUs each on (2-3, 4-5, 6-7). Cosine-LR
# decay 1e-3 → 1e-5 over 800 epochs to escape the constant-LR SGD noise
# floor. Reduced (50 dirs, 50 coeffs) to fit the deadline; per-seed
# wall ~4.15hr (2.2 train + 1.94 perturb).
_COSINE_ENV = {
    "OVN_LR": "1e-3",
    "OVN_COSINE_MIN_LR": "1e-5",
    "OVN_EPOCHS": "800",
    "OVN_N_COEFFS": "50",
}
SEED_CONFIGS = [
    {
        "label": "seed1",
        "model_seed": 1, "data_seed": 1,
        "num_directions": 50, "perturbation_seed_base": 1,
        "gpu_offset": 2, "num_train_gpus": 2, "num_perturb_gpus": 2,
        "output_dir": EXPT_ROOT / "model_1_data_1",
    },
    {
        "label": "seed2",
        "model_seed": 2, "data_seed": 2,
        "num_directions": 50, "perturbation_seed_base": 51,
        "gpu_offset": 4, "num_train_gpus": 2, "num_perturb_gpus": 2,
        "output_dir": EXPT_ROOT / "model_2_data_2",
    },
    {
        "label": "seed3",
        "model_seed": 3, "data_seed": 3,
        "num_directions": 50, "perturbation_seed_base": 101,
        "gpu_offset": 6, "num_train_gpus": 2, "num_perturb_gpus": 2,
        "output_dir": EXPT_ROOT / "model_3_data_3",
    },
]


def _spawn(cfg):
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(_COSINE_ENV)
    env.update({
        "OVN_DATA_SEED": str(cfg["data_seed"]),
        "OVN_MODEL_SEED": str(cfg["model_seed"]),
        "OVN_NUM_DIRECTIONS": str(cfg["num_directions"]),
        "OVN_PERTURBATION_SEED_BASE": str(cfg["perturbation_seed_base"]),
        "OVN_NUM_TRAIN_GPUS": str(cfg["num_train_gpus"]),
        "OVN_NUM_PERTURB_GPUS": str(cfg["num_perturb_gpus"]),
        "OVN_GPU_OFFSET": str(cfg["gpu_offset"]),
        "OVN_OUTPUT_DIR": str(cfg["output_dir"]),
    })
    log_path = cfg["output_dir"] / "orchestrator.log"
    log = open(log_path, "w")
    p = subprocess.Popen(
        [sys.executable, str(OVERNIGHT)],
        env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"[multiseed] {cfg['label']} pid={p.pid} GPUs "
          f"{cfg['gpu_offset']}..{cfg['gpu_offset']+cfg['num_train_gpus']-1}  "
          f"log={log_path}")
    return p, log


def main():
    t0 = time.time()
    procs = []
    for cfg in SEED_CONFIGS:
        procs.append((cfg, *_spawn(cfg)))

    failed = []
    for cfg, p, log in procs:
        rc = p.wait()
        log.close()
        print(f"[multiseed] {cfg['label']} exit={rc}")
        if rc != 0:
            failed.append(cfg["label"])
    print(f"[multiseed] training+perturb+merge+estimate elapsed "
          f"{time.time()-t0:.1f}s; failed={failed}")
    if failed:
        sys.exit(1)

    # Cutoffs for each seed (cheap; sequential on a single GPU is fine).
    for cfg in SEED_CONFIGS:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_offset"])
        log_path = cfg["output_dir"] / "cutoffs.log"
        with open(log_path, "w") as log:
            rc = subprocess.call(
                [sys.executable, str(COMPUTE_CUTOFFS),
                 "--seed-dir", str(cfg["output_dir"])],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        print(f"[multiseed] {cfg['label']} cutoffs exit={rc}  log={log_path}")

    print(f"\n=== multiseed_run complete in {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
