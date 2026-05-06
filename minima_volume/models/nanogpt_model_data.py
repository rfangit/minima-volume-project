"""
Model + dataset definition for nanoGPT (character-level shakespeare).

Same hook surface as the other model_data modules so the training and
volume-estimation notebooks can swap between datasets without changes:

    get_dataset(device)                  -> x_base, y_base, x_test, y_test
    get_model(device, seed, **kwargs)    -> nn.Module (returns full-sequence logits)
    get_loss_fn()                        -> callable(logits, targets) -> scalar
    get_additional_metrics()             -> {'accs': callable}
    verify_model_results(...)            -> prints per-model train/test diagnostics

A "sample" here is one block_size-length window of token ids. x has shape
(N, block_size) of int64 token ids; y is the same window shifted by one
(the next-token target). prepare_datasets in the existing pipeline then
draws non-overlapping random subsets of these rows for the dataset-size
sweep -- so the "random sample from shakespeare" semantics is implicit
in the existing index-shuffling, no special handling needed here.

Run nanogpt/data/shakespeare_char/prepare.py once before calling
get_dataset (it produces train.bin / val.bin / meta.pkl).
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Vendored nanoGPT lives at <repo>/nanogpt/. This file is at
# <repo>/minima_volume/models/nanogpt_model_data.py, so go up two parents.
NANOGPT_DIR = Path(__file__).resolve().parents[2] / "nanogpt"
if str(NANOGPT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOGPT_DIR))

from model import GPT, GPTConfig  # noqa: E402  (vendored nanoGPT)


# ---- defaults match nanogpt/config/train_shakespeare_char.py (~10M params) ----
DATA_DIR = NANOGPT_DIR / "data" / "shakespeare_char"
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.0   # turned off so we can overfit cleanly to ~0 train loss
BIAS = False

# Pool size for x_base / x_test. prepare_datasets draws random subsets of
# size up to (base_data_size + max(dataset_quantities)) from the pool.
DEFAULT_TRAIN_WINDOWS = 5000
DEFAULT_TEST_WINDOWS = 500


# --------------------------
# Data loading
# --------------------------

def _sample_windows(bin_path: Path, n_windows: int, block_size: int, seed: int):
    """Sample n_windows random (overlapping) start offsets from a token bin file
    and return (x, y) with y = x shifted right by one position."""
    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    max_start = len(arr) - block_size - 1
    if n_windows > max_start + 1:
        raise ValueError(
            f"Requested {n_windows} windows from {bin_path} but only "
            f"{max_start + 1} valid starting positions exist."
        )
    rng = np.random.default_rng(seed)
    starts = rng.choice(max_start + 1, size=n_windows, replace=False)
    x = np.stack([np.asarray(arr[s : s + block_size], dtype=np.int64) for s in starts])
    y = np.stack([np.asarray(arr[s + 1 : s + 1 + block_size], dtype=np.int64) for s in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


def get_dataset(device,
                n_train_windows: int = DEFAULT_TRAIN_WINDOWS,
                n_test_windows: int = DEFAULT_TEST_WINDOWS,
                block_size: int = BLOCK_SIZE,
                data_dir=None,
                seed: int = 42):
    """Load shakespeare_char as fixed pools of (x, y) windows.

    Returns four tensors of shape (N, block_size) on `device`. The training
    notebook then passes `x_base`, `y_base` to prepare_datasets which slices
    out base + additional rows.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    if not train_bin.exists() or not val_bin.exists():
        raise FileNotFoundError(
            f"Missing {train_bin} or {val_bin}. "
            f"Run nanogpt/data/shakespeare_char/prepare.py first."
        )

    x_base, y_base = _sample_windows(train_bin, n_train_windows, block_size, seed)
    x_test, y_test = _sample_windows(val_bin, n_test_windows, block_size, seed + 1)
    return x_base.to(device), y_base.to(device), x_test.to(device), y_test.to(device)


# --------------------------
# Model
# --------------------------

class FullLogitsGPT(nn.Module):
    """Wraps nanoGPT's GPT so forward(idx) always returns logits at every
    position, shape (B, T, V). nanoGPT's own forward returns last-position-
    only when targets is None (an inference-time optimisation that breaks
    cross-entropy over the full sequence)."""

    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt

    def forward(self, idx):
        _, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok = self.gpt.transformer.wte(idx)
        pemb = self.gpt.transformer.wpe(pos)
        x = self.gpt.transformer.drop(tok + pemb)
        for block in self.gpt.transformer.h:
            x = block(x)
        x = self.gpt.transformer.ln_f(x)
        return self.gpt.lm_head(x)


def _load_vocab_size(data_dir: Path) -> int:
    meta_path = data_dir / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Need {meta_path} or explicit vocab_size; "
            f"run nanogpt/data/shakespeare_char/prepare.py first."
        )
    with open(meta_path, "rb") as f:
        return pickle.load(f)["vocab_size"]


def get_model(device="cpu",
              seed: int = 0,
              n_layer: int = N_LAYER,
              n_head: int = N_HEAD,
              n_embd: int = N_EMBD,
              dropout: float = DROPOUT,
              bias: bool = BIAS,
              block_size: int = BLOCK_SIZE,
              vocab_size: int = None,
              data_dir=None):
    """Build a nanoGPT GPT, wrapped in FullLogitsGPT, seeded for reproducibility.

    Note on weight tying: nanoGPT shares lm_head.weight with transformer.wte.weight
    via a post-init reassignment. PyTorch's named_parameters deduplicates by
    parameter id, so the tied weight appears under exactly one name
    ("transformer.wte.weight") and will be perturbed once -- which is what we
    want for volume estimation."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    if vocab_size is None:
        vocab_size = _load_vocab_size(data_dir)

    torch.manual_seed(seed)
    cfg = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )
    gpt = GPT(cfg)
    return FullLogitsGPT(gpt).to(device)


# --------------------------
# Loss + metrics
# --------------------------

def loss_fn(logits, targets):
    """logits (B, T, V) and targets (B, T) -> scalar token-level cross entropy."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def get_loss_fn():
    return loss_fn


def accuracy_fn(logits, targets):
    """Fraction of next-token positions where argmax matches the target."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def get_additional_metrics():
    return {"accs": accuracy_fn}


# --------------------------
# Verification
# --------------------------

def verify_model_results(all_models,
                         x_base_train, y_base_train,
                         x_additional, y_additional,
                         x_test, y_test,
                         dataset_quantities,
                         dataset_type,
                         eval_windows: int = 16):
    """Print per-model train/test loss + token accuracy on a small eval slice.
    Mirrors the role of the verify_model_results in the image model_data
    modules but skips the per-image plotting -- token-level visualisation
    isn't useful and full corpus eval would be slow."""
    if dataset_type != "data":
        print(f"[nanogpt verify] dataset_type={dataset_type!r}: only 'data' "
              f"is meaningful for language modeling; skipping deeper checks.")
    print(f"=== nanoGPT verify ({dataset_type}) ===")
    device = next(all_models[0]["model"].parameters()).device

    for additional_data, model_data in zip(dataset_quantities, all_models):
        model = model_data["model"].eval()
        x_train_full = torch.cat([x_base_train, x_additional[:additional_data]], dim=0)
        y_train_full = torch.cat([y_base_train, y_additional[:additional_data]], dim=0)

        n = min(eval_windows, len(x_train_full), len(x_test))
        with torch.no_grad():
            xt = x_train_full[:n].to(device)
            yt = y_train_full[:n].to(device)
            xv = x_test[:n].to(device)
            yv = y_test[:n].to(device)
            train_logits = model(xt)
            test_logits = model(xv)
            tr_loss = loss_fn(train_logits, yt).item()
            te_loss = loss_fn(test_logits, yv).item()
            tr_acc = accuracy_fn(train_logits, yt)
            te_acc = accuracy_fn(test_logits, yv)
        print(f"  model_additional={additional_data:>6}: "
              f"train loss/acc = {tr_loss:.4f}/{tr_acc:.4f} | "
              f"test  loss/acc = {te_loss:.4f}/{te_acc:.4f}")
