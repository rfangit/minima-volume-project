# shakespeare_char (nanoGPT)

Volume-vs-data-size sweep for character-level shakespeare with the vendored
nanoGPT (`<repo>/nanogpt/`). Same scientific question as the
`MNIST/low data MLP` and `CIFAR10/low data` experiments; the only
nanoGPT-specific code is in `minima_volume/models/nanogpt_model_data.py`.

## Setup (once)

From the repo root:

```
python nanogpt/data/shakespeare_char/prepare.py
```

This produces `nanogpt/data/shakespeare_char/{train.bin,val.bin,meta.pkl}`,
which `nanogpt_model_data.py` loads.

## Single-run workflow

Templates live in `base folder/`. To run one (model_seed, data_seed) pair:

```
cd "experiments/shakespeare_char/base folder"
python train_models.py        # produces models_and_data/
python random_perturbs.py     # produces data_<N>/data_<M>.npz files
python volume_estimation.py   # produces data_<N>/loss_<T>/results.json + plots
```

These scripts are direct ports of the corresponding notebooks in
`experiments/CIFAR10/template - CNN/base folder/`. The interface
(`prepare_datasets`, `analyze_wiggles_metrics_large`,
`analyze_and_plot_model_landscape`) is unchanged; only the model_data
module differs.

## Multi-seed / multi-GPU

Not wired up yet. Same pattern as the existing experiments would apply
(clone `base folder` per seed via `make_dataset_notebooks.ipynb`-style
helper, then shard direction seeds across processes for
`random_perturbs.py`). The 8x H100 layout will likely shard
`perturbation_seed` ranges and merge `.npz` files before
`volume_estimation.py`.

## Notes

- A "sample" is one `block_size`-length window of token ids (default 256).
  `dataset_quantities=[0, 150, 450, 950, 2950]` on top of `base_data_size=50`
  spans roughly 12.8K -> 768K training tokens.
- Models train to ~0 train loss (heavily overparameterized: 10M params vs
  at most ~3000 fixed windows). Default `epochs=2000` with constant lr=1e-3.
  Bump if you see train loss stuck above ~1e-3.
- nanoGPT ties `lm_head.weight = transformer.wte.weight`. PyTorch's
  `named_parameters` deduplicates by id, so the tied weight is perturbed
  exactly once during volume estimation -- which is what we want.
