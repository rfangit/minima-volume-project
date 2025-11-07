# Experiments

This directory contains all experiment pipelines used to generate the results in the paper.
Each experiment folder corresponds to a **dataset or training regime** (e.g., MNIST, CIFAR10, SVHN, SAM, poisoning, class imbalance, etc.).
All experiments follow the **same volume estimation workflow**, based on training models, generating perturbations, and measuring loss thresholds.

The core logic for models, datasets, and perturbation routines lives in the **`minima_volume`** package, which all notebooks import from.

---

## 🧱 Standard Experiment Workflow

Each experiment ultimately consists of running the following **four main notebooks** inside an **individual experiment run folder** (usually named something like `model_<seed>_data_<seed>`):

| Notebook                             | Purpose                                                                                                                                                         |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Train Low Test Models.ipynb**      | Trains one or more models on a dataset configuration (e.g., low-data regime, poison ratio, architecture choice). Saves the trained weights + dataset splits.    |
| **Random Perturbs.ipynb**            | Generates random perturbation directions in parameter space and evaluates the loss along those directions. No optimization—these are *fixed random* directions. |
| **Volume Cutoff.ipynb**              | Determines the **loss threshold** beyond which a perturbation is considered to have “exited” the minimum region.                                                |
| **Volume Estimation Pipeline.ipynb** | Uses the perturbation trajectories + cutoff estimates to compute an approximate **minimum volume** for each trained model.                                      |

Additionally:

| Notebook                | Purpose                                                                                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Test Accuracy.ipynb** | Computes and visualizes test accuracy across training configurations (e.g., dataset size, poison fraction). Useful for comparing volume vs. performance. |

---

## 📦 Folder Layout Pattern (Inside Each Dataset Folder)

```
experiments/
└── CIFAR10/                  # Example dataset experiment folder
    ├── template/             # Base notebooks for creating new experiment runs
    ├── low_data/             # (Example) experiment category
    │   ├── base/             # Model architecture or training baseline
    │   │   ├── model_0_data_0/
    │   │   │   ├── Train Low Test Models.ipynb
    │   │   │   ├── Random Perturbs.ipynb
    │   │   │   ├── Volume Cutoff.ipynb
    │   │   │   ├── Volume Estimation Pipeline.ipynb
    │   │   │   └── Test Accuracy.ipynb
    │   │   └── model_1_data_2/
    │   │       └── ...
    │   └── large/
    │       └── ...
    └── poison/
        └── ...
```

### Naming Conventions

* **Top-level experiment categories** = training condition (e.g., `low_data/`, `poison/`, `sam/`, etc.)
* **Subfolders** = different **architectures** or experiment variations (e.g., `base/`, `large/`)
* **Run folders** = individual model/data seed trials (e.g., `model_0_data_2/`)

The exact meaning of folder names varies slightly by experiment, but the structure is consistent:
→ *Each run folder contains the 4 core notebooks listed above.*

---

## 🏗 Creating a New Experiment

1. **Start from the `template/` folder** in your dataset.
2. Copy the template notebooks into a new experiment subfolder matching your setup.
3. In:

   * `Train Low Test Models.ipynb` → select model + dataset subset or corruption regime.
   * `Random Perturbs.ipynb` and `Volume Cutoff.ipynb` → reference your trained model checkpoint.
4. Run the notebooks in order:

```
Train Low Test Models → Random Perturbs → Volume Cutoff → Volume Estimation Pipeline
```

5. (Optional) Use `Test Accuracy.ipynb` to analyze dataset-size vs. accuracy trends.

---

## 📊 Output File Structure *(to be documented later)*

This section will describe:

* How trained models are stored
* How perturbations are logged
* Format of cutoff data
* Volume estimation summary outputs

> **TODO — to be filled in later.**

---

## ⚙️ Batch + Automation Utilities (Optional / Advanced)

Most experiment folders include helper notebooks/scripts for **running many experiments at once**, such as:

| Script / Notebook                    | Purpose                                                            |
| ------------------------------------ | ------------------------------------------------------------------ |
| `full_pipeline.ipynb`                | Runs the entire workflow end-to-end in sequence.                   |
| `make_dataset_notebooks.ipynb`       | Clones and configures run folders for multiple seeds / datasets.   |
| `cleanup_files.ipynb`                | Removes large cached files to reclaim space.                       |
| `propagate_file.ipynb`               | Copies a single file to all run directories (e.g., patch updates). |
| `Run Random Perturbs Parallel.ipynb` | Parallelizes the perturbation step across runs.                    |
| `Run Volume Parallel.ipynb`          | Parallelizes volume estimation.                                    |

These tools are not required for small-scale experimentation, but they are helpful when scaling to dozens or hundreds of runs.