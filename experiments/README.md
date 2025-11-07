# Experiments

Directory for experiments in the paper. Each folder contains a dataset or training regime, and follows the same workflow (train model, generate perturbations, measure loss threshold volumes).

---

## 🧱 Experiment Workflow

Each experiment runs  **five notebooks** inside a folder (usually named something like `model_<seed>_data_<seed>`):

| Notebook                             | Purpose                                                                                                                                                      |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Train Low Test Models.ipynb**      | Trains models on a dataset configuration (e.g., low-data regime, poison ratio, architecture choice). Saves the trained weights + dataset splits.             |
| **Random Perturbs.ipynb**            | Generates random perturbation directions and evaluates the perturbed model loss along those directions for a dataset. Fixed seeds for random perturbations.  |
| **Volume Estimation Pipeline.ipynb** | Uses the loss along the perturbations + a user chosen cutoff to compute the  **minimum volume** for each trained model.                                      |
| **Volume Cutoff.ipynb**              | Determines the dataset size at which a minima has 0 volume, purely for nicer looking graphs. Results are saved to cutoffs and visualized in loss_curves.png  |                                             |
| **Test Accuracy.ipynb**              | Obtains the test accuracy of models and saves it in test_accuracies.npz. Used only for nicer looking graphs.                                                 |

Only the first three are important, the last two are for nicer visuals and plots.

---

## Folder Naming Conventions

* **Top-level experiment categories** = training condition (e.g., `low_data/`, `poison/`, `sam/`, etc.) + a template for making new experiments
* **Subfolders** = different **architectures** or experiment variations (e.g., `base/`, `large/`), doesn't always exist
* **Run folders** = individual model/data seed trials (e.g., `model_0_data_2/`) and the analysis folder, containing the final plots.

The meaning of folder names varies slightly by experiment, but all run folders contain the 5 core notebooks.

---

## 🏗 Creating New Experiments

1. If the model + dataset are not supported, you need to modify the 5 notebooks to use the architecture and dataset required. Else, start from a **`template/` folder**.
2. Copy the template notebooks into a new experiment subfolder matching your setup.
3. Run the five notebooks in sequence.

```
Train Low Test Models → Random Perturbs → Volume Estimation Pipeline → Volume Cutoff → Test Accuracy
```

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

Experiment folders also include helper notebooks/scripts for **running many experiments at once**, such as:

| Script / Notebook                    | Purpose                                                               |
| ------------------------------------ | --------------------------------------------------------------------- |
| `full_pipeline.ipynb`                | Runs the entire workflow end-to-end in sequence.                      |
| `make_dataset_notebooks.ipynb`       | Clones and run folders for multiple seeds / datasets.                 |
| `cleanup_files.ipynb`                | Removes files in all experiments (eg, cleaning up datasets for space) |
| `propagate_file.ipynb`               | Copies a single file to all run directories.                          |
| `Run Random Perturbs Parallel.ipynb` | Parallelizes random perturbs across runs.                             |
| `Run Volume Parallel.ipynb`          | Parallelizes volume estimation, test accuracy, volume cutoffs         |

These tools are not required for small-scale experimentation, but they are helpful when scaling to dozens or hundreds of runs.