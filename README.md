# Minima Volume Experiments

This repository contains code and experiments for analyzing **loss landscape minima volumes** under different training conditions (low data, poisoning, SAM, class imbalance, etc.).  
The core logic lives in the `minima_volume` package, while experiment-specific code is organized in dedicated folders.

---

## Repository Structure

### Core
- **`minima_volume/`**  
  Core package containing main functions, utilities, and analysis code.

### Experiment Folders
- **`CIFAR10/`** – Experiments on CIFAR-10 dataset.  
- **`MNIST/`** – Experiments on MNIST (standard, CNN, SAM, etc.).  
- **`modulo_arithmetic/`** – Synthetic modulo arithmetic experiments.  
- **`sam/`** – Sharpness-Aware Minimization (SAM) experiments.  
- **`swiss_roll/`** – Swiss roll experiments for geometric visualization.  
- **`imbalanced_classes/`** – Experiments with artificially imbalanced class distributions.  

### Supporting / Utility Folders
- **`figs/`** – Figures for the paper.  
- **`videos/`** – Code for generating videos of experiments.  
- **`toy_models/`** – Code for generating simple visual diagrams (not full experiments).  
- **`tests/`** – Legacy testing code (may be outdated).  
- **`to_propagate/`** – Placeholder for revisions to common scripts (not currently in use).  

---

## How to Use

Each **experiment folder** follows the same structure:

### 1. Template Folder
- Contains Jupyter notebooks that serve as **experiment templates**.  
- These allow you to:
  - Copy and duplicate notebooks for new runs.  
  - Swap hyperparameters and parameters.  
  - Launch multiple experiments in sequence.  
- This setup is somewhat ad-hoc but enables quick iteration and scaling.

### 2. Base Folder
The **base** folder contains the **important notebooks** for the experiment.  
These represent the core experimental loop:

- **Training models with different data levels**  
  - Imports models and datasets from `minima_volume`.  
  - Trains models with varying dataset sizes.  
  - Saves trained models alongside the dataset used.  

- **Random perturbations**  
  - Applies fixed random perturbations to model parameters.  
  - Evaluates loss changes along those directions.  
  - Purely random directions — no binary search or optimization.  

- **Loss threshold evaluation**  
  - Evaluates trained models on the dataset.  
  - Determines when loss grows too large (usually right after unseen data).  

- **Volume estimation**  
  - Uses perturbation results to estimate when perturbations cross a cutoff.  
  - Collects radii and estimates approximate minima volumes.  

### Running New Experiments
To run a new experiment:  
1. **Select a template** from the template folder.  
2. **Modify the notebooks** in the base folder:  
   - Set the model architecture in:
     - `train_low_test_models`  
     - `random_perturbs`  
     - `volume_cutoff`  
   - Pass in the dataset to `train_low_test_models`.  
3. **Run training → perturbations → cutoff evaluation → volume estimation**.  
4. Results (models, perturbations, radii, figures) will be saved in the respective experiment folder.  

---

## Experiment Categories & Progress

The experiments are grouped into four categories. Each may include multiple architectures (x5, x6, CNN, Large).

### Poison Experiments
- **Swiss Poison** *(Update)*  
  - x5 *(Done)*  
  - x6 *(Done)*  
- **MNIST Poison** *(Done — remember to change model name!)*  
- **CIFAR Poison** *(Done — 100 poisons didn’t ruin it)*  

### Low Data Experiments
- **Swiss Data**  
  - x5 *(Done)*  
  - x6 *(Done)*  
- **MNIST Data**  
  - Base *(Done)*  
  - Large *(Done)*  
- **MNIST CNN Data** *(relationship is much more mild)*  
- **CIFAR Data**  
  - Base *(Done — only one run)*  
- **Modulo Arithmetic Data**  
  - Base *(Done — Update to compare loss landscape plots)*  

### SAM Experiments
- **MNIST SAM Data** *(Done)*  
- **Swiss SAM** *(Done)*  
- **Modulo Arithmetic Grokking** *(Done — needs updates)*  

### Class Imbalance Experiments
- **Class Imbalance MNIST**  
- **Class Imbalance CIFAR**  

---

## Figures & Videos

- All **figures** generated during experiments are in [`figs/`](figs/).  
- **Videos** of experiment dynamics can be generated via scripts in [`videos/`](videos/).  

---

## Notes

- Some outdated code stored **models and datasets directly inside experiment folders**.  
  - These have been updated to import from `minima_volume`.  
- For reproducibility, seeds are logged for each run.  

---

## License

[MIT License](LICENSE)
