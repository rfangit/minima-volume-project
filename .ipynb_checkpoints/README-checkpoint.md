# 🎢 Minima Volume Experiments

⚠️ **Note:** The **models** and **data folders** are missing, as are the **random perturbations results**.  
Only the resulting **volumes** from volume estimation are present.  
You can still run `analyze_experiments_data.ipynb` to explore results.

---

This repository contains code and experiments for analyzing **loss landscape minima volumes** under different training conditions (low data, poisoning, SAM, class imbalance, etc.).  
The core logic lives in the `minima_volume` package, while experiment-specific code is organized in dedicated folders.

---

## 📂 Repository Structure

### 🔧 Core
- **`minima_volume/`** – Core package containing main functions, utilities, and analysis code.

### 🧪 Experiment Folders
- **`CIFAR10/`** – Experiments on CIFAR-10 dataset.  
- **`MNIST/`** – Experiments on MNIST (standard, CNN, SAM, etc.).  
- **`modulo_arithmetic/`** – Synthetic modulo arithmetic experiments.  
- **`sam/`** – Sharpness-Aware Minimization (SAM) experiments.  
- **`swiss_roll/`** – Swiss roll experiments for geometric visualization.  
- **`imbalanced_classes/`** – Experiments with artificially imbalanced class distributions.  

### 📎 Supporting / Utility Folders
- **`figs/`** – Figures for the paper.  
- **`videos/`** – Code for generating videos of experiments.  
- **`toy_models/`** – Code for generating simple visual diagrams (not full experiments).  
- **`tests/`** – Legacy testing code (may be outdated).  
- **`to_propagate/`** – Placeholder for revisions to common scripts (not currently in use).  

---

## 🚀 How to Use

Each **experiment folder** follows the same structure:

### 1️⃣ Template Folder
- Contains Jupyter notebooks that serve as **experiment templates**.  
- These allow you to:
  - 📑 Copy and duplicate notebooks for new runs.  
  - 🔄 Swap hyperparameters and parameters.  
  - 🏃 Launch multiple experiments in sequence.  

👉 This setup is somewhat ad-hoc but enables quick iteration and scaling.

---

### 2️⃣ Base Folder
The **base** folder contains the **important notebooks** for the experimental loop:

- **📘 Train Low Test Models.ipynb**  
  - Imports models and datasets from `minima_volume`.  
  - Trains models with varying dataset sizes.  
  - Saves trained models alongside the dataset used.  

- **📘 Random Perturbs.ipynb**  
  - Applies fixed random perturbations to model parameters.  
  - Evaluates loss changes along those directions.  
  - Purely random directions — no binary search or optimization.  

- **📘 Volume Cutoff.ipynb**  
  - Evaluates trained models on the dataset.  
  - Determines when loss grows too large (often after encountering unseen data).  

- **📘 Volume Estimation Pipeline.ipynb**  
  - Uses perturbation results to estimate when perturbations cross a cutoff.  
  - Collects radii and computes approximate minima volumes.  

---

### 🏗 Running New Experiments
To run a new experiment:  

1. **Select a template** from the template folder.  
2. **Modify the notebooks** in the base folder:  
   - Set the model architecture in:
     - `Train Low Test Models.ipynb`  
     - `Random Perturbs.ipynb`  
     - `Volume Cutoff.ipynb`  
   - Pass in the dataset to `Train Low Test Models.ipynb`.  
3. **Run in sequence:**  
   - 🟢 Training →  
   - 🔵 Perturbations →  
   - 🟡 Cutoff evaluation →  
   - 🟣 Volume estimation  
4. Results (models, perturbations, radii, figures) will be saved in the respective experiment folder.  

---

## 📊 Experiment Categories & Progress

✅ = Done 🔄 = Needs update ⬜ = Not started  

### 🧨 Poison Experiments
- **Swiss Poison** 🔄 *(Update)*  
  - x5 ✅  
  - x6 ✅  
- **MNIST Poison** ✅ 
- **CIFAR Poison** ✅ *(Volume actually increased at 100 poison...)*  

### 📉 Low Data Experiments
- **Swiss Data**  
  - x5 ✅  
  - x6 ✅  
- **MNIST Data**  
  - Base ✅  
  - Large ✅  
- **MNIST CNN Data** 🔄 *(relationship is more mild than MLP)*  
- **CIFAR Data**  
  - Base ✅ 
- **Modulo Arithmetic Data** 🔄 *(compare grokking later)*
  - Base ✅
  - High Epoch ✅


### ⚡ SAM Experiments
- **MNIST SAM Data** ✅  
- **Swiss SAM Data** ✅

### ⚡ Grokking Experiments
- Not analyzed yet

### ⚖️ Class Imbalance Experiments
- **Class Imbalance MNIST** ⬜  
- **Class Imbalance CIFAR** ⬜  

---

## 🖼 Figures & 🎥 Videos

- All **figures** are in [`figs/`](figs/) for use in the paper.  
- **Videos** can be generated via scripts in [`videos/`](videos/).  

---

## 📝 Notes

- 🕰 Some outdated code stored **models/datasets directly inside experiment folders**.  
  - These have been updated to import from `minima_volume`.  
- 🎲 Random seeds are logged for reproducibility.  

---

## 📜 License

[MIT License](LICENSE)
