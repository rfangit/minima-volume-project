# **Sharp Minima Can Generalize: A Loss Landscape Perspective on Data**

### *Minima Volume Project (Code & Experiments)*

**Paper:** Link (Currently N/A)

**Tutorial Colab:** [Link](https://colab.research.google.com/drive/1JNbk8Sau-M31mLVOQv19GR2dlwW7xwLd)

---

<p align="center">
  <img src="videos\combined_figure.png" width="600">
</p>

---

This repository contains code and experiments for the paper **“Sharp Minima Can Generalize: A Loss Landscape Perspective on Data.”**
It has tools to measure **the volume of loss landscape minima** in different loss landscapes (formed by different datasets).

In the paper we mainly study minima trained on large datasets, observing how their volumes behave in smaller datasets.
However, our code can also study the volumes of minima from poisoned datasets (as was done in past experiments) and recreate past results on the effects of batch size on flatness and generalization.

The main idea is to estimate **the volume of a minimum** by:

1. Training a model to reach a local minimum of the loss.
2. Generating **random perturbations** to the model parameters.
3. Measuring how far one can move in random directions before the loss exceeds a preset threshold.
4. Estimate **volume** using the distances.

---

## 🎓 Quick Start (Recommended)

For a simple minima volume experiment, we recommend starting with the [**interactive Colab tutorial**](https://colab.research.google.com/drive/1JNbk8Sau-M31mLVOQv19GR2dlwW7xwLd). 
The tutorial estimates the volumes on MNIST. Experiments in our code are scalable versions of the same workflow.

---

## 📦 What This Repository Contains

* Code to **train models** under controlled dataset manipulations
* Scripts to **apply random perturbations** and measure loss thresholds
* Tools to **estimate minima volumes** and analyze scaling trends
* Plotting utilities used to generate figures in the paper

**Note:**
This repository is missing:

* Final trained models
* Raw perturbation sweeps
* Full datasets

However, it includes **volume results** used for the figures in the paper, letting you recreate the plots.
If you wish to regenerate full experimental results, you will need to rerun the training and perturbation pipelines.

---

## 📂 Repository Structure

```
minima-volume-project/
│
├── experiments/             # All experiment pipelines (MNIST, CIFAR10, SAM, SVHN, etc.)
├── minima_volume/           # Core codebase: models, datasets, utilities, analysis logic
├── minima_volume.egg-info/  # Package
├── tests/                   # Misc. testing scripts (not actively maintained)
├── videos/                  # Tools for rendering loss landscape visualizations and animations
├── visualizations/          # Main paper figures + scripts for generating diagrams
│
├── pyproject.toml           # Build + dependency configuration
└── requirements.txt         # Python package requirements
```

### 🔧 Core Package

**`minima_volume/`**

This is the **main library** used across all experiments. It contains:

* Models + Datasets (MLP, CNNs for MNIST, CIFAR10, etc.)
* Dataset loading and preprocessing utilities
* Training utilities (standard + SAM)
* Random perturbations and volume estimation code
* Analysis and plotting helpers

All experiment folders import from here.

---

### 🧪 Experiments

**`experiments/`**

This folder contains **all experimental pipelines**. Each subfolder corresponds to a **training regime or dataset**

| Folder                | Description                                                           |
| --------------------- | --------------------------------------------------------------------- |
| `MNIST/`              | Standard MNIST experiments (MLP, CNN, low-data regimes, poisoning)    |
| `CIFAR10/`            | CIFAR-10 experiments (low-data, CNN, poisoning.)                      |
| `SVHN/`               | Experiments on street-view house numbers dataset                      |
| `modulo_arithmetic/`  | modulo arithmetic (low data, high epoch and grokking)                 |
| `swiss_roll/`         | swiss roll experiments                                                |
| `imbalanced_classes/` | Experiments with class imbalanced datasets                            |
| `sam/`                | Sharpness-Aware Minimization experiments                              |

> Each experiment subdirectory follows a **common workflow**:
> train model → evaluate perturbations → estimate volume via cutoffs.
> A dedicated README in `experiments/` explains this pipeline in detail.

---

### 🎥 Landscape Visualizations

**`videos/`**

Contains scripts for **rendering 2D / 3D visualizations** of the slices of the loss landscape.
Not really related to the main volume work, but generates nice visuals.

---

### 🖼 Figures and Diagrams

**`visualizations/`**

Includes:

* Final figures used in the paper
* Scripts for generating plots, summary graphs, and illustrative diagrams

---

### 🧪 Tests (Legacy)

**`tests/`**

Contains older verification scripts for internal functionality.
These are **not guaranteed to be up to date** and are not required to run experiments.

---

### 📦 Environment Configuration

* **`requirements.txt`** — Quickly install required dependencies.
* **`pyproject.toml`** — Allows package installation via `pip install -e .` for development.
