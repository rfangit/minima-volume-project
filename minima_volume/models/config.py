# config.py
from pathlib import Path
import os

# Root project directory (the folder containing config.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# ======================
# Dataset Directories
# ======================
DATASETS_ROOT = PROJECT_ROOT / "datasets"

DATA_DIRS = {
    "MNIST": DATASETS_ROOT / "MNIST",
    "FASHION_MNIST": DATASETS_ROOT / "FashionMNIST",
    "CIFAR10": DATASETS_ROOT / "CIFAR10",
    "SVHN": DATASETS_ROOT / "SVHN",
}
