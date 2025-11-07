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

## 📊 Output File Structure

Each notebook produces files with their own structure which we describe here. 

**Train Low Test Models.ipynb** saves models and datasets typically in a folder called 'models_and_data', organized as follows

```
models_and_data/
│
├── dataset.pt
├── model_additional_0.pt
├── model_additional_10.pt
├── model_additional_50.pt
└── ...
```

**dataset.pt** is a version of the training dataset that tracks the data variant used (eg, poisoned data, or the specific additional data).
This is used to recreate the datasets used to train models and evaluate their volumes in that loss landscape.

| Key                            | Description                                                                   |
| ------------------------------ | ----------------------------------------------------------------------------- |
| `x_base_train`, `y_base_train` | Base training set used for initial training.                                  |
| `x_additional`, `y_additional` | Additional data added later (e.g., perturbation augmentation).                |
| `x_test`, `y_test`             | Final held-out test set.                                                      |
| `dataset_quantities`           | Count metadata (e.g., number of added samples).                               |
| `dataset_type`                 | String describing how the dataset was generated (e.g., `"poison"`, `"data"`). |

**model_additional_x.pt** contains the model trained on x additional data samples and useful metadata.

| Key                       | Description                                                        |
| ------------------------- | ------------------------------------------------------------------ |
| `state_dict`              | Model weights (compatible with the same model class).              |
| `train_loss`, `test_loss` | Lists of loss values across epochs.                                |
| `train_accs`, `test_accs` | Accuracy curves across training.                                   |
| `additional_data`         | Number of extra training samples used for this model.              |
| `dataset_type`            | Dataset configuration this model was trained on.                   |
| `model_class`             | Name of the model class (for reference when reloading).            |
| additional kwargs         | Some experiments have additional kwargs, like training epoch count |

Calling 'load_models_and_data' loads these in and also recreates the exact model from the weights.

**Random Perturbs.ipynb** loads the previous models and dataset, and generates perturbations to model weights.
It records the loss values along these perturbations and saves the results in a folder describing the loss landscape.

```
<base_output_dir>/
└── <dataset_type>_<additional_data_used>/
      ├── <dataset_type>_<model_trained_data>.npz
      ├── <dataset_type>_<model_trained_data>.npz
      └── ...
```

Eg, data_0 refers to the loss landscape of the dataset without any additional data. It contains files such as

```
data_0/
  ├── data_0.npz      ← model trained on 0 extra samples, evaluated on 0 extra example landscape
  ├── data_540.npz    ← model trained on 540 extra samples, evaluated on 0 extra example landscape
  └── data_5940.npz   ← model trained on 5940 extra samples, evaluated on 0 extra example landscape
```

Each .npz file contains wiggle results, a list of dictionaries of the form

| Key                 | Description                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| `loss`              | Loss values along the perturbation coefficients (one value per coefficient).                        |
| `coefficients`      | The coefficients used to scale the perturbation direction (0 corresponds to the unperturbed model). |
| `accs`              | Accuracy along the same coefficients (not used in main analysis).                                   |
| `perturbation_seed` | Random seed used to generate the perturbation direction.                                            |
| `perturbation_norm` | Norm of the perturbation vector.                                                                    |

There are num_direction such dictionaries in the list.
The .npz file also contains in addition to this list of dictionaries of resutls for each perturbation, metadata of the form

| Key                      | Description                                                           |
| ------------------------ | --------------------------------------------------------------------- |
| `additional_data`        | Number of extra samples used when *evaluating* the landscape.         |
| `model_trained_data`     | Number of extra samples the model was originally trained with.        |
| `dataset_type`           | Dataset variant label, e.g., `"data"`, `"noise"`, `"poison"`.         |
| `base_dataset_size`      | Size of the base dataset used during evaluation.                      |
| `test_loss`, `test_accs` | Final test-set performance metrics of the trained model.              |
| `num_params`             | Number of parameters affected by perturbations.                       |
| `state_dict`             | The full model weights (allows exact reproduction of the landscapes). |

To estimate volumes, you need to go through every dictionary in the list wiggle results, and check for the coefficient value that reaches the loss threshold.

**Volume Estimation Pipeline** loads in the various .npz files made from Random Perturbs, and finds the coefficients at which the loss exceeds a threshold.
On top of plots for the experiment and a histogram (which can be made in more detail with code in the visualization folder at the root), it saves a results.json
This results.json is used to plot averages over many seeds. It contains

#### **Primary Data (used in downstream analysis)**

| Key                                   | Description                                                                                                                       |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `test_loss_values`                    | Test loss of each trained model. Used for comparing generalization across training conditions.                                    |
| `log_exp_r_n_values`                  | Estimated log-volume of each minima (`log(r^n)`), where `n` is the number of model parameters. This is the primary volume metric. |
| `log_exp_r_n_acc_values` *(optional)* | Same volume estimate but using accuracy threshold instead of loss threshold. Only present if `acc_threshold` was specified.       |
| `model_data_levels`                   | The number of additional training samples for each model. Serves as the x-axis in volume plots.                                   |
| `loss_landscape_data_param`           | Number of extra samples used when *evaluating* the landscape (not when training).                                                 |
| `train_loss_values`                   | Final train-loss of each model.                                                                                                   |
| `num_params`                          | Number of parameters that were perturbed during the volume estimation.                                                            |
| `dataset_type`                        | The dataset variant used (e.g., `"data"`, `"noise"`, `"poison"`).                                                                 |
| `top_r_vals`                          | Largest perturbation radii for each model (shape: models × top_k). Captures the most volume-enlarging directions.                 |
| `top_r_val_seeds`                     | The random seeds corresponding to the top perturbation directions — allows reproduction of the extreme perturbations.             |

---

#### **Secondary Evaluation Outputs (not used in final plots, included for diagnostics)**

| Key                                         | Meaning / Purpose                                                                                             |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `loss_predicts_minimas`                     | Boolean indicating whether ordering minima by volume matches ordering by training data (expected: True).      |
| `loss_ordering_violations`                  | Number of times the ordering of `log(r^n)` disagrees with the ordering of test loss — measures inconsistency. |
| `test_loss_ranks`                           | Rank ordering of models by test loss.                                                                         |
| `log_exp_r_n_ranks`                         | Rank ordering of models by volume.                                                                            |
| `data_level_ranks`                          | Rank ordering of models by training data amount.                                                              |
| `accuracy_predicts_minimas` *(optional)*    | Same as `loss_predicts_minimas` but using accuracy-based radii.                                               |
| `accuracy_ordering_violations` *(optional)* | Same as `loss_ordering_violations` but accuracy-based.                                                        |
| `log_exp_r_n_acc_ranks` *(optional)*        | Rank ordering of volume according to accuracy thresholding.                                                   |

There is a notebook for plotting results from multiple seeds called **analyze_experiments_data.ipynb**. This primarily uses 'test_loss_values', 'log_exp_r_n_values',
'model_data_levels' and 'loss_landscape_data_param'.

The remaining notebooks, **Volume Cutoff.ipynb** and **Test Accuracy.ipynb** generate results in the folder 'cutoffs' or directly generate **test_accuracies.npz**.
These help analyze_experiments_data make better looking plots. They are both fairly self explanatory.

Cutoffs contains a dictionary, where the amount of data the model was trained on serves as a key for an entry that contains the additional data it was trained on,
the base dataset size, and the loss values as more data is added. Test accuracies denotes how much data a model was trained on, and it's corresponding accuracy.

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