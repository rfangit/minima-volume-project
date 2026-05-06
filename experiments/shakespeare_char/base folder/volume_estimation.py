"""
Stage 3: turn the per-direction loss curves saved by random_perturbs.py
into log-volumes and a results.json per dataset-size landscape.

Mirrors experiments/CIFAR10/template - CNN/base folder/Volume Estimation Pipeline.ipynb.
Cheap to re-run with different thresholds -- the curves on disk don't change.
"""
import os

from minima_volume.volume_funcs import analyze_and_plot_model_landscape

# ============================================================================
# Configuration
# ============================================================================
# Paper used loss_threshold=0.1 for image classification (training reaches
# ~0). nanoGPT char-shakespeare overfit train loss can also approach 0,
# but volumes are sensitive: start with a few candidates and pick after
# inspecting curves.
loss_thresholds = [0.1, 0.05, 0.02]
# Char-level next-token accuracy at memorisation: near 1.0. Pick thresholds
# below the trained accuracy floor.
accuracy_thresholds = [0.95, 0.9, 0.85]

display_options = {"loss_plots": False, "accuracy_plots": False}


def main():
    if len(loss_thresholds) != len(accuracy_thresholds):
        raise ValueError("loss_thresholds and accuracy_thresholds must align.")

    valid_starts = ("data", "poison", "noise")
    directories_to_analyze = [
        d for d in os.listdir(".")
        if os.path.isdir(d) and d.startswith(valid_starts)
    ]
    print("Directories to analyze:", directories_to_analyze)

    for directory in directories_to_analyze:
        print(f"\n{'=' * 60}\n{directory}\n{'=' * 60}")
        for loss_th, acc_th in zip(loss_thresholds, accuracy_thresholds):
            print(f"\n-- loss={loss_th}  acc={acc_th} --")
            save_paths = {
                "log_volume": f"{directory}/loss_{loss_th}/log_volume.png",
                "log_volume_generalization": f"{directory}/loss_{loss_th}/gen_log_volume.png",
                "model_modification_vs_test_loss": f"{directory}/loss_{loss_th}/test_vs_data.png",
                "average_radius_loss": f"{directory}/loss_{loss_th}/other_plots/avg_radius.png",
                "radius_histogram_loss": f"{directory}/loss_{loss_th}/other_plots/radius_hist.png",
                "average_radius_acc": f"{directory}/acc_{acc_th}/avg_radius.png",
                "radius_histogram_acc": f"{directory}/acc_{acc_th}/radius_hist.png",
                "log_volume_acc": f"{directory}/acc_{acc_th}/log_volume.png",
                "log_volume_generalization_acc": f"{directory}/acc_{acc_th}/gen_log_volume.png",
                "results.json": f"{directory}/loss_{loss_th}",
            }
            try:
                analyze_and_plot_model_landscape(
                    directory=directory,
                    loss_threshold=loss_th,
                    acc_threshold=acc_th,
                    verbose=True,
                    display_options=display_options,
                    save_paths_dict=save_paths,
                )
            except Exception as e:
                print(f"Error analyzing {directory} loss={loss_th} acc={acc_th}: {e}")
                continue


if __name__ == "__main__":
    main()
