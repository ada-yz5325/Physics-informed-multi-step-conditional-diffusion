#!/usr/bin/env python3

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "results/final_figures"


INC_LOGS = {
    "SS baseline":
        "runs/2D_Inc/"
        "128_acdm-r20_puv_phys_ablation_diffonly_10h_00/"
        "log.txt",

    "SS + vort.":
        "runs/2D_Inc/"
        "128_acdm-r20_puv_phys_ablation_vort_10h_00/"
        "log.txt",

    r"MS ($C=P=2$) + vort.":
        "runs/2D_Inc/"
        "128_acdm-r20_puv_vor001_multistep_p2_10h_00/"
        "log.txt",
}


TRA_LOGS = {
    "SS + cont.":
        "runs/2D_Tra/"
        "128_acdm-r20_tra_continuity02_10h_00/"
        "log.txt",

    r"MS ($C=P=2$) + cont.":
        "runs/2D_Tra/"
        "128_acdm-r20_tra_multistep_p2_cont02_fixed_10h_00/"
        "log.txt",

    r"MS ($C=P=4$) + cont.":
        "runs/2D_Tra/"
        "128_acdm-r20_tra_multistep_p4_cont02_full_10h_00/"
        "log.txt",
}


# Example:
# Training Epoch 70 (5.18 min): 0.0027
PATTERN = re.compile(
    r"Training Epoch\s+(\d+).*?:\s+([0-9eE+\-.]+)"
)


def load_training_loss(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    epochs = []
    losses = []

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))

    if not epochs:
        raise RuntimeError(
            f"No Training Epoch entries found in {path}"
        )

    return epochs, losses


def plot_group(logs, out_png, out_pdf):

    fig, ax = plt.subplots(
        figsize=(7.6, 4.8),
        constrained_layout=True,
    )

    for label, path in logs.items():
        epochs, losses = load_training_loss(path)

        print(
            f"{label:35s} "
            f"epochs={len(epochs):3d} "
            f"first={losses[0]:.6e} "
            f"last={losses[-1]:.6e} "
            f"min={min(losses):.6e}"
        )

        ax.plot(
            epochs,
            losses,
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel(
        "Epoch",
        fontsize=11,
    )

    ax.set_ylabel(
        "Training loss",
        fontsize=11,
    )

    ax.set_yscale("log")

    ax.grid(
        True,
        which="both",
        alpha=0.20,
        linewidth=0.6,
    )

    ax.legend(
        fontsize=9,
        frameon=False,
    )

    fig.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():

    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    print("===== INC TRAINING CONVERGENCE =====")

    plot_group(
        INC_LOGS,
        os.path.join(
            OUT_DIR,
            "fig_A5a_inc_training_loss.png",
        ),
        os.path.join(
            OUT_DIR,
            "fig_A5a_inc_training_loss.pdf",
        ),
    )

    print()
    print("===== TRA TRAINING CONVERGENCE =====")

    plot_group(
        TRA_LOGS,
        os.path.join(
            OUT_DIR,
            "fig_A5b_tra_training_loss.png",
        ),
        os.path.join(
            OUT_DIR,
            "fig_A5b_tra_training_loss.pdf",
        ),
    )

    print()
    print("Saved figures to:", OUT_DIR)


if __name__ == "__main__":
    main()
