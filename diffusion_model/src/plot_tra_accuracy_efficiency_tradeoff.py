#!/usr/bin/env python3

import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FILES = {
    r"MS ($C=P=2$) + cont.":
        "results/sampling_tra_ms_p2_cont02_fixed_10h_500step_metrics/"
        "Tralong_500step/"
        "acdm-r20-tra-ms-p2-cont02-fixed-10h-500step_metrics.csv",

    r"MS ($C=P=3$) + cont.":
        "results/sampling_tra_ms_p3_cont02_full_10h_500step_metrics/"
        "Tralong_500step/"
        "acdm-r20-tra-ms-p3-cont02-full-10h-500step_metrics.csv",

    r"MS ($C=P=4$) + cont.":
        "results/sampling_tra_ms_p4_cont02_full_10h_500step_metrics/"
        "Tralong_500step/"
        "acdm-r20-tra-ms-p4-cont02-full-10h-500step_metrics.csv",
}


OUT_DIR = "results/final_figures"

OUT_PNG = os.path.join(
    OUT_DIR,
    "fig_tra_accuracy_efficiency_tradeoff.png",
)

OUT_PDF = os.path.join(
    OUT_DIR,
    "fig_tra_accuracy_efficiency_tradeoff.pdf",
)


def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []

    for label, path in FILES.items():

        df = pd.read_csv(path)

        mse = float(df.loc[0, "predMSE"])
        time_min = float(df.loc[0, "inferenceTimeMin"])

        rows.append(
            {
                "label": label,
                "time": time_min,
                "mse": mse,
            }
        )

        print(
            f"{label:32s} "
            f"time={time_min:.6f} min  "
            f"predMSE={mse:.8f}"
        )


    # Plot


    fig, ax = plt.subplots(
        figsize=(8.4, 5.6),
        constrained_layout=True,
    )

    # Explicit plotting order:
    # P2, P3, P4
    #
    # Keep default Matplotlib colour cycle.
    for row in rows:

        point = ax.scatter(
            row["time"],
            row["mse"],
            s=75,
            zorder=3,
        )

        # Match annotation colour to point colour
        colour = point.get_facecolor()[0]

        label = row["label"]

        # Custom label placement so nothing collides
        # with axes / boundaries.
        if "P=2" in label:
            offset = (-10, 8)
            ha = "right"
            va = "bottom"

        elif "P=3" in label:
            offset = (10, 8)
            ha = "left"
            va = "bottom"

        else:  # P=4
            offset = (10, -8)
            ha = "left"
            va = "top"

        ax.annotate(
            label,
            xy=(row["time"], row["mse"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=10.5,
            ha=ha,
            va=va,
            color=colour,
        )



    # Axes
 
    ax.set_xlim(
        1.10,
        2.05,
    )

    ax.set_ylim(
        0.02145,
        0.02218,
    )

    ax.set_xlabel(
        "Inference time (min)",
        fontsize=11,
    )

    ax.set_ylabel(
        "Prediction MSE",
        fontsize=11,
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25,
        linewidth=0.7,
    )



    fig.savefig(
        OUT_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("Saved:")
    print(" ", OUT_PNG)
    print(" ", OUT_PDF)


if __name__ == "__main__":
    main()
