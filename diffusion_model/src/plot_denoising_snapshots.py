#!/usr/bin/env python3
import os
import glob
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


"""
Plot denoising snapshots from denoising_trace_*.npz.

Input npz should contain:
  steps
  trace_denorm
  prediction_denorm
  ground_truth_denorm
  channel_names

The output is a row of images:
  noise / step 15 / step 10 / step 5 / final / ground truth

Supported fields:
  pressure
  u
  v
  vorticity
"""


def find_latest_npz(input_path):
    if os.path.isfile(input_path):
        return input_path

    matches = sorted(
        glob.glob(os.path.join(input_path, "denoising_trace_*.npz")),
        key=os.path.getmtime,
        reverse=True,
    )

    if not matches:
        raise FileNotFoundError(f"No denoising_trace_*.npz found in {input_path}")

    return matches[0]


def get_channel_index(channel_names, name):
    channel_names = [str(x) for x in channel_names]

    if name == "u":
        return channel_names.index("u")
    if name == "v":
        return channel_names.index("v")
    if name in ["pressure", "p", "pres"]:
        if "p" in channel_names:
            return channel_names.index("p")
        if "pressure" in channel_names:
            return channel_names.index("pressure")
        if "pres" in channel_names:
            return channel_names.index("pres")

    raise ValueError(f"Could not find channel {name} in {channel_names}")


def compute_vorticity(field_chw):
    """
    field_chw shape:
      [C, W, H]

    channel convention:
      0 = u
      1 = v

    vorticity = dv/dx - du/dy
    """
    u = field_chw[0]
    v = field_chw[1]

    dvdx = np.gradient(v, axis=-2)
    dudy = np.gradient(u, axis=-1)

    return dvdx - dudy


def extract_field(field_chw, field_name, channel_names):
    if field_name == "vorticity":
        return compute_vorticity(field_chw)

    if field_name in ["pressure", "p", "pres"]:
        idx = get_channel_index(channel_names, "pressure")
    elif field_name == "u":
        idx = get_channel_index(channel_names, "u")
    elif field_name == "v":
        idx = get_channel_index(channel_names, "v")
    else:
        raise ValueError(f"Unknown field: {field_name}")

    return field_chw[idx]


def robust_vmin_vmax(arrays):
    values = np.concatenate([a.reshape(-1) for a in arrays])
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0, 1.0

    vmin = np.percentile(values, 2)
    vmax = np.percentile(values, 98)

    if abs(vmax - vmin) < 1e-12:
        vmin = np.min(values)
        vmax = np.max(values)

    if abs(vmax - vmin) < 1e-12:
        vmin -= 1.0
        vmax += 1.0

    return float(vmin), float(vmax)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="results/denoising_process",
        help="Either a denoising_trace_*.npz file or a directory containing one.",
    )

    parser.add_argument(
        "--out-dir",
        default="results/denoising_process/figures",
        help="Output directory for figures.",
    )

    parser.add_argument(
        "--field",
        default="pressure",
        choices=["pressure", "u", "v", "vorticity"],
        help="Field to visualise.",
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index within trace batch_seq dimension to visualise.",
    )

    parser.add_argument(
        "--include-ground-truth",
        action="store_true",
        help="Add the ground-truth final frame as the last panel.",
    )

    parser.add_argument(
        "--transpose",
        action="store_true",
        help="Transpose each image before plotting.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    npz_path = find_latest_npz(args.input)

    print("Loading:", npz_path)
    data = np.load(npz_path, allow_pickle=True)

    steps = data["steps"]
    trace = data["trace_denorm"]
    pred = data["prediction_denorm"]
    gt = data["ground_truth_denorm"]
    channel_names = [str(x) for x in data["channel_names"]]

    print("steps:", steps.tolist())
    print("trace shape:", trace.shape)
    print("prediction shape:", pred.shape)
    print("ground truth shape:", gt.shape)
    print("channel_names:", channel_names)

    sample_index = args.sample_index

    if sample_index >= trace.shape[1]:
        raise IndexError(
            f"sample-index {sample_index} is out of range for trace shape {trace.shape}"
        )

    panels = []
    labels = []

    # trace shape:
    # [num_steps, batch_seq, C, W, H]
    for i, step in enumerate(steps):
        field_chw = trace[i, sample_index]
        arr = extract_field(field_chw, args.field, channel_names)
        panels.append(arr)
        labels.append(f"step {int(step)}")

    if args.include_ground_truth:
        # gt shape from model input:
        # [B, T, C, W, H]
        # final frame is gt[0, -1]
        gt_field = extract_field(gt[0, -1], args.field, channel_names)
        panels.append(gt_field)
        labels.append("ground truth")

    if args.transpose:
        panels = [p.T for p in panels]

    vmin, vmax = robust_vmin_vmax(panels)

    n = len(panels)

    # Wider figure and fixed right margin for colorbar.
    fig_width = max(3.0 * n + 2.0, 14)
    fig_height = 4.2

    fig, axes = plt.subplots(1, n, figsize=(fig_width, fig_height))

    if n == 1:
        axes = [axes]

    # Leave enough space on the right for a standalone colorbar.
    fig.subplots_adjust(
        left=0.03,
        right=0.88,
        bottom=0.12,
        top=0.78,
        wspace=0.38,
    )

    ims = []

    for ax, arr, label in zip(axes, panels, labels):
        im = ax.imshow(
            arr,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            aspect="equal",
        )
        ims.append(im)
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # Manual colorbar axis: [left, bottom, width, height]
    # This avoids colorbar being inserted between subplots.
    cax = fig.add_axes([0.905, 0.23, 0.012, 0.45])
    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.set_label(args.field, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    title = f"Denoising process snapshots: {args.field}"
    fig.suptitle(title, fontsize=13)

    base = os.path.splitext(os.path.basename(npz_path))[0]

    transpose_suffix = "_transpose" if args.transpose else ""
    gt_suffix = "_with_gt" if args.include_ground_truth else ""

    out_png = os.path.join(
        args.out_dir,
        f"{base}_{args.field}_snapshots{transpose_suffix}{gt_suffix}.png",
    )

    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print("Saved figure:")
    print("  ", out_png)


if __name__ == "__main__":
    main()
