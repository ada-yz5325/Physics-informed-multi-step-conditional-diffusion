#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRED_KEYS = [
    "prediction_denorm",
    "predictions_denorm",
    "pred_denorm",
    "prediction",
    "predictions",
    "pred",
    "rollout_pred",
    "pred_rollout",
    "preds",
    "output",
    "outputs",
]

GT_KEYS = [
    "ground_truth_denorm",
    "groundtruth_denorm",
    "gt_denorm",
    "ground_truth",
    "groundtruth",
    "gt",
    "target",
    "targets",
    "truth",
    "rollout_gt",
    "gt_rollout",
]

CHANNEL_KEYS = [
    "channel_names",
    "channels",
    "field_names",
]


def unwrap_npz(path):
    data = np.load(path, allow_pickle=True)

    # Case 1: normal named arrays
    if set(data.files) != {"arr_0"}:
        return {k: data[k] for k in data.files}

    arr = data["arr_0"]
    print("Found only arr_0")
    print("arr_0 type:", type(arr), "shape:", arr.shape, "dtype:", arr.dtype)

    # Case 2: arr_0 is a dict stored as object
    if arr.dtype == object:
        try:
            obj = arr.item()
            if isinstance(obj, dict):
                print("arr_0 is dict with keys:", list(obj.keys()))
                return obj
        except Exception:
            pass

        # Case 3: arr_0 is object array/list
        out = {"arr_0": arr}
        try:
            for i, x in enumerate(arr):
                out[f"item_{i}"] = x
                print(f"item_{i}:", type(x), getattr(x, "shape", None), getattr(x, "dtype", None))
        except Exception:
            pass
        return out

    # Case 4: arr_0 is a numeric array
    return {"arr_0": arr}


def find_npz(folder):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.npz"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {folder}")
    print("Using npz:", files[0])
    return files[0]


def get_first_existing(obj, keys):
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k], k
    return None, None


def standardize_rollout_array(a, name="array"):
    a = np.asarray(a)
    print(f"{name} original shape:", a.shape, "dtype:", a.dtype)

    # unwrap object array if possible
    if a.dtype == object:
        try:
            a = np.asarray(a.item())
            print(f"{name} after item shape:", a.shape, "dtype:", a.dtype)
        except Exception:
            pass

    # remove singleton leading dimensions
    while a.ndim > 4 and a.shape[0] == 1:
        a = a[0]

    # [B,T,C,H,W] -> [T,C,H,W]
    if a.ndim == 5:
        a = a[0]

    # [T,H,W,C] -> [T,C,H,W]
    if a.ndim == 4 and a.shape[-1] <= 8 and a.shape[1] > 8:
        a = np.moveaxis(a, -1, 1)

    if a.ndim != 4:
        raise ValueError(f"{name}: expected 4D after processing, got {a.shape}")

    print(f"{name} standard shape:", a.shape)
    return a


def guess_pred_gt_from_dict(obj):
    pred, pred_key = get_first_existing(obj, PRED_KEYS)
    gt, gt_key = get_first_existing(obj, GT_KEYS)
    ch, ch_key = get_first_existing(obj, CHANNEL_KEYS)

    if pred is not None and gt is not None:
        return pred, pred_key, gt, gt_key, ch, ch_key

    print("\nCould not find named pred/gt keys.")
    print("Available keys and shapes:")
    candidates = []
    for k, v in obj.items():
        try:
            arr = np.asarray(v)
            print(" ", k, arr.shape, arr.dtype)
            if arr.ndim >= 4:
                candidates.append((k, v, arr.shape))
        except Exception as e:
            print(" ", k, type(v), e)

    # Fallback:
    # If arr_0 is a numeric ndarray, it may contain [prediction, ground truth] or similar.
    if "arr_0" in obj:
        arr = np.asarray(obj["arr_0"])
        print("\nTrying arr_0 fallback. shape:", arr.shape, "dtype:", arr.dtype)

        # Common case: arr_0 shape [2, T, C, H, W], first pred, second gt
        if arr.ndim == 5 and arr.shape[0] == 2:
            print("Assuming arr_0[0] = prediction, arr_0[1] = ground truth")
            return arr[0], "arr_0[0]", arr[1], "arr_0[1]", None, None

        # Common case: arr_0 shape [N, ...] with N>=2 and rest is rollout
        if arr.ndim >= 5 and arr.shape[0] >= 2:
            print("Assuming arr_0[0] = prediction, arr_0[1] = ground truth")
            return arr[0], "arr_0[0]", arr[1], "arr_0[1]", None, None

    raise RuntimeError("Could not infer prediction and ground truth arrays.")


def to_channel_names(x):
    if x is None:
        return ["u", "v", "pres"]
    try:
        return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in x]
    except Exception:
        return ["u", "v", "pres"]


def get_channel_indices(channel_names):
    names = [s.lower() for s in channel_names]

    def find_one(candidates, default):
        for cand in candidates:
            for i, name in enumerate(names):
                if cand == name or cand in name:
                    return i
        return default

    u_idx = find_one(["u", "velx", "velocity_x"], 0)
    v_idx = find_one(["v", "vely", "velocity_y"], 1)
    p_idx = find_one(["pres", "pressure", "p"], 2)
    return u_idx, v_idx, p_idx


def get_field(arr, field, channel_names, transpose=False):
    u_idx, v_idx, p_idx = get_channel_indices(channel_names)

    if field == "u":
        out = arr[:, u_idx]
    elif field == "v":
        out = arr[:, v_idx]
    elif field in ["p", "pressure", "pres"]:
        out = arr[:, p_idx]
    elif field in ["vort", "vorticity"]:
        u = arr[:, u_idx]
        v = arr[:, v_idx]
        dv_dx = np.gradient(v, axis=-2)
        du_dy = np.gradient(u, axis=-1)
        out = dv_dx - du_dy
    else:
        raise ValueError(f"Unknown field: {field}")

    if transpose:
        out = np.transpose(out, (0, 2, 1))

    return out


def plot_one_comparison(pred, gt, steps, title, out_path):
    n = len(steps)
    fig, axes = plt.subplots(3, n, figsize=(4.0 * n, 8.5), constrained_layout=True)

    if n == 1:
        axes = axes.reshape(3, 1)

    for j, step in enumerate(steps):
        idx = step - 1
        idx = max(0, min(idx, pred.shape[0] - 1))

        gt_img = gt[idx]
        pred_img = pred[idx]
        err_img = np.abs(pred_img - gt_img)

        vmin = np.nanmin([gt_img.min(), pred_img.min()])
        vmax = np.nanmax([gt_img.max(), pred_img.max()])

        im0 = axes[0, j].imshow(gt_img, origin="lower", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"GT, step {step}")

        im1 = axes[1, j].imshow(pred_img, origin="lower", vmin=vmin, vmax=vmax)
        axes[1, j].set_title(f"Prediction, step {step}")

        im2 = axes[2, j].imshow(err_img, origin="lower")
        axes[2, j].set_title(f"|Error|, step {step}")

        for i in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

        fig.colorbar(im0, ax=axes[0, j], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1, j], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[2, j], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--dataset", default="highRey", choices=["lowRey", "highRey"])
    parser.add_argument("--field", default="vorticity")
    parser.add_argument("--steps", default="100,300,500")
    parser.add_argument("--label", default="")
    parser.add_argument("--out-dir", default="results/long_rollout_snapshots")
    parser.add_argument("--transpose", action="store_true")
    args = parser.parse_args()

    folder = os.path.join(args.result_dir, args.dataset)
    if not os.path.isdir(folder):
        folder = args.result_dir

    npz_path = find_npz(folder)
    obj = unwrap_npz(npz_path)

    pred, pred_key, gt, gt_key, channel_names, ch_key = guess_pred_gt_from_dict(obj)

    channel_names = to_channel_names(channel_names)
    print("Prediction key:", pred_key)
    print("Ground truth key:", gt_key)
    print("Channel names:", channel_names)

    pred = standardize_rollout_array(pred, "prediction")
    gt = standardize_rollout_array(gt, "ground truth")

    pred_field = get_field(pred, args.field, channel_names, transpose=args.transpose)
    gt_field = get_field(gt, args.field, channel_names, transpose=args.transpose)

    steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]

    os.makedirs(args.out_dir, exist_ok=True)

    safe_label = args.label.replace(" ", "_").replace("+", "plus").replace("/", "_")
    if not safe_label:
        safe_label = os.path.basename(args.result_dir.rstrip("/"))

    out_path = os.path.join(
        args.out_dir,
        f"{safe_label}_{args.dataset}_{args.field}_snapshots.png"
    )

    title = f"{args.label} | {args.dataset} | {args.field}"
    plot_one_comparison(pred_field, gt_field, steps, title, out_path)


if __name__ == "__main__":
    main()
