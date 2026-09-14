#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_npz_array(path):
    d = np.load(path, allow_pickle=True)

    if "arr_0" in d.files:
        return np.asarray(d["arr_0"])

    if len(d.files) == 1:
        return np.asarray(d[d.files[0]])

    raise RuntimeError(f"Cannot identify array in {path}. Keys: {d.files}")


def find_frame_file(sim_dir, prefixes, frame):
    candidates = []

    for prefix in prefixes:
        candidates.extend([
            os.path.join(sim_dir, f"{prefix}_{frame:06d}.npz"),
            os.path.join(sim_dir, f"{prefix}_{frame:05d}.npz"),
            os.path.join(sim_dir, f"{prefix}_{frame:04d}.npz"),
            os.path.join(sim_dir, f"{prefix}_{frame:03d}.npz"),
            os.path.join(sim_dir, f"{prefix}_{frame}.npz"),
        ])

    for path in candidates:
        if os.path.exists(path):
            return path

    for prefix in prefixes:
        matches = sorted(glob.glob(os.path.join(sim_dir, f"{prefix}*{frame:06d}*.npz")))
        if matches:
            return matches[0]

    return None


def load_prediction(result_dir, dataset):
    folder = os.path.join(result_dir, dataset)
    npz_files = sorted(glob.glob(os.path.join(folder, "*.npz")))

    if not npz_files:
        raise FileNotFoundError(f"No npz found in {folder}")

    path = npz_files[0]
    arr = np.load(path, allow_pickle=True)["arr_0"]

    print("Loaded prediction:", path)
    print("Raw prediction shape:", arr.shape)

    # Expected raw shape:
    # [1, 1, 5, T, 4, 128, 64]
    while arr.ndim > 5 and arr.shape[0] == 1:
        arr = arr[0]

    while arr.ndim > 5 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 5:
        raise ValueError(f"Expected prediction shape [N,T,C,H,W], got {arr.shape}")

    print("Standard prediction shape:", arr.shape)
    return arr


def get_sim_ids(dataset):
    if dataset == "highRey":
        return [0, 2, 4, 6, 8]
    if dataset == "lowRey":
        return [82, 84, 86, 88, 90]
    raise ValueError(f"Unknown dataset: {dataset}")


def load_one_sim_ground_truth(data_root, sim_id, frame_start, rollout_length):
    sim_dir = os.path.join(data_root, "128_inc", f"sim_{sim_id:06d}")

    if not os.path.isdir(sim_dir):
        raise FileNotFoundError(f"Simulation folder not found: {sim_dir}")

    frames = []

    # Prediction step 1 usually corresponds to frame_start + 1.
    # If the visual comparison looks shifted by one frame, set this to 0.
    gt_offset = 1

    for t in range(rollout_length):
        frame = frame_start + gt_offset + t

        pressure_path = find_frame_file(
            sim_dir,
            prefixes=["pressure", "pres", "p"],
            frame=frame,
        )

        velocity_path = find_frame_file(
            sim_dir,
            prefixes=["velocity", "vel", "velo", "velocities"],
            frame=frame,
        )

        if pressure_path is None:
            raise FileNotFoundError(
                f"Cannot find pressure file for sim {sim_id}, frame {frame} in {sim_dir}"
            )

        if velocity_path is None:
            raise FileNotFoundError(
                f"Cannot find velocity file for sim {sim_id}, frame {frame} in {sim_dir}. "
                f"Run: ls -lh {sim_dir} | head -50"
            )

        p = load_npz_array(pressure_path)
        vel = load_npz_array(velocity_path)

        # pressure usually [1,H,W] or [H,W]
        p = np.asarray(p)
        if p.ndim == 3 and p.shape[0] == 1:
            p = p[0]
        elif p.ndim == 2:
            pass
        else:
            p = np.squeeze(p)
            if p.ndim != 2:
                raise ValueError(f"Unexpected pressure shape {p.shape} from {pressure_path}")

        # velocity usually [2,H,W] or [H,W,2]
        vel = np.asarray(vel)

        if vel.ndim == 4 and vel.shape[0] == 1:
            vel = vel[0]

        if vel.ndim == 3 and vel.shape[0] == 2:
            u = vel[0]
            v = vel[1]
        elif vel.ndim == 3 and vel.shape[-1] == 2:
            u = vel[..., 0]
            v = vel[..., 1]
        else:
            vel = np.squeeze(vel)

            if vel.ndim == 3 and vel.shape[0] == 2:
                u = vel[0]
                v = vel[1]
            elif vel.ndim == 3 and vel.shape[-1] == 2:
                u = vel[..., 0]
                v = vel[..., 1]
            else:
                raise ValueError(f"Unexpected velocity shape {vel.shape} from {velocity_path}")

        # Re is not needed for vorticity, but keep a 4-channel format consistent with prediction.
        rey = np.zeros_like(p)

        state = np.stack([u, v, p, rey], axis=0)  # [4,H,W]
        frames.append(state)

    out = np.stack(frames, axis=0)  # [T,4,H,W]
    print(f"Loaded GT sim {sim_id}, shape:", out.shape)
    return out


def load_ground_truth(data_root, dataset, frame_start, rollout_length):
    sim_ids = get_sim_ids(dataset)

    gt_list = []
    for sim_id in sim_ids:
        gt_list.append(
            load_one_sim_ground_truth(
                data_root=data_root,
                sim_id=sim_id,
                frame_start=frame_start,
                rollout_length=rollout_length,
            )
        )

    gt = np.stack(gt_list, axis=0)  # [N,T,4,H,W]
    print("GT all shape:", gt.shape)
    return gt


def compute_vorticity(x):
    """
    x: [T,C,H,W] or [N,T,C,H,W]
    channel 0 = u, channel 1 = v

    The derivatives are computed in grid units using finite differences.
    """
    u = x[..., 0, :, :]
    v = x[..., 1, :, :]

    dv_dx = np.gradient(v, axis=-2)
    du_dy = np.gradient(u, axis=-1)

    return dv_dx - du_dy


def get_field(x, field):
    if field == "u":
        return x[..., 0, :, :]

    if field == "v":
        return x[..., 1, :, :]

    if field in ["p", "pressure", "pres"]:
        return x[..., 2, :, :]

    if field in ["vort", "vorticity"]:
        return compute_vorticity(x)

    raise ValueError(f"Unknown field: {field}")


def plot_snapshots(
    pred,
    gt,
    sim_index,
    sim_id,
    steps,
    field,
    label,
    dataset,
    out_dir,
    transpose,
):
    pred_sim = pred[sim_index]  # [T,C,H,W]
    gt_sim = gt[sim_index]      # [T,C,H,W]

    pred_field = get_field(pred_sim, field)
    gt_field = get_field(gt_sim, field)

    if transpose:
        pred_field = np.transpose(pred_field, (0, 2, 1))
        gt_field = np.transpose(gt_field, (0, 2, 1))

    # Convert requested steps to valid zero-based indices.
    step_indices = []
    valid_steps = []

    for step in steps:
        idx = step - 1
        idx = max(0, min(idx, pred_field.shape[0] - 1))
        step_indices.append(idx)
        valid_steps.append(step)

    # Shared colour scales
    field_values = []
    error_values = []

    for idx in step_indices:
        gt_img = gt_field[idx]
        pred_img = pred_field[idx]
        err_img = np.abs(pred_img - gt_img)

        field_values.append(gt_img)
        field_values.append(pred_img)
        error_values.append(err_img)

    vmin_field = min(float(np.nanmin(a)) for a in field_values)
    vmax_field = max(float(np.nanmax(a)) for a in field_values)

    vmin_error = 0.0
    vmax_error = max(float(np.nanmax(a)) for a in error_values)

    # Avoid degenerate colour scale.
    if np.isclose(vmin_field, vmax_field):
        vmin_field -= 1e-6
        vmax_field += 1e-6

    if np.isclose(vmin_error, vmax_error):
        vmax_error += 1e-6

    print("Shared field colour scale:", vmin_field, vmax_field)
    print("Shared error colour scale:", vmin_error, vmax_error)

    # Plot
    n = len(valid_steps)
    fig, axes = plt.subplots(
        3,
        n,
        figsize=(4.2 * n, 8.5),
        constrained_layout=True,
    )

    if n == 1:
        axes = axes.reshape(3, 1)

    field_im = None
    error_im = None

    for j, (step, idx) in enumerate(zip(valid_steps, step_indices)):
        gt_img = gt_field[idx]
        pred_img = pred_field[idx]
        err_img = np.abs(pred_img - gt_img)

        field_im = axes[0, j].imshow(
            gt_img,
            origin="lower",
            vmin=vmin_field,
            vmax=vmax_field,
        )
        axes[0, j].set_title(f"GT, step {step}")

        axes[1, j].imshow(
            pred_img,
            origin="lower",
            vmin=vmin_field,
            vmax=vmax_field,
        )
        axes[1, j].set_title(f"Prediction, step {step}")

        error_im = axes[2, j].imshow(
            err_img,
            origin="lower",
            vmin=vmin_error,
            vmax=vmax_error,
        )
        axes[2, j].set_title(f"|Error|, step {step}")

        for i in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    # Shared colourbars:
    # one for GT + Prediction rows, one for Error row.
    cbar_field = fig.colorbar(
        field_im,
        ax=axes[0:2, :],
        fraction=0.025,
        pad=0.02,
    )
    cbar_field.set_label(field)

    cbar_error = fig.colorbar(
        error_im,
        ax=axes[2, :],
        fraction=0.025,
        pad=0.02,
    )
    cbar_error.set_label("|Error|")

    title = f"{label} | {dataset} | sim_{sim_id:06d} | {field}"
    fig.suptitle(title, fontsize=16)

    os.makedirs(out_dir, exist_ok=True)

    safe_label = (
        label.replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
        .replace("|", "_")
    )

    out_path = os.path.join(
        out_dir,
        f"{safe_label}_{dataset}_sim_{sim_id:06d}_{field}_snapshots_shared_scale.png",
    )

    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--dataset", choices=["lowRey", "highRey"], default="highRey")

    default_data_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", default_data_root),
    )

    parser.add_argument("--frame-start", type=int, default=300)
    parser.add_argument("--rollout-length", type=int, default=500)

    parser.add_argument("--field", default="vorticity")
    parser.add_argument("--steps", default="100,300,500")

    # sim-index is the index inside the test group, not the physical simulation id.
    # highRey: [0, 2, 4, 6, 8]
    # lowRey:  [82, 84, 86, 88, 90]
    parser.add_argument("--sim-index", type=int, default=0)

    parser.add_argument("--label", default="")
    parser.add_argument("--out-dir", default="results/long_rollout_snapshots")
    parser.add_argument("--transpose", action="store_true")

    args = parser.parse_args()

    sim_ids = get_sim_ids(args.dataset)

    if args.sim_index < 0 or args.sim_index >= len(sim_ids):
        raise ValueError(
            f"--sim-index must be between 0 and {len(sim_ids)-1} for {args.dataset}. "
            f"Available sim ids: {sim_ids}"
        )

    sim_id = sim_ids[args.sim_index]

    print("Dataset:", args.dataset)
    print("Available simulation ids:", sim_ids)
    print("Selected sim index:", args.sim_index)
    print("Selected simulation id:", sim_id)

    pred = load_prediction(args.result_dir, args.dataset)

    gt = load_ground_truth(
        data_root=args.data_root,
        dataset=args.dataset,
        frame_start=args.frame_start,
        rollout_length=args.rollout_length,
    )

    n = min(pred.shape[0], gt.shape[0])
    t = min(pred.shape[1], gt.shape[1])

    pred = pred[:n, :t]
    gt = gt[:n, :t]

    print("Final pred shape:", pred.shape)
    print("Final gt shape:", gt.shape)

    steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]

    plot_snapshots(
        pred=pred,
        gt=gt,
        sim_index=args.sim_index,
        sim_id=sim_id,
        steps=steps,
        field=args.field,
        label=args.label,
        dataset=args.dataset,
        out_dir=args.out_dir,
        transpose=args.transpose,
    )


if __name__ == "__main__":
    main()
