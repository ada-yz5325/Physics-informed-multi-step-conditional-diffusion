#!/usr/bin/env python3
import os
import re
import glob
import csv
import argparse
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


"""
Plot training loss curves from PBS output logs.

This script reads training log files such as:
  puv_phys_10h.o*
  train_multistep_p2_4h.o*
  sample100.o*

It extracts lines like:
  [70, 4935] (11.06 min): 0.0003
  Training Epoch 70 (xx.xx min): 0.004 RecMSE ...
  Test Low Reynolds 100-200 Epoch 10 (xx.xx min): 0.123 RecMSE ... PredMSE 0.456 ...
  Test High Reynolds 900-1000 Epoch 10 (xx.xx min): 0.123 RecMSE ... PredMSE 0.456 ...

Outputs:
  - batch_training_loss.csv
  - epoch_training_loss.csv
  - test_metrics.csv
  - batch_training_loss.png
  - epoch_training_loss.png
  - test_predmse.png
  - training_loss_summary.txt
"""


BATCH_LOSS_RE = re.compile(
    r"^\[\s*(?P<epoch>\d+)\s*,\s*(?P<batch>\d+)\s*\]\s*"
    r"\((?P<minutes>[-+0-9.eE]+)\s*min\)\s*:\s*"
    r"(?P<loss>[-+0-9.eE]+)"
)

TRAIN_EPOCH_RE = re.compile(
    r"^Training Epoch\s+(?P<epoch>\d+)\s*"
    r"\((?P<minutes>[-+0-9.eE]+)\s*min\)\s*:\s*"
    r"(?P<loss>[-+0-9.eE]+)"
)

TEST_RE = re.compile(
    r"^Test\s+(?P<name>.+?)\s+Epoch\s+(?P<epoch>\d+)\s*"
    r"\((?P<minutes>[-+0-9.eE]+)\s*min\)\s*:\s*"
    r"(?P<loss>[-+0-9.eE]+)"
)

KEY_VALUE_RE = re.compile(
    r"(?P<key>RecMSE|RecLSIM|PredMSE|PredLSIM)\s+"
    r"(?P<value>[-+0-9.eE]+)"
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_float(value: str):
    try:
        return float(value)
    except Exception:
        return None


def read_lines(paths: List[str]) -> List[str]:
    lines = []
    for path in paths:
        print(f"Reading log file: {path}")
        try:
            with open(path, "r", errors="ignore") as f:
                lines.extend(f.readlines())
        except Exception as e:
            print(f"Warning: failed to read {path}: {e}")
    return lines


def parse_logs(lines: List[str]):
    batch_rows: List[Dict[str, Any]] = []
    epoch_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    global_step = 0

    for raw_line in lines:
        line = raw_line.strip()

        # Batch-level training loss:
        # [70, 4935] (11.06 min): 0.0003
        m = BATCH_LOSS_RE.match(line)
        if m:
            global_step += 1
            row = {
                "global_step": global_step,
                "epoch": int(m.group("epoch")),
                "batch": int(m.group("batch")),
                "minutes": safe_float(m.group("minutes")),
                "loss": safe_float(m.group("loss")),
            }
            batch_rows.append(row)
            continue

        # Epoch-level training loss:
        # Training Epoch 70 (xx.xx min): 0.004 RecMSE ...
        m = TRAIN_EPOCH_RE.match(line)
        if m:
            row = {
                "epoch": int(m.group("epoch")),
                "minutes": safe_float(m.group("minutes")),
                "loss": safe_float(m.group("loss")),
            }

            for kv in KEY_VALUE_RE.finditer(line):
                row[kv.group("key")] = safe_float(kv.group("value"))

            epoch_rows.append(row)
            continue

        # Test metrics:
        # Test Low Reynolds 100-200 Epoch 10 ... PredMSE 0.123 ...
        m = TEST_RE.match(line)
        if m:
            row = {
                "test_name": m.group("name").strip(),
                "epoch": int(m.group("epoch")),
                "minutes": safe_float(m.group("minutes")),
                "loss": safe_float(m.group("loss")),
            }

            for kv in KEY_VALUE_RE.finditer(line):
                row[kv.group("key")] = safe_float(kv.group("value"))

            test_rows.append(row)
            continue

    return batch_rows, epoch_rows, test_rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"No rows to write for {path}")
        return

    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {path}")


def plot_batch_loss(batch_rows: List[Dict[str, Any]], out_dir: str) -> None:
    if not batch_rows:
        print("No batch-level training loss found. Skipping batch loss plot.")
        return

    x = [r["global_step"] for r in batch_rows]
    y = [r["loss"] for r in batch_rows]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=0.8)
    plt.xlabel("Optimisation step")
    plt.ylabel("Training loss")
    plt.title("Batch-level training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "batch_training_loss.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot: {out_path}")


def plot_epoch_loss(epoch_rows: List[Dict[str, Any]], out_dir: str) -> None:
    if not epoch_rows:
        print("No epoch-level training loss found. Skipping epoch loss plot.")
        return

    x = [r["epoch"] for r in epoch_rows]
    y = [r["loss"] for r in epoch_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=1.2)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Epoch-level training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "epoch_training_loss.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot: {out_path}")


def plot_test_predmse(test_rows: List[Dict[str, Any]], out_dir: str) -> None:
    pred_rows = [r for r in test_rows if "PredMSE" in r]

    if not pred_rows:
        print("No test PredMSE found. Skipping test PredMSE plot.")
        return

    test_names = sorted(set(r["test_name"] for r in pred_rows))

    plt.figure(figsize=(8, 5))

    for test_name in test_names:
        rows = [r for r in pred_rows if r["test_name"] == test_name]
        rows = sorted(rows, key=lambda r: r["epoch"])

        x = [r["epoch"] for r in rows]
        y = [r["PredMSE"] for r in rows]

        plt.plot(x, y, marker="o", linewidth=1.2, label=test_name)

    plt.xlabel("Epoch")
    plt.ylabel("Test PredMSE")
    plt.title("Testing prediction MSE over training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, "test_predmse.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot: {out_path}")


def write_summary(
    out_dir: str,
    log_paths: List[str],
    batch_rows: List[Dict[str, Any]],
    epoch_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
) -> None:
    out_path = os.path.join(out_dir, "training_loss_summary.txt")

    with open(out_path, "w") as f:
        f.write("Training loss parsing summary\n")
        f.write("=============================\n\n")

        f.write("Input log files:\n")
        for p in log_paths:
            f.write(f"  {p}\n")

        f.write("\nParsed rows:\n")
        f.write(f"  Batch-level loss rows: {len(batch_rows)}\n")
        f.write(f"  Epoch-level loss rows: {len(epoch_rows)}\n")
        f.write(f"  Test metric rows: {len(test_rows)}\n")

        if batch_rows:
            f.write("\nBatch-level training loss:\n")
            f.write(f"  First step loss: {batch_rows[0]['loss']}\n")
            f.write(f"  Last step loss: {batch_rows[-1]['loss']}\n")
            f.write(f"  Total optimisation steps parsed: {len(batch_rows)}\n")

        if epoch_rows:
            f.write("\nEpoch-level training loss:\n")
            f.write(f"  First epoch: {epoch_rows[0]['epoch']}, loss={epoch_rows[0]['loss']}\n")
            f.write(f"  Last epoch: {epoch_rows[-1]['epoch']}, loss={epoch_rows[-1]['loss']}\n")

        if test_rows:
            f.write("\nTest rows found:\n")
            for r in test_rows[-10:]:
                f.write(f"  {r}\n")

    print(f"Saved summary: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs",
        nargs="+",
        default=None,
        help="Log file paths or glob patterns, e.g. puv_phys_10h.o*",
    )
    parser.add_argument(
        "--out",
        default="results/training_loss_curves",
        help="Output directory for CSV files and figures.",
    )
    parser.add_argument(
        "--label",
        default="baseline",
        help="Optional label used as a subfolder name.",
    )

    args = parser.parse_args()

    if args.logs is None:
        # Default: try common PBS output logs.
        patterns = [
            "puv_phys_10h.o*",
            "train_puv_phys_10h.o*",
            "baseline*.o*",
            "*.o*",
        ]
    else:
        patterns = args.logs

    log_paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        log_paths.extend(matched)

    # Remove duplicates while preserving order.
    seen = set()
    log_paths_unique = []
    for p in log_paths:
        if p not in seen:
            seen.add(p)
            log_paths_unique.append(p)

    log_paths = log_paths_unique

    if not log_paths:
        raise FileNotFoundError(
            "No log files found. Try for example:\n"
            "  python src/plot_training_loss_from_log.py --logs 'puv_phys_10h.o*'\n"
            "or:\n"
            "  python src/plot_training_loss_from_log.py --logs 'sample*.o*'\n"
        )

    out_dir = os.path.join(args.out, args.label)
    ensure_dir(out_dir)

    lines = read_lines(log_paths)
    batch_rows, epoch_rows, test_rows = parse_logs(lines)

    write_csv(os.path.join(out_dir, "batch_training_loss.csv"), batch_rows)
    write_csv(os.path.join(out_dir, "epoch_training_loss.csv"), epoch_rows)
    write_csv(os.path.join(out_dir, "test_metrics.csv"), test_rows)

    plot_batch_loss(batch_rows, out_dir)
    plot_epoch_loss(epoch_rows, out_dir)
    plot_test_predmse(test_rows, out_dir)

    write_summary(out_dir, log_paths, batch_rows, epoch_rows, test_rows)

    print("\nDone.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
