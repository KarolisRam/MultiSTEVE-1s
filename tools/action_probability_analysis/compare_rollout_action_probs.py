import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np


def _load_array(run_dir, name):
    path = os.path.join(run_dir, name)
    return np.load(path, allow_pickle=True)


def _summarize_modality(original, replayed):
    if original.shape != replayed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs replayed {replayed.shape}"
        )

    diff = replayed - original
    abs_diff = np.abs(diff)
    flat_orig = original.reshape(original.shape[0], -1)
    flat_replay = replayed.reshape(replayed.shape[0], -1)

    per_step_mean_abs = abs_diff.reshape(abs_diff.shape[0], -1).mean(axis=1)
    per_step_max_abs = abs_diff.reshape(abs_diff.shape[0], -1).max(axis=1)
    per_step_l2 = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)

    orig_argmax = flat_orig.argmax(axis=1)
    replay_argmax = flat_replay.argmax(axis=1)

    result = {
        "shape": list(original.shape),
        "dtype": str(original.dtype),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "mean_l2_per_step": float(per_step_l2.mean()),
        "max_l2_per_step": float(per_step_l2.max()),
        "exact_equal": bool(np.array_equal(original, replayed)),
        "allclose_atol_1e-6_rtol_1e-5": bool(np.allclose(original, replayed, atol=1e-6, rtol=1e-5)),
        "argmax_match_rate": float(np.mean(orig_argmax == replay_argmax)),
        "top_mean_abs_steps": [int(i) for i in np.argsort(-per_step_mean_abs)[:10]],
        "top_max_abs_steps": [int(i) for i in np.argsort(-per_step_max_abs)[:10]],
    }

    per_step_rows = []
    for step in range(original.shape[0]):
        per_step_rows.append(
            {
                "step": step,
                "mean_abs_diff": float(per_step_mean_abs[step]),
                "max_abs_diff": float(per_step_max_abs[step]),
                "l2_diff": float(per_step_l2[step]),
                "orig_argmax": int(orig_argmax[step]),
                "replay_argmax": int(replay_argmax[step]),
                "argmax_match": bool(orig_argmax[step] == replay_argmax[step]),
            }
        )

    return result, per_step_rows


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_run_dir", required=True)
    parser.add_argument("--replayed_run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary = {
        "created_at": datetime.now().isoformat(),
        "original_run_dir": os.path.abspath(args.original_run_dir),
        "replayed_run_dir": os.path.abspath(args.replayed_run_dir),
    }

    button_summary, button_rows = _summarize_modality(
        _load_array(args.original_run_dir, "buttons.npy"),
        _load_array(args.replayed_run_dir, "buttons.npy"),
    )
    camera_summary, camera_rows = _summarize_modality(
        _load_array(args.original_run_dir, "camera.npy"),
        _load_array(args.replayed_run_dir, "camera.npy"),
    )

    summary["buttons"] = button_summary
    summary["camera"] = camera_summary

    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    _write_csv(os.path.join(args.output_dir, "buttons_per_step.csv"), button_rows)
    _write_csv(os.path.join(args.output_dir, "camera_per_step.csv"), camera_rows)

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        for modality in ("buttons", "camera"):
            data = summary[modality]
            f.write(f"{modality}\n")
            f.write(f"  shape: {tuple(data['shape'])}\n")
            f.write(f"  exact_equal: {data['exact_equal']}\n")
            f.write(f"  allclose_atol_1e-6_rtol_1e-5: {data['allclose_atol_1e-6_rtol_1e-5']}\n")
            f.write(f"  mean_abs_diff: {data['mean_abs_diff']:.10f}\n")
            f.write(f"  max_abs_diff: {data['max_abs_diff']:.10f}\n")
            f.write(f"  rmse: {data['rmse']:.10f}\n")
            f.write(f"  argmax_match_rate: {data['argmax_match_rate']:.10f}\n")
            f.write(f"  top_mean_abs_steps: {data['top_mean_abs_steps']}\n")
            f.write(f"  top_max_abs_steps: {data['top_max_abs_steps']}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
