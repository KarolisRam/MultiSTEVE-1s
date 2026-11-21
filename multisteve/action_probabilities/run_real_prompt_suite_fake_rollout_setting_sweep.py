import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multisteve.action_probabilities import run_fake_rollout_setting_sweep as base


REPO_ROOT = Path(os.environ.get("MULTISTEVE_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUTPUT_ROOT = Path(
    os.environ.get(
        "MULTISTEVE_ACTION_PROB_OUTPUT_ROOT",
        "outputs/action_probabilities/model_group_comparisons/real_prompt_suite_setting_sweep",
    )
).resolve()
DEFAULT_REAL_PROMPT_MANIFEST = Path(
    os.environ.get("MULTISTEVE_REAL_PROMPT_MANIFEST", "outputs/action_probabilities/real_prompt_suite/latest/manifest.json")
).resolve()
MAX_ATTEMPTS_PER_ROLLOUT = 3


def _python_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{REPO_ROOT}{':' + existing if existing else ''}"
    return env


def _load_prompt_suite(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    prompt_text_by_slug = {item["slug"]: item["text"] for item in manifest["prompts"]}
    ordered_results = []
    for prompt in manifest["prompts"]:
        slug = prompt["slug"]
        matching = [item for item in manifest["results"] if item["prompt_slug"] == slug]
        if len(matching) != 1:
            raise ValueError(f"Expected exactly one result for prompt {slug}, found {len(matching)}")
        result = matching[0]
        observations_zip = Path(result["run_dir"]).resolve() / "observations.zip"
        if not observations_zip.exists():
            raise FileNotFoundError(f"Missing observations zip for prompt {slug}: {observations_zip}")
        ordered_results.append(
            {
                "prompt_slug": slug,
                "prompt_text": prompt_text_by_slug[slug],
                "rollout_seed": int(result["rollout_seed"]),
                "run_dir": str(Path(result["run_dir"]).resolve()),
                "save_dir": str(Path(result["save_dir"]).resolve()),
                "observations_zip": str(observations_zip),
            }
        )
    return {
        "manifest_path": str(manifest_path.resolve()),
        "suite_root": str(manifest_path.parent.resolve()),
        "episodes_root": manifest["episodes_root"],
        "prompt_seed": int(manifest["prompt_seed"]),
        "rollout_seed_mode": manifest.get("rollout_seed_mode"),
        "results": ordered_results,
    }


def _launch_rollout(model_spec: base.ModelSpec, prompt_result: dict, prompt_seed: int, family_rollouts_dir: Path) -> dict:
    rollout_root = family_rollouts_dir / model_spec.label
    rollout_root.mkdir(parents=True, exist_ok=True)

    try:
        run_dir = base._resolve_run_dir(rollout_root)
        status = "reused_existing"
    except ValueError:
        log_path = rollout_root / "launcher.log"
        run_dir = None
        for attempt_index in range(1, MAX_ATTEMPTS_PER_ROLLOUT + 1):
            cmd = [
                sys.executable,
                "-m",
                "steve1.run_agent.run_agent_with_logging",
                "--in_model",
                base.IN_MODEL,
                "--in_weights",
                model_spec.weights_path,
                "--prior_weights",
                base.PRIOR_WEIGHTS,
                "--text_cond_scale",
                "6.0",
                "--custom_text_prompt",
                prompt_result["prompt_text"],
                "--custom_text_seed",
                str(prompt_seed),
                "--gameplay_length",
                str(base.GAMEPLAY_LENGTH),
                "--save_dirpath",
                str(rollout_root),
                "--seed",
                str(prompt_result["rollout_seed"]),
                "--save_additional_data",
                "--fake_env_zip",
                prompt_result["observations_zip"],
            ]
            start_time = time.time()
            with open(log_path, "a") as log_file:
                log_file.write(
                    f"[{datetime.now().isoformat()}] Launching rollout attempt {attempt_index} "
                    f"for {prompt_result['prompt_slug']} {model_spec.family_name} {model_spec.label}\n"
                )
                log_file.write(" ".join(cmd) + "\n\n")
                log_file.flush()
                try:
                    subprocess.run(
                        cmd,
                        cwd=REPO_ROOT,
                        env=_python_env(),
                        check=True,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
                    elapsed = time.time() - start_time
                    log_file.write(f"\n[{datetime.now().isoformat()}] Completed in {elapsed:.2f}s\n")
                    run_dir = base._resolve_run_dir(rollout_root)
                    status = "ran_new" if attempt_index == 1 else "ran_new_after_retry"
                    break
                except subprocess.CalledProcessError as exc:
                    elapsed = time.time() - start_time
                    log_file.write(
                        f"\n[{datetime.now().isoformat()}] Attempt {attempt_index} failed after "
                        f"{elapsed:.2f}s with code {exc.returncode}\n"
                    )
        if run_dir is None:
            raise RuntimeError(
                f"Rollout failed for prompt {prompt_result['prompt_slug']} and model {model_spec.label}; see {log_path}"
            )

    metadata = {
        "family_name": model_spec.family_name,
        "family_display_name": model_spec.family_display_name,
        "label": model_spec.label,
        "model_dir": model_spec.model_dir,
        "weights_path": model_spec.weights_path,
        "rollout_root": str(rollout_root),
        "run_dir": str(run_dir),
        "status": status,
        "prompt_slug": prompt_result["prompt_slug"],
        "prompt_text": prompt_result["prompt_text"],
        "source_observations_zip": prompt_result["observations_zip"],
        "source_real_run_dir": prompt_result["run_dir"],
        "source_rollout_seed": prompt_result["rollout_seed"],
        "prompt_seed": prompt_seed,
    }
    (rollout_root / "model_spec.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def _plot_dimensions():
    return base.DEFAULT_SCREEN_WIDTH * 0.9 / base.DEFAULT_DPI, base.DEFAULT_SCREEN_HEIGHT * 0.9 / base.DEFAULT_DPI, base.DEFAULT_DPI


def _write_prompt_summary(
    output_dir: Path,
    prompt_result: dict,
    prompt_seed: int,
    family_summaries: list[dict],
    skipped_by_family: dict[str, list[dict]],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for summary in family_summaries:
        family_name = summary["family_name"]
        summary_rows.append(
            {
                "prompt_slug": prompt_result["prompt_slug"],
                "prompt_text": prompt_result["prompt_text"],
                "family_name": family_name,
                "family_display_name": summary["family_display_name"],
                "num_models": summary["num_models"],
                "num_pairs": len(summary["pair_labels"]),
                "skipped_count": len(skipped_by_family.get(family_name, [])),
                "exact_full_action_agreement": summary[base.EXACT_ACTION_METRIC]["mean_pairwise"],
                "semantic_button_group_agreement": summary[base.SEMANTIC_METRIC]["mean_pairwise"],
                "camera_action_agreement": summary[base.CAMERA_METRIC]["mean_pairwise"],
            }
        )

    csv_path = output_dir / "family_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "prompt_slug",
                "prompt_text",
                "family_name",
                "family_display_name",
                "num_models",
                "num_pairs",
                "skipped_count",
                "exact_full_action_agreement",
                "semantic_button_group_agreement",
                "camera_action_agreement",
            ]
        )
        for row in summary_rows:
            writer.writerow(
                [
                    row["prompt_slug"],
                    row["prompt_text"],
                    row["family_name"],
                    row["family_display_name"],
                    row["num_models"],
                    row["num_pairs"],
                    row["skipped_count"],
                    f"{row['exact_full_action_agreement']:.8f}",
                    f"{row['semantic_button_group_agreement']:.8f}",
                    f"{row['camera_action_agreement']:.8f}",
                ]
            )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    base._plot_cross_family_metric_bars(family_summaries, plots_dir / "family_metric_comparison.png")

    summary_json = {
        "created_at": datetime.now().isoformat(),
        "prompt_slug": prompt_result["prompt_slug"],
        "prompt_text": prompt_result["prompt_text"],
        "prompt_seed": prompt_seed,
        "source_rollout_seed": prompt_result["rollout_seed"],
        "source_observations_zip": prompt_result["observations_zip"],
        "source_real_run_dir": prompt_result["run_dir"],
        "families": summary_rows,
        "skipped_models": skipped_by_family,
    }
    json_path = output_dir / "family_summary.json"
    json_path.write_text(json.dumps(summary_json, indent=2, sort_keys=True))

    txt_lines = [
        f"Prompt Sweep Summary: {prompt_result['prompt_slug']}",
        f"Prompt text: {prompt_result['prompt_text']}",
        f"Prompt seed: {prompt_seed}",
        f"Source rollout seed: {prompt_result['rollout_seed']}",
        f"Source fake env zip: {prompt_result['observations_zip']}",
        "",
    ]
    for row in summary_rows:
        txt_lines.append(f"{row['family_display_name']} ({row['num_models']} models, {row['num_pairs']} pairs)")
        txt_lines.append(f"  Exact full-action agreement: {row['exact_full_action_agreement']:.4f}")
        txt_lines.append(f"  Semantic button-group agreement: {row['semantic_button_group_agreement']:.4f}")
        txt_lines.append(f"  Camera-action agreement: {row['camera_action_agreement']:.4f}")
        if skipped_by_family.get(row["family_name"]):
            skipped_names = ", ".join(item["model_dir_name"] for item in skipped_by_family[row["family_name"]])
            txt_lines.append(f"  Skipped: {skipped_names}")
        txt_lines.append("")
    txt_path = output_dir / "summary.txt"
    txt_path.write_text("\n".join(txt_lines))

    manifest = {
        "created_at": datetime.now().isoformat(),
        "prompt_slug": prompt_result["prompt_slug"],
        "summary_csv": str(csv_path),
        "summary_json": str(json_path),
        "summary_txt": str(txt_path),
        "plots": {"family_metric_comparison_png": str(plots_dir / "family_metric_comparison.png")},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    summary_json["manifest_path"] = str(manifest_path)
    return summary_json


def _plot_family_metric_mean_bars(aggregate_rows: list[dict], output_path: Path) -> None:
    width_in, height_in, dpi = _plot_dimensions()
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    family_labels = [item["family_name"].split("_", 1)[0] for item in aggregate_rows]
    x = np.arange(len(family_labels))
    bar_width = 0.22

    for idx, metric_name in enumerate(base.METRIC_ORDER):
        mean_key = f"{metric_name}_mean"
        std_key = f"{metric_name}_std"
        values = [item[mean_key] for item in aggregate_rows]
        errors = [item[std_key] for item in aggregate_rows]
        offset = (idx - 1) * bar_width
        ax.bar(
            x + offset,
            values,
            yerr=errors,
            capsize=4,
            width=bar_width,
            color=base.METRIC_COLORS[metric_name],
            label=base.METRIC_TITLES[metric_name],
        )
        for xpos, value in zip(x + offset, values):
            ax.text(xpos, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(family_labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Mean pairwise agreement")
    ax.set_title("Agreement Across Prompts")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_prompt_family_heatmap(
    values: np.ndarray,
    prompt_labels: list[str],
    family_labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    width_in, height_in, dpi = _plot_dimensions()
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(family_labels)))
    ax.set_xticklabels(family_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(prompt_labels)))
    ax.set_yticklabels(prompt_labels)
    ax.set_title(title)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            text_color = "white" if value < 0.55 else "black"
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Agreement")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_aggregate_summary(
    output_dir: Path,
    prompt_suite: dict,
    prompt_family_summaries: list[dict],
    skipped_by_family: dict[str, list[dict]],
    family_names: list[str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_prompt_rows = []
    per_family_metric_values = defaultdict(lambda: defaultdict(list))
    family_display_names = {}
    family_num_models = {}
    prompt_order = [item["prompt_slug"] for item in prompt_suite["results"]]

    for prompt_summary in prompt_family_summaries:
        for row in prompt_summary["families"]:
            per_prompt_rows.append(row)
            family_name = row["family_name"]
            family_display_names[family_name] = row["family_display_name"]
            family_num_models[family_name] = row["num_models"]
            per_family_metric_values[family_name][base.EXACT_ACTION_METRIC].append(row["exact_full_action_agreement"])
            per_family_metric_values[family_name][base.SEMANTIC_METRIC].append(row["semantic_button_group_agreement"])
            per_family_metric_values[family_name][base.CAMERA_METRIC].append(row["camera_action_agreement"])

    per_prompt_csv = output_dir / "per_prompt_family_metrics.csv"
    with open(per_prompt_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "prompt_slug",
                "prompt_text",
                "family_name",
                "family_display_name",
                "num_models",
                "num_pairs",
                "skipped_count",
                "exact_full_action_agreement",
                "semantic_button_group_agreement",
                "camera_action_agreement",
            ]
        )
        for row in per_prompt_rows:
            writer.writerow(
                [
                    row["prompt_slug"],
                    row["prompt_text"],
                    row["family_name"],
                    row["family_display_name"],
                    row["num_models"],
                    row["num_pairs"],
                    row["skipped_count"],
                    f"{row['exact_full_action_agreement']:.8f}",
                    f"{row['semantic_button_group_agreement']:.8f}",
                    f"{row['camera_action_agreement']:.8f}",
                ]
            )

    aggregate_rows = []
    for family_name in family_names:
        metric_values = per_family_metric_values[family_name]
        aggregate_rows.append(
            {
                "family_name": family_name,
                "family_display_name": family_display_names[family_name],
                "num_models": family_num_models[family_name],
                "num_prompts": len(metric_values[base.EXACT_ACTION_METRIC]),
                f"{base.EXACT_ACTION_METRIC}_mean": float(np.mean(metric_values[base.EXACT_ACTION_METRIC])),
                f"{base.EXACT_ACTION_METRIC}_std": float(np.std(metric_values[base.EXACT_ACTION_METRIC])),
                f"{base.SEMANTIC_METRIC}_mean": float(np.mean(metric_values[base.SEMANTIC_METRIC])),
                f"{base.SEMANTIC_METRIC}_std": float(np.std(metric_values[base.SEMANTIC_METRIC])),
                f"{base.CAMERA_METRIC}_mean": float(np.mean(metric_values[base.CAMERA_METRIC])),
                f"{base.CAMERA_METRIC}_std": float(np.std(metric_values[base.CAMERA_METRIC])),
                "skipped_count": len(skipped_by_family.get(family_name, [])),
            }
        )

    family_csv = output_dir / "family_aggregate_summary.csv"
    with open(family_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "family_name",
                "family_display_name",
                "num_models",
                "num_prompts",
                "skipped_count",
                "exact_full_action_agreement_mean",
                "exact_full_action_agreement_std",
                "semantic_button_group_agreement_mean",
                "semantic_button_group_agreement_std",
                "camera_action_agreement_mean",
                "camera_action_agreement_std",
            ]
        )
        for row in aggregate_rows:
            writer.writerow(
                [
                    row["family_name"],
                    row["family_display_name"],
                    row["num_models"],
                    row["num_prompts"],
                    row["skipped_count"],
                    f"{row[f'{base.EXACT_ACTION_METRIC}_mean']:.8f}",
                    f"{row[f'{base.EXACT_ACTION_METRIC}_std']:.8f}",
                    f"{row[f'{base.SEMANTIC_METRIC}_mean']:.8f}",
                    f"{row[f'{base.SEMANTIC_METRIC}_std']:.8f}",
                    f"{row[f'{base.CAMERA_METRIC}_mean']:.8f}",
                    f"{row[f'{base.CAMERA_METRIC}_std']:.8f}",
                ]
            )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_family_metric_mean_bars(aggregate_rows, plots_dir / "family_metric_mean_across_prompts.png")

    family_short_labels = [name.split("_", 1)[0] for name in family_names]
    rows_by_prompt_and_family = {(row["prompt_slug"], row["family_name"]): row for row in per_prompt_rows}
    for metric_name, value_key, title, short_name in [
        (base.EXACT_ACTION_METRIC, "exact_full_action_agreement", "Exact Full-Action Agreement Across Prompts", "exact_action"),
        (base.SEMANTIC_METRIC, "semantic_button_group_agreement", "Semantic Button-Group Agreement Across Prompts", "semantic_buttons"),
        (base.CAMERA_METRIC, "camera_action_agreement", "Camera-Action Agreement Across Prompts", "camera_action"),
    ]:
        values = np.zeros((len(prompt_order), len(family_names)), dtype=np.float64)
        for prompt_idx, prompt_slug in enumerate(prompt_order):
            for family_idx, family_name in enumerate(family_names):
                values[prompt_idx, family_idx] = rows_by_prompt_and_family[(prompt_slug, family_name)][value_key]
        _plot_prompt_family_heatmap(
            values,
            prompt_order,
            family_short_labels,
            title,
            plots_dir / f"{short_name}_prompt_family_heatmap.png",
        )

    summary_json = {
        "created_at": datetime.now().isoformat(),
        "source_real_prompt_manifest": prompt_suite["manifest_path"],
        "source_suite_root": prompt_suite["suite_root"],
        "prompt_seed": prompt_suite["prompt_seed"],
        "rollout_seed_mode": prompt_suite["rollout_seed_mode"],
        "num_prompts": len(prompt_order),
        "families": aggregate_rows,
        "per_prompt_rows": per_prompt_rows,
        "skipped_models": skipped_by_family,
    }
    json_path = output_dir / "family_aggregate_summary.json"
    json_path.write_text(json.dumps(summary_json, indent=2, sort_keys=True))

    txt_lines = [
        "Real Prompt Suite Fake-Rollout Setting Sweep Summary",
        f"Source real prompt manifest: {prompt_suite['manifest_path']}",
        f"Prompt seed: {prompt_suite['prompt_seed']}",
        f"Rollout seed mode: {prompt_suite['rollout_seed_mode']}",
        f"Num prompts: {len(prompt_order)}",
        "",
    ]
    for row in aggregate_rows:
        txt_lines.append(f"{row['family_display_name']} ({row['num_models']} models across {row['num_prompts']} prompts)")
        txt_lines.append(
            f"  Exact full-action agreement: {row[f'{base.EXACT_ACTION_METRIC}_mean']:.4f} ± {row[f'{base.EXACT_ACTION_METRIC}_std']:.4f}"
        )
        txt_lines.append(
            f"  Semantic button-group agreement: {row[f'{base.SEMANTIC_METRIC}_mean']:.4f} ± {row[f'{base.SEMANTIC_METRIC}_std']:.4f}"
        )
        txt_lines.append(
            f"  Camera-action agreement: {row[f'{base.CAMERA_METRIC}_mean']:.4f} ± {row[f'{base.CAMERA_METRIC}_std']:.4f}"
        )
        if skipped_by_family.get(row["family_name"]):
            skipped_names = ", ".join(item["model_dir_name"] for item in skipped_by_family[row["family_name"]])
            txt_lines.append(f"  Skipped: {skipped_names}")
        txt_lines.append("")
    txt_path = output_dir / "summary.txt"
    txt_path.write_text("\n".join(txt_lines))

    manifest = {
        "created_at": datetime.now().isoformat(),
        "per_prompt_family_metrics_csv": str(per_prompt_csv),
        "family_aggregate_summary_csv": str(family_csv),
        "family_aggregate_summary_json": str(json_path),
        "summary_txt": str(txt_path),
        "plots": {
            "family_metric_mean_across_prompts_png": str(plots_dir / "family_metric_mean_across_prompts.png"),
            "exact_action_prompt_family_heatmap_png": str(plots_dir / "exact_action_prompt_family_heatmap.png"),
            "semantic_buttons_prompt_family_heatmap_png": str(plots_dir / "semantic_buttons_prompt_family_heatmap.png"),
            "camera_action_prompt_family_heatmap_png": str(plots_dir / "camera_action_prompt_family_heatmap.png"),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    summary_json["manifest_path"] = str(manifest_path)
    return summary_json


def run_sweep(
    output_dir: Path,
    real_prompt_manifest: Path,
    prompt_slugs: list[str] | None = None,
    family_names: list[str] | None = None,
) -> dict:
    prompt_suite = _load_prompt_suite(real_prompt_manifest)
    prompt_seed = prompt_suite["prompt_seed"]
    selected_prompt_slugs = set(prompt_slugs) if prompt_slugs else None
    selected_family_names = family_names if family_names else list(base.FAMILY_SPECS.keys())

    if selected_prompt_slugs is not None:
        prompt_suite["results"] = [item for item in prompt_suite["results"] if item["prompt_slug"] in selected_prompt_slugs]
    if not prompt_suite["results"]:
        raise ValueError("No prompts selected for sweep")

    prompts_root = output_dir / "prompts"
    aggregate_root = output_dir / "aggregate"
    prompts_root.mkdir(parents=True, exist_ok=True)
    aggregate_root.mkdir(parents=True, exist_ok=True)

    skipped_by_family = {}
    model_specs_by_family = {}
    for family_name in selected_family_names:
        model_specs, skipped = base._discover_model_specs(family_name, base.SNAPSHOT_NAME)
        model_specs_by_family[family_name] = model_specs
        skipped_by_family[family_name] = skipped

    prompt_family_summaries = []
    prompt_manifests = {}

    for prompt_result in prompt_suite["results"]:
        prompt_dir = prompts_root / prompt_result["prompt_slug"]
        rollouts_root = prompt_dir / "rollouts"
        comparisons_root = prompt_dir / "comparisons"
        summary_root = prompt_dir / "summary"
        rollouts_root.mkdir(parents=True, exist_ok=True)
        comparisons_root.mkdir(parents=True, exist_ok=True)
        summary_root.mkdir(parents=True, exist_ok=True)

        family_summaries = []
        prompt_rollout_metadata = {}
        for family_name in selected_family_names:
            family_rollouts_dir = rollouts_root / family_name
            family_rollouts_dir.mkdir(parents=True, exist_ok=True)

            rollout_metadata = []
            for model_spec in model_specs_by_family[family_name]:
                print(
                    f"[rollout] {prompt_result['prompt_slug']} {family_name} "
                    f"{model_spec.label} -> {model_spec.weights_path}",
                    flush=True,
                )
                rollout_metadata.append(_launch_rollout(model_spec, prompt_result, prompt_seed, family_rollouts_dir))
            prompt_rollout_metadata[family_name] = rollout_metadata

            family_comparison_dir = comparisons_root / family_name
            print(f"[compare] {prompt_result['prompt_slug']} {family_name} ({len(rollout_metadata)} models)", flush=True)
            family_summaries.append(base._compare_family_rollouts(family_name, rollout_metadata, family_comparison_dir))

        prompt_summary = _write_prompt_summary(summary_root, prompt_result, prompt_seed, family_summaries, skipped_by_family)
        prompt_family_summaries.append(prompt_summary)
        prompt_manifests[prompt_result["prompt_slug"]] = {
            "prompt_slug": prompt_result["prompt_slug"],
            "prompt_text": prompt_result["prompt_text"],
            "source_observations_zip": prompt_result["observations_zip"],
            "source_rollout_seed": prompt_result["rollout_seed"],
            "rollouts_root": str(rollouts_root),
            "comparisons_root": str(comparisons_root),
            "summary_root": str(summary_root),
            "summary_manifest": str(summary_root / "manifest.json"),
            "families": prompt_rollout_metadata,
        }

    aggregate_summary = _write_aggregate_summary(
        aggregate_root,
        prompt_suite,
        prompt_family_summaries,
        skipped_by_family,
        selected_family_names,
    )
    top_manifest = {
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "source_real_prompt_manifest": prompt_suite["manifest_path"],
        "source_suite_root": prompt_suite["suite_root"],
        "prompts_root": str(prompts_root),
        "aggregate_root": str(aggregate_root),
        "prompt_seed": prompt_suite["prompt_seed"],
        "rollout_seed_mode": prompt_suite["rollout_seed_mode"],
        "snapshot_name": base.SNAPSHOT_NAME,
        "prompt_manifests": prompt_manifests,
        "aggregate_summary_manifest": str(aggregate_root / "manifest.json"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(top_manifest, indent=2, sort_keys=True))
    return aggregate_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_prompt_manifest", type=Path, default=DEFAULT_REAL_PROMPT_MANIFEST)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--prompt_slug", action="append", default=None)
    parser.add_argument("--family", action="append", default=None, choices=list(base.FAMILY_SPECS.keys()))
    args = parser.parse_args()

    source_id = args.real_prompt_manifest.resolve().parent.name
    output_dir = args.output_dir.resolve() if args.output_dir else OUTPUT_ROOT / source_id / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_sweep(output_dir, args.real_prompt_manifest.resolve(), prompt_slugs=args.prompt_slug, family_names=args.family)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
