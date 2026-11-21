import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(os.environ.get("MULTISTEVE_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUTPUT_ROOT = Path(os.environ.get("MULTISTEVE_ACTION_PROB_OUTPUT_ROOT", "outputs/action_probabilities/real_prompt_suite")).resolve()
MODEL_WEIGHTS = Path(
    os.environ.get(
        "MULTISTEVE_MODEL_WEIGHTS",
        "data/model_zoo/03_init_and_data/dataset_seed1-weight_init_seed1/STEVE-1_snapshot_160005120.weights",
    )
).resolve()
IN_MODEL = "data/weights/vpt/2x.model"
PRIOR_WEIGHTS = "data/weights/steve1/steve1_prior.pt"
TEXT_COND_SCALE = 6.0
GAMEPLAY_LENGTH = 1000
PROMPT_SEED = 49
ROLLOUT_SEED_MODE = "random"
MAX_ATTEMPTS_PER_PROMPT = 5

PROMPTS = [
    ("dig", "dig as far as possible"),
    ("dirt", "get dirt"),
    ("sky", "look at the sky"),
    ("leaves", "break leaves"),
    ("wood", "chop a tree"),
    ("seeds", "collect seeds"),
    ("flower", "break a flower"),
    ("explore", "go explore"),
    ("swim", "go swimming"),
    ("underwater", "go underwater"),
    ("inventory", "open inventory"),
    ("dirt_engineered", "get dirt, dig hole, dig dirt, gather a ton of dirt, collect dirt"),
    ("wood_engineered", "chop down the tree, gather wood, pick up wood, chop it down, break tree"),
    ("seeds_engineered", "break tall grass, break grass, collect seeds, punch the ground, run around in circles getting seeds from bushes"),
]


@dataclass(frozen=True)
class PromptSpec:
    slug: str
    text: str


def _python_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{REPO_ROOT}{':' + existing if existing else ''}"
    return env


def _resolve_run_dir(save_dir: Path) -> Path | None:
    candidates = [
        child
        for child in sorted(save_dir.iterdir())
        if child.is_dir() and (child / "video.mp4").exists() and (child / "command.log").exists()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    raise ValueError(f"Expected at most one run directory under {save_dir}, found {len(candidates)}")


def _sample_rollout_seeds(num_prompts: int) -> list[int]:
    rng = random.SystemRandom()
    chosen = set()
    seeds = []
    while len(seeds) < num_prompts:
        seed = rng.randrange(0, 2**31 - 1)
        if seed in chosen:
            continue
        chosen.add(seed)
        seeds.append(seed)
    return seeds


def _sample_rollout_seed(exclude: set[int]) -> int:
    rng = random.SystemRandom()
    while True:
        seed = rng.randrange(0, 2**31 - 1)
        if seed not in exclude:
            return seed


def _launch_prompt(prompt: PromptSpec, initial_rollout_seed: int, prompts_root: Path) -> dict:
    save_dir = prompts_root / prompt.slug
    save_dir.mkdir(parents=True, exist_ok=True)

    existing_run = _resolve_run_dir(save_dir)
    if existing_run is not None:
        command = shlex.split((existing_run / "command.log").read_text())
        # Preserve the seed of the saved episode, not a newly sampled seed.
        rollout_seed = int(command[command.index("--seed") + 1]) if "--seed" in command else None
        return {
            "prompt_slug": prompt.slug,
            "prompt_text": prompt.text,
            "rollout_seed": rollout_seed,
            "save_dir": str(save_dir),
            "run_dir": str(existing_run),
            "attempts": [],
            "status": "reused_existing",
        }

    launcher_log = save_dir / "launcher.log"
    attempts = []
    used_seeds = set()
    rollout_seed = initial_rollout_seed
    run_dir = None

    for attempt_index in range(1, MAX_ATTEMPTS_PER_PROMPT + 1):
        used_seeds.add(rollout_seed)
        cmd = [
            sys.executable,
            "-m",
            "steve1.run_agent.run_agent_with_logging",
            "--in_model",
            IN_MODEL,
            "--in_weights",
            str(MODEL_WEIGHTS),
            "--prior_weights",
            PRIOR_WEIGHTS,
            "--text_cond_scale",
            str(TEXT_COND_SCALE),
            "--gameplay_length",
            str(GAMEPLAY_LENGTH),
            "--save_dirpath",
            str(save_dir),
            "--custom_text_prompt",
            prompt.text,
            "--custom_text_seed",
            str(PROMPT_SEED),
            "--seed",
            str(rollout_seed),
            "--save_additional_data",
        ]
        start_time = time.time()
        attempt_record = {
            "attempt_index": attempt_index,
            "rollout_seed": rollout_seed,
            "command": " ".join(cmd),
        }
        with open(launcher_log, "a") as log_file:
            log_file.write(f"[{datetime.now().isoformat()}] Launching prompt {prompt.slug} attempt {attempt_index} seed {rollout_seed}\n")
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
                attempt_record["status"] = "completed"
                attempt_record["elapsed_seconds"] = elapsed
                attempts.append(attempt_record)
                run_dir = _resolve_run_dir(save_dir)
                break
            except subprocess.CalledProcessError as exc:
                elapsed = time.time() - start_time
                attempt_record["status"] = "failed"
                attempt_record["elapsed_seconds"] = elapsed
                attempt_record["return_code"] = exc.returncode
                attempts.append(attempt_record)
                log_file.write(f"\n[{datetime.now().isoformat()}] Attempt {attempt_index} failed after {elapsed:.2f}s with code {exc.returncode}\n")
        rollout_seed = _sample_rollout_seed(used_seeds)

    if run_dir is None:
        raise RuntimeError(f"Prompt {prompt.slug} failed after {MAX_ATTEMPTS_PER_PROMPT} attempts; see {launcher_log}")

    return {
        "prompt_slug": prompt.slug,
        "prompt_text": prompt.text,
        "rollout_seed": attempts[-1]["rollout_seed"],
        "save_dir": str(save_dir),
        "run_dir": str(run_dir),
        "attempts": attempts,
        "status": "ran_new" if len(attempts) == 1 else "ran_new_after_retry",
    }


def run_suite(output_dir: Path) -> dict:
    prompts_root = output_dir / "episodes"
    prompts_root.mkdir(parents=True, exist_ok=True)

    prompt_specs = [PromptSpec(slug=slug, text=text) for slug, text in PROMPTS]
    rollout_seeds = _sample_rollout_seeds(len(prompt_specs))
    results = []
    for prompt, rollout_seed in zip(prompt_specs, rollout_seeds):
        print(f"[episode] {prompt.slug}: {prompt.text} | env_seed={rollout_seed}", flush=True)
        results.append(_launch_prompt(prompt, rollout_seed, prompts_root))

    summary = {
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "episodes_root": str(prompts_root),
        "model_weights": str(MODEL_WEIGHTS),
        "in_model": IN_MODEL,
        "prior_weights": PRIOR_WEIGHTS,
        "text_cond_scale": TEXT_COND_SCALE,
        "gameplay_length": GAMEPLAY_LENGTH,
        "prompt_seed": PROMPT_SEED,
        "rollout_seed_mode": ROLLOUT_SEED_MODE,
        "max_attempts_per_prompt": MAX_ATTEMPTS_PER_PROMPT,
        "num_prompts": len(prompt_specs),
        "prompts": [asdict(prompt) for prompt in prompt_specs],
        "results": results,
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve() if args.output_dir else OUTPUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_suite(output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
