"""Run baseline, controlled GRPO arms, checkpoint evaluation and reporting."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from countdown_rl.data import generate_dataset
from countdown_rl.report import build_report


def run_command(command: list[str], cwd: Path) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/full.json"))
    parser.add_argument("--output", type=Path, default=Path("results/full"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs/full"))
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    results_dir, outputs_dir = args.output.resolve(), args.outputs.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.collect_only:
        build_report(results_dir, outputs_dir, config)
        return

    metadata = generate_dataset(root / "data", config["data"])
    print(f"Dataset: {metadata['train_size']} train, {metadata['test_size']} test, overlap=0")
    evaluation = config["evaluation"]
    limit = evaluation.get("limit")

    def evaluate(adapter, condition, seed, step, destination):
        summary_path = destination / "summary.json"
        if summary_path.exists():
            return
        command = [sys.executable, "-m", "countdown_rl.evaluate", "--base-model", config["model"],
                   "--test-data", "data/test.jsonl", "--output", str(destination),
                   "--condition", condition, "--seed", str(seed), "--step", str(step),
                   "--batch-size", str(evaluation["batch_size"]),
                   "--max-new-tokens", str(evaluation["max_new_tokens"])]
        if adapter is not None:
            command += ["--adapter", str(adapter)]
        if limit is not None:
            command += ["--limit", str(limit)]
        run_command(command, root)

    base_dir = results_dir / "base" / "step_0"
    evaluate(None, "base", 0, 0, base_dir)
    baseline = json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))
    full_protocol = int(baseline["n_tasks"]) >= 200 and int(config["training"]["max_steps"]) >= 50
    if full_protocol and not 0.10 <= baseline["accuracy"] <= 0.50:
        raise RuntimeError(
            f"Base accuracy {baseline['accuracy']:.3f} is outside the required 0.10–0.50 range. "
            "Change data difficulty, regenerate the fixed split and rerun baseline before training."
        )

    for condition in config["training"]["conditions"]:
        for seed in config["training"]["seeds"]:
            run_dir = outputs_dir / condition / f"seed_{seed}"
            if not (run_dir / "final" / "adapter_config.json").exists():
                run_command([sys.executable, "-m", "countdown_rl.train_grpo", "--config", str(args.config),
                             "--condition", condition, "--seed", str(seed), "--output", str(run_dir)], root)
            checkpoints = []
            for path in run_dir.glob("checkpoint-*"):
                try:
                    checkpoints.append((int(path.name.split("-")[-1]), path))
                except ValueError:
                    pass
            checkpoints.append((int(config["training"]["max_steps"]), run_dir / "final"))
            for step, adapter in sorted(checkpoints):
                evaluate(adapter, condition, seed, step, results_dir / condition / f"seed_{seed}" / f"step_{step}")
    build_report(results_dir, outputs_dir, config)
    print(f"Done: {results_dir}")


if __name__ == "__main__":
    main()
