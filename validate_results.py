"""Validate generated research artifacts against each saved full configuration."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require_files(folder: Path, names: list[str]) -> None:
    for name in names:
        path = folder / name
        assert path.is_file() and path.stat().st_size > 0, f"Missing or empty: {path}"


def validate_track1(root: Path) -> None:
    folder = root / "track1_acrobot" / "results" / "full"
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    episodes = csv_rows(folder / "evaluation_episodes.csv")
    expected = 3 * len(config["seeds"]) * len(config["eval_noise_levels"]) * config["eval_episodes"]
    assert len(episodes) == expected, f"Track 1: {len(episodes)} episodes, expected {expected}"
    assert {int(row["seed"]) for row in episodes if row["condition"] == "fixed"} == set(config["seeds"])
    require_files(folder, ["RESULTS.md", "summary.csv", "learning_curves.png", "robustness_return.png",
                           "robustness_success.png", "fixed_seed_11_noise_0.5.gif",
                           "randomized_seed_11_noise_0.5.gif", "runtime.json"])
    print(f"Track 1 artifacts OK: {expected} evaluation episodes")


def validate_track2(root: Path) -> None:
    folder = root / "track2_inventory" / "results" / "full"
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    episodes = csv_rows(folder / "evaluation_episodes.csv")
    expected = 4 * len(config["seeds"]) * len(config["eval_demand_means"]) * config["eval_episodes"]
    assert len(episodes) == expected, f"Track 2: {len(episodes)} episodes, expected {expected}"
    assert {int(row["seed"]) for row in episodes if row["condition"] == "fixed"} == set(config["seeds"])
    require_files(folder, ["RESULTS.md", "summary.csv", "learning_curves.png", "demand_shift_profit.png",
                           "demand_shift_service.png", "trajectories.png", "policy_heatmap.png", "runtime.json"])
    print(f"Track 2 artifacts OK: {expected} evaluation episodes")


def validate_track3(root: Path) -> None:
    folder = root / "track3_countdown_grpo" / "data"
    train = [json.loads(line) for line in (folder / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    test = [json.loads(line) for line in (folder / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train) == 1200 and len(test) >= 200
    train_keys = {(tuple(sorted(row["numbers"])), row["target"]) for row in train}
    test_keys = {(tuple(sorted(row["numbers"])), row["target"]) for row in test}
    assert len(train_keys) == len(train) and len(test_keys) == len(test)
    assert train_keys.isdisjoint(test_keys)
    print(f"Track 3 data OK: {len(train)} train, {len(test)} held-out, no overlap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-full", action="store_true", help="Fail if full Track 1 results are not ready")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    track1_ready = (root / "track1_acrobot" / "results" / "full" / "runtime.json").exists()
    if track1_ready or args.require_full:
        validate_track1(root)
    validate_track2(root)
    validate_track3(root)
    print("Artifact validation passed.")


if __name__ == "__main__":
    main()
