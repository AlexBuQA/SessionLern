"""Deterministic generation of solvable, disjoint Countdown puzzles."""
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path

from countdown_rl.reward import verify_answer

OPERATIONS = ("+", "-", "*", "/")


def build_solvable_expression(rng: random.Random, numbers: list[int]):
    items = [(Fraction(number), str(number)) for number in numbers]
    while len(items) > 1:
        i, j = sorted(rng.sample(range(len(items)), 2), reverse=True)
        right_value, right_expression = items.pop(i)
        left_value, left_expression = items.pop(j)
        operation = rng.choice(OPERATIONS)
        if operation == "+":
            value = left_value + right_value
        elif operation == "-":
            value = left_value - right_value
        elif operation == "*":
            value = left_value * right_value
        else:
            if right_value == 0:
                return None
            value = left_value / right_value
        items.append((value, f"({left_expression} {operation} {right_expression})"))
    return items[0]


def make_puzzle(rng: random.Random, n_numbers: int, low: int, high: int, max_target: int):
    numbers = [rng.randint(low, high) for _ in range(n_numbers)]
    built = build_solvable_expression(rng, numbers)
    if built is None:
        return None
    value, expression = built
    if value.denominator != 1 or not 1 <= value <= max_target:
        return None
    target = int(value)
    if target in numbers or target == sum(numbers):
        return None
    rng.shuffle(numbers)
    # Self-check through the same strict verifier used for training rewards.
    check = verify_answer(f"<answer>{expression}</answer>", numbers, target)
    if not check.strict_correct:
        return None
    return {"numbers": numbers, "target": target, "solution": expression, "n_numbers": n_numbers}


def generate_split(rng: random.Random, size: int, settings: dict, used_keys: set, split: str):
    puzzles = []
    attempts = 0
    max_attempts = size * 500
    while len(puzzles) < size and attempts < max_attempts:
        attempts += 1
        n_numbers = 3 if rng.random() < settings["three_number_fraction"] else 4
        high = settings["max_number_3"] if n_numbers == 3 else settings["max_number_4"]
        puzzle = make_puzzle(rng, n_numbers, settings["min_number"], high, settings["max_target"])
        if puzzle is None:
            continue
        key = (tuple(sorted(puzzle["numbers"])), puzzle["target"])
        if key in used_keys:
            continue
        used_keys.add(key)
        puzzle.update(id=f"{split}_{len(puzzles):04d}", split=split)
        puzzles.append(puzzle)
    if len(puzzles) != size:
        raise RuntimeError(f"Could only generate {len(puzzles)}/{size} unique {split} puzzles")
    return puzzles


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generate_dataset(data_dir: Path, settings: dict) -> dict:
    rng = random.Random(settings["seed"])
    used_keys: set = set()
    train = generate_split(rng, settings["train_size"], settings, used_keys, "train")
    test = generate_split(rng, settings["test_size"], settings, used_keys, "test")
    write_jsonl(data_dir / "train.jsonl", train)
    write_jsonl(data_dir / "test.jsonl", test)
    metadata = {
        "settings": settings,
        "train_size": len(train),
        "test_size": len(test),
        "overlap_count": 0,
        "generation_attempt_note": "Only provably solvable, non-trivial, globally unique puzzles are retained.",
    }
    (data_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/full.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(generate_dataset(args.data_dir, config["data"]), indent=2))


if __name__ == "__main__":
    main()
