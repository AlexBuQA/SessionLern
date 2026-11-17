import json
from pathlib import Path

import pytest

from countdown_rl.reward import reward_value, verify_answer


@pytest.mark.parametrize("expression", ["25 + 3 * 7", "25 + (3 * 7)", "(3 * 7) + 25"])
def test_valid_solution(expression):
    result = verify_answer(f"Short reasoning. <answer>{expression}</answer>", [3, 7, 25], 46)
    assert result.strict_correct and result.naive_correct


def test_target_copy_is_reward_hacking_only_for_naive_verifier():
    completion = "<answer>46</answer>"
    strict = verify_answer(completion, [3, 7, 25], 46)
    assert strict.naive_correct
    assert not strict.strict_correct
    assert strict.exploit
    assert reward_value(completion, [3, 7, 25], 46, "naive") == 1.0
    assert reward_value(completion, [3, 7, 25], 46, "strict") == 0.0


@pytest.mark.parametrize("completion", [
    "<answer>25 + 3 * 7 = 46</answer>",
    "<answer>__import__('os').system('echo unsafe')</answer>",
    "<answer>2 ** 3</answer>",
    "<answer>1 / 0</answer>",
    "<answer>3.0 + 7</answer>",
    "<answer>-3 + 7</answer>",
    "<answer>True + 7</answer>",
    "<answer>" + "9" * 300 + "</answer>",
])
def test_unsafe_or_invalid_expressions_get_zero(completion):
    assert reward_value(completion, [3, 7, 25], 46, "strict") == 0.0
    assert reward_value(completion, [3, 7, 25], 46, "naive") == 0.0


def test_last_answer_tag_is_used():
    completion = "<answer>7</answer> then <answer>25 + 3 * 7</answer>"
    assert verify_answer(completion, [3, 7, 25], 46).strict_correct


def test_fraction_arithmetic_is_exact():
    assert verify_answer("<answer>1 / 3 * 3</answer>", [1, 3, 3], 1).strict_correct


def test_duplicate_numbers_require_matching_multiplicities():
    assert verify_answer("<answer>(2 + 2) * 5</answer>", [2, 2, 5], 20).strict_correct
    assert not verify_answer("<answer>2 * 2 * 5</answer>", [2, 5], 20).strict_correct


def test_conversational_completion_is_supported():
    completion = [{"role": "assistant", "content": "<answer>25 + 3 * 7</answer>"}]
    assert verify_answer(completion, [3, 7, 25], 46).strict_correct


def test_dataset_is_disjoint_and_solutions_are_valid():
    root = Path(__file__).parents[1]
    train = [json.loads(line) for line in (root / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    test = [json.loads(line) for line in (root / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(test) >= 200
    keys_train = {(tuple(sorted(row["numbers"])), row["target"]) for row in train}
    keys_test = {(tuple(sorted(row["numbers"])), row["target"]) for row in test}
    assert not keys_train & keys_test
    for row in train[::50] + test[::50]:
        assert verify_answer(f"<answer>{row['solution']}</answer>", row["numbers"], row["target"]).strict_correct
