"""Prompt construction shared by GRPO training and greedy evaluation."""

SYSTEM_PROMPT = (
    "You solve Countdown arithmetic puzzles precisely. Use every supplied number exactly once. "
    "Only +, -, *, / and parentheses are allowed. Do not concatenate digits or add constants. "
    "Put only the final expression inside <answer></answer> tags. Think briefly without long reasoning."
)


def user_prompt(numbers: list[int], target: int) -> str:
    joined = ", ".join(map(str, numbers))
    return (
        f"Numbers: {joined}. Target: {target}.\n"
        "Use each number exactly once and produce an expression exactly equal to the target. "
        "Do not append '= target'. Return <answer>expression</answer>."
    )


def messages(numbers: list[int], target: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(numbers, target)},
    ]
