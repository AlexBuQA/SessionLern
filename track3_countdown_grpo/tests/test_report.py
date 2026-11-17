"""Report aggregation test with explicitly synthetic temporary metrics (never research results)."""
import json
from pathlib import Path

from countdown_rl.data import write_jsonl
from countdown_rl.report import build_report


def _summary(condition, seed, step, accuracy, naive_accuracy, exploit_rate):
    return {
        "condition": condition, "seed": seed, "step": step, "n_tasks": 240,
        "accuracy": accuracy, "naive_accuracy": naive_accuracy, "exploit_rate": exploit_rate,
        "format_rate": 1.0, "syntax_rate": 1.0, "numbers_ok_rate": 1.0 - exploit_rate,
        "mean_response_tokens": 12.0, "mean_response_chars": 30.0,
    }


def test_report_pipeline_uses_metrics_and_writes_required_artifacts(tmp_path):
    results, outputs = tmp_path / "results", tmp_path / "outputs"
    config = {"training": {"conditions": ["strict", "naive"], "seeds": [42],
                           "max_steps": 300, "beta": 0.01}}
    for condition, reward in [("strict", 0.4), ("naive", 0.8)]:
        run = outputs / condition / "seed_42"
        run.mkdir(parents=True)
        (run / "trainer_log.json").write_text(json.dumps([
            {"step": 50, "reward": reward, "kl": 0.1, "completions/mean_length": 12.0}
        ]), encoding="utf-8")
    base_dir = results / "base" / "step_0"
    base_dir.mkdir(parents=True)
    (base_dir / "summary.json").write_text(json.dumps(_summary("base", 0, 0, 0.2, 0.2, 0.0)), encoding="utf-8")
    base_rows = [
        {"id": "test_0", "numbers": [3, 7, 25], "target": 46, "response": "<answer>7</answer>", "strict_correct": False},
        {"id": "test_1", "numbers": [2, 2, 5], "target": 20, "response": "<answer>(2+2)*5</answer>", "strict_correct": True},
    ]
    write_jsonl(base_dir / "predictions.jsonl", base_rows)
    for condition, accuracy, naive_accuracy, exploit in [("strict", 0.5, 0.5, 0.0), ("naive", 0.3, 0.9, 0.6)]:
        folder = results / condition / "seed_42" / "step_50"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps(_summary(condition, 42, 50, accuracy, naive_accuracy, exploit)), encoding="utf-8")
        rows = [dict(row, response="<answer>46</answer>" if condition == "naive" else row["response"],
                     strict_correct=bool(row["strict_correct"]) if condition == "strict" else False,
                     exploit=condition == "naive") for row in base_rows]
        write_jsonl(folder / "predictions.jsonl", rows)
    build_report(results, outputs, config)
    assert (results / "RESULTS.md").exists()
    assert (results / "answer_comparison.csv").exists()
    assert (results / "answer_examples.md").exists()
    for name in ["training_reward.png", "heldout_accuracy.png", "kl_divergence.png", "response_length.png"]:
        assert (results / name).stat().st_size > 0
