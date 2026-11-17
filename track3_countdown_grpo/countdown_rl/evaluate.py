"""Greedy held-out evaluation of the base model or a saved LoRA adapter."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from countdown_rl.data import read_jsonl, write_jsonl
from countdown_rl.prompts import messages
from countdown_rl.reward import verify_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--test-data", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import torch
    import pandas as pd
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(20261117)
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, attn_implementation="sdpa"
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    puzzles = read_jsonl(args.test_data)
    if args.limit is not None:
        puzzles = puzzles[:args.limit]
    rows = []
    for start in range(0, len(puzzles), args.batch_size):
        batch = puzzles[start:start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                messages(row["numbers"], row["target"]),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in batch
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = generated[:, encoded["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
        for puzzle, completion, token_ids in zip(batch, texts, completions):
            result = verify_answer(completion, puzzle["numbers"], puzzle["target"])
            rows.append({
                "id": puzzle["id"], "numbers": puzzle["numbers"], "target": puzzle["target"],
                "reference_solution": puzzle["solution"], "response": completion,
                "response_tokens": int((token_ids != tokenizer.pad_token_id).sum().item()),
                "response_chars": len(completion), **result.to_dict(),
            })
        print(f"Evaluated {min(start + len(batch), len(puzzles))}/{len(puzzles)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "predictions.jsonl", rows)
    pd.DataFrame(rows).to_csv(args.output / "predictions.csv", index=False)
    count = max(1, len(rows))
    summary = {
        "condition": args.condition, "seed": args.seed, "step": args.step,
        "n_tasks": len(rows),
        "accuracy": sum(row["strict_correct"] for row in rows) / count,
        "naive_accuracy": sum(row["naive_correct"] for row in rows) / count,
        "exploit_rate": sum(row["exploit"] for row in rows) / count,
        "format_rate": sum(row["format_ok"] for row in rows) / count,
        "syntax_rate": sum(row["syntax_ok"] for row in rows) / count,
        "numbers_ok_rate": sum(row["numbers_ok"] for row in rows) / count,
        "mean_response_tokens": sum(row["response_tokens"] for row in rows) / count,
        "mean_response_chars": sum(row["response_chars"] for row in rows) / count,
        "base_model": args.base_model, "adapter": str(args.adapter) if args.adapter else None,
        "decoding": "greedy", "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft"]},
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
