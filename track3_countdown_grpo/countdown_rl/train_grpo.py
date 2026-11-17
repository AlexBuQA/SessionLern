"""GRPO + LoRA training for one verifier condition and one seed."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from countdown_rl.data import read_jsonl
from countdown_rl.prompts import messages
from countdown_rl.reward import make_trl_reward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/full.json"))
    parser.add_argument("--condition", choices=["strict", "naive"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, default=Path("data/train.jsonl"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    settings = config["training"]

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        print("WARNING: GRPO is running on CPU; use Colab T4 or another CUDA GPU for the full study.", flush=True)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], dtype=dtype, attn_implementation="sdpa"
    )
    model.config.use_cache = False

    puzzles = read_jsonl(args.train_data)
    dataset_rows = [{
        "prompt": messages(row["numbers"], row["target"]),
        "numbers": row["numbers"],
        "target": row["target"],
        "puzzle_id": row["id"],
    } for row in puzzles]
    dataset = Dataset.from_list(dataset_rows)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    training_args = GRPOConfig(
        output_dir=str(args.output),
        run_name=f"countdown_{args.condition}_seed_{args.seed}",
        max_steps=int(settings["max_steps"]),
        num_train_epochs=1,
        learning_rate=float(settings["learning_rate"]),
        per_device_train_batch_size=int(settings["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(settings["gradient_accumulation_steps"]),
        num_generations=int(settings["num_generations"]),
        max_completion_length=int(settings["max_completion_length"]),
        temperature=1.0,
        top_p=1.0,
        beta=float(settings["beta"]),
        loss_type="dapo",
        scale_rewards="group",
        gradient_checkpointing=True,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=int(settings["logging_steps"]),
        logging_first_step=True,
        save_strategy="steps",
        save_steps=int(settings["save_steps"]),
        save_only_model=True,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        use_vllm=bool(settings.get("use_vllm", False)),
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        chat_template_kwargs={"enable_thinking": False},
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_trl_reward(args.condition),
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output / "final")
    tokenizer.save_pretrained(args.output / "final")
    (args.output / "trainer_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metadata = {
        "condition": args.condition,
        "seed": args.seed,
        "base_model": config["model"],
        "settings": settings,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["torch", "trl", "transformers", "peft", "datasets", "accelerate"]},
        "cuda": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
