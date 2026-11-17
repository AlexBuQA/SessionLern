import json
from pathlib import Path

import pytest


def test_installed_grpo_config_accepts_experiment_parameters(tmp_path):
    pytest.importorskip("trl")
    from trl import GRPOConfig

    root = Path(__file__).parents[1]
    config = json.loads((root / "configs" / "full.json").read_text(encoding="utf-8"))
    settings = config["training"]
    args = GRPOConfig(
        output_dir=str(tmp_path),
        max_steps=settings["max_steps"],
        learning_rate=settings["learning_rate"],
        per_device_train_batch_size=settings["per_device_train_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        num_generations=settings["num_generations"],
        max_completion_length=settings["max_completion_length"],
        beta=settings["beta"],
        loss_type="dapo",
        scale_rewards="group",
        save_only_model=True,
        chat_template_kwargs={"enable_thinking": False},
        use_vllm=False,
        use_cpu=True,
        bf16=False,
        fp16=False,
        report_to=[],
    )
    assert args.beta == 0.01
    generation_batch = settings["per_device_train_batch_size"] * settings["gradient_accumulation_steps"]
    assert generation_batch % settings["num_generations"] == 0
