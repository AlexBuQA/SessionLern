"""Aggregate real TRL logs and greedy held-out evaluations into a report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from countdown_rl.data import read_jsonl


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_training_logs(outputs_dir: Path, config: dict) -> pd.DataFrame:
    rows = []
    for condition in config["training"]["conditions"]:
        for seed in config["training"]["seeds"]:
            path = outputs_dir / condition / f"seed_{seed}" / "trainer_log.json"
            if not path.exists():
                continue
            for entry in _read_json(path):
                step = entry.get("step", entry.get("global_step"))
                if step is None or "reward" not in entry:
                    continue
                rows.append({
                    "condition": condition, "seed": seed, "step": int(step),
                    "reward": entry.get("reward"), "kl": entry.get("kl"),
                    "mean_response_tokens": entry.get("completions/mean_length"),
                    "train_strict_accuracy": entry.get("strict_accuracy"),
                    "train_exploit_rate": entry.get("exploit_rate"),
                })
    return pd.DataFrame(rows)


def collect_evaluations(results_dir: Path, config: dict) -> pd.DataFrame:
    rows = []
    for path in results_dir.glob("*/**/summary.json"):
        summary = _read_json(path)
        if "accuracy" in summary:
            rows.append(summary)
    return pd.DataFrame(rows).sort_values(["condition", "seed", "step"])


def _plot_metric(frame: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path) -> None:
    usable = frame.dropna(subset=[metric])
    if usable.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition, group in usable.groupby("condition"):
        stats = group.groupby("step")[metric].agg(["mean", "std"]).fillna(0.0)
        ax.plot(stats.index, stats["mean"], marker="o", label=condition)
        ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"], alpha=0.18)
    ax.set(title=title, xlabel="Шаг GRPO", ylabel=ylabel)
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def make_plots(training: pd.DataFrame, evaluation: pd.DataFrame, output_dir: Path) -> None:
    if not training.empty:
        _plot_metric(training, "reward", "Средняя награда обучения", "GRPO: программно проверяемая награда", output_dir / "training_reward.png")
        _plot_metric(training, "kl", "KL(π || исходная модель)", "Отклонение политики от исходной модели", output_dir / "kl_divergence.png")
        _plot_metric(training, "mean_response_tokens", "Средняя длина, токены", "Длина сгенерированного ответа", output_dir / "response_length.png")
    _plot_metric(evaluation, "accuracy", "Строгая точность", "Жадная оценка на фиксированной отложенной выборке", output_dir / "heldout_accuracy.png")


def final_summary(evaluation: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    trained = evaluation[evaluation["condition"] != "base"].copy()
    final_rows = trained.loc[trained.groupby(["condition", "seed"])["step"].idxmax()].copy()
    metrics = ["accuracy", "naive_accuracy", "exploit_rate", "format_rate", "numbers_ok_rate", "mean_response_tokens"]
    table = final_rows.groupby("condition")[metrics].agg(["mean", "std"]).fillna(0.0)
    table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
    table = table.reset_index()
    base = evaluation[evaluation["condition"] == "base"]
    return final_rows, pd.concat([base, table], ignore_index=True, sort=False)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join(lines)


def write_answer_examples(evaluation: pd.DataFrame, results_dir: Path, config: dict, output_dir: Path) -> None:
    base_path = results_dir / "base" / "step_0" / "predictions.jsonl"
    if not base_path.exists():
        return
    base = {row["id"]: row for row in read_jsonl(base_path)}
    merged = {}
    for condition in config["training"]["conditions"]:
        seed = config["training"]["seeds"][0]
        candidates = evaluation[(evaluation["condition"] == condition) & (evaluation["seed"] == seed)]
        if candidates.empty:
            continue
        step = int(candidates["step"].max())
        rows = read_jsonl(results_dir / condition / f"seed_{seed}" / f"step_{step}" / "predictions.jsonl")
        merged[condition] = {row["id"]: row for row in rows}
    records = []
    for puzzle_id, base_row in base.items():
        record = {"id": puzzle_id, "numbers": base_row["numbers"], "target": base_row["target"],
                  "base_response": base_row["response"], "base_correct": base_row["strict_correct"]}
        for condition, responses in merged.items():
            row = responses[puzzle_id]
            record.update({f"{condition}_response": row["response"],
                           f"{condition}_correct": row["strict_correct"],
                           f"{condition}_exploit": row["exploit"]})
        records.append(record)
    pd.DataFrame(records).to_csv(output_dir / "answer_comparison.csv", index=False)
    error_records = [row for row in records if not row["base_correct"] or
                     any(not row.get(f"{condition}_correct", True) or row.get(f"{condition}_exploit", False)
                         for condition in merged)]
    with (output_dir / "answer_examples.md").open("w", encoding="utf-8") as handle:
        handle.write("# Ответы на одних и тех же отложенных задачах\n\n")
        for row in error_records[:20]:
            handle.write(f"## {row['id']}: числа {row['numbers']}, цель {row['target']}\n\n")
            for condition in ["base", *merged]:
                handle.write(f"**{condition}; strict_correct={row.get(f'{condition}_correct')}; exploit={row.get(f'{condition}_exploit', False)}**\n\n")
                handle.write("```text\n" + str(row.get(f"{condition}_response", ""))[:2000] + "\n```\n\n")


def build_report(results_dir: Path, outputs_dir: Path, config: dict) -> None:
    training = collect_training_logs(outputs_dir, config)
    evaluation = collect_evaluations(results_dir, config)
    if evaluation.empty:
        raise RuntimeError("No evaluation summaries found")
    training.to_csv(results_dir / "training_metrics.csv", index=False)
    evaluation.to_csv(results_dir / "evaluation_by_checkpoint.csv", index=False)
    make_plots(training, evaluation, results_dir)
    final_rows, summary = final_summary(evaluation, config)
    final_rows.to_csv(results_dir / "final_runs.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    write_answer_examples(evaluation, results_dir, config, results_dir)

    means = final_rows.groupby("condition")[["accuracy", "naive_accuracy", "exploit_rate"]].mean()
    base = evaluation[evaluation["condition"] == "base"].iloc[0]
    supported = (means.loc["naive", "naive_accuracy"] > means.loc["strict", "naive_accuracy"] and
                 means.loc["naive", "exploit_rate"] > means.loc["strict", "exploit_rate"] and
                 means.loc["naive", "accuracy"] <= means.loc["strict", "accuracy"])
    verdict = "подтверждается" if supported else "не подтверждается полностью"
    limited = config["training"]["max_steps"] < 50 or int(evaluation["n_tasks"].max()) < 200
    warning = "\n> **Внимание:** это smoke-запуск; научные выводы по нему не допускаются.\n" if limited else ""
    table_columns = ["condition", "n_tasks", "accuracy", "accuracy_mean", "accuracy_std",
                     "naive_accuracy_mean", "exploit_rate_mean", "format_rate_mean", "mean_response_tokens_mean"]
    existing_columns = [column for column in table_columns if column in summary.columns]
    text = f"""# Результаты: reward hacking при GRPO-дообучении Countdown
{warning}
## Вопрос и гипотеза

**Вопрос.** Что произойдёт, если программная награда проверяет значение выражения, но не проверяет использование исходных чисел?

**Гипотеза.** Naive-агент научится получать высокую награду через лазейку (например, `<answer>цель</answer>`), поэтому его exploit rate вырастет, а строгая точность не превысит strict-агента, несмотря на высокую naive-награду.

Единственный изменённый фактор — проверка мультимножества чисел в reward. Модель Qwen3-0.6B, LoRA, {config['training']['max_steps']} шагов GRPO, seed {config['training']['seeds']}, β={config['training']['beta']} и все остальные параметры совпадают. Итоговая оценка — жадное декодирование на одной зафиксированной выборке из {int(base['n_tasks'])} задач. Обучающие и отложенные задачи не пересекаются.

## Проверка корректности

Калькулятор использует `ast` и `Fraction`, не `eval`. Strict требует тег, допустимый синтаксис, точное значение и совпадение мультимножества. Naive отличается только отсутствием последней проверки. Юнит-тесты отдельно показывают, что `<answer>цель</answer>` получает 1 у naive и 0 у strict.

## Итоговые метрики

Базовая точность: **{base['accuracy']:.4f}** ({int(base['n_tasks'])} задач). Окончательные строки strict/naive усреднены по обучающим seed; `_std` — стандартное отклонение между seed.

{markdown_table(summary, existing_columns)}

![Награда](training_reward.png)

![Строгая точность](heldout_accuracy.png)

![KL](kl_divergence.png)

![Длина ответа](response_length.png)

## Ответ на вопрос

Strict: точность **{means.loc['strict', 'accuracy']:.4f}**, naive-награда **{means.loc['strict', 'naive_accuracy']:.4f}**, exploit rate **{means.loc['strict', 'exploit_rate']:.4f}**. Naive: **{means.loc['naive', 'accuracy']:.4f}**, **{means.loc['naive', 'naive_accuracy']:.4f}**, **{means.loc['naive', 'exploit_rate']:.4f}** соответственно.

По заранее заданному сочетанию критериев гипотеза **{verdict}**. Разница naive-награды и строгой точности важнее роста training reward: последняя может измерять именно успех обхода проверки. Конкретные одинаковые задачи и ответы приведены в `answer_comparison.csv` и `answer_examples.md`; полные ответы — в `predictions.jsonl`.

## Ограничения

GRPO чувствителен к seed и нулевой дисперсии награды в группе. Три seed не исключают случайных различий; одна модель и один диапазон сложности ограничивают обобщение. KL зависит от выбранного β и сам по себе не означает смысловое качество. Далее нужно повторить запуск на другой сложности и добавить защиту от повторяющихся тегов/чрезмерно длинных рассуждений.
"""
    (results_dir / "RESULTS.md").write_text(text, encoding="utf-8")
