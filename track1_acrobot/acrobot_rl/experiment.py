"""Training, matched evaluation, plots and report for the Acrobot study."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from acrobot_rl.environment import make_acrobot


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_episode(model, env: gym.Env, seed: int, random_rng=None) -> dict:
    observation, info = env.reset(seed=seed)
    total_reward, steps, terminated = 0.0, 0, False
    while True:
        if model is None:
            action = int(random_rng.integers(env.action_space.n))
        else:
            action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
    return {
        "return": total_reward,
        "steps": steps,
        "success": bool(terminated),
        "episode_seed": seed,
        "applied_noise": float(env.unwrapped.torque_noise_max),
    }


def evaluate_at_noise(model, noise: float, episodes: int, seed_base: int) -> list[dict]:
    env = make_acrobot(torque_noise=noise)
    rng = np.random.default_rng(seed_base + 77)
    rows = []
    for episode in range(episodes):
        seed = seed_base + episode
        row = run_episode(model, env, seed, rng)
        row.update(noise=float(noise), episode=episode)
        rows.append(row)
    env.close()
    return rows


class PeriodicHeldOutEval(BaseCallback):
    """Evaluate on one fixed noise distribution, not on randomized train returns."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.rows: list[dict] = []
        self.next_step = int(config["learning_eval_interval"])

    def _evaluate(self, step: int) -> None:
        rows = evaluate_at_noise(
            self.model,
            float(self.config["learning_eval_noise"]),
            int(self.config["learning_eval_episodes"]),
            900_000,
        )
        values = [row["return"] for row in rows]
        self.rows.append({
            "step": step,
            "mean_return": float(np.mean(values)),
            "std_return": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "success_rate": float(np.mean([row["success"] for row in rows])),
        })

    def _on_training_start(self) -> None:
        self._evaluate(0)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_step:
            self._evaluate(int(self.num_timesteps))
            self.next_step += int(self.config["learning_eval_interval"])
        return True


def train_one(condition: str, seed: int, config: dict, output_dir: Path) -> tuple[PPO, list[dict]]:
    set_global_seed(seed)
    randomization = tuple(config["randomization_range"]) if condition == "randomized" else None
    env = make_acrobot(randomization_range=randomization, monitor=True)
    env.reset(seed=seed)
    eval_callback = PeriodicHeldOutEval(config)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, **config["ppo"])
    model.learn(total_timesteps=int(config["total_timesteps"]), callback=eval_callback)
    run_dir = output_dir / condition / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model.zip")
    pd.DataFrame(eval_callback.rows).assign(condition=condition, seed=seed).to_csv(
        run_dir / "learning_curve.csv", index=False
    )
    env.close()
    return model, eval_callback.rows


def save_gif(model, condition: str, seed: int, noise: float, output_dir: Path) -> None:
    env = make_acrobot(torque_noise=noise, render_mode="rgb_array")
    observation, _ = env.reset(seed=777_000)
    frames = [env.render()]
    for _ in range(500):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break
    env.close()
    imageio.mimsave(
        output_dir / f"{condition}_seed_{seed}_noise_{noise:.1f}.gif",
        frames[::2],
        duration=2 / 15,
        loop=0,
    )


def aggregate_final(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = (
        episodes.groupby(["condition", "seed", "noise"], as_index=False)
        .agg(mean_return=("return", "mean"), success_rate=("success", "mean"),
             mean_steps=("steps", "mean"), episodes=("return", "size"))
    )
    overall = (
        per_seed.groupby(["condition", "noise"], as_index=False)
        .agg(mean_return=("mean_return", "mean"),
             return_std_between_seeds=("mean_return", "std"),
             success_rate=("success_rate", "mean"),
             success_std_between_seeds=("success_rate", "std"),
             mean_steps=("mean_steps", "mean"), seeds=("seed", "nunique"))
        .fillna(0.0)
    )
    return per_seed, overall


def make_plots(curves: pd.DataFrame, overall: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition, group in curves.groupby("condition"):
        stats = group.groupby("step")["mean_return"].agg(["mean", "std"]).fillna(0.0)
        ax.plot(stats.index, stats["mean"], marker="o", label=condition)
        ax.fill_between(stats.index, stats["mean"] - stats["std"],
                        stats["mean"] + stats["std"], alpha=0.18)
    ax.set(title="Acrobot: качество на фиксированном шуме 0.2 во время обучения",
           xlabel="Шаги взаимодействия со средой", ylabel="Средняя награда за эпизод")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "learning_curves.png", dpi=160); plt.close(fig)

    trained = overall[overall["condition"].isin(["fixed", "randomized"])]
    for metric, ylabel, filename, title in [
        ("mean_return", "Средняя награда", "robustness_return.png", "Устойчивость к шуму момента силы"),
        ("success_rate", "Доля успешных эпизодов", "robustness_success.png", "Успех на знакомом и новом уровнях шума"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for condition, group in trained.groupby("condition"):
            group = group.sort_values("noise")
            error = group["return_std_between_seeds"] if metric == "mean_return" else group["success_std_between_seeds"]
            ax.errorbar(group["noise"], group[metric], yerr=error, marker="o", capsize=4, label=condition)
        ax.axvspan(0.3, float(overall["noise"].max()), color="grey", alpha=0.1, label="новые уровни шума")
        ax.set(title=title, xlabel="Максимальный шум момента силы (Н·м)", ylabel=ylabel)
        ax.legend(); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(output_dir / filename, dpi=160); plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join(lines)


def write_report(overall: pd.DataFrame, config: dict, output_dir: Path) -> None:
    trained = overall[overall["condition"].isin(["fixed", "randomized"])]
    high = trained[trained["noise"] >= 0.4].groupby("condition")["mean_return"].mean()
    nominal = trained[trained["noise"] == 0.0].set_index("condition")["mean_return"]
    high_success = trained[trained["noise"] >= 0.4].groupby("condition")["success_rate"].mean()
    supported = high.get("randomized", -np.inf) > high.get("fixed", -np.inf)
    verdict = "подтверждается" if supported else "не подтверждается"
    per_seed_path = output_dir / "evaluation_by_seed.csv"
    seed_note, conditional_note = "", ""
    if per_seed_path.exists():
        per_seed = pd.read_csv(per_seed_path)
        nominal_runs = per_seed[(per_seed["condition"].isin(["fixed", "randomized"])) &
                                (per_seed["noise"] == 0.0)]
        weak = nominal_runs[nominal_runs["success_rate"] < 0.5]
        if not weak.empty:
            details = "; ".join(
                f"{row.condition}/seed_{int(row.seed)}: успех {row.success_rate:.2f}, награда {row.mean_return:.1f}"
                for row in weak.itertuples()
            )
            seed_note = (
                "**Анализ неудачных запусков.** Несмотря на усреднение, отдельные обучающие запуски не сформировали "
                f"надёжную политику даже без шума (успех < 50%): {details}. Это показывает, что разброс между seed "
                "связан не только с малыми колебаниями награды: возможна полная неудача обучения при том же бюджете."
            )
            successful_keys = set(zip(nominal_runs.loc[nominal_runs["success_rate"] >= 0.5, "condition"],
                                      nominal_runs.loc[nominal_runs["success_rate"] >= 0.5, "seed"]))
            row_keys = list(zip(per_seed["condition"], per_seed["seed"]))
            successful_high = per_seed[[key in successful_keys and noise >= 0.4
                                        for key, noise in zip(row_keys, per_seed["noise"])]]
            conditional = successful_high.groupby("condition")["mean_return"].mean()
            if {"fixed", "randomized"}.issubset(conditional.index):
                conditional_note = (
                    "**Post-hoc проверка альтернативного объяснения.** После исключения запусков, не решивших исходную "
                    f"среду, средняя награда на новом шуме: fixed {conditional['fixed']:.2f}, randomized "
                    f"{conditional['randomized']:.2f}. Это исследовательский, а не основной критерий: он помогает "
                    "отделить устойчивость уже обученной политики к шуму от вероятности неудачи самого обучения."
                )
        else:
            seed_note = "Во всех обучающих запусках доля успеха без шума составила не менее 50%."
    text = f"""# Результаты: устойчивость PPO к шуму управления

## Исследовательский вопрос и гипотеза

**Вопрос.** Помогает ли случайный шум момента силы во время обучения сохранять качество Acrobot на уровнях шума, которых не было при обучении?

**Гипотеза.** Агент, обученный при шуме из диапазона [0, 0.3] Н·м, может немного уступать без шума, но будет иметь более высокую награду и долю успеха при шуме 0.5–1.5 Н·м.

До обучения зафиксированы: один алгоритм PPO, бюджет {config['total_timesteps']:,} шагов на запуск, seed {config['seeds']}, {config['eval_episodes']} оценочных эпизодов на уровень шума. Меняется только режим шума при обучении. Во время оценки обучение и исследовательские действия отключены; seed эпизодов совпадают между условиями.

## Метрики

- награда эпизода: `-число шагов`, при достижении цели последний шаг даёт 0;
- доля успехов: эпизод завершился по `terminated`, а не по лимиту 500 шагов;
- кривая обучения: средняя награда на одном и том же наборе эпизодов с шумом {config['learning_eval_noise']} Н·м;
- полосы и погрешности: стандартное отклонение между средними значениями отдельных seed, не между отдельными эпизодами.

## Итоговое сравнение

{markdown_table(overall.round(4))}

![Кривые обучения](learning_curves.png)

![Награда при разном шуме](robustness_return.png)

![Доля успеха при разном шуме](robustness_success.png)

## Ответ на вопрос

При новых уровнях шума 0.5–1.5 средняя награда fixed составила **{high.get('fixed', float('nan')):.2f}**, randomized — **{high.get('randomized', float('nan')):.2f}**. Средние доли успеха — **{high_success.get('fixed', float('nan')):.3f}** и **{high_success.get('randomized', float('nan')):.3f}** соответственно. Без шума награды равны **{nominal.get('fixed', float('nan')):.2f}** и **{nominal.get('randomized', float('nan')):.2f}**.

На этом бюджете и этих seed гипотеза **{verdict}** по заранее выбранному критерию: большей средней награде randomized на новых уровнях шума. Это результат ограниченного эксперимента, а не доказательство универсальности доменной рандомизации.

{seed_note}

{conditional_note}

Для случайного бейзлайна одни и те же общие оценочные эпизоды повторяются под тремя метками seed для согласования с RL-запусками. Его нулевой разброс между этими метками не является оценкой статистической неопределённости; разброс обучения интерпретируется для fixed и randomized.

## Визуализация поведения

В каталоге сохранены GIF-анимации для обоих агентов при шуме {config['video_noise']} Н·м. В таблице `evaluation_episodes.csv` можно проверить каждый эпизод и его seed.

## Ограничения и следующий шаг

Только три обучающих seed и одна среда; доверительные выводы о небольших различиях нельзя делать по числу эпизодов, потому что основная неопределённость связана с обучением. Далее стоит повторить эксперимент на 5–10 seed и проверить, сохраняется ли эффект при другом диапазоне рандомизации.
"""
    (output_dir / "RESULTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/full.json"))
    parser.add_argument("--output", type=Path, default=Path("results/full"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    import torch
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_curves, all_episodes = [], []
    final_models = {}
    for condition in ["fixed", "randomized"]:
        for seed in config["seeds"]:
            print(f"Training {condition}, seed={seed}", flush=True)
            model, curve_rows = train_one(condition, seed, config, output_dir)
            final_models[(condition, seed)] = model
            all_curves.extend([dict(row, condition=condition, seed=seed) for row in curve_rows])
            for noise in config["eval_noise_levels"]:
                rows = evaluate_at_noise(model, float(noise), int(config["eval_episodes"]), 200_000 + int(noise * 100) * 1000)
                all_episodes.extend([dict(row, condition=condition, seed=seed) for row in rows])
        save_gif(final_models[(condition, config["seeds"][0])], condition, config["seeds"][0], float(config["video_noise"]), output_dir)

    # An untrained random policy is an additional reference, evaluated on identical seeds.
    for seed in config["seeds"]:
        for noise in config["eval_noise_levels"]:
            rows = evaluate_at_noise(None, float(noise), int(config["eval_episodes"]), 200_000 + int(noise * 100) * 1000)
            all_episodes.extend([dict(row, condition="random", seed=seed) for row in rows])

    curves = pd.DataFrame(all_curves)
    episodes = pd.DataFrame(all_episodes)
    per_seed, overall = aggregate_final(episodes)
    curves.to_csv(output_dir / "learning_curves.csv", index=False)
    episodes.to_csv(output_dir / "evaluation_episodes.csv", index=False)
    per_seed.to_csv(output_dir / "evaluation_by_seed.csv", index=False)
    overall.to_csv(output_dir / "summary.csv", index=False)
    make_plots(curves, overall, output_dir)
    write_report(overall, config, output_dir)
    versions = {name: importlib.metadata.version(name) for name in
                ["gymnasium", "stable-baselines3", "torch", "numpy", "pandas", "matplotlib"]}
    (output_dir / "runtime.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(), "packages": versions
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
