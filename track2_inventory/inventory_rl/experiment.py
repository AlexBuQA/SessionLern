"""Train tabular Q-learning agents and test transfer across demand regimes."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inventory_rl.environment import DemandRandomizer, InventoryEnv


def epsilon_at(episode: int, config: dict) -> float:
    decay_episodes = config["episodes"] * config["epsilon_decay_fraction"]
    fraction = min(1.0, episode / max(1.0, decay_episodes))
    return config["epsilon_start"] + fraction * (config["epsilon_end"] - config["epsilon_start"])


def greedy_action(q_table: np.ndarray, observation: np.ndarray) -> int:
    inventory_position, previous_demand = map(int, observation)
    return int(np.argmax(q_table[inventory_position, previous_demand]))


def run_episode(env: InventoryEnv, action_fn, seed: int, initial_inventory=None) -> tuple[dict, list[dict]]:
    options = {} if initial_inventory is None else {"initial_inventory": initial_inventory}
    observation, reset_info = env.reset(seed=seed, options=options)
    total_profit, done = 0.0, False
    history = []
    while not done:
        action = int(action_fn(observation))
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_profit += reward
        history.append(dict(info, reward=reward))
    totals = {
        "profit": total_profit,
        "demand": sum(row["demand"] for row in history),
        "sales": sum(row["sales"] for row in history),
        "lost_sales": sum(row["lost_sales"] for row in history),
        "orders": sum(row["actual_order"] for row in history),
        "holding_units": sum(row["inventory_after_sales"] for row in history),
        "initial_inventory": reset_info["initial_inventory"],
        "episode_seed": seed,
    }
    totals["service_level"] = totals["sales"] / totals["demand"] if totals["demand"] else 1.0
    return totals, history


def evaluate_q(q_table: np.ndarray, demand_mean: float, episodes: int, seed_base: int, config: dict) -> list[dict]:
    env = InventoryEnv(demand_mean=demand_mean, horizon=config["horizon"])
    rows = []
    for episode in range(episodes):
        row, _ = run_episode(env, lambda obs: greedy_action(q_table, obs), seed_base + episode)
        row.update(episode=episode, demand_mean=demand_mean)
        rows.append(row)
    env.close()
    return rows


def train_one(condition: str, seed: int, config: dict, output_dir: Path) -> tuple[np.ndarray, list[dict], list[dict]]:
    base_env = InventoryEnv(demand_mean=config["training_demand_mean"], horizon=config["horizon"])
    env = DemandRandomizer(base_env, *config["randomization_range"]) if condition == "randomized" else base_env
    q_shape = [int(n) for n in env.observation_space.nvec] + [env.action_space.n]
    q_table = np.zeros(q_shape, dtype=np.float64)
    rng = np.random.default_rng(seed)
    curve, train_rows = [], []
    for episode in range(int(config["episodes"])):
        observation, _ = env.reset(seed=seed * 100_000 + episode)
        epsilon = epsilon_at(episode, config)
        episode_profit, done = 0.0, False
        while not done:
            state = tuple(map(int, observation))
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                action = int(np.argmax(q_table[state]))
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = tuple(map(int, next_observation))
            target = reward if terminated else reward + config["gamma"] * np.max(q_table[next_state])
            q_table[state][action] += config["alpha"] * (target - q_table[state][action])
            observation = next_observation
            episode_profit += reward
        train_rows.append({"episode": episode, "training_profit": episode_profit,
                           "epsilon": epsilon, "condition": condition, "seed": seed})
        if episode == 0 or (episode + 1) % int(config["learning_eval_interval"]) == 0:
            rows = evaluate_q(q_table, config["training_demand_mean"],
                              int(config["learning_eval_episodes"]), 900_000, config)
            curve.append({
                "episode": episode + 1,
                "mean_profit": np.mean([row["profit"] for row in rows]),
                "service_level": np.mean([row["service_level"] for row in rows]),
                "condition": condition, "seed": seed,
            })
    run_dir = output_dir / condition / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "q_table.npy", q_table)
    env.close()
    return q_table, curve, train_rows


def evaluate_all_policies(q_tables: dict, config: dict) -> pd.DataFrame:
    rows = []
    for condition in ["fixed", "randomized", "base_stock", "random"]:
        for seed in config["seeds"]:
            for demand_mean in config["eval_demand_means"]:
                env = InventoryEnv(demand_mean=demand_mean, horizon=config["horizon"])
                seed_base = 300_000 + int(demand_mean * 10) * 10_000
                random_rng = np.random.default_rng(seed_base + 99)
                for episode in range(int(config["eval_episodes"])):
                    if condition in q_tables:
                        q_table = q_tables[condition][seed]
                        action_fn = lambda obs, q=q_table: greedy_action(q, obs)
                    elif condition == "base_stock":
                        action_fn = lambda obs: max(0, config["base_stock_target"] - int(obs[0]))
                    else:
                        action_fn = lambda obs: int(random_rng.integers(env.action_space.n))
                    row, _ = run_episode(env, action_fn, seed_base + episode)
                    row.update(condition=condition, seed=seed, demand_mean=demand_mean, episode=episode)
                    rows.append(row)
                env.close()
    return pd.DataFrame(rows)


def aggregate(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = episodes.groupby(["condition", "seed", "demand_mean"], as_index=False).agg(
        mean_profit=("profit", "mean"), service_level=("sales", "sum"),
        demand=("demand", "sum"), lost_sales=("lost_sales", "mean"),
        holding_units=("holding_units", "mean"), orders=("orders", "mean"),
    )
    per_seed["service_level"] = per_seed["service_level"] / per_seed.pop("demand")
    overall = per_seed.groupby(["condition", "demand_mean"], as_index=False).agg(
        mean_profit=("mean_profit", "mean"), profit_std_between_seeds=("mean_profit", "std"),
        service_level=("service_level", "mean"), service_std_between_seeds=("service_level", "std"),
        lost_sales=("lost_sales", "mean"), holding_units=("holding_units", "mean"),
        orders=("orders", "mean"), seeds=("seed", "nunique"),
    ).fillna(0.0)
    return per_seed, overall


def make_plots(curves: pd.DataFrame, overall: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition, group in curves.groupby("condition"):
        stats = group.groupby("episode")["mean_profit"].agg(["mean", "std"]).fillna(0.0)
        ax.plot(stats.index, stats["mean"], label=condition)
        ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"], alpha=0.18)
    ax.set(title="Q-learning: контрольная оценка при среднем спросе 6",
           xlabel="Обучающий эпизод", ylabel="Прибыль за эпизод, ден. ед.")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=160); plt.close(fig)

    trained = overall[overall["condition"].isin(["fixed", "randomized", "base_stock", "random"])]
    for metric, error_column, ylabel, filename, title in [
        ("mean_profit", "profit_std_between_seeds", "Средняя прибыль, ден. ед.", "demand_shift_profit.png", "Перенос стратегий на другой спрос"),
        ("service_level", "service_std_between_seeds", "Уровень обслуживания", "demand_shift_service.png", "Доля спроса, закрытого продажами"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for condition, group in trained.groupby("condition"):
            group = group.sort_values("demand_mean")
            ax.errorbar(group["demand_mean"], group[metric], yerr=group[error_column],
                        marker="o", capsize=4, label=condition)
        ax.axvspan(9, 15, color="grey", alpha=0.1, label="сдвиг спроса")
        ax.set(title=title, xlabel="Средний дневной спрос λ, единиц", ylabel=ylabel)
        ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160); plt.close(fig)


def plot_trajectories_and_policy(q_tables: dict, config: dict, output_dir: Path) -> None:
    histories = {}
    seed = 777_000
    for condition in ["fixed", "randomized"]:
        env = InventoryEnv(demand_mean=config["trajectory_demand_mean"], horizon=config["horizon"])
        q = q_tables[condition][config["seeds"][0]]
        _, history = run_episode(env, lambda obs: greedy_action(q, obs), seed, initial_inventory=8)
        histories[condition] = history
        env.close()
        pd.DataFrame(history).to_csv(output_dir / f"trajectory_{condition}.csv", index=False)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for condition, history in histories.items():
        days = [row["day"] for row in history]
        axes[0].plot(days, [row["demand"] for row in history], marker="o", label=condition)
        axes[1].plot(days, [row["inventory_position"] for row in history], marker="o", label=condition)
        axes[2].plot(days, [row["actual_order"] for row in history], marker="o", label=condition)
    axes[0].set_ylabel("Спрос, ед."); axes[1].set_ylabel("Складская позиция, ед.")
    axes[2].set_ylabel("Заказ, ед."); axes[2].set_xlabel("День")
    fig.suptitle(f"Поведение на одном эпизоде при λ={config['trajectory_demand_mean']}")
    for ax in axes: ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "trajectories.png", dpi=160); plt.close(fig)

    q = q_tables["randomized"][config["seeds"][0]]
    policy = np.argmax(q, axis=2).T
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(policy, origin="lower", aspect="auto", cmap="viridis")
    ax.set(title="Randomized-агент: выбранный заказ", xlabel="Складская позиция, ед.",
           ylabel="Спрос предыдущего дня, ед.")
    fig.colorbar(image, ax=ax, label="Заказ, единиц")
    fig.tight_layout(); fig.savefig(output_dir / "policy_heatmap.png", dpi=160); plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join(lines)


def write_report(overall: pd.DataFrame, config: dict, output_dir: Path) -> None:
    high = overall[overall["demand_mean"] >= 12].groupby("condition")["mean_profit"].mean()
    nominal = overall[overall["demand_mean"] == config["training_demand_mean"]].set_index("condition")
    supported = high.get("randomized", -np.inf) > high.get("fixed", -np.inf)
    verdict = "подтверждается" if supported else "не подтверждается"
    text = f"""# Результаты: перенос RL-стратегии управления запасами

## Вопрос, гипотеза и план

**Вопрос.** Улучшает ли случайное изменение среднего спроса λ ∈ [3, 9] во время обучения прибыль и уровень обслуживания при незнакомом высоком спросе λ = 12 и 15?

**Гипотеза.** Randomized-агент может уступать fixed при исходном λ = 6, но получит более высокую среднюю прибыль при λ = 12–15, поскольку чаще видел дефицит и необходимость страхового запаса.

Сравниваются только два режима обучения: постоянный λ = 6 и эпизодическая рандомизация λ ∈ [3, 9]. Алгоритм, {config['episodes']:,} эпизодов по {config['horizon']} дней, seed {config['seeds']} и оценочные seed совпадают. Дополнительные ориентиры: order-up-to с S = {config['base_stock_target']} и случайные заказы.

## Среда и экономическая награда

Действие — заказ 0–15 единиц, поставка приходит на следующий день. Спрос пуассоновский; незакрытый спрос теряется. Наблюдение — складская позиция (остаток + поставка) и спрос предыдущего дня, ограниченный значением 25. Эпизод длится {config['horizon']} дней. Заказ автоматически ограничен вместимостью 30 единиц.

Награда в денежных единицах: `12 × продажи − 5 × заказ − 1 × остаток − 8 × потерянный спрос`. Она напрямую связывает решение с маржой, хранением и стоимостью дефицита. Уровень обслуживания — суммарные продажи / суммарный спрос; он показывает, не достигается ли прибыль за счёт отказа клиентам.

## Результаты

{markdown_table(overall.round(4))}

![Кривые обучения](learning_curves.png)

![Прибыль при сдвиге спроса](demand_shift_profit.png)

![Уровень обслуживания](demand_shift_service.png)

## Ответ и интерпретация

При λ = 12–15 средняя прибыль fixed — **{high.get('fixed', float('nan')):.2f}**, randomized — **{high.get('randomized', float('nan')):.2f}**, order-up-to — **{high.get('base_stock', float('nan')):.2f}**. При λ = 6 прибыль fixed — **{nominal['mean_profit'].get('fixed', float('nan')):.2f}**, randomized — **{nominal['mean_profit'].get('randomized', float('nan')):.2f}**, order-up-to — **{nominal['mean_profit'].get('base_stock', float('nan')):.2f}**.

По заранее выбранному критерию прибыли на новом высоком спросе гипотеза **{verdict}**. Поведение на одинаковом эпизоде показано в `trajectories.png`, а выбранные заказы — в `policy_heatmap.png`. Низкая прибыль случайной стратегии сама по себе не доказывает качество RL; содержательное сравнение — с order-up-to и между двумя RL-агентами на одних начальных состояниях и спросе. Сопоставление прибыли, уровня обслуживания, потерянных продаж, хранения и заказов позволяет увидеть, за счёт чего меняется результат, а не только итоговую награду.

Для `base_stock` и `random` одинаковые общие оценочные эпизоды повторяются под тремя метками seed, чтобы сохранить согласование с тремя RL-запусками. Поэтому их нулевой разброс между этими метками **не является оценкой статистической неопределённости**. Разброс обучения имеет смысл интерпретировать прежде всего для fixed и randomized.

## Ограничения

Модель предполагает известную экономику и пуассоновский спрос, не моделирует возвраты и сроки годности. Всего три seed; разброс эпизодов не заменяет разброс обучения. Для реального бизнеса выводы носят демонстрационный характер. Далее стоит проверить выбросы спроса и сравнить с оптимальной политикой, рассчитанной динамическим программированием на малой сетке.
"""
    (output_dir / "RESULTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/full.json"))
    parser.add_argument("--output", type=Path, default=Path("results/full"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    q_tables = {"fixed": {}, "randomized": {}}
    all_curves, all_training = [], []
    for condition in ["fixed", "randomized"]:
        for seed in config["seeds"]:
            print(f"Training {condition}, seed={seed}", flush=True)
            q, curves, training = train_one(condition, seed, config, output_dir)
            q_tables[condition][seed] = q
            all_curves.extend(curves); all_training.extend(training)
    episodes = evaluate_all_policies(q_tables, config)
    per_seed, overall = aggregate(episodes)
    curves = pd.DataFrame(all_curves)
    curves.to_csv(output_dir / "learning_curves.csv", index=False)
    pd.DataFrame(all_training).to_csv(output_dir / "training_returns.csv", index=False)
    episodes.to_csv(output_dir / "evaluation_episodes.csv", index=False)
    per_seed.to_csv(output_dir / "evaluation_by_seed.csv", index=False)
    overall.to_csv(output_dir / "summary.csv", index=False)
    make_plots(curves, overall, output_dir)
    plot_trajectories_and_policy(q_tables, config, output_dir)
    write_report(overall, config, output_dir)
    versions = {name: importlib.metadata.version(name) for name in ["gymnasium", "numpy", "pandas", "matplotlib"]}
    (output_dir / "runtime.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(), "packages": versions
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
