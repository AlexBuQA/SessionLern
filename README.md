# Обучение с подкреплением — сессионное задание

Автор: **Александра Бужор**, Telegram: [@Angry_C_F_O](https://t.me/Angry_C_F_O).

Репозиторий содержит три автономных решения с воспроизводимыми экспериментами. В каждом треке заранее сформулированы вопрос, гипотеза, метрика и контролируемое сравнение; финальные выводы строятся по CSV из оценочных запусков без обучения.

| Папка | Среда / модель | Исследуемый фактор | Основная оценка |
|---|---|---|---|
| `track1_acrobot` | Acrobot-v1, PPO | случайный шум момента силы при обучении | 20 эпизодов × 6 уровней шума × 3 seed, награда и успех |
| `track2_inventory` | собственная InventoryEnv, Q-learning | рандомизация среднего спроса λ при обучении | 50 эпизодов × 5 режимов спроса × 3 seed, прибыль и сервис |
| `track3_countdown_grpo` | Countdown, Qwen3-0.6B, GRPO + LoRA | отсутствие проверки исходных чисел в награде | 240 непересекающихся отложенных задач, strict accuracy и exploit rate |

## Быстрый старт в VS Code

Откройте корневую папку репозитория. Для каждого трека создаётся отдельное окружение, чтобы локальные CPU-эксперименты не зависели от тяжёлого стека языковых моделей.

```powershell
# Пример для трека 1; аналогичные команды есть в README каждой папки.
cd track1_acrobot
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q
python run.py --config configs/smoke.json --output results/smoke
```

- Инструкция и отчёт трека 1: `track1_acrobot/README.md`, `track1_acrobot/results/full/RESULTS.md`.
- Инструкция и отчёт трека 2: `track2_inventory/README.md`, `track2_inventory/results/full/RESULTS.md`.
- Инструкция трека 3: `track3_countdown_grpo/README.md`.

**Важно:** третий трек содержит полный обучающий и оценочный pipeline, фиксированный датасет и 19 автоматических проверок, но фактическое GRPO-обучение требует GPU. Оно не представлено как выполненное на CPU. Запустите `Countdown_GRPO_Colab.ipynb` на T4 или локальный `run.py`; только после этого появятся численные результаты.

## Проверка всего проекта

Из корня, при активном окружении с `pytest` и зависимостями треков:

```powershell
python run_checks.py
python validate_results.py --require-full
# Необязательно: собрать чистый архив без весов и виртуальных окружений.
python package_project.py
```

Smoke-конфигурации проверяют сквозной pipeline. Научические выводы делаются только по `results/full`, не по smoke.

## Воспроизводимость

- обучающие seed треков 1–2: **11, 22, 33**;
- GRPO seed: **42, 43, 44**;
- seed генерации Countdown: **20261117**;
- одинаковые оценочные seed согласуют начальные состояния и случайность среды между условиями;
- точные версии каждого трека перечислены в его `requirements.txt`;
- `requirements-lock.txt` — freeze фактического окружения, в котором проходила проверка проекта;
- `runtime.json` в результатах фиксирует Python, ОС и основные версии конкретного запуска.

Полосы на графиках первых двух треков показывают стандартное отклонение между средними значениями обучающих seed. Это важно: увеличение числа оценочных эпизодов не заменяет независимые запуски обучения.

## Структура

```text
track1_acrobot/           # среда, PPO, тесты, конфиги, отчёт и GIF
track2_inventory/         # собственная среда, Q-learning, бейзлайны и траектории
track3_countdown_grpo/    # безопасный AST/Fraction-верификатор, данные, GRPO и Colab
.vscode/                  # готовые конфигурации запуска
run_checks.py             # единый запуск тестов
requirements-lock.txt     # версии проверочного окружения
```

## Загрузка на GitHub

После проверки результатов и замены плейсхолдера в Colab-ноутбуке:

```powershell
git init
git add .
git commit -m "Solve three RL research tracks"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/rl-session-assignment.git
git push -u origin main
```

Ссылка на репозиторий должна открываться без запроса доступа. Перед публикацией проверьте `git status`: `.gitignore` исключает `.venv`, веса, чекпоинты и крупные архивы. Небольшие CSV, PNG, GIF и Countdown JSONL следует оставить — они являются результатами и основанием для выводов.
