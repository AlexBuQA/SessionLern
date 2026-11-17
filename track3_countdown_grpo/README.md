# Трек 3. GRPO-дообучение Qwen3-0.6B на Countdown

Исследование посвящено **reward hacking**, а не простому сравнению learning rate. Strict-награда проверяет формат, безопасный синтаксис, точное значение и мультимножество исходных чисел. Naive-награда отличается только тем, что не проверяет исходные числа. Поэтому `<answer>46</answer>` для цели 46 получает 1 у naive и 0 у strict.

## Статус результатов

**Исследование выполнено на Colab Tesla T4 в FP16.** Отчёт: [RESULTS.md](RESULTS.md); исходные таблицы, ответы и графики: [results/deadline](results/deadline/). Получено 6 обучающих запусков (strict/naive × seed 42, 43, 44) по 50 шагов GRPO; базовая модель и чекпоинты 25/50 оценены на одних 240 отложенных задачах.

Короткий исследовательский протокол `deadline` показал reward hacking: naive-награда 99,44 %, строгая точность naive-модели 0 %; strict-точность 18,61 % против 16,67 % у базовой модели. Исходный протокол на 300 шагов сохранён отдельно в `configs/full.json`.

## Подготовка в VS Code

```powershell
cd track3_countdown_grpo
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# На CUDA-машине выберите подходящую сборку PyTorch: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python -m countdown_rl.data --config configs/full.json
pytest -q
```

## Воспроизведение выполненного эксперимента

В Colab откройте `Countdown_GRPO_Colab.ipynb`, подключите T4 и выберите `MODE = 'deadline'`. Локально на CUDA GPU:

```powershell
python run.py --config configs/deadline.json --output results/deadline --outputs outputs/deadline
```

## Исходный расширенный протокол

```powershell
python run.py --config configs/full.json --output results/full --outputs outputs/full
```

Последовательность:

1. детерминированно создаются 1200 обучающих и 240 отложенных решаемых задач без пересечений;
2. базовая Qwen3-0.6B оценивается жадно на всех 240 задачах;
3. если базовая точность вне 10–50%, обучение останавливается с просьбой изменить сложность;
4. strict и naive обучаются с одинаковыми seed 42, 43, 44, 300 шагами GRPO и LoRA;
5. каждые 50 шагов сохраняются чекпоинты; все они независимо оцениваются на одной выборке;
6. создаются графики reward / held-out accuracy / KL / длины, итоговая таблица и ответы на одинаковых задачах.

Процессы обучения и оценки запускаются отдельно, чтобы освобождать VRAM. Готовые summary позволяют не повторять завершённую оценку при повторном запуске оркестратора. Прерванное обучение запускается заново, поскольку чекпоинты сохраняются без состояния оптимизатора. Модель скачивается с Hugging Face; нужен доступ в интернет и достаточно места на диске.

## Быстрая GPU-проверка

```powershell
python run.py --config configs/smoke.json --output results/smoke --outputs outputs/smoke
```

Это 2 шага и 8 оценочных задач. Smoke проверяет совместимость версий и память, но не подтверждает и не опровергает гипотезу.

## Память и скорость

Конфигурация по умолчанию не использует vLLM, чтобы уменьшить число зависимостей. На Linux с совместимым CUDA можно установить `vllm`, выставить `training.use_vllm=true` и протестировать colocate-режим отдельно. При нехватке VRAM уменьшайте `max_completion_length` или batch size, но затем применяйте одну и ту же изменённую конфигурацию к обоим условиям и всем seed.

## Файлы результата

- `evaluation_by_checkpoint.csv`, `heldout_accuracy.png`;
- `training_metrics.csv`, `training_reward.png`, `kl_divergence.png`, `response_length.png`;
- `summary.csv`, `final_runs.csv`;
- `answer_comparison.csv`, `answer_examples.md`;
- `predictions.jsonl` с полными ответами и причинами ошибок;
- `RESULTS.md` с выводом по заранее заданным критериям.

Все модели оцениваются одним строгим верификатором. `naive_accuracy` оставлен отдельной метрикой, чтобы показать разрыв «награда проверяющего — реальное качество». Модельные веса и чекпоинты исключены из Git; для сдачи нужны код, фиксированные данные и небольшие результаты, а не веса LoRA.
