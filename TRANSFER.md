# Как перенести результаты с CPU и продолжить на GPU

Краткая инструкция для стационарного ПК. Если обучение уже шло на другой
машине (без видеокарты), копируйте из её рабочего каталога нужные файлы,
не удаляя оригинал.

---

## 1. Что уже есть в репозитории, а что нет

В репозитории — код, конфиги и документация.
Скачайте отдельно:

- веса BENDR в `weights/` (см. README, раздел 3);
- нужных пациентов CHB-MIT в `data/chbmit/` (раздел 2);
- Python-окружение с CUDA (раздел 1).

Текущие CPU-результаты в репозиторий тоже не входят: их копируют с машины, где шло обучение.

---

## 2. Какие файлы копировать с CPU-машины

Корень исходного проекта — каталог, в котором выполнялись CPU-прогоны
(в нём должны быть `results/feature_cache/`, `results/checkpoints_probe/` и т.д.).

Обязательно, если хотите опереться на уже полученные цифры и веса головы:

| Откуда | Куда в этом репозитории | Зачем |
|---|---|---|
| `results/feature_cache/*.npz` и `*.npz.meta.json` | `results/feature_cache/` | кэш замороженных признаков: не извлекать заново на GPU. Нужны `{patient}_window_variants.npz` (попытка 3, выбранный вариант) и при желании `*_global_variants.npz`, `*_fold.npz` (первый токен, попытка 2). Вместе около 190 МБ |
| `results/checkpoints_probe/*.pt` | `results/checkpoints/` или `results/from_cpu/checkpoints_probe/` | чекпоинты зонда. Файлы `*_ctx_mean_window.pt` (~6 КБ, только голова, попытка 3) — основные. `chb01_frac100.pt` (~600 МБ) — полная модель попытки 2, для сравнения. `*_cached.pt` — голова с кэша первого токена |
| `results/lopo_probe_variants/` | `results/from_cpu/lopo_probe_variants/` | таблицы и рисунки попытки 3 (`compact_before_after.csv`, метрики до/после голосования) |
| `results/lopo_probe/` | `results/from_cpu/lopo_probe/` | метрики попытки 2 (первый токен) |
| `results/lopo_probe_fractions/` | `results/from_cpu/lopo_probe_fractions/` | доли 25/50/100 % на кэше первого токена |
| `results/EXPERIMENT_ATTEMPTS.md`, журналы `*.log` | `results/from_cpu/` | журнал попыток и логи |
| `scripts_final/config_probe.yaml` | `results/from_cpu/config_probe.yaml` | точный конфиг CPU; в пакете ему соответствует `configs/cpu_probe.yaml` |

Полезно, но не обязательно:

| Откуда | Куда | Зачем |
|---|---|---|
| `results/checkpoints/*.pt` | `results/from_cpu/lopo_run1_degenerate/checkpoints/` | чекпоинты первой попытки (полное дообучение, 3 эпохи). Прогон признан вырожденным, в отчёт как успех не идёт |
| `results/lopo_run1_degenerate/` | `results/from_cpu/lopo_run1_degenerate/` | цифры и графики первой попытки |
| `results/mock/` | `results/from_cpu/mock/` | синтетическая проверка конвейера |
| `results/EXPERIMENT_ATTEMPTS.md` | `results/from_cpu/` | журнал попыток |
| `NOTES.md` | `results/from_cpu/NOTES.md` | гиперпараметры и ограничения CPU |

Данные `data/chbmit/` и веса `weights/` копировать имеет смысл только если
на новом ПК нет быстрого интернета. На CPU-машине скачаны **только chb01–chb03**
и оба файла весов v0.1-alpha. Для `gpu_full.yaml` остальных пациентов всё равно
нужно докачать с PhysioNet (~43 ГБ на весь набор).

После копирования структура может выглядеть так:

```
results/
  from_cpu/                 всё, что приехало с CPU
  checkpoints/              сюда же или новые GPU-чекпоинты
  lopo_gpu/                 выход нового обучения
  lopo_eval/                выход повторной оценки
```

---

## 3. Что умеют текущие чекпоинты, а чего не умеют

Формат CPU-чекпоинта (первый и второй прогоны): словарь PyTorch с ключами
`model`, `val_loss`, `epoch`. Статистик нормализатора, имени пациента и режима
в файле нет. Оптимизатор Adam и планировщик **не сохранялись**.

Следствия:

1. **Продолжить обучение с эпохи N+1** с тем же состоянием Adam нельзя.
   На GPU либо запускают обучение заново от официальных весов BENDR, либо
   только оценивают уже полученную голову / сеть.
2. **Повторная оценка** старого чекпоинта возможна командой `eval`.
   Для попытки 3 указывайте `configs/cpu_quick.yaml` (или `cpu_probe.yaml`):
   там `feature_variant: ctx_mean`, `norm: window`, порог `youden`.
   Нормализатор для `norm: window` не нужен. Для старых глобальных чекпоинтов
   без статистик z-score они пересчитываются по обучающим пациентам
   (нужны те же записи и `seed: 42`). Имя файла вида `chb01_frac100_ctx_mean_window.pt`
   задаёт пациента; иначе укажите `--patient`.
3. Новые прогоны этого пакета дописывают в `.pt` статистики z-score (если norm=global),
   имя пациента, долю, `mode`, `feature_variant`, `norm` и порог Youden.
   Это упрощает `eval`, но по-прежнему не даёт «resume» оптимизатора.

Чекпоинты зонда (`mode: head_only`, `cpu_quick.yaml` / `cpu_probe.yaml`)
и чекпоинты полного дообучения (`mode: full`) оцениваются **своим** конфигом.
Не подавайте зонд в `gpu_full.yaml` и наоборот.

**Как переиспользовать кэш признаков, не считая BENDR заново.**
Положите `*.npz` и `*.meta.json` в `results/feature_cache/` и запустите:

```bash
# попытка 3: четыре варианта, норма window, голова + Youden
python -m bendr_seizure diagnose --config configs/cpu_quick.yaml --device cuda

# только доли 25/50/100 % на кэше одного варианта ({patient}_fold.npz)
python -m bendr_seizure fractions --config configs/cpu_quick.yaml --device cuda
```

`diagnose` подхватывает `{patient}_{norm}_variants.npz`, если рядом есть `.meta.json`.
Имеет смысл для проверки метрик и постобработки, но не как путь к рабочему детектору:
на CPU test AUC остался около 0,5 (см. README, раздел 8). Для итогового эксперимента
на GPU запускайте полное дообучение (`configs/gpu_full.yaml`, `mode: full`).

---

## 4. Повторная оценка CPU-чекпоинтов на новом ПК

Поставьте окружение, веса и как минимум тех пациентов, которые участвовали
в прогоне (для зонда — chb01, chb02, chb03). Затем:

```bash
# Linux / macOS
export PYTHONPATH=.
python -m bendr_seizure eval \
  --config configs/cpu_quick.yaml \
  --checkpoint results/from_cpu/checkpoints_probe/chb01_frac100_ctx_mean_window.pt \
  --device cuda
```

```powershell
# Windows PowerShell
$env:PYTHONPATH="."
python -m bendr_seizure eval `
  --config configs/cpu_quick.yaml `
  --checkpoint results/from_cpu/checkpoints_probe/chb01_frac100_ctx_mean_window.pt `
  --device cuda
```

Все зонд-чекпоинты сразу (положите их в `results/checkpoints/` либо временно
поправьте `paths.checkpoints_dir` в копии конфига):

```bash
python -m bendr_seizure eval --config configs/cpu_quick.yaml --device cuda
```

Выход: `results/lopo_cpu/eval_results.csv` (каталог задаётся
`paths.experiment_dir` в конфиге), `metrics_*.json`, `cm_*.png`.
В JSON есть метрики `raw` (до голосования) и `smooth` (после).

---

## 5. Новое обучение на GPU

Для полного эксперимента по диссертации откройте `configs/gpu_full.yaml`
и при необходимости поправьте:

- `data.patients` — начните с `[chb01, chb02, chb03]`, затем расширьте;
  полный список в файле — chb01–chb24, см. комментарий про chb24;
- `training.data_fractions` — например, сначала `[1.0]`, потом `[0.25, 0.5, 1.0]`;
- `training.freeze_encoder` / `freeze_contextualizer` / `mode` —
  по умолчанию полное дообучение (`full`, оба блока разморожены);
  для зонда на GPU можно поставить `true` / `true` / `head_only` и `lr: 0.001`;
- `training.max_epochs` — 120, сработает ранняя остановка (patience 15);
- `training.batch_size` — 256; если не хватает видеопамяти, уменьшите
  (это уже отступление от диссертации, пометьте в отчёте).

Запуск:

```bash
python -m bendr_seizure train --config configs/gpu_full.yaml --device cuda
```

Это **новое** обучение от официальных `encoder.pt` и `contextualizer.pt`,
а не продолжение CPU-прогона. Так и нужно: на CPU не набиралось ни полного
набора пациентов, ни 120 эпох, ни штатного batch 256, а замороженные признаки
не перенеслись между пациентами (test AUC ≈ 0,5). Полное дообучение как раз
и проверяет, даст ли разморозка энкодера переносимость.

---

## 6. Как заново собрать рисунки и таблицы для отчёта

После `train` или `eval` в каталоге эксперимента уже лежат готовые файлы:

- `lopo_results.csv` / `lopo_summary.csv` (или `eval_results.csv`) — таблицы
  по пациентам и средние по долям данных;
- те же таблицы в Markdown (`*.md`);
- `loss_*.png` — кривые SSWCE (появляются только после обучения, не после `eval`);
- `cm_*.png` — матрицы ошибок после голосования;
- `metrics_*.json` — полное содержимое, включая метрики до постобработки.

Их вставляют в раздел 3 отчёта и в приложения.
Если рисунок нужно переименовать или пересчитать подписи, достаточно повторного
`eval` (матрицы) или повторного `train` (кривые потерь). Отдельный скрипт
построения графиков из JSON не требуется: `metrics.py` пишет PNG при оценке.

Рекомендуемый набор для отчёта с GPU:

1. Таблица LOPO: пациент, доля данных, чувствительность, специфичность, FP/ч,
   сгруппированные FP/ч, событийная чувствительность — до и после голосования.
2. Строка средних по каждой доле 25/50/100 %.
3. Кривые потерь нескольких типичных фолдов (не обязательно все 24×3).
4. Две–три матрицы ошибок (удачный, средний и неудачный фолд).
5. Короткий комментарий, если цифры плохие: не приукрашивать, указать баланс
   классов и длительность межприступного периода на тесте.

CPU-цифры из `lopo_run1_degenerate` в основные таблицы не ставить: модель
схлопнулась в константный класс. Их можно коротко описать как неудачную попытку.
Цифры зонда (`lopo_probe`) — промежуточный результат на трёх пациентах;
после GPU-прогона их заменяют или оставляют как сравнение «голова / полная сеть».

---

## 7. Контрольный список на новом ПК

1. Клонировать репозиторий, поставить Python и torch с CUDA, затем `requirements.txt`.
2. `python -m bendr_seizure --help` — справка открывается.
3. Скачать веса, проверить SHA-256 (`python scripts/download_weights.py`).
4. Скачать хотя бы chb01–chb03, прогнать `python -m bendr_seizure smoke --config configs/cpu_quick.yaml`.
5. Скопировать `results/feature_cache/` и `checkpoints_probe/`, при желании прогнать `diagnose` (подхватит кэш) или `eval`.
6. Докачать остальных пациентов и запустить `train` с `configs/gpu_full.yaml`.
7. Забрать `lopo_results.csv`, `lopo_summary.csv` и PNG в черновик отчёта.
