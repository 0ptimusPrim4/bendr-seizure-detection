# Детекция эпилептических приступов на ЭЭГ: дообучение BENDR на CHB-MIT

Магистерская НИР-3 (Самарский университет, кафедра лазерных и биотехнических систем).
Адаптация подхода Benfenati (2023): предобученная модель BENDR (веса SPOClab-ca v0.1-alpha)
дообучается на базе CHB-MIT с функцией потерь SSWCE. Оценка ведётся по схеме LOPO
(leave-one-patient-out) или patient-wise, затем идёт пост-обработка (голосование
большинством, группировка ложных срабатываний).

Запуск сведён к трём скриптам:

| Скрипт | Что делает |
|---|---|
| `prepare_data.py` | скачивает CHB-MIT и веса BENDR (по желанию), строит кэш окон: Баттерворт 0,5–50 Гц, 20 каналов 10-20, 256 Гц, окна 4 с / 1024 отсчёта, метки приступов |
| `train.py` | дообучение: заморозка/разморозка энкодера, доли данных 25/50/100 %, LOPO или patient-wise, балансировка, AMP, чекпоинты, ранняя остановка, порог Юдена |
| `evaluate.py` | метрики до и после пост-обработки, сводные таблицы CSV/JSON |

`common.py` — общий модуль (модель, предобработка, SSWCE, метрики), отдельно не запускается.
Конфиги: `configs/gpu.yaml` (полный прогон на GPU) и `configs/cpu_quick.yaml` (быстрая проверка, только голова).

## 1. Установка

Нужен Python 3.10–3.13 и NVIDIA GPU с драйвером CUDA. На Windows удобнее Python 3.11 или 3.12.

```bash
git clone https://github.com/0ptimusPrim4/bendr-seizure-detection.git
cd bendr-seizure-detection

# вариант А: conda
conda env create -f environment.yml
conda activate bendr-seizure

# вариант Б: venv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch ставится отдельно, под свою версию CUDA (точная команда — на https://pytorch.org/get-started/locally/):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # должно быть True
```

## 2. Данные и веса

- **CHB-MIT Scalp EEG Database** v1.0.0, PhysioNet: https://physionet.org/content/chbmit/1.0.0/
  (открытый доступ, около 43 ГБ). Записи кладутся в `data/chbmit/chbXX/` вместе с `chbXX-summary.txt`.
- **Веса BENDR v0.1-alpha** (`encoder.pt` 15,9 МБ и `contextualizer.pt` 612,7 МБ):
  https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha. Кладутся в `weights/`.

Оба пункта делает `prepare_data.py`:

```bash
# веса + записи всех 23 пациентов из configs/gpu.yaml + кэш окон
python prepare_data.py --config configs/gpu.yaml --weights --download --workers 4

# только несколько пациентов
python prepare_data.py --config configs/gpu.yaml --patients chb01 chb02 chb03 --download

# проверить, что всё на месте (веса, EDF, кэш)
python prepare_data.py --config configs/gpu.yaml --check
```

Если данные уже скачаны (например, `wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/`),
укажите путь: `--set paths.raw_dir=D:/data/chbmit`. Повторный запуск пропускает готовых пациентов;
кэш пересчитывается сам, если в конфиге поменялись параметры фильтра, окон или каналов.

Кэш хранится в `data/cache/<пациент>/` в float32: по три `.npy` на файл EDF и `meta.json`
(число окон, приступы, пропущенные файлы, статистики каналов). Одно окно занимает 80 КБ,
весь CHB-MIT — порядка 70 ГБ на диске (оценка: ~980 ч записей); читается через memmap, в ОЗУ целиком не грузится.
Файлы, где не хватает нужных отведений, пропускаются и перечисляются в `meta.json`
(`data.channels.on_missing: skip_file`; `zero` — заполнить нулями, `error` — остановиться).

## 3. Обучение на GPU

Все параметры берутся из `configs/gpu.yaml`; флаги командной строки их переопределяют,
любой другой ключ меняется через `--set ключ=значение`.

```bash
# полный LOPO по 23 пациентам, полное дообучение (энкодер разморожен), 100 % данных,
# сразу с оценкой
python train.py --config configs/gpu.yaml --evaluate

# варианты 25/50/100 % обучающей выборки
python train.py --config configs/gpu.yaml --fractions 0.25 0.5 1.0 --run-name lopo_full_fr --evaluate

# заморожен свёрточный энкодер, учится контекстуализатор и голова
python train.py --config configs/gpu.yaml --freeze encoder --evaluate

# частичная разморозка: последние 2 слоя трансформера + голова
python train.py --config configs/gpu.yaml --freeze partial --unfreeze-last 2 --evaluate

# только голова на замороженных признаках BENDR (быстро)
python train.py --config configs/gpu.yaml --freeze backbone --fractions 0.25 0.5 1.0 --evaluate

# patient-wise: отдельная модель на пациента, тест — отложенные файлы этого пациента
python train.py --config configs/gpu.yaml --scheme patient_wise --evaluate

# только некоторые фолды LOPO (обучение всё равно на всех остальных пациентах)
python train.py --config configs/gpu.yaml --test-patients chb01 chb02 chb03

# эксперимент на 5 пациентах
python train.py --config configs/gpu.yaml --patients chb01 chb02 chb03 chb04 chb05 --run-name lopo5
```

Режимы `--freeze`:

| Режим | Что обучается | Обучаемых параметров |
|---|---|---|
| `full` | всё (как в диссертации) | 157,1 млн |
| `encoder` | контекстуализатор + голова | 153,2 млн |
| `partial` | последние `--unfreeze-last` слоёв трансформера, выходная проекция, голова | 38,6 млн при 2 слоях |
| `backbone` | только голова Linear(512, 2) | 1026 |

Другие полезные флаги: `--balancing sampler|class_weights|none`, `--norm global|window`,
`--feature ctx_first|ctx_mean|enc_mean`, `--epochs`, `--batch-size`, `--lr`, `--no-amp`,
`--workers`, `--force` (пересчитать готовые фолды). Прерванный прогон можно запустить той же
командой: готовые фолды (есть `preds.npz`) пропускаются.

Доли 25/50/100 % относятся к объёму обучающей выборки (стратифицированно по классу), а не к
размеру сети: опубликованные веса BENDR есть только для полной архитектуры
(6 свёрточных блоков, 8 слоёв, 8 голов).

Если не хватает видеопамяти — уменьшите батч (`--batch-size 64`). Полный LOPO на 23 пациентах
с полным дообучением идёт долго: эпоха проходит все обучающие окна. Чтобы сократить эпоху,
можно задать `--set training.samples_per_epoch=200000`.

## 4. Оценка и пост-обработка

`train.py --evaluate` вызывает оценку сам. Отдельно (например, с другими окнами голосования):

```bash
python evaluate.py --run-dir results/lopo_full
python evaluate.py --run-dir results/lopo_full --vote-widths 1 3 5 7
python evaluate.py --run-dir results/lopo_full --threshold 0.5
```

Окно 1 — метрики до пост-обработки, 3/5/7 — после голосования большинством.
Голосование и группировка ложных срабатываний идут внутри каждого файла EDF. Считаются:
событийная чувствительность, чувствительность и специфичность по окнам,
FP/ч = (1 − специфичность) · 3600 / 4 (формула диссертации), FP/ч после группировки, AUC.

## 5. Где результаты

```
results/<run_name>/                 по умолчанию <схема>_<режим>, например lopo_full
  config_used.yaml                  итоговый конфиг с учётом флагов
  train.log                         журнал (эпохи, размеры выборок, пороги)
  <пациент>_frac<доля>/
    best.pt                         лучший по val_loss чекпоинт (только обучаемые параметры)
    preds.npz                       вероятности на тесте и валидации
    fold.json                       порог Юдена, val AUC, история эпох, тестовые файлы
  metrics_all.csv                   все фолды × доли × окна голосования
  summary.csv                       среднее и медиана по фолдам
  metrics.json                      то же в JSON
```

## 6. Быстрая проверка на CPU

`configs/cpu_quick.yaml` повторяет настройки CPU-экспериментов НИР-3: 5 пациентов,
замороженный BENDR, голова на признаках `ctx_mean`, нормализация внутри окна, 25 эпох.

```bash
python prepare_data.py --config configs/cpu_quick.yaml
python train.py --config configs/cpu_quick.yaml --threads 4 --evaluate
```

## 7. Откуда параметры

Из диссертации Benfenati (2023) и исходного `configs/default.yaml`: Баттерворт порядка 5,
0,5–50 Гц; 256 Гц; окна 4 с; исключение 15 мин постиктала; SSWCE с α = 0,3, β = 0,5;
Adam, lr = 1e-4, weight decay 0,01; ReduceLROnPlateau (0,1; 5 эпох); ранняя остановка по
val_loss (15 эпох); batch 256; до 120 эпох; валидация 20 %; голосование в окне 3.
В конфигах помечено «выбрано автором» то, чего в диссертации нет: канал SCALE как
относительная амплитуда, пропуск файлов с неполным монтажом, порог Юдена, окна
голосования 5 и 7, AMP, режим `partial`, доля тестовых файлов для patient-wise.

## 8. Прочее

- `src/`, `tests/`, `notebooks/` и `configs/default.yaml` — исходный каркас проекта, для запуска трёх скриптов не нужны
  (тесты: `pip install pytest && pytest`).
- Кэши и чекпоинты прежних CPU-прогонов (`results/feature_cache/*.npz`) с этими скриптами
  несовместимы: кэш строится заново через `prepare_data.py`.

Источники: Benfenati L. Unsupervised and Self-Supervised Machine-Learning for Epilepsy
Detection on EEG Data, Politecnico di Torino, 2023; Kostas D. et al. BENDR, Front. Hum.
Neurosci., 2021, doi:10.3389/fnhum.2021.653659; Shoeb A. CHB-MIT, PhysioNet.
