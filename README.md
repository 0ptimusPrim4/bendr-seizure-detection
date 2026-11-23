# Дообучение BENDR на CHB-MIT

Пакет для воспроизводимого дообучения предобученной модели BENDR
на записях ЭЭГ CHB-MIT и оценки детекции эпилептических приступов.

Основной код — пакет `bendr_seizure/`: конфиги, кэш признаков, обучение
головы или полное дообучение, инференс с постобработкой. Каталог `src/`
и часть скриптов в `scripts/` — исходный каркас репозитория; его можно
использовать для предобработки одного пациента, но новые эксперименты
запускайте через `python -m bendr_seizure` или обёртки в `scripts/`.

Код опирается на официальную реализацию
[SPOClab-ca/BENDR](https://github.com/SPOClab-ca/BENDR).
Гиперпараметры — из диссертации L. Benfenati (2023); значения, введённые
из-за ограничений CPU, помечены в конфигах как «выбрано автором».

В репозиторий **не входят** записи CHB-MIT (~43 ГБ) и веса BENDR (~600 МБ).
Их нужно скачать отдельно, см. ниже.

---

## Состав каталога

```
bendr_seizure/          основной пакет (модель, предобработка, обучение, метрики)
configs/
  cpu_quick.yaml        короткий прогон: 3 пациента, замороженный энкодер, только голова
  cpu_probe.yaml        расширенный CPU-зонд (варианты признаков, порог Youden)
  gpu_full.yaml         полный прогон: 23 пациента (chb01–chb23), 120 эпох, batch 256, энкодер разморожен
  default.yaml          конфиг исходного каркаса (src/)
scripts/
  download_chbmit.py    загрузка выбранных пациентов CHB-MIT
  download_weights.py   загрузка encoder.pt и contextualizer.pt
  train.py              обёртка над python -m bendr_seizure train
  evaluate.py           обёртка над python -m bendr_seizure eval
  diagnose_and_train.py обёртка над python -m bendr_seizure diagnose
  run_fractions_from_cache.py  обёртка над python -m bendr_seizure fractions
  run_lopo_cv.py        обёртка над python -m bendr_seizure train
  download_data.py      обёртка над download_chbmit.py
  preprocess_patient.py проверка и предобработка одного пациента (исходный каркас)
  run_preprocessing.py  полный конвейер одного пациента (исходный каркас)
  train_single_patient.py  короткий тестовый прогон на одном пациенте (исходный каркас)
src/                    исходный каркас (предобработка, модель, заготовки обучения)
tests/                  тесты исходного каркаса
data/chbmit/            сюда кладутся записи (в репозитории пусто)
data/raw/               каталог исходного каркаса (тоже пустой)
weights/                сюда кладутся веса BENDR (в репозитории пусто)
results/                метрики, рисунки, чекпоинты
TRANSFER.md             как перенести результаты с CPU и продолжить на GPU
```

Заглушки `scripts/train.py`, `evaluate.py` и `run_lopo_cv.py` заменены обёртками
над пакетом `bendr_seizure`. `scripts/download_data.py` направляет к
`download_chbmit.py`. Для новых экспериментов используйте конфиги
`cpu_quick.yaml`, `cpu_probe.yaml` или `gpu_full.yaml`, а не `default.yaml`.

---

## 1. Установка окружения

Нужны Python 3.10–3.13, pip и (для полного прогона) GPU NVIDIA с CUDA.
Исходное окружение НИР-3 было на **Python 3.13.5** и `torch==2.15.0+cpu`.
На Windows, если для 3.13 ещё нет колеса PyTorch с вашей CUDA, ставьте **3.11 или 3.12**.

Команды ниже предполагают, что вы находитесь в корне репозитория.

### 1.1. Вариант с venv (Linux / macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Сначала PyTorch **под вашу CUDA** (не копируйте сборку `+cpu` из исходного окружения).
Актуальную строку возьмите на [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
Пример для CUDA 12.8 (индекс и версию проверьте на сайте, они меняются):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Проверка:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Запуск модулей без установки пакета:

```bash
export PYTHONPATH=.
python -m bendr_seizure --help
```

Либо редактируемая установка:

```bash
pip install -e .
```

### 1.2. Вариант с venv (Windows, PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
$env:PYTHONPATH="."
python -m bendr_seizure --help
```

Если политика выполнения скриптов запрещает `Activate.ps1`:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

В cmd.exe активация: `.venv\Scripts\activate.bat`.

### 1.3. Вариант с conda

```bash
conda env create -f environment.yml
conda activate bendr-seizure
# затем отдельно torch с CUDA, как на pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

В `environment.yml` указан `python=3.13`. На Windows при необходимости замените на `python=3.12`.

---

## 2. Данные CHB-MIT

Источник: [https://physionet.org/content/chbmit/1.0.0/](https://physionet.org/content/chbmit/1.0.0/)

По карточке PhysioNet несжатый объём полного набора — **42,6 ГБ** (около 43 ГБ на диске).
Файлы лежат по адресу `https://physionet.org/files/chbmit/1.0.0/`.
Конвейер ожидает структуру:

```
data/chbmit/
  RECORDS
  RECORDS-WITH-SEIZURES
  SUBJECT-INFO
  chb01/chb01-summary.txt
  chb01/chb01_01.edf
  chb01/chb01_03.edf
  chb01/chb01_03.edf.seizures
  ...
```

### 2.1. Полный набор через wget (Linux)

```bash
wget -r -N -c -np -nH --cut-dirs=3 -P data/chbmit https://physionet.org/files/chbmit/1.0.0/
```

Флаги: `-r` рекурсия, `-N` не качать более новый локальный файл, `-c` докачка, `-np` не подниматься выше каталога, `-nH --cut-dirs=3` убирает префикс `physionet.org/files/chbmit/1.0.0`.

На Windows wget можно взять из [GnuWin32](https://gnuwin32.sourceforge.net/packages/wget.htm), Git for Windows или WSL. В PowerShell 7+ иногда доступен `wget` как псевдоним `Invoke-WebRequest` — **этот псевдоним не подходит**, используйте `wget.exe` или скрипт ниже.

### 2.2. Выбранные пациенты (Python, Linux и Windows)

Сначала индекс, затем нужные папки (пример — три пациента, как в CPU-прогоне):

```bash
python scripts/download_chbmit.py --patients chb01 chb02 chb03
```

Все 24 случая (долго, ~43 ГБ):

```bash
python scripts/download_chbmit.py --all
```

### 2.3. Выбранные пациенты (PowerShell)

```powershell
$base = "https://physionet.org/files/chbmit/1.0.0"
$dest = "data/chbmit"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
foreach ($name in @("RECORDS","RECORDS-WITH-SEIZURES","SUBJECT-INFO","SHA256SUMS.txt","ANNOTATORS")) {
  curl.exe -L --create-dirs -o "$dest\$name" "$base/$name"
}
# Пример: только chb01. Список файлов пациента есть в RECORDS.
New-Item -ItemType Directory -Force -Path "$dest\chb01" | Out-Null
curl.exe -L -o "$dest\chb01\chb01-summary.txt" "$base/chb01/chb01-summary.txt"
# Дальше удобнее вызвать python scripts/download_chbmit.py --patients chb01
```

В индексном файле `SUBJECT-INFO` перечислены **chb01–chb23**. Папка `chb24` есть в наборе PhysioNet и добавлена позже. Исходный репозиторий использует 23 случая. В `configs/gpu_full.yaml` по умолчанию стоят chb01–chb23; чтобы добавить chb24, допишите его в `data.patients`.

---

## 3. Веса BENDR

Официальный релиз **v0.1-alpha** репозитория [SPOClab-ca/BENDR](https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha).
Прямые ссылки проверены 23.11.2026 (`curl -sI` дал HTTP 302 на `release-assets.githubusercontent.com`; размеры — из GitHub API):

| Файл | URL | Размер |
|---|---|---|
| `encoder.pt` | https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/encoder.pt | 15 893 001 байт (~16 МиБ) |
| `contextualizer.pt` | https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/contextualizer.pt | 612 693 181 байт (~585 МиБ) |

SHA-256 копий, использованных в НИР-3:

```
encoder.pt         32c913b1778d8bdf7f08bbe8ef5e6bef2954087b482851816deefc4d0fcf84cb
contextualizer.pt  b8ee0c8c8403809d65114ae88485cdf841dd1860a88ebebb4cdee2bf26f91526
```

Скрипт сам скачает оба файла в `weights/` и сверит размер и хеш:

```bash
python scripts/download_weights.py
```

Вручную (Linux / Git Bash / PowerShell, нужен `curl`):

```bash
curl -L -o weights/encoder.pt https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/encoder.pt
curl -L -o weights/contextualizer.pt https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/contextualizer.pt
```

Архитектура в конфиге выровнена под эти веса: чётные ядра свёртки в коде делаются нечётными, как в официальном `ConvEncoderBENDR`. Иначе ключи `encoder.pt` не сходятся.

---

## 4. Запуск обучения

Все команды — из корня проекта, с `PYTHONPATH=.` (или после `pip install -e .`).

### Дымовой тест (EDF + прямой проход, без обучения)

```bash
python -m bendr_seizure smoke --config configs/cpu_quick.yaml
```

Нужны веса и хотя бы один пациент из списка в конфиге (для `cpu_quick.yaml` — `chb01`).

### Синтетическая проверка цикла обучения

```bash
python -m bendr_seizure mock --config configs/cpu_quick.yaml --epochs 2
```

Данные CHB-MIT для `mock` не нужны, веса — нужны.

### LOPO на CPU (замороженный энкодер, только голова)

```bash
python -m bendr_seizure train --config configs/cpu_quick.yaml
# то же самое:
python scripts/train.py --config configs/cpu_quick.yaml
```

### LOPO на GPU (полное дообучение, параметры диссертации)

```bash
python -m bendr_seizure train --config configs/gpu_full.yaml --device cuda
```

Устройство по умолчанию в обоих конфигах — `auto` (CUDA, если есть, иначе CPU).
Явно задать можно так: `--device cpu` или `--device cuda`.

Переопределить число эпох, не правя YAML:

```bash
python -m bendr_seizure train --config configs/gpu_full.yaml --epochs 40
```

### Диагностика вариантов признаков и голова с порогом Youden

На CPU это основной сценарий попытки 3: четыре варианта признаков, две нормы,
кэш на диск, порог Юдена на валидации, голосование и группировка ЛС.

```bash
python -m bendr_seizure diagnose --config configs/cpu_quick.yaml
# или
python scripts/diagnose_and_train.py --config configs/cpu_quick.yaml
```

Если признаки уже лежат в `results/feature_cache/*_variants.npz`, скрипт их подхватит
и не будет заново гонять энкодер.

### Доли 25/50/100 % с уже посчитанного кэша (один вариант признаков)

```bash
python -m bendr_seizure fractions --config configs/cpu_quick.yaml --fractions 0.25,0.5,1.0
python scripts/run_fractions_from_cache.py --config configs/cpu_quick.yaml
```

Кэш формата `{patient}_fold.npz` (первый токен) или, для `diagnose`,
`{patient}_{norm}_variants.npz`. См. `TRANSFER.md`.

Ожидаемое время. На CPU полный прогон по `gpu_full.yaml` (23 пациента, 120 эпох, 157 млн параметров) практически нереален. `cpu_quick.yaml` считает признаки BENDR один раз и учит только линейную голову (25 эпох, 3 пациента). На GPU с batch 256 полный LOPO по 23 пациентам и трём долям данных (25/50/100 %) занимает много часов; имеет смысл начать с доли 100 % и двух–трёх пациентов, затем расширить список в конфиге. Замороженный зонд на CPU не перенёсся между пациентами (см. раздел 8): на GPU стоит пробовать полное дообучение, а не те же замороженные признаки.


---

## 5. Только инференс / оценка с постобработкой

Обучение не запускается. Загружается чекпоинт, считаются метрики до и после
мажоритарного голосования и группировки ложных тревог, пишутся JSON, CSV и рисунки.

Один файл:

```bash
python -m bendr_seizure eval --config configs/gpu_full.yaml --checkpoint results/checkpoints/chb01_frac100.pt
# или
python scripts/evaluate.py --config configs/gpu_full.yaml --checkpoint results/checkpoints/chb01_frac100.pt
```

Все `*.pt` из `paths.checkpoints_dir`:

```bash
python -m bendr_seizure eval --config configs/gpu_full.yaml
```

Имя файла вида `chb01_frac100.pt` задаёт тестового пациента и долю данных.
Если имя другое, укажите `--patient chb01`. Новые чекпоинты этого пакета хранят
статистики z-score. У старых CPU-чекпоинтов их может не быть: тогда нормализатор
пересчитывается по остальным пациентам из конфига (нужны их записи и тот же `seed`).

Для оценки зонда (`mode: head_only`) передавайте тот же конфиг, которым учили
(например `configs/cpu_quick.yaml`). Для полного дообучения — `gpu_full.yaml`.

---

## 6. Основные параметры конфига

Править удобнее всего YAML в `configs/`. Ниже — ключи, которые обычно меняют.

| Ключ | Смысл |
|---|---|
| `device` | `auto`, `cpu` или `cuda` |
| `paths.raw_dir` | каталог CHB-MIT (`data/chbmit`) |
| `paths.weights_dir` | каталог `encoder.pt` / `contextualizer.pt` |
| `paths.results_dir`, `paths.checkpoints_dir`, `paths.experiment_dir` | куда писать результаты |
| `data.patients` | список фолдов LOPO, например `[chb01, chb02, chb03]` |
| `data.max_background_per_file`, `data.max_background_files` | усечение фона на обучении (`null` — без усечения) |
| `data.test_max_background_*` | то же для теста (для честного FP/ч лучше `null`) |
| `data.filter.low_hz` / `high_hz` / `order` | Баттерворт, по диссертации 0,5–50 Гц, порядок 5 |
| `data.window_sec` | длина окна, 4 с (1024 отсчёта при 256 Гц) |
| `loss.alpha`, `loss.beta` | веса SSWCE, по диссертации 0,3 и 0,5 |
| `training.lr` | шаг Adam: `1e-4` для полного дообучения, `1e-3` в CPU-зонде (выбор автора) |
| `training.weight_decay` | 0,01 |
| `training.max_epochs` | верхняя граница: 120 в диссертации, 25 в CPU-зонде |
| `training.batch_size` | 256 (GPU) |
| `training.batch_size_cpu` | размер пакета на CPU |
| `training.early_stopping_patience` | 15 в диссертации, 8 в CPU-зонде |
| `training.freeze_encoder`, `training.freeze_contextualizer` | заморозка блоков BENDR |
| `training.mode` | `full` — учим всю сеть; `head_only` — признаки один раз, учим голову |
| `training.data_fractions` | доли обучающей выборки, например `[0.25, 0.5, 1.0]` |
| `training.lopo.val_fraction` | доля валидации внутри обучающих пациентов, 0,2 |
| `feature_variant` | представление замороженных признаков: `ctx_first`, `ctx_last`, `ctx_mean`, `enc_mean` (только `head_only` / `diagnose`) |
| `norm` | `global` — z-score по обучающей выборке (диссертация); `window` — z-score внутри каждого окна (выбор автора) |
| `postprocessing.threshold.mode` | `argmax` (порог 0,5) или `youden` (порог по валидации, выбор автора) |
| `features.cache_dir` | каталог кэша признаков, по умолчанию `results/feature_cache` |
| `postprocessing.smoothing.window` | нечётная ширина мажоритарного голосования, 3 |

Доли 25/50/100 % относятся к **объёму обучающей выборки**, а не к «урезанной» архитектуре BENDR.
Предобученные веса соответствуют полной модели (6 свёрточных блоков, 8 слоёв, 8 голов). Масштабы 25 % / 55 % из исходного репозитория здесь не используются: под них нет этих весов.

---

## 7. Что появляется в `results/`

После `train` / `lopo`:

```
results/
  <experiment_dir>/
    loss_<пациент>_frac<доля>.png     кривые функции потерь
    cm_<пациент>_frac<доля>.png       матрица ошибок (после голосования)
    metrics_<пациент>_frac<доля>.json история эпох и метрики raw/smooth
    lopo_results.csv / .md            таблица по всем фолдам
    lopo_summary.csv / .md            средние по долям данных
  checkpoints/
    <пациент>_frac<доля>.pt           лучший по val_loss чекпоинт фолда
    <пациент>_frac<доля>_<вариант>_<норма>.pt   чекпоинты головы из diagnose
  feature_cache/
    <пациент>_fold.npz                кэш одного варианта (обычно первый токен)
    <пациент>_<норма>_variants.npz    четыре варианта + train/val/test
  lopo_probe_variants/                таблицы и рисунки команды diagnose
  mock/                               выход команды mock
```

После `eval` в каталоге эксперимента появляются `eval_results.csv`, `metrics_*.json` и `cm_*.png`.
В JSON для каждого фолда есть блоки `raw` (до постобработки) и `smooth` (после голосования):
чувствительность, специфичность, FP/ч, сгруппированные FP/ч, событийная чувствительность.

Чекпоинт (этот пакет) содержит `model`, `val_loss`, `epoch`, статистики нормализатора,
имя тестового пациента, долю данных и режим (`full` / `head_only`).
Оптимизатор в файл **не пишется**, поэтому «продолжить эпоху N+1» из старого Adam-состояния нельзя: либо оценка, либо обучение заново с предобутых весов BENDR (см. `TRANSFER.md`).

---


## 8. Результаты на CPU и что проверить на GPU

Кратко по журналу `results/EXPERIMENT_ATTEMPTS.md` (прогоны на chb01–chb03, 23.11.2026).
Цифры не приукрашены: замороженный BENDR на этом подмножестве **не дал рабочего детектора**.

**Попытка 1.** Полное дообучение всех ~157 млн параметров, 3 эпохи, усечённый фон.
Классификатор схлопнулся (почти всегда «приступ» или всегда «фон»). Тестовая выборка
была искусственно обогащена приступами, формальные 900 FP/ч — это срабатывание
на каждом 4-секундном окне. Результаты в `lopo_run1_degenerate/`, в итоговые
таблицы отчёта их ставить не нужно.

**Попытка 2.** Заморожены энкодер и контекстуализатор, училась только голова
на первом токене (`ctx_first`). Сам токен оказался почти константой
(стандартное отклонение по осям 1e-6…1e-5, косинус между окнами ≈ 1).
LogisticRegression: val AUC 0,486, test AUC 0,543 — уровень случая.
Голова SSWCE схлопывалась в константный класс. Больше эпох на этом признаке
бесполезны. В `main.py` исправлена ошибка `del model, trainer` в режиме head-only.

**Попытка 3 (итог CPU).** В одном проходе считались четыре варианта
(`ctx_first`, `ctx_last`, `ctx_mean`, `enc_mean`) и две нормы (глобальный z-score
и z-score внутри окна). Лучшее на валидации: **window + `ctx_mean`**
(val AUC 0,677 / 0,615 / 0,628 на chb01 / chb02 / chb03). Первый токен
по-прежнему почти константа (val AUC ≈ 0,49–0,55).

На отложенном пациенте те же признаки дают **test AUC ≈ 0,55 / 0,49 / 0,54** —
снова уровень случая. Линейная голова SSWCE+WRS на 25 % и 50 % данных
схлопнулась (порог Youden t=0, всегда «приступ», spec=0, FP/ч=900).
На 100 % у chb02 и chb03 порог t=0,51 даёт ненулевую специфичность,
но ранжирование на чужом пациенте не работает. Вывод: **замороженные признаки
официального BENDR линейно не переносятся между пациентами CHB-MIT**
на этом подмножестве. Доля 100 % выбрана только как наименее вырожденная,
не как приемлемое качество.

**Что имеет смысл проверить на GPU.** Не продолжать зонд на тех же замороженных
признаках в расчёте на «ещё эпох». Имеет смысл **полное дообучение**
(`configs/gpu_full.yaml`: энкодер и контекстуализатор разморожены, `mode: full`,
до 120 эпох, batch 256, LOPO по 23 пациентам, доли 25/50/100 %).
Если видеопамяти не хватит — уменьшить batch и пометить это в отчёте как
отступление от диссертации. Кэш признаков и чекпоинты головы с CPU можно
перенести и заново оценить (см. `TRANSFER.md`), но ждать от них рабочего
детектора не стоит.

Таблицы попытки 3: `results/lopo_probe_variants/compact_before_after.csv`.

## 9. Замечания

- Чтение EDF идёт через `mne`. Пакет `pyedflib` не обязателен.
- Пакет `dn3` не используется: модель в `bendr_seizure/model.py` совместима с официальными весами напрямую.
- Баланс классов на обучении — `WeightedRandomSampler` (веса, обратные частоте класса). В журнале печатается фактическое соотношение приступ/фон для train/val/test.
- Постобработка: мажоритарное голосование в окне нечётной длины и группировка подряд идущих ложных тревог.
- Метрика FP/ч: `(1 − specificity) · 3600 / window_sec`. Для окна 4 с множитель равен 900. Имеет смысл только если тестовый фон не усечён искусственно.

Если что-то из данных или гиперпараметров в исходных материалах не зафиксировано, в конфиге это не выдумано, а помечено «# выбрано автором».
