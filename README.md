# Система автоматической детекции эпилептических приступов на основе BENDR

Магистерский проект: обнаружение эпилептических приступов по многоканальной
электроэнцефалограмме (ЭЭГ, EEG). Используется архитектура BENDR
(BErt-inspired Neural Data Representations), предобученная авторами на корпусе
TUEG (Temple University Hospital EEG Corpus). Здесь модель не обучается с нуля:
загружаются готовые веса, затем выполняется тонкая настройка (fine-tuning)
на базе CHB-MIT.

Статья: Kostas D., Aroca-Ouellette S., Rudzicz F. «BENDR: Using Transformers and a
Contrastive Self-Supervised Learning Task to Learn From Massive Amounts of EEG Data»,
Frontiers in Human Neuroscience, 2021. DOI: [10.3389/fnhum.2021.653659](https://doi.org/10.3389/fnhum.2021.653659).

## Конвейер

Сырые EDF-файлы → предварительная обработка (preprocessing) → предобученный
энкодер BENDR → тонкая настройка классификатора → постобработка (postprocessing)
→ клинические метрики.

Целевые показатели:

| Метрика | Порог |
| --- | --- |
| Чувствительность (sensitivity) | ≥ 65 % |
| Специфичность (specificity) | ≥ 99 % |
| Ложные срабатывания в час (false positives per hour, FP/h) | ≤ 1 |
| Событийная чувствительность (event-based sensitivity) | ≥ 70 % |

Все гиперпараметры лежат в `configs/default.yaml`.

## Установка

Нужен Python 3.9 или новее. Для полного обучения — PyTorch 2.0+ и GPU
с видеопамятью не меньше 16 ГБ. Чтение данных и тесты этого шага работают на CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Данные

Основной набор — [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)
(23 пациента, `chb01`–`chb23`, частота 256 Гц, формат EDF). Файлы в git не хранятся.

Ожидаемое расположение:

```text
data/raw/chb01/chb01-summary.txt
data/raw/chb01/chb01_03.edf
```

Предобученные веса энкодера и контекстуализатора публикует репозиторий
[SPOClab-ca/BENDR, релиз v0.1-alpha](https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha).
Их нужно положить в `weights/`. `BENDRClassifier.load_pretrained` подставляет
совпавшие тензоры энкодера и оставляет голову случайной.

## Структура

```text
configs/default.yaml              гиперпараметры
scripts/preprocess_patient.py     проверка сводки и полный конвейер одного пациента
src/preprocessing/                сводки, EDF, каналы, фильтр, окна, нормализация
src/models/bendr.py               BENDRClassifier: энкодер и голова
src/training/lopo_cv.py           разбиение «один пациент вне выборки»
src/training/trainer.py           ранняя остановка
```

## Проверка одного пациента

Из корня репозитория. Режим задаётся явно: `--quick`, `--edf` или `--all`.

Быстрое чтение сводки и, если файл есть, одного EDF:

```bash
python scripts/preprocess_patient.py --quick --patient chb01 --data-root data/raw
```

Скрипт печатает число файлов и приступов из `chb01-summary.txt`. Если в каталоге
есть EDF, читает первый файл с приступом, приводит частоту к 256 Гц и показывает
форму тензора `[C, T]` и имена каналов. Без EDF сводка всё равно проверяется,
а в конце будет сказано, какой файл положить рядом.

Проверить конкретную запись, не запуская фильтр и окна:

```bash
python scripts/preprocess_patient.py --quick --patient chb01 --edf data/raw/chb01/chb01_03.edf --require-edf
```

Коды возврата: `0` — запрошенное чтение удалось, `1` — ошибка режима, разбора или чтения,
`2` — сводка не найдена.

Полный конвейер одного файла (каналы, фильтр 0.5–50 Гц, окна 4 с, метки,
нормализация) печатает форму `[N, 20, 1024]` и число окон классов 0 и 1:

```bash
python scripts/preprocess_patient.py --patient chb01 --edf data/raw/chb01/chb01_03.edf --data-root data/raw
```

Тот же конвейер для всех EDF пациента, которые уже лежат на диске:

```bash
python scripts/preprocess_patient.py --all --patient chb01 --data-root data/raw
```

Тесты парсера, конвейера, классификатора и разбиения LOPO-CV:

```bash
pytest
```

## Статус

Фаза 1 закрыта: каркас репозитория, конфиг, разбор сводок, чтение EDF,
сопоставление 20 каналов, полосовой фильтр, окна с метками, нормализация
по обучающей выборке и разбиение «один пациент вне выборки»
(leave-one-patient-out, LOPO-CV). `BENDRClassifier` уже принимает окно
`[B, 20, 1024]` и возвращает вероятности двух классов; ранняя остановка
живёт в `src/training/trainer.py`. Цикл обучения, постобработка и ноутбук
анализа впереди.
