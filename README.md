# Система автоматической детекции эпилептических приступов на основе BENDR

Магистерский проект: обнаружение эпилептических приступов по многоканальной
электроэнцефалограмме (ЭЭГ, EEG). Используется архитектура BENDR
(BErt-inspired Neural Data Representations), предобученная авторами на корпусе
TUEG (Temple University Hospital EEG Corpus). Здесь модель не обучается с нуля:
загружаются готовые веса, затем выполняется тонкая настройка (fine-tuning)
на базе CHB-MIT.

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

## Быстрый тест

Две эпохи на одном пациенте: EDF, каналы, фильтр, окна, нормализация по train,
случайная голова и шаг Adam. Если `weights/encoder.pt` нет, скрипт пишет
«Веса не найдены, инициализация случайная» и всё равно учится. Без GPU берётся
`batch_size_cpu` из конфига.

```bash
python scripts/train_single_patient.py --patient chb01
```

В конце каждой эпохи строка вида `Epoch 1 | Train Loss ... | Val Loss ... | Sensitivity ... | Specificity ...`.
Чекпоинт появляется в `checkpoints/test_chb01.pt`. Это не полный LOPO-CV:
тот вход ещё в `scripts/run_lopo_cv.py`.

## Статус

Фаза 1 закрыта: каркас репозитория, конфиг, разбор сводок, чтение EDF,
сопоставление 20 каналов, полосовой фильтр, окна с метками, нормализация
по обучающей выборке и разбиение «один пациент вне выборки»
(leave-one-patient-out, LOPO-CV). Модель, лосс и короткий прогон на одном
пациенте есть. Постобработка и полный LOPO-CV впереди.
