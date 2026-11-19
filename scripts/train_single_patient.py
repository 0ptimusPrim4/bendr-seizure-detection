"""Короткий прогон обучения на одном пациенте.

Проверяет, что окна, модель, лосс и шаг оптимизатора сходятся.
Полный LOPO-CV здесь не запускается.

    python scripts/train_single_patient.py --patient chb01
"""

# NOTE: это тестовый прогон на 2 эпохи, для полного LOPO-CV см. run_lopo_cv.py

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess_patient import _jobs, _locate_summary, _resolve_raw_dir
from src.models.bendr import BENDRClassifier, PretrainedLoadError
from src.preprocessing.channel_mapping import MappingError, map_channels
from src.preprocessing.edf_loader import EdfLoadError, load_edf
from src.preprocessing.filtering import FilterError, bandpass_filter
from src.preprocessing.normalization import Normalizer, NormalizerError
from src.preprocessing.segmentation import SegmentationError, make_windows
from src.training.trainer import Trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description="Тест обучения BENDR на одном пациенте.")
    parser.add_argument("--patient", default="chb01", help="Например chb01.")
    parser.add_argument("--data-root", default=None, help="Каталог с пациентами. По умолчанию paths.raw_dir.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--epochs", type=int, default=2, help="2 или 3. Дольше этот скрипт не учит.")
    args = parser.parse_args(argv)

    if not 1 <= args.epochs <= 3:
        print("Это тестовый прогон: эпох может быть от 1 до 3.", file=sys.stderr)
        return 1

    try:
        config = _load_config(Path(args.config))
        raw_dir = _resolve_raw_dir(args.data_root, config)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Не удалось прочитать конфиг: {exc}", file=sys.stderr)
        return 1

    patient = args.patient.strip().lower()
    summary_path = _locate_summary(raw_dir, patient)
    if summary_path is None:
        print(
            f"Сводка не найдена для {patient}. Ожидался файл "
            f"{raw_dir / patient / (patient + '-summary.txt')}.",
            file=sys.stderr,
        )
        return 2

    try:
        windows, labels = _patient_windows(summary_path, raw_dir, patient, config)
        train_idx, val_idx = _split_indices(labels, config)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Ошибка подготовки данных: {exc}", file=sys.stderr)
        return 1

    # FIXME: fit только на train, иначе val увидит своё же среднее
    normalizer = Normalizer(config)
    try:
        x_train = normalizer.fit_transform(windows[train_idx])
        x_val = normalizer.transform(windows[val_idx])
    except NormalizerError as exc:
        print(f"Ошибка нормализации: {exc}", file=sys.stderr)
        return 1
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    print(f"Пациент: {patient}", flush=True)
    print(f"Окна train [N, C, T]: {list(x_train.shape)}, приступов {int((y_train == 1).sum())}", flush=True)
    print(f"Окна val [N, C, T]: {list(x_val.shape)}, приступов {int((y_val == 1).sum())}", flush=True)

    torch.manual_seed(int(config["seed"]))
    model = BENDRClassifier(config)
    _try_pretrained(model, config)
    trainer = Trainer(model, config)
    batch_size = _batch_size(config)
    print(f"Устройство: {trainer.device}, батч: {batch_size}", flush=True)

    checkpoint = _checkpoint_path(config, patient)
    grad_norm = None
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _, seen_grad = _pass(
            trainer,
            _batches(x_train, y_train, batch_size, shuffle=True, seed=int(config["seed"]) + epoch),
            train=True,
            watch_grad=grad_norm is None,
        )
        if seen_grad is not None:
            grad_norm = seen_grad
        val_loss, sensitivity, specificity, _ = _pass(
            trainer,
            _batches(x_val, y_val, batch_size, shuffle=False, seed=0),
            train=False,
            watch_grad=False,
        )
        trainer.scheduler.step(val_loss)
        trainer.save_checkpoint(checkpoint, val_loss)
        print(
            f"Epoch {epoch} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"Sensitivity {_fmt(sensitivity)} | Specificity {_fmt(specificity)}",
            flush=True,
        )

    if grad_norm is None or grad_norm == 0.0:
        print("Градиент головы нулевой, обучение по факту не идёт.", file=sys.stderr)
        return 1
    print(f"Градиент головы после первого батча: {grad_norm:.4f}")
    print(f"Чекпоинт: {checkpoint}")
    return 0


def _patient_windows(summary_path, raw_dir, patient, config):
    from src.preprocessing.annotations import parse_summary

    summary = parse_summary(summary_path, patient_id=patient)
    jobs = _jobs(summary, raw_dir, patient, None)
    if not jobs:
        raise RuntimeError(f"Для {patient} не найден ни один EDF в {raw_dir / patient}.")

    target_sfreq = float(config["data"]["sampling_rate_hz"])
    windows_list = []
    labels_list = []
    for record, edf_path in jobs:
        try:
            recording = load_edf(edf_path, target_sfreq)
            mapped = map_channels(recording.data, recording.channel_names, config)
            filtered = bandpass_filter(mapped, recording.sfreq, config)
            windows, labels = make_windows(filtered, recording.sfreq, record.seizures, config)
        except (MappingError, FilterError, SegmentationError, EdfLoadError, OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(str(exc)) from exc
        if len(labels):
            windows_list.append(windows)
            labels_list.append(labels)
    if not labels_list:
        raise RuntimeError(f"У {patient} не осталось окон после предобработки.")
    return np.concatenate(windows_list, axis=0), np.concatenate(labels_list, axis=0)


def _split_indices(labels, config):
    val_fraction = float(config["training"]["lopo"]["val_fraction"])
    seed = int(config["seed"])
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction должен быть между 0 и 1, получено {val_fraction}.")
    if len(labels) < 2:
        raise ValueError("Для train/val нужно хотя бы два окна.")
    counts = np.bincount(labels.astype(int), minlength=2)
    # TODO: стратификация из LOPO. Если приступов нет, делим как получится
    if counts.min() >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
    else:
        print("В пациенте мало окон одного класса, val беру без стратификации.")
        splitter = ShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels))))
    return train_idx, val_idx


def _try_pretrained(model, config):
    weights = config.get("weights") or {}
    filename = str(weights.get("encoder_filename", "encoder.pt"))
    directory = Path(str(config["paths"]["weights_dir"]))
    if not directory.is_absolute():
        directory = ROOT / directory
    path = directory / filename
    if not path.is_file():
        print("Веса не найдены, инициализация случайная", flush=True)
        return
    try:
        model.load_pretrained(path)
    except PretrainedLoadError as exc:
        print("Веса не найдены, инициализация случайная")
        print(exc)
        return
    print(f"Загружены веса энкодера: {path}")


def _batch_size(config):
    training = config["training"]
    prefer_cuda = bool((config.get("device") or {}).get("prefer_cuda", True))
    if prefer_cuda and torch.cuda.is_available():
        return int(training["batch_size"])
    return int(training["batch_size_cpu"])


def _batches(windows, labels, batch_size, shuffle, seed):
    order = np.arange(len(labels))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        pick = order[start:start + batch_size]
        yield torch.from_numpy(np.ascontiguousarray(windows[pick])), torch.from_numpy(labels[pick].astype(np.int64))


def _pass(trainer, batches, train, watch_grad):
    trainer.model.train(train)
    total = 0.0
    seen = 0
    tp = fn = tn = fp = 0
    grad_norm = None
    for x, y in batches:
        x = x.to(trainer.device)
        y = y.to(trainer.device).long()
        if train:
            trainer.optimizer.zero_grad()
        probs = trainer.model(x)
        loss = trainer.criterion(probs, y)
        if train:
            loss.backward()
            if watch_grad and grad_norm is None and trainer.model.head.linear.weight.grad is not None:
                grad_norm = float(trainer.model.head.linear.weight.grad.detach().norm())
            trainer.optimizer.step()
        pred = probs.argmax(dim=-1)
        tp += int(((pred == 1) & (y == 1)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())
        tn += int(((pred == 0) & (y == 0)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        total += float(loss.detach()) * int(y.shape[0])
        seen += int(y.shape[0])
    if seen == 0:
        raise RuntimeError("Пустой набор окон.")
    sensitivity = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    return total / seen, sensitivity, specificity, grad_norm


def _checkpoint_path(config, patient):
    directory = Path(str(config["paths"]["checkpoints_dir"]))
    if not directory.is_absolute():
        directory = ROOT / directory
    return directory / f"test_{patient}.pt"


def _fmt(value):
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _load_config(path):
    config_path = path if path.is_absolute() else ROOT / path
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"конфиг {config_path} должен быть словарём.")
    return loaded


if __name__ == "__main__":
    sys.exit(main())
