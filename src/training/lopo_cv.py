"""Перекрёстная проверка «один пациент вне выборки» (LOPO-CV).

Для каждого пациента тест — все его окна. Окна остальных пациентов делятся
на обучение и валидацию. При ``lopo.stratify: true`` разбиение стратифицировано
по метке окна, поэтому в валидации есть и приступ, и фон. Доля валидации
берётся из ``lopo.val_fraction``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from src.config_lookup import require_section


class LopoSplitError(ValueError):
    """Собрать фолд LOPO-CV по этим пациентам и меткам нельзя."""


class WindowSubset:
    """Ссылки на окна одного фолда.

    Сами тензоры здесь не копируются. ``patient_ids[i]`` и ``local_indices[i]``
    указывают на окно ``i`` в массиве этого пациента. ``labels[i]`` — его метка.
    """

    def __init__(
        self,
        patient_ids: np.ndarray,
        local_indices: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.patient_ids = np.asarray(patient_ids)
        self.local_indices = np.asarray(local_indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        if not (len(self.patient_ids) == len(self.local_indices) == len(self.labels)):
            raise LopoSplitError("Части выборки окон имеют разную длину.")

    def __len__(self) -> int:
        return int(self.labels.shape[0])


class LopoSplit(tuple):
    """Тройка ``(train, val, test)`` и имя пациента, оставленного на тест."""

    def __new__(
        cls,
        train: WindowSubset,
        val: WindowSubset,
        test: WindowSubset,
        test_patient: str,
    ) -> LopoSplit:
        split = super().__new__(cls, (train, val, test))
        split.test_patient = test_patient
        return split

    @property
    def train(self) -> WindowSubset:
        """Обучающие окна."""
        return self[0]

    @property
    def val(self) -> WindowSubset:
        """Валидационные окна."""
        return self[1]

    @property
    def test(self) -> WindowSubset:
        """Окна пациента, оставленного вне обучения."""
        return self[2]


def generate_lopo_splits(
    patients: list[str],
    config: dict,
    labels: Mapping[str, np.ndarray],
) -> list[LopoSplit]:
    """Построить фолды, в каждом из которых один пациент остаётся тестом.

    Args:
        patients: Идентификаторы пациентов в порядке фолдов.
        config: Полный конфиг или раздел ``lopo``. Нужен также ``seed`` полного конфига.
        labels: Метки окон по пациентам, массив формы ``[N_пациента]`` из 0 и 1.

    Returns:
        Список троек ``(train, val, test)``. У каждой тройки есть поле
        ``test_patient``.

    Raises:
        LopoSplitError: Нет меток, один класс на всех остальных пациентах
            или стратифицированное разбиение невозможно.
    """
    if not patients:
        raise LopoSplitError("Список пациентов пуст.")
    section = require_section(config, "lopo")
    val_fraction = float(section["val_fraction"])
    if not 0.0 < val_fraction < 1.0:
        raise LopoSplitError(f"lopo.val_fraction должен быть между 0 и 1, получено: {val_fraction}.")
    if not bool(section.get("stratify", True)):
        raise LopoSplitError("В этом протоколе lopo.stratify должен быть true.")
    if "seed" not in config:
        raise LopoSplitError("В конфиге нет seed: без него разбиение невоспроизводимо.")
    seed = int(config["seed"])

    prepared = {patient: _labels_of(patient, labels) for patient in patients}
    folds: list[LopoSplit] = []
    for target in patients:
        others = [patient for patient in patients if patient != target]
        if not others:
            raise LopoSplitError(
                f"Для пациента {target} не осталось данных обучения: в списке только он."
            )
        train_val_ids, train_val_index, train_val_labels = _concat(others, prepared)
        train_rows, val_rows = _stratified_rows(train_val_labels, val_fraction, seed, target)
        test_index = np.arange(len(prepared[target]), dtype=np.int64)
        test_ids = np.full(len(test_index), target)
        folds.append(
            LopoSplit(
                _subset(train_val_ids, train_val_index, train_val_labels, train_rows),
                _subset(train_val_ids, train_val_index, train_val_labels, val_rows),
                WindowSubset(test_ids, test_index, prepared[target]),
                target,
            )
        )
    return folds


def collect_windows(windows_by_patient: Mapping[str, np.ndarray], subset: WindowSubset) -> np.ndarray:
    """Собрать тензор окон ``[N, C, T]`` по ссылкам выборки.

    Args:
        windows_by_patient: Окна каждого пациента формы ``[N_i, C, T]``.
        subset: Ссылки, которые вернул ``generate_lopo_splits``.

    Returns:
        Окна в том же порядке, что и ``subset.labels``.
    """
    if len(subset) == 0:
        return np.empty((0, 0, 0), dtype=np.float32)
    rows = [
        np.asarray(windows_by_patient[str(patient)][int(index)])
        for patient, index in zip(subset.patient_ids, subset.local_indices)
    ]
    return np.stack(rows, axis=0)


def _labels_of(patient: str, labels: Mapping[str, np.ndarray]) -> np.ndarray:
    if patient not in labels:
        raise LopoSplitError(f"Нет меток окон для пациента {patient}.")
    values = np.asarray(labels[patient], dtype=np.int64).reshape(-1)
    if values.size and not np.isin(values, (0, 1)).all():
        raise LopoSplitError(f"У пациента {patient} метки должны быть только 0 и 1.")
    return values


def _concat(
    patients: Sequence[str],
    prepared: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for patient in patients:
        current = prepared[patient]
        ids.append(np.full(len(current), patient))
        indices.append(np.arange(len(current), dtype=np.int64))
        labels.append(current)
    if not labels or sum(len(item) for item in labels) == 0:
        raise LopoSplitError("У остальных пациентов нет окон для обучения и валидации.")
    return np.concatenate(ids), np.concatenate(indices), np.concatenate(labels)


def _stratified_rows(
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise LopoSplitError(
            f"Фолд пациента {target}: среди остальных пациентов только один класс, "
            "стратифицированное разбиение невозможно."
        )
    smallest = int(counts.min())
    val_count = int(round(len(labels) * val_fraction))
    if smallest < 2 or val_count < len(classes) or len(labels) - val_count < len(classes):
        raise LopoSplitError(
            f"Фолд пациента {target}: слишком мало окон класса для доли валидации {val_fraction}."
        )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_rows, val_rows = next(splitter.split(np.zeros(len(labels)), labels))
    return train_rows.astype(np.int64), val_rows.astype(np.int64)


def _subset(
    patient_ids: np.ndarray,
    local_indices: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
) -> WindowSubset:
    return WindowSubset(patient_ids[rows], local_indices[rows], labels[rows])
