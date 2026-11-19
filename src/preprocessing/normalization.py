"""Нормализация окон. Статистики только с train, иначе в тест утекает среднее."""

import numpy as np

from src.config_lookup import require_section


class NormalizerError(RuntimeError):
    """Нормализатор ещё не обучен или конфиг ему не подходит."""


class Normalizer:
    """z-score или min-max по каждому каналу."""

    def __init__(self, config):
        section = require_section(config, "normalization")
        method = str(section["method"]).strip().lower()
        if method not in {"z-score", "min-max"}:
            raise NormalizerError(
                f"normalization.method должен быть «z-score» или «min-max», получено: {method}."
            )
        fit_on = str(section.get("fit_on", "train")).strip().lower()
        if fit_on != "train":
            # NOTE: fit_on: train — единственный допустимый режим, val/test не смотрим
            raise NormalizerError(
                "Статистики нормализации считаются только по обучающей выборке "
                f"(fit_on: train). В конфиге указано: {fit_on}."
            )
        self.method = method
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None

    @property
    def is_fitted(self):
        return self.mean_ is not None or self.min_ is not None

    def fit(self, x_train):
        """Считает статистики по обучающим окнам. Возвращает self."""
        values = self._as_channels(x_train)
        flat = values.reshape(values.shape[0], -1)
        if self.method == "z-score":
            self.mean_ = flat.mean(axis=1).astype(np.float64)
            std = flat.std(axis=1).astype(np.float64)
            # TODO: нулевой std заменяем на 1, чтобы не делить на ноль. В диссертации этого нет
            std[std == 0.0] = 1.0
            self.std_ = std
            self.min_ = None
            self.max_ = None
        else:
            self.min_ = flat.min(axis=1).astype(np.float64)
            span = flat.max(axis=1).astype(np.float64) - self.min_
            # FIXME: константный канал получает span 1, min-max его почти не двигает
            span[span == 0.0] = 1.0
            self.max_ = self.min_ + span
            self.mean_ = None
            self.std_ = None
        return self

    def transform(self, data):
        """Применяет уже посчитанные статистики. Сам ничего не оценивает."""
        if not self.is_fitted:
            raise NormalizerError("Сначала вызовите fit на обучающих окнах.")
        original_ndim = np.asarray(data).ndim
        values = self._as_channels(data)
        if self.method == "z-score":
            if self.mean_ is None or self.std_ is None:
                raise NormalizerError("Статистики z-score не запомнены.")
            self._check_channels(values.shape[0], len(self.mean_))
            scaled = (values - self._column(self.mean_, values.ndim)) / self._column(self.std_, values.ndim)
        else:
            if self.min_ is None or self.max_ is None:
                raise NormalizerError("Статистики min-max не запомнены.")
            self._check_channels(values.shape[0], len(self.min_))
            scaled = (values - self._column(self.min_, values.ndim)) / self._column(self.max_ - self.min_, values.ndim)
        # hack: канал уезжал на ось 0 ради broadcast, для [N, C, T] возвращаем назад
        if original_ndim == 3:
            scaled = np.moveaxis(scaled, 0, 1)
        return np.ascontiguousarray(scaled, dtype=np.float32)

    def fit_transform(self, x_train):
        return self.fit(x_train).transform(x_train)

    def _as_channels(self, data):
        values = np.asarray(data, dtype=np.float64)
        if values.ndim == 2:
            arranged = values
        elif values.ndim == 3:
            if values.shape[0] == 0:
                raise NormalizerError("В выборке нет окон.")
            arranged = np.moveaxis(values, 1, 0)
        else:
            raise NormalizerError(f"Ожидалась форма [N, C, T] или [C, T], получено ndim={values.ndim}.")
        if arranged.shape[0] == 0 or arranged.shape[1] == 0:
            raise NormalizerError("В выборке нет каналов или отсчётов.")
        return arranged

    def _column(self, stats, ndim_after_move):
        if ndim_after_move == 2:
            return stats[:, np.newaxis]
        return stats[:, np.newaxis, np.newaxis]

    def _check_channels(self, actual, expected):
        if actual != expected:
            raise NormalizerError(f"В transform {actual} каналов, а fit запомнил {expected}.")
