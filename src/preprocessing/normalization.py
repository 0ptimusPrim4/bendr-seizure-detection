"""Нормализация окон ЭЭГ без утечки из валидации и теста.

Статистики считаются методом ``fit`` только по обучающим окнам. ``transform``
применяет уже запомненные значения и сам ничего не оценивает заново.
По каждому каналу отдельно берутся среднее и стандартное отклонение
(z-score) либо минимум и максимум (min-max). Метод задаётся
``normalization.method``.
"""

from __future__ import annotations

import numpy as np

from src.config_lookup import require_section


class NormalizerError(RuntimeError):
    """Нормализатор ещё не обучен или конфиг ему не подходит."""


class EEGNormalizer:
    """Нормализатор, запоминающий статистики обучающей выборки.

    Ожидаемая форма входа: ``[N, C, T]`` или ``[C, T]``. Статистики имеют
    форму ``[C]`` и вычитаются по оси каналов.
    """

    def __init__(self, config: dict) -> None:
        section = require_section(config, "normalization")
        method = str(section["method"]).strip().lower()
        if method not in {"z-score", "min-max"}:
            raise NormalizerError(
                f"normalization.method должен быть «z-score» или «min-max», получено: {method}."
            )
        fit_on = str(section.get("fit_on", "train")).strip().lower()
        if fit_on != "train":
            raise NormalizerError(
                "Статистики нормализации считаются только по обучающей выборке "
                f"(fit_on: train). В конфиге указано: {fit_on}."
            )
        self.method = method
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        """Признак того, что ``fit`` уже вызывался."""
        return self.mean_ is not None or self.min_ is not None

    def fit(self, x_train: np.ndarray) -> EEGNormalizer:
        """Оценить статистики по обучающим окнам.

        Args:
            x_train: Окна обучающего фолда, ``[N, C, T]`` или ``[C, T]``.

        Returns:
            Этот же нормализатор.
        """
        values = self._as_channels(x_train)
        channel_axis = values.reshape(values.shape[0], -1)
        if self.method == "z-score":
            self.mean_ = channel_axis.mean(axis=1).astype(np.float64)
            std = channel_axis.std(axis=1).astype(np.float64)
            std[std == 0.0] = 1.0
            self.std_ = std
            self.min_ = None
            self.max_ = None
        else:
            self.min_ = channel_axis.min(axis=1).astype(np.float64)
            maximum = channel_axis.max(axis=1).astype(np.float64)
            span = maximum - self.min_
            span[span == 0.0] = 1.0
            self.max_ = self.min_ + span
            self.mean_ = None
            self.std_ = None
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Применить статистики обучающей выборки.

        Args:
            data: Окна любой выборки той же размерности каналов, что и при ``fit``.

        Returns:
            Новый массив ``float32``. Вход не изменяется.

        Raises:
            NormalizerError: ``fit`` ещё не вызывался или число каналов другое.
        """
        if not self.is_fitted:
            raise NormalizerError("Сначала вызовите fit на обучающих окнах.")
        original_ndim = np.asarray(data).ndim
        values = self._as_channels(data)
        n_channels = values.shape[0]
        if self.method == "z-score":
            if self.mean_ is None or self.std_ is None:
                raise NormalizerError("Статистики z-score не запомнены.")
            self._check_channels(n_channels, len(self.mean_))
            centered = values - self._column(self.mean_, values.ndim)
            scaled = centered / self._column(self.std_, values.ndim)
        else:
            if self.min_ is None or self.max_ is None:
                raise NormalizerError("Статистики min-max не запомнены.")
            self._check_channels(n_channels, len(self.min_))
            shifted = values - self._column(self.min_, values.ndim)
            scaled = shifted / self._column(self.max_ - self.min_, values.ndim)
        if original_ndim == 3:
            scaled = np.moveaxis(scaled, 0, 1)
        return np.ascontiguousarray(scaled, dtype=np.float32)

    def fit_transform(self, x_train: np.ndarray) -> np.ndarray:
        """Оценить статистики и сразу преобразовать обучающие окна."""
        return self.fit(x_train).transform(x_train)

    def _as_channels(self, data: np.ndarray) -> np.ndarray:
        values = np.asarray(data, dtype=np.float64)
        if values.ndim == 2:
            arranged = values
        elif values.ndim == 3:
            if values.shape[0] == 0:
                raise NormalizerError("В выборке нет окон.")
            arranged = np.moveaxis(values, 1, 0)
        else:
            raise NormalizerError(
                f"Ожидалась форма [N, C, T] или [C, T], получено ndim={values.ndim}."
            )
        if arranged.shape[0] == 0 or arranged.shape[1] == 0:
            raise NormalizerError("В выборке нет каналов или отсчётов.")
        return arranged

    def _column(self, stats: np.ndarray, ndim_after_move: int) -> np.ndarray:
        if ndim_after_move == 2:
            return stats[:, np.newaxis]
        return stats[:, np.newaxis, np.newaxis]

    def _check_channels(self, actual: int, expected: int) -> None:
        if actual != expected:
            raise NormalizerError(
                f"В transform {actual} каналов, а fit запомнил {expected}."
            )
