"""Цикл тонкой настройки и ранняя остановка.

``EarlyStopping`` следит за ``val_loss``: терпение, минимальное улучшение
и имя метрики читаются из раздела ``training``. Сам цикл Adam и
планировщик ReduceLROnPlateau подключаются вместе с обучением фолдов.
"""

from __future__ import annotations


class EarlyStopping:
    """Останавливает обучение, когда контролируемая метрика перестаёт убывать.

    Первое значение запоминается как лучшее. Дальше улучшением считается
    уменьшение больше чем на ``early_stopping_min_delta``. После
    ``early_stopping_patience`` эпох без такого улучшения ``step`` возвращает
    ``True``.
    """

    def __init__(self, config: dict) -> None:
        section = _training_section(config)
        self.monitor = str(section["early_stopping_monitor"])
        self.patience = int(section["early_stopping_patience"])
        self.min_delta = float(section["early_stopping_min_delta"])
        if self.patience < 1:
            raise ValueError(f"early_stopping_patience должен быть не меньше 1, получено: {self.patience}.")
        if self.min_delta < 0:
            raise ValueError(f"early_stopping_min_delta не может быть отрицательным, получено: {self.min_delta}.")
        self.best: float | None = None
        self.epochs_without_improvement = 0
        self.should_stop = False

    def step(self, metrics: dict[str, float]) -> bool:
        """Учесть метрики эпохи и сообщить, пора ли остановиться.

        Args:
            metrics: Словарь метрик эпохи. Обязательно содержит ``monitor``.

        Returns:
            ``True``, если число эпох без улучшения достигло терпения.
        """
        if self.monitor not in metrics:
            raise KeyError(f"В метриках эпохи нет «{self.monitor}».")
        value = float(metrics[self.monitor])
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        self.should_stop = self.epochs_without_improvement >= self.patience
        return self.should_stop


def _training_section(config: dict) -> dict:
    if not isinstance(config, dict):
        raise TypeError("Конфиг ранней остановки должен быть словарём.")
    if "early_stopping_patience" in config and "early_stopping_min_delta" in config:
        return config
    training = config.get("training")
    if isinstance(training, dict):
        return training
    raise KeyError("В конфиге нет раздела training с параметрами ранней остановки.")
