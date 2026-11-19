"""Одна эпоха тонкой настройки и ранняя остановка по val_loss."""

from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.training.loss import SensitivitySpecificityWeightedCrossEntropy


class EarlyStopping:
    """Считает эпохи без улучшения val_loss. patience и min_delta из конфига."""

    def __init__(self, config):
        section = _training_section(config)
        self.monitor = str(section["early_stopping_monitor"])
        self.patience = int(section["early_stopping_patience"])
        self.min_delta = float(section["early_stopping_min_delta"])
        if self.patience < 1:
            raise ValueError(f"early_stopping_patience должен быть не меньше 1, получено: {self.patience}.")
        if self.min_delta < 0:
            raise ValueError(f"early_stopping_min_delta не может быть отрицательным, получено: {self.min_delta}.")
        self.best = None
        self.epochs_without_improvement = 0
        self.should_stop = False

    def step(self, metrics):
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


class Trainer:
    """Adam, ReduceLROnPlateau и сохранение лучшего чекпоинта."""

    def __init__(self, model, config):
        self.config = config
        self.device = _pick_device(config)
        self.model = model.to(self.device)
        train_cfg = _training_section(config)
        self.criterion = SensitivitySpecificityWeightedCrossEntropy(config)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        # NOTE: ReduceLROnPlateau.step ждёт val_loss, голый step() без числа ему нельзя
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=str(train_cfg.get("scheduler_mode", "min")),
            factor=float(train_cfg["scheduler_factor"]),
            patience=int(train_cfg["scheduler_patience"]),
        )
        self.early_stopping = EarlyStopping(config)

    def run_epoch(self, train_batches, val_batches, checkpoint_path):
        """Один проход train, потом val. Возвращает (train_loss, val_loss, надо_ли_стоп)."""
        train_loss = self._run(train_batches, train=True)
        val_loss = self._run(val_batches, train=False)
        self.scheduler.step(val_loss)
        stop = self.early_stopping.step({"val_loss": val_loss})
        if self.early_stopping.epochs_without_improvement == 0:
            self.save_checkpoint(checkpoint_path, val_loss)
        return train_loss, val_loss, stop

    def save_checkpoint(self, path, val_loss):
        # TODO: оптимизатор в чекпоинт не кладём, продолжить обучение с него пока нельзя
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "val_loss": float(val_loss)}, path)

    def _run(self, batches, train):
        self.model.train(train)
        total = 0.0
        seen = 0
        for batch in batches:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device).long()
            if train:
                self.optimizer.zero_grad()
            probs = self.model(x)
            loss = self.criterion(probs, y)
            if train:
                loss.backward()
                self.optimizer.step()
            batch_size = int(y.shape[0])
            total += float(loss.detach()) * batch_size
            seen += batch_size
        if seen == 0:
            raise RuntimeError("В эпохе нет батчей.")
        return total / seen


def _training_section(config):
    if not isinstance(config, dict):
        raise TypeError("Конфиг обучения должен быть словарём.")
    if "early_stopping_patience" in config:
        return config
    training = config.get("training")
    if isinstance(training, dict):
        return training
    raise KeyError("В конфиге нет раздела training.")


def _pick_device(config):
    # FIXME: batch_size_cpu в конфиге есть, но нарезку батча делает вызывающий код, не тренер
    prefer_cuda = True
    device_cfg = config.get("device")
    if isinstance(device_cfg, dict):
        prefer_cuda = bool(device_cfg.get("prefer_cuda", True))
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
